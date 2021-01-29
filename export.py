import tensorflow as tf
import torch

from deepclean_prod.nn.net import DeepClean
from mldet.net import Net as BBHNet

from exportlib.model_repository import ModelRepository
from exportlib.platform import PlatformName


FS = 4000
KERNEL_STRIDE = 0.002
BATCH_SIZE = 1

class PostProcessor(torch.nn.Module):
    def forward(self, strain, noise_h, noise_l):
        # TODO: needs to add:
        #    - filtering
        #    - de-centering
        #    - any preprocessing for bbh
        noise = torch.stack([noise_h, noise_l], dim=1)
        return strain - noise


@tf.keras.utils.register_keras_serializable(name="Snapshotter")
class Snapshotter(tf.keras.layers.Layer):
  def __init__(self, size, **kwargs):
    super().__init__(**kwargs)
    self.size = size

  def build(self, input_shapes):
    if isinstance(input_shapes, tf.TensorShape):
        input_shapes = [input_shapes]

    self.snapshots = []
    for i, shape in enumerate(input_shapes):
        if shape[0] is None:
            raise ValueError
        self.snapshots.append(self.add_weight(
            name=f"snapshot_{i}",
            shape=(shape[0], shape[1], self.size),
            dtype=tf.float32,
            initializer="zeros",
            trainable=False
        ))
    self.update_size = shape[2]

  def call(self, inputs):
    if isinstance(inputs, tf.Tensor):
        inputs = [inputs]
    outputs = []
    for x, snapshot in zip(inputs, self.snapshots):
        update = tf.concat(
            [snapshot[:, :, self.update_size:], x], axis=-1
        )
        snapshot.assign(update)
        name = x.name.split(":")[0].replace("stream", "snapshot")
        outputs.append(
            tf.identity(snapshot, name=name)
        )
    return outputs

  def compute_output_shape(self, input_shapes):
    if isinstance(input_shapes, tf.TensorShape):
        input_shapes = [input_shapes]
    return [tf.TensorShape([shape[0], self.size]) for shape in input_shapes]

  def get_config(self):
    config = {"size": self.size}
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))


def main(
    platform="onnx",
    count=1
):
    streaming_inputs = []
    for stream in ["witness_h", "witness_l", "strain"]:
        streaming_inputs.append(tf.keras.Input(
        name=f"{stream}_stream",
        dtype=tf.float32,
        shape=(2 if stream == "strain" else 21, int(KERNEL_STRIDE*FS),),
        batch_size=BATCH_SIZE
    ))
    snapshots = Snapshotter(FS)(streaming_inputs)
    input_model = tf.keras.Model(
        inputs=streaming_inputs, outputs=snapshots
    )

    deepclean_h = DeepClean(21)
    deepclean_h.train(False)
    deepclean_l = DeepClean(21)
    deepclean_l.train(False)
    postprocessor = PostProcessor()

    bbh_params = {
        "filters": (3, 3, 3),
        "kernels": (8, 16, 32),
        "pooling": (4, 4, 4, 4),
        "dilations": (1, 1, 1, 1),
        "pooling_type": "max",
        "pooling_first": True,
        "bn": True,
        "linear": (64, 32),
        "dropout": 0.5,
        "weight_init": None,
        "bias_init": None
    }
    bbh = BBHNet((2, FS), bbh_params)
    bbh.train(False)

    export_kwargs = {}
    try:
        platform, precision = platform.split("_")
        if precision == "fp16":
            export_kwargs["use_fp16"] = True
    except ValueError:
        pass
    platform = PlatformName.__members__[platform.upper()].value

    # create a repository and add our models to it, using
    # designated platform for deepclean models
    repo = ModelRepository("/repo")
    deepclean_h_model = repo.create_model("deepclean_h", platform=platform)
    deepclean_l_model = repo.create_model("deepclean_l", platform=platform)
    pp_model = repo.create_model("postproc", platform=PlatformName.ONNX)
    bbh_model = repo.create_model("bbh", platform=PlatformName.ONNX)

    # create possibly multiple version of each model
    for model in repo.models.values():
        model.config.add_instance_group(count=count)

    # export a version of each model
    deepclean_h_model.export_version(
        deepclean_h,
        input_shapes={"witness": (BATCH_SIZE, 21, FS)},
        output_names=["noise"],
        **export_kwargs
    )
    deepclean_l_model.export_version(
        deepclean_l,
        input_shapes={"witness": (BATCH_SIZE, 21, FS)},
        output_names=["noise"],
        **export_kwargs
    )
    pp_model.export_version(
        postprocessor,
        input_shapes={
            "strain": (BATCH_SIZE, 2, FS),
            "noise_h": (BATCH_SIZE, FS),
            "noise_l": (BATCH_SIZE, FS)
        },
        output_names=["cleaned"]
    )
    bbh_model.export_version(
        bbh,
        input_shapes={"strain": (BATCH_SIZE, 2, FS)},
        output_names=["prob"]
    )
    # hardcoding some of the TF stuff until I have
    # support, but this will allow me to use the
    # native ensemble tools
    from tritonclient.grpc import model_config_pb2 as model_config
    tf_model_name = "snapshotter"
    config = model_config.ModelConfig(
        name=tf_model_name,
        platform="tensorflow_savedmodel",
        sequence_batching=model_config.ModelSequenceBatching(
            max_sequence_idle_microseconds=5000000,
            direct=model_config.ModelSequenceBatching.StrategyDirect()
        )
    )
    for input in streaming_inputs:
        config.input.append(model_config.ModelInput(
            name=input.name.split(":")[0],
            dims=tuple(input.shape),
            data_type=model_config.TYPE_FP32
        ))
    for n, output in enumerate(input_model.outputs):
        postfix = "" if n == 0 else f"_{n}"
        config.output.append(model_config.ModelOutput(
            name="snapshotter" + postfix,
            dims=tuple(output.shape),
            data_type=model_config.TYPE_FP32
        ))
    tf.io.gfile.makedirs(repo.path + f"/{tf_model_name}/1")
    input_model.save(repo.path + f"/{tf_model_name}/1/model.savedmodel")
    with open(repo.path + f"/{tf_model_name}/config.pbtxt", "w") as f:
        f.write(str(config))

    # reinitialize repo to include tf model
    repo = ModelRepository(repo.path)

    # build an ensemble model from these constituent parts
    ensemble = repo.create_model("gwe2e", platform=PlatformName.ENSEMBLE)
    ensemble.config.add_input(
        "witness_h",
        shape=(BATCH_SIZE, 21, int(FS*KERNEL_STRIDE)),
        dtype="float32"
    )
    ensemble.config.add_input(
        "witness_l",
        shape=(BATCH_SIZE, 21, int(FS*KERNEL_STRIDE)),
        dtype="float32"
    )
    ensemble.config.add_input(
        "strain",
        shape=(BATCH_SIZE, 2, int(FS*KERNEL_STRIDE)),
        dtype="float32"
    )
    ensemble.config.add_output(
        "noise_h",
        shape=(BATCH_SIZE, FS),
        dtype="float32"
    )
    ensemble.config.add_output(
        "noise_l",
        shape=(BATCH_SIZE, FS),
        dtype="float32"
    )
    ensemble.config.add_output(
        "prob",
        shape=(BATCH_SIZE, 1),
        dtype="float32"
    )

    ensemble.config.add_model(repo.models["snapshotter"])
    ensemble.config.add_model(repo.models["deepclean_h"])
    ensemble.config.add_model(repo.models["deepclean_l"])
    ensemble.config.add_model(repo.models["postproc"])
    ensemble.config.add_model(repo.models["bbh"])

    # pipe the inputs and outputs of these constituent
    # models between one another
    for name in ["witness_h", "witness_l", "strain"]:
        ensemble.config.pipe(f"INPUT.{name}", f"snapshotter.{name}_stream")
    inputs = ["deepclean_h.witness", "deepclean_l.witness", "postproc.strain"]
    for n, input in enumerate(inputs):
        postfix = "" if n == 0 else f"_{n}"
        ensemble.config.pipe("snapshotter.snapshotter" + postfix, input)

    ensemble.config.pipe("deepclean_h.noise", "postproc.noise_h", name="noise_h")
    ensemble.config.pipe("deepclean_l.noise", "postproc.noise_l", name="noise_l")
    ensemble.config.pipe("deepclean_h.noise", "OUTPUT.noise_h")
    ensemble.config.pipe("deepclean_l.noise", "OUTPUT.noise_l")

    ensemble.config.pipe("postproc.cleaned", "bbh.strain")
    ensemble.config.pipe("bbh.prob", "OUTPUT.prob")
    ensemble.export_version()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--platform",
        type=str,
        choices=("onnx", "trt_fp16", "trt_fp32"),
        default="onnx",
        help="Format to export deepclean models in"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of model instances to place on GPU"
    )
    flags = parser.parse_args()
    main(**vars(flags))

