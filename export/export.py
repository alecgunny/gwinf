import tensorflow as tf
import torch

from deepclean_prod.nn.net import DeepClean
from mldet.net import Net as BBHNet

from exportlib.model_repository import ModelRepository
from exportlib.platform import PlatformName


tf.config.set_visible_devices([], 'GPU')

BATCH_SIZE = 1


def main(
    platform: str = "onnx",
    count: int = 1,
    kernel_stride: float = 0.002,
    fs: int = 4000
):
    repo = ModelRepository("/repo")

    # do some parsing of the deepclean export platform
    deepclean_export_kwargs = {
        "output_names": ["noise"]
    }
    url = None
    try:
        platform, url = platform.split(":", maxsplit=1)
    except ValueError:
        pass

    try:
        platform, precision = platform.split("_")
    except ValueError:
        pass
    else:
        if precision == "fp16":
            deepclean_export_kwargs["use_fp16"] = True
        deepclean_export_kwargs["url"] = url
    platform = PlatformName.__members__[platform.upper()].value

    witness_channels = {"h": 21, "l": 2}
    deepcleans = {}
    for detector, num_channels in witness_channels.items():
        arch = DeepClean(num_channels)
        arch.eval()

        model = repo.create_model(f"deepclean_{detector}")
        model.config.add_instance_group(count)
        model.export_version(
            arch,
            input_names={"witness": (BATCH_SIZE, num_channels, fs)},
            **deepclean_export_kwargs
        )
        deepcleans[detector] = model

    postprocessor = PostProcessor()
    postprocessor.eval()

    pp_model = repo.create_model("postproc", platform=PlatformName.ONNX)
    pp_model.config.add_instance_group(count=count)
    pp_model.export_version(
        postprocessor,
        input_shapes={
            "strain": (BATCH_SIZE, 2, fs),
            "noise_h": (BATCH_SIZE, fs),
            "noise_l": (BATCH_SIZE, fs)
        },
        output_names=["cleaned"]
    )

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
    bbh = BBHNet((2, fs), bbh_params)
    bbh.eval()

    bbh_model = repo.create_model("bbh", platform=PlatformName.ONNX)
    bbh_model.config.add_instance_group(count=count)
    bbh_model.export_version(
        bbh,
        input_shapes={"strain": (BATCH_SIZE, 2, fs)},
        output_names=["prob"]
    )

    ensemble = model_repository.create_model(
        "gwe2e", platform=PlatformName.ENSEMBLE
    )
    ensemble.add_streaming_inputs(
        inputs=[
            deepcleans["h"].inputs["witness"],
            deepcleans["l"].inputs["witness"],
            postprocessor.inputs["strain"]
        ],
        stream_size=int(kernel_stride*fs),
    )


    for detector, model in deepcleans.items():
        ensemble.pipe(
            model.outputs["noise"]
            postprocessor.inputs[f"noise_{detector}"],
            name=f"noise_{detector}"
        )
        ensemble.add_output(model.outputs["noise"])
    ensemble.pipe(
        postprocessor.outputs["cleaned"],
        bbh.inputs["strain"]
    )
    ensemble.add_output(bbh.outputs["prob"])
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
    parser.add_argument(
        "--kernel-stride",
        type=float,
        default=0.002,
        help="Time between frame snapshots"
    )
    # TODO: generalize by making type float,
    # including kernel-size, and mapping
    # to int after multiplying
    parser.add_argument(
        "--fs",
        type=int,
        default=4000,
        help="Samples in a frame"
    )
    flags = parser.parse_args()
    main(**vars(flags))
