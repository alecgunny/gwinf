import argparse
import logging
import time

from tritonclient import grpc as triton

from stillwater import pipe, sync_recv
from stillwater.data_generator import DummyDataGenerator, LowLatencyFrameGenerator


log = logging.getLogger()
console = logging.StreamHandler()
console.setFormatter(
    logging.Formatter("%(asctime)s\t%(message)s")
)
log.addHandler(console)
log.setLevel(logging.INFO)

def cleanup(processes):
    for process in processes:
        process.stop()
        process.join(0.5)

        try:
            process.close()
        except ValueError:
            process.terminate()
            time.sleep(0.1)
            process.close()
            print(f"Process {process.name} couldn't join")


def main(
    url: str,
    model_name: str,
    model_version: int,
    sequence_id: int,
    kernel_stride: float,
    channels: typing.Optional[typing.Dict[str, typing.List[str]]] = None,
    data_dirs: typing.Optional[typing.Dict[str, str]] = None,
    file_patterns: typing.Optional[typing.Dict[str, str]] = None,
    use_dummy: bool = False,
    num_iterations: int = 10000,
    num_warm_ups: int = 50
):
    client = StreamingInferenceClient(
        url=url,
        model_name=model_name,
        model_version=model_version,
        name="client",
        sequence_id=sequence_id
    )
    processes = [client]
    for name, x in client.inputs.items():
        if use_dummy:
            data_gen = DummyDataGenerator(
                x.shape()[1:], name, kernel_stride
            )
        else:
            data_gen = LowLatencyFrameGenerator(
                data_dirs[name],
                channels[name],
                sample_rate=4000,
                kernel_stride=kernel_stride,
                t0=None,
                file_pattern=file_patterns[name],
                name=name
            )
        pipe(data_gen, client)
        processes.append(data_gen)

    out_pipes = {}
    for output in client.outputs:
        out_pipes[name] = pipe(client, output.name())

    for process in processes:
        process.start()

    packages = 0
    log.info(f"Warming up for {num_warm_ups} batches")
    for i in range(num_warm_ups):
        try:
            packages = sync_recv(out_pipes, timeout=1)
        except Exception:
            cleanup(processes)
            raise

        if packages is None:
            time.sleep(0.1)
            continue
        packages += 1

    if packages == 0:
        cleanup(processes)
        raise RuntimeError("Nothing ever showed up!")
    log.info(f"Warmed up with {packages}")

    average_latency = 0
    for i in range(num_iterations):
        try:
            package = sync_recv(out_pipes, timeout=0.1)
        except Exception:
            cleanup(processes)
            raise

        if packages is None:
            continue

        latency, throughput = package.t0
        average_latency += (latency - average_latency) / (i + 1)

        msg = "Average latency: {} us, Average Throughput: {} frames/s".format(
            int(average_latency * 10**6), throughput
        )
        if i < (num_iterations - 1):
            print(msg, end="\r", flush=True)
        else:
            log.info(msg)

    # spin everything down
    cleanup(processes)

    # report on how individual models in the
    # ensemble did
    client = triton.InferenceServerClient(url)
    inference_stats = client.get_inference_statistics().model_stats
    for stat in inference_stats:
        name = stat.name.upper()
        for field, data in stat.inference_stats.ListFields():
            if field.name == "fail":
                continue

            field = field.name
            time = int(data.ns / data.count / 1000)
            msg = f"{name}\tAverage {field} time: {time}"
            logging.info(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    client_parser = parser.add_argument_group(
        title="Client",
        help=(
            "Arguments for instantiation the Triton "
            "client instance"
        )
    )
    client_parser.add_argument(
        "--url",
        type=str,
        default="localhost:8001",
        help="Server URL"
    )
    client_parser.add_argument(
        "--model-name",
        type=str,
        default="gwe2e",
        help="Name of model to send requests to"
    )
    client_parser.add_argument(
        "--model-version",
        type=int,
        default=1,
        help="Model version to send requests to"
    )
    client_parser.add_argument(
        "--sequence-id",
        type=int,
        default=1001,
        help="Sequence identifier to use for the client stream"
    )

    data_parser = parser.add_argument_group(
        title="Data",
        help="Arguments for instantiating the client data sources"
    )
    data_parser.add_argument(
        "--kernel-stride",
        type=float,
        required=True,
        help="Time between frame snapshosts"
    )
    data_parser.add_argument(
        "--use-dummy",
        action="store_true",
        help=(
            "If set, the client will generate and send requests "
            "using random dummy data. All data directories and "
            "filename formats will accordingly be ignored."
        )
    )

    inputs = ["hanford witness", "livingston witness", "strain"]
    for input in inputs:
        name = input if input == "strain" else "witness-" + input[0]
        input_group = data_parser.add_argument_group(input.title())
        input_group.add_argument(
            f"--{name}-data-dir",
            type=str,
            default=None,
            help=(
                "Directory for LLF files corresponding to "
                f"{input} channels"
            )
        )
        input_group.add_argument(
            f"--{name}-file-pattern",
            type=str,
            default=None,
            help=(
                "File pattern for timestamped LLF files corresponding "
                f"to {input} channels"
            )
        )
        input_group.add_argument(
            f"--{name}-channels",
            type=str,
            nargs="+",
            required=True,
            help=f"Names of channels to use for {input} stream"
        )

    runtime_parser = parser.add_argument_group(
        title="Run Options",
        help="Arguments parameterizing client run"
    )
    runtime_parser.add_argument(
        "--num-iterations",
        type=int,
        default=10000,
        help="Number of requests to get for profiling"
    )
    runtime_parser.add_argument(
        "--num-warm-ups",
        type=int,
        default=50,
        help="Number of warm up requests"
    )
    flags = vars(parser.parse_args())

    file_formats, data_dirs, channels = {}, {}, {}
    def _check_arg(d, arg, name):
        try:
            d[name] = flags.pop(f"{name}_{arg}")
        except KeyError:
            if not flags.use_dummy:
                raise argparse.ArgumentError(
                    "Must provide {} for {} input stream".format(
                        arg.replace("_", " "), name
                    )
                )

    for input in inputs:
        name = input if input == "strain" else "witness_" + input[0]

        _check_arg(channels, "channels", name)
        _check_arg(file_patterns, "file_pattern", name)
        _check_arg(data_dirs, "data_dir", name)

    flags["channels"] = channels
    flags["file_patterns"] = file_patterns
    flags["data_dirs"] = data_dirs
    main(**flags)
