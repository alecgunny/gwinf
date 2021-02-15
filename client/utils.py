import argparse
import logging
import time
import typing

import attr

from stillwater import sync_recv

if typing.TYPE_CHECKING:
    from multiprocessing import Process
    from multiprocessing.connection import Connection


# set up logger
log = logging.getLogger()
console = logging.StreamHandler()
console.setFormatter(
    logging.Formatter("%(asctime)s\t%(message)s")
)
log.addHandler(console)
log.setLevel(logging.INFO)


@attr.s(auto_attribs=True)
class Pipeline:
    processes: typing.List["Process"]
    out_pipes: typing.Dict[str, "Connection"]

    def cleanup(self):
        for process in self.processes:
            if process.is_alive():
                process.stop()
                process.join(0.5)

            try:
                process.close()
            except ValueError:
                process.terminate()
                time.sleep(0.1)
                process.close()
                print(f"Process {process.name} couldn't join")

    def get(self, timeout=None):
        try:
            package = sync_recv(self.out_pipes, timeout=timeout)
        except Exception:
            self.cleanup()
            raise
        return package

    def __enter__(self):
        for process in self.processes:
            if not process.is_alive() and process.exitcode is not None:
                process.start()
            # or else what?
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.cleanup()


def get_inference_stats(client):
    stats = defaultdict(defaultdict(dict))
    for stat in client.get_inference_stats():
        name = stat.name.upper()
        for field, data in stat.inference_stats.ListFields():
            if field.name == "fail":
                continue

            field = field.name
            stats[name][field]["ns"] = data.ns
            stats[name][field]["count"] = data.count
    return stats


def parse_args():
    parser = argparse.ArgumentParser()

    client_parser = parser.add_argument_group(
        title="Client",
        description=(
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
        description="Arguments for instantiating the client data sources"
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
        description="Arguments parameterizing client run"
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

    file_patterns, data_dirs, channels = {}, {}, {}

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
    return flags
