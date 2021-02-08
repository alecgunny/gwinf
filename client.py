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

# TODO: add as command line args
DATA_DIR = "/dev/shm/llhoft/H1"
FILE_PATTERN = "H-H1_llhoft-{}-1.gwf"
CHANNELS = """
H1:GDS-CALIB_STRAIN
H1:PEM-CS_MAINSMON_EBAY_1_DQ
H1:ASC-INP1_P_INMON
H1:ASC-INP1_Y_INMON
H1:ASC-MICH_P_INMON
H1:ASC-MICH_Y_INMON
H1:ASC-PRC1_P_INMON
H1:ASC-PRC1_Y_INMON
H1:ASC-PRC2_P_INMON
H1:ASC-PRC2_Y_INMON
H1:ASC-SRC1_P_INMON
H1:ASC-SRC1_Y_INMON
H1:ASC-SRC2_P_INMON
H1:ASC-SRC2_Y_INMON
H1:ASC-DHARD_P_INMON
H1:ASC-DHARD_Y_INMON
H1:ASC-CHARD_P_INMON
H1:ASC-CHARD_Y_INMON
H1:ASC-DSOFT_P_INMON
H1:ASC-DSOFT_Y_INMON
H1:ASC-CSOFT_P_INMON
H1:ASC-CSOFT_Y_INMON
"""
CHANNELS = [x for x in CHANNELS.split("\n") if x]


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
    kernel_stride: float,
    url: str,
    num_iterations: int = 10000,
    num_warm_ups: int = 50,
    use_dummy=True
):
    client = StreamingInferenceClient(
        url, "gwe2e", 1, "client", sequence_id=1001
    )
    processes = [client]
    for name, x in client.inputs.items():
        if use_dummy:
            data_gen = DummyDataGenerator(
                x.shape()[1:], name, kernel_stride
            )
        else:
            if name == "strain":
                channels = [CHANNELS[0], CHANNELS[0]]
            else:
                channels = channels[1:]
            data_gen = LowLatencyFrameGenerator(
                DATA_DIR,
                channels,
                sample_rate=4000,
                kernel_stride=kernel_stride,
                t0=None,
                file_pattern=FILE_PATTERN,
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
    parser.add_argument(
        "--kernel-stride",
        type=float,
        required=True,
        help="Time between frame snapshosts"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="localhost:8001",
        help="Server URL"
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=10000,
        help="Number of requests to get for profiling"
    )
    parser.add_argument(
        "--num-warm-ups",
        type=int,
        default=50,
        help="Number of warm up requests"
    )
    parser.add_argument(
        "--use-dummy",
        action="store_true",
        help="Whether to use dummy data generators"
    )

    flags = parser.parse_args()
    main(**vars(flags))
