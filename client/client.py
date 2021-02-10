import time
import typing

from tritonclient import grpc as triton

from stillwater import pipe
from stillwater.client import StreamingInferenceClient
from stillwater.data_generator import DummyDataGenerator, LowLatencyFrameGenerator

from utils import log, parse_args, Pipeline


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
    t0 = None
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
                t0=t0,
                file_pattern=file_patterns[name],
                name=name
            )
            t0 = data_gen._generator_fn.t0
        pipe(data_gen, client)
        processes.append(data_gen)

    out_pipes = {}
    for output in client.outputs:
        out_pipes[output.name()] = pipe(client, output.name())

    pipeline = Pipeline(processes, out_pipes)
    for process in processes:
        process.start()

    packages_recvd = 0
    log.info(f"Warming up for {num_warm_ups} batches")
    for i in range(num_warm_ups):
        package = pipeline.get(timeout=1)
        if package is None:
            time.sleep(0.1)
            continue
        packages_recvd += 1

    if packages_recvd == 0:
        cleanup(processes)
        raise RuntimeError("Nothing ever showed up!")
    log.info(f"Warmed up with {packages_recvd}")

    average_latency, packages_recvd = 0, 0
    while packages_recvd < num_iterations:
        package = pipeline.get(timeout=0.1)
        if package is None:
            continue

        latency, throughput = 0, 0
        for pack in package.values():
            l, t = pack.t0
            throughput += t
            latency += l
        throughput /= len(package)
        latency /= len(package)
        average_latency += (latency - average_latency) / (i + 1)

        msg = "Average latency: {} us, Average Throughput: {} frames/s".format(
            int(average_latency * 10**6), throughput
        )
        print(msg, end="\r", flush=True)
        packages_recvd += 1

    print("\r")
    log.info(msg)

    # spin everything down
    pipeline.cleanup()

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
            average_time = int(data.ns / data.count / 1000)
            msg = f"{name}\tAverage {field} time: {average_time}"
            log.info(msg)


if __name__ == "__main__":
    flags = parse_args()
    main(**flags)
