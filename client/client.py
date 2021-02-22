import time
import typing
from collections import defaultdict

from tritonclient import grpc as triton
from stillwater.client import StreamingInferenceClient

from utils import get_inference_stats, log, parse_args, Pipeline



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

    # do imports here in case you don't have gwpy
    if use_dummy:
        from stillwater.data_generator import DummyDataGenerator
        def _get_data_gen(name):
            return DummyDataGenerator(
                (client.inputs[name].shape()[1], int(kernel_stride*4000)),
                name,
                sample_rate=4000
            )
    else:
        from stillwater.data_generator import LowLatencyFrameGenerator
        def _get_data_gen(name):
            return LowLatencyFrameGenerator(
                data_dirs[name],
                channels[name],
                sample_rate=4000,
                kernel_stride=kernel_stride,
                t0=t0,
                file_pattern=file_patterns[name],
                name=name
            )

    for input_name in client.inputs:
        data_gen = _get_data_gen(input_name)
        if not use_dummy:
            t0 = data_gen._generator_fn.t0
        client.add_parent(data_gen)
        processes.append(data_gen)

    out_pipes = {}
    for output in client.outputs:
        out_pipes[output.name()] = client.add_child(output.name())

    with Pipeline(processes, out_pipes) as pipeline:
        packages_recvd = 0
        log.info(f"Warming up for {num_warm_ups} batches")
        for i in range(num_warm_ups):
            package = pipeline.get(timeout=1)
            if package is None:
                time.sleep(0.1)
                continue
            packages_recvd += 1

        if packages_recvd == 0:
            raise RuntimeError("Nothing ever showed up!")
        log.info(f"Warmed up with {packages_recvd}")

        initial_server_stats = get_inference_stats(client)
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

    # report on how individual models in the ensemble did
    final_server_stats = get_inference_stats(client)
    for model, fields in final_server_stats.items():
        for field, stats in fields.items():
            init_stats = initial_server_stats[model][field]
            total_time = stats["ns"] - init_stats["ns"]
            total_count = stats["count"] - init_stats["count"]
            average_time = int(total_time / total_count / 100)

            log.info(f"{model}\tAverage {field} time: {average_time} us")


if __name__ == "__main__":
    flags = parse_args()
    main(**flags)
