import queue
import time
import typing
from collections import defaultdict

import pandas as pd
from stillwater.client import StreamingInferenceClient

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
import utils


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
    num_warm_ups: typing.Optional[int] = None
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
                client.inputs[name].shape()[1:],
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

    with utils.Pipeline(processes, out_pipes) as pipeline:
        packages_recvd = 0
        utils.log.info(f"Warming up for {num_warm_ups} batches")
        for i in range(num_warm_ups):
            package = pipeline.get(timeout=1)
            if package is None:
                time.sleep(0.5)
                continue
            packages_recvd += 1

        if packages_recvd == 0:
            raise RuntimeError("Nothing ever showed up!")
        utils.log.info(f"Warmed up with {packages_recvd}")

        pipeline.reset()

        initial_server_stats = utils.get_inference_stats(client)
        metrics = defaultdict(utils.StreamingAverageStat)
        packages_recvd = 0
        while packages_recvd < num_iterations:
            package = pipeline.get(timeout=1)
            for i in range(50):
                try:
                    metric_name, value = client._metric_q.get_nowait()
                except queue.Empty:
                    break
                if metric_name == "throughput":
                    metrics[metric_name] = value
                else:
                    metrics[metric_name].update(value)

            if package is None:
                continue

            if num_warm_ups is None:
                msg = (
                    "Average latency: {} us, "
                    "Average throughput: {} frames/s".format(
                        int(metrics["latency"].value * 10**6),
                        metrics["throughput"]
                    )
                )
                print(msg, end="\r", flush=True)

            packages_recvd += 1

    if num_warm_ups is not None:
        return

    print("\n")
    utils.log.info(msg)

    # report on how individual models in the ensemble did
    final_server_stats = utils.get_inference_stats(client)
    data = defaultdict(list)
    for model, fields in final_server_stats.items():
        for field, stats in fields.items():
            init_stats = initial_server_stats[model][field]
            total_time = stats["ns"] - init_stats["ns"]
            total_count = stats["count"] - init_stats["count"]
            average_time = int(total_time / total_count / 1000)

            data["model"].append(model)
            data["process"].append(field)
            data["time (us)"].append(average_time)

            # log.info(f"{model}\tAverage {field} time: {average_time} us")
    df = pd.DataFrame(data)
    df["throughput"] = metrics["throughput"]
    df["preproc"] = int(metrics["preproc"].value * 10**6)
    df["round_trip"] = int(metrics["round_trip"].value * 10**6)
    df["latency"] = int(metrics["latency"].value * 10**6)
    return df


if __name__ == "__main__":
    flags = utils.parse_args()
    main(**flags)
