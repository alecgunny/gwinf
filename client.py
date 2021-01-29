import queue
import time
from multiprocessing import Process, Queue, Event

from collections import defaultdict

from tsinfer.pipeline import DummyDataGenerator
from tsinfer.pipeline import Preprocessor



def postprocessor(batch_start_time, result, error):
    return time.time() - batch_start_time

q_out = Queue(100)
targets = {"preprocessor": Preprocessor(
    "localhost:8001",
    "gwe2e",
    1,
    batch_size=8,
    kernel_size=1.0,
    kernel_stride=0.005,
    fs=4000,
    profile=True,
    q_out=q_out,
    qsize=100,
    postprocessor=postprocessor
)}

processes = {}
for name, x in targets["preprocessor"].inputs.items():
    q = targets["preprocessor"].q_in[name]
    targets[name] = DummyDataGenerator(x.shape()[1], q_out=q)
    processes[name] = Process(target=targets[name], name=name)
    processes[name].start()
processes["preprocessor"] = Process(
    target=targets["preprocessor"], name="preprocessor"
)
processes["preprocessor"].start()


def cleanup():
    for target in targets.values():
        target.stop()
    for process in processes.values():
        if process.is_alive():
          process.join(0.1)
        try:
            process.close()
        except ValueError:
            process.terminate()
            time.sleep(0.1)
            process.close()
            print(f"Process {process.name} couldn't join")


def do_a_get(timeout=1e-3):
    try:
        result = q_out.get(timeout=timeout)
    except queue.Empty:
        raise
    if isinstance(result, Exception):
        cleanup()
        raise result
    return result


warm_up_batches = 50
packages = 0
print(f"Warming up for {warm_up_batches} batches...")
for i in range(warm_up_batches):
    try:
        package = do_a_get(timeout=0.5)
        packages += 1
    except queue.Empty:
        continue

if packages == 0:
    cleanup()
    raise RuntimeError("Nothing ever showed up!")
print(f"Warmed up with {packages}")

average_latency = 0
for i in range(1000):
    try:
        latency, throughput = do_a_get(timeout=0.5)
    except queue.Empty:
        cleanup()
        raise RuntimeError

    average_latency += (latency - average_latency) / (i+1)

    msg = "Average latency: {} us, Average Throughput: {} frames/s".format(
        int(average_latency * 10**6), 8*throughput
    )
    print(msg, end="\r", flush=True)
cleanup()

