import time

from inference_process import (
    DummyDataGenerator,
    pipe,
    StreamingInferenceClient
)


KERNEL_STRIDE = 0.002

client = StreamingInferenceClient(
    "localhost:8001", "gwe2e", 1, "client"
)
data_generators = []
for name, x in client.input.items():
    data_gen = DummyDataGenerator(x.shape()[1:], name, KERNEL_STRIDE)
    pipe(data_gen, client)
    data_generators.append(data_gen)
    data_gen.start()

out_pipes = []
for output in client.output.items():
    class Output:
        name = output.name()
        _parents = {}
    out_pipes.append(Output())
    pipe(client, out_pipes[-1])

client.start()


def cleanup():
    for process in data_generators + [client]:
        process.stop()
        process.join(0.5)

        try:
            process.close()
        except ValueError:
            process.terminate()
            time.sleep(0.1)
            process.close()
            print(f"Process {process.name} couldn't join")


class Empty(Exception):
    pass


def do_a_get(timeout=1e-3):
    start_time = time()
    while time.time() - start_time < timeout:
        for p in out_pipes:
            if p.poll():
                result = p.recv()
                break
        else:
            continue
        break
    else:
        raise Empty
    if isinstance(result, Exception):
        cleanup()
        raise result
    return result.t0


warm_up_batches = 50
packages = 0
print(f"Warming up for {warm_up_batches} batches...")
for i in range(warm_up_batches):
    try:
        package = do_a_get(timeout=0.5)
        packages += 1
    except Empty:
        continue

if packages == 0:
    cleanup()
    raise RuntimeError("Nothing ever showed up!")
print(f"Warmed up with {packages}")

average_latency = 0
for i in range(1000):
    try:
        latency, throughput = do_a_get(timeout=0.5)
    except Empty:
        cleanup()
        raise RuntimeError

    average_latency += (latency - average_latency) / (i + 1)

    msg = "Average latency: {} us, Average Throughput: {} frames/s".format(
        int(average_latency * 10**6), 8 * throughput
    )
    print(msg, end="\r", flush=True)
cleanup()
