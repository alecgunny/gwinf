import random
import time
import typing
from multiprocessing import Event, Pipe, Process, Queue
from multiprocessing.connection import Connection

import attr
import numpy as np

from tritonclient import grpc as triton
from tritonclient.grpc import model_config_pb2 as model_config


@attr.s(auto_attribs=True)
class Relative:
    name: "InferenceProcess"
    conn: Connection


@attr.s(auto_attribs=True)
class Package:
    x: np.ndarray
    t0: float


class InferenceProcess(Process):
    def __init__(self, name: str) -> None:
        self.name = name

        self._parents = {}
        self._children = {}

        self._pause_event = Event()
        self._stop_event = Event()

        self._emergecy_q = Queue()
        super().__init__()

    def add_parent(self, parent: Relative):
        name = getattr(parent, "name", None)
        self._parents[name] = parent.conn

    def add_child(self, child: Relative):
        name = getattr(child, "name", None)
        self._children[name] = child.conn

    @property
    def stopped(self):
        return self._stop_event.is_set()

    def stop(self):
        self._stop_event.set()

    def _break_glass(self, exception):
        self.stop()
        if len(self._children) == 0:
            self._emergency_q.put(exception)
        for child in self._children.values():
            child.send(exception)

    def _get(self, parent, timeout=None):
        if parent.poll(timeout):
            obj = parent.recv()
        else:
            return

        if isinstance(obj, Exception):
            raise obj
        return obj

    def run(self):
        try:
            self._main_loop()
        except Exception as e:
            self._break_glass(e)

    def _get_data(self):
        ready_objs = {}
        for name, parent in self._parents.items():
            obj = self._get(parent)
            if obj is not None:
                ready_objs[name] = obj

        if ready_objs and len(ready_objs) < len(self._parents):
            start_time = time.time()
            _TIMEOUT = 1
            while (time.time() - start_time) < _TIMEOUT:
                for name in set(self._parents) - set(ready_objs):
                    obj = self._get(self._parents[name])
                    if obj is not None:
                        ready_objs[name] = obj
                if len(ready_objs) == len(self._parents):
                    break
            else:
                unfinished = set(self._parents) - set(ready_objs)
                raise RuntimeError(
                    "Parent processes {} stopped providing data".format(
                        ", ".join(unfinished)
                    )
                )
        return ready_objs

    def _main_loop(self):
        while not self.stopped:
            ready_objs = self._get_data()
            if ready_objs is None:
                continue
            self.do_stuff_with_data(ready_objs)

    def do_stuff_with_data(self, objs):
        raise NotImplementedError

    def __getstate__(self):
        # capture what is normally pickled
        state = self.__dict__.copy()

        # remove unpicklable/problematic variables 
        state['_parents'] = None
        state['_children'] = None
        return state


class DummyDataGenerator(InferenceProcess):
    def __init__(
        self,
        shape: typing.Tuple[int, ...],
        name: str,
        kernel_stride: typing.Optional[float]
    ) -> None:
        self.shape = shape
        self.kernel_stride = kernel_stride
        self.last_time = time.time()
        super().__init__(shape)

    def _get_data(self):
        if (
            self.kernel_stride is not None
            and time.time() - self.last_time < self.kernel_stride
        ):
            return
        x = np.random.randn(*self.shape).astype(np.float32)
        package = Package(x=x, t0=time.time())
        self.last_time = package.t0
        return package

    def do_stuff_with_data(self, objs):
        for child in self._children.values():
            child.send(objs)


class StreamingInferenceClient(InferenceProcess):
    def __init__(
        self,
        url: str,
        model_name: str,
        model_version: int,
        name: str
    ):
        # do a few checks on the server to make sure
        # we'll be good to go
        try:
            client = triton.InferenceServerClient(url)
        except triton.InferenceServerException:
            raise RuntimeError(
                "Couldn't connect to server at specified "
                "url {}".format(url)
            )
        if not client.is_server_live():
            raise RuntimeError(
                "Server at url {} isn't live".format(url)
            )
        if not client.is_model_ready(model_name):
            raise RuntimeError(
                "Model {} isn't ready at server url {}".format(
                    model_name, url
                )
            )
        super().__init__(name)
        self.url = url
        self.model_name = model_name
        self.model_version = model_version
        self._start_times = {}

        model_metadata = client.get_model_metadata(model_name)
        self.inputs = {}
        for input in model_metadata.inputs:
            self.inputs[input.name] = triton.InferInput(
                input.name, tuple(input.shape), input.datatype
            )
        self.outputs = [
            triton.InferRequestedOutput(output.name)
            for output in model_metadata.outputs
        ]

    def add_parent(self, parent:  Relative):
        name = getattr(parent, "name", None)
        if name not in self.inputs:
            raise ValueError(
                "Tried to add data source named {} "
                "to inference client expecting "
                "sources {}".format(
                    name, ", ".join(self.inputs.keys())
                )
            )
        super().add_parent(parent)

    def add_child(self, child: Relative):
        name = getattr(child, "name", None)
        if name not in [x.name() for x in self.outputs]:
            raise ValueError(
                "Tried to add output named {} "
                "to inference client expecting "
                "outputs {}".format(
                    name, ", ".join(self.inputs.keys())
                )
            )
        super().add_child(child)

    def _callback(self, result, error):
        id = int(result.get_response().id)
        t0 = self._start_times.pop(id)
        end_time = time.time()
        latency = end_time - t0
        throughput = id / (end_time - self._start_time)
        for name, conn in self._children:
            x = result.as_numpy(name)
            conn.send(Package(x, (latency, thoughput)))

    def _main_loop(self):
        missing_sources = set(self.inputs) - set(self._parents)
        if not len(missing_sources) == 0:
            raise RuntimeError(
                "Couldn't start inference client process, "
                "missing sources {}".format(
                    ", ".join(missing_sources)
                )
            )
        output_names = set([x.name() for x in self.outputs])
        missing_outputs = output_names - set(self._children)
        if not len(missing_outputs) == 0:
            raise RuntimeError(
                "Couldn't start inference client process, "
                "missing outputs {}".format(
                    ", ".join(missing_outputs)
                )
            )

        with triton.InferenceServerClient(url=self.url) as self.client:
            client.start_stream(callback=self._callback, stream_timeout=60)
            self._start_time = time.time()
            self._request_id = 0
            super()._main_loop()

        def do_stuff_with_data(objs):
            t0 = 0
            for name, package in objs.items():
                inputs[name].set_data_from_numpy(package.x)
                start_time += package.t0
            t0 /= len(objs)

            # request_id = "".join(random.choices(string.ascii_letters, k=16))
            self._request_id += 1
            self._start_times[self._request_id+0] = t0

            self.client.async_stream_infer(
                self.model_name,
                inputs=list(self.inputs.values()),
                outputs=self.outputs,
                request_id=str(request_id),
                sequence_id=1001
            )


def pipe(parent: InferenceProcess, child: InferenceProcess):
    parent_conn, child_conn = Pipe()
    parent.add_child(Relative(child, child_conn))
    child.add_parent(Relative(parent, parent_conn))
