import re
import time
from contextlib import contextmanager
from functools import wraps

from google.oauth2 import service_account
from google.cloud import storage
from google.cloud import container_v1 as container


def snakeify(name):
    return re.sub("(?<!^)(?=[A-Z])", "_", name).lower()


def wait_print(msg, wait_time=1):
    for i in range(3):
        dots = "."*(i+1)
        spaces = " "*(2-i)
        print(msg + dots + spaces, end="\r", flush=True)
        time.sleep(wait_time)
    return msg


def request_throttle(f):
    @wraps(f)
    def wrapper(self, *args, **kwargs):
        while (time.time() - self._last_request_time) < self.throttle:
            time.sleep(0.1)
        self._last_request_time = time.time()
        return f(self, *args, **kwargs)
    return wrapper


class Resource:
    def __init__(self, resource, parent, throttle=2):
        self.resource = resource
        self.parent = parent.name
        self.client = parent.client

        self.throttle = throttle
        self._last_request_time = time.time()

    @property
    def resource_type(self):
        return type(self.resource).__name__

    @property
    def name(self):
        camel = self.resource_type[0].lower() + self.resource_type[1:]
        return self.parent + "/{}/{}".format(
            camel, self.resource.name
        )

    @request_throttle
    def create(self):
        create_request_cls = getattr(
            container, f"Create{self.resource_type}Request"
        )

        resource_type = snakeify(self.resource_type)
        kwargs = {
            resource_type: self.resource,
            "parent": self.parent
        }
        create_request = create_request_cls(**kwargs)

        create_request_fn = getattr(
            self.client, f"create_{resource_type}"
        )
        return create_request_fn(request=create_request)

    @request_throttle
    def delete(self):
        delete_request_cls = getattr(
            container, f"Delete{self.resource_type}Request"
        )
        delete_request = delete_request_cls(name=self.name)
        delete_request_fn = getattr(
            self.client, "delete_{}".format(
                snakeify(self.resource_type)
            )
        )
        return delete_request_fn(request=delete_request)

    @request_throttle
    def get(self, timeout=None):
        get_request_cls = getattr(
            container, f"Get{self.resource_type}Request"
        )
        get_request = get_request_cls(name=self.name)
        get_request_fn = getattr(
            self.client, "get_{}".format(
                snakeify(self.resource_type)
            )
        )
        return get_request_fn(request=get_request, timeout=timeout)


class GKEClusterManager:
    def __init__(
            self,
            service_account_key_file: str,
            project: str,
            zone: str
        ):
        self.client = container.ClusterManagerClient.from_service_account_file(
            filename=service_account_key_file
        )
        self.name = f"projects/{project}/locations/{zone}"

    @contextmanager
    def create_temporary_resource(self, resource, parent=None):
        parent = parent or self
        resource = Resource(resource, parent)

        try:
            response = resource.create()
        except Exception as e:
            try:
                if e.code != 409:
                    raise
            except AttributeError:
                raise e

        while True:
            msg = wait_print(
                f"Waiting for resource {resource.name} to become ready"
            )
            try:
                status = resource.get(timeout=5).status
            except:
                # TODO: something to catch here?
                raise
            if status == 2:
                break
            elif status > 2:
                raise RuntimeError
        print(f"\nResource {resource.name} ready", flush=True)

        def delete_resource():
            while True:
                # first try to submit the delete request,
                # possibly waiting for the resource to
                # become available to be deleted if we
                # need to
                msg = None
                try:
                    resource.delete()
                except Exception as e:
                    try:
                        if e.code == 404:
                            # resource is gone, we're good
                            break
                        elif e.code != 400:
                            # resource is tied up, wait to
                            # try again in a bit
                            raise
                        else:
                            msg = wait_print(
                                f"Waiting for resource {resource.name} to "
                                "become available to delete"
                            )
                    except AttributeError:
                        # the exception didn't have a `.code`
                        # attribute, so evidently something
                        # else went wrong, raise it
                        raise e
                else:
                    # response went off ok, so we're good
                    break
            if msg is not None:
                print("\n", flush=True)
            print(
                f"Resource {resource.name} delete request submitted",
                flush=True
            )

            while True:
                # now wait for the delete request to
                # be completed. TODO: can we use the
                # operation id from the response to
                # check in on the status of the operations?
                msg = wait_print(
                    f"Waiting for resource {resource.name} to delete"
                )
                try:
                    status = resource.get(timeout=5).status
                except Exception as e:
                    try:
                        if e.code == 404:
                            # resource is gone, so we're good
                            # to exit
                            break
                        # some other error occured, raise it
                        raise
                    except AttributeError:
                        # a non-HTTP error occurred, raise it
                        raise e

                if status > 4:
                    # something bad happened to the resource,
                    # raise the issue
                    raise RuntimeError(status)
            print(f"\nResource {resource.name} deleted", flush=True)

        try:
            yield resource
        except:
            print(
                f"Encountered error, removing resource {resource.name}",
                flush=True
            )
            delete_resource()
            raise
        delete_resource()


def t4_node_config(vcpus=8, gpus=1):
    return container.NodeConfig(
        machine_type=f"n1-standard-{vcpus}",
        accelerators=[cloud.container.AcceleratorConfig(
            accelerator_count=gpus,
            accelerator_type="nvidia-tesla-t4"
        )]
    )
