import base64
import os
import re
import requests
import time
import typing
import yaml
from contextlib import contextmanager
from functools import wraps
from tempfile import NamedTemporaryFile

from google import auth
from google.auth.transport.requests import Request as AuthRequest
from google.oauth2 import service_account
from google.cloud import storage
from google.cloud import container_v1 as container

import kubernetes


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
        self.resources = {}

    @contextmanager
    def manage_resource(self, resource, parent=None, keep=False):
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

        resource_type = snakeify(resource.resource_type).replace("_", " ")
        while True:
            msg = wait_print(
                f"Waiting for {resource_type} {resource.name} to become ready"
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
        print(f"\n{resource_type} {resource.name} ready", flush=True)

        def delete_resource():
            if keep:
                return

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
                                f"Waiting for {resource_type} {resource.name} "
                                "to become available to delete"
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
                f"{resource_type} {resource.name} delete request submitted",
                flush=True
            )

            while True:
                # now wait for the delete request to
                # be completed. TODO: can we use the
                # operation id from the response to
                # check in on the status of the operations?
                msg = wait_print(
                    f"Waiting for {resource_type} {resource.name} to delete"
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

            self.resources.pop(resource.name)
            print(f"\n{resource_type} {resource.name} deleted", flush=True)

        self.resources[resource.name] = resource
        try:
            yield resource
        except:
            print(
                f"Encountered error, removing {resource_type} {resource.name}",
                flush=True
            )
            delete_resource()
            raise
        delete_resource()


def make_k8s_app_client(cluster):
    response = cluster.get()

    credentials, project = auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials.refresh(AuthRequest())

    configuration = kubernetes.client.Configuration()
    configuration.host = f"https://{response.endpoint}"
    with NamedTemporaryFile(delete=False) as ca_cert:
        ca_cert.write(
            base64.b64decode(response.master_auth.cluster_ca_certificate)
        )
    configuration.ssl_ca_cert = ca_cert.name
    configuration.api_key_prefix["authorization"] = "Bearer"
    configuration.api_key["authorization"] = credentials.token

    return kubernetes.client.ApiClient(configuration)


@contextmanager
def _get_file(file, repo, branch, ignore_if_exists):
    if repo is not None:
        # TODO: use main as defualt? or at least try to?
        branch = branch or "master"
        url_header = "https://raw.githubusercontent.com"
        url = f"{url_header}/{repo}/{branch}/{file}"

        yaml_content = requests.get(url).content.decode("utf-8")
        with NamedTemporaryFile(mode="w", delete=False) as f:
            f.write(yaml_content)
            file = f.name

    try:
        try:
            yield file
        except Exception as e:
            if not ignore_if_exists:
                raise

            try:
                info = str(e).split("(Conflict): ")[1]
                info = yaml.safe_load(info)
            except Exception:
                raise e
            if info["reason"] != "AlreadyExists":
                raise
    finally:
        if repo is not None:
            os.remove(file)


def create_deployment_from_yaml(
    client: kubernetes.client.ApiClient,
    file: str,
    repo: typing.Optional[str] = None,
    branch: typing.Optional[str] = None,
    wait: bool = False,
    ignore_if_exists: bool = False
):
    response = None
    with _get_file(file, repo, branch, ignore_if_exists) as file:
        with open(file, "r") as f:
            name = yaml.safe_load(f)["metadata"]["name"]
        response = kubernetes.utils.create_from_yaml(client, file)

    if not wait:
        return response

    # TODO: we need to a get if response is None
    # we also might need to do integer indexing since
    # sometimes the response can be a list of list
    # (e.g. for daemonset, not sure what it is for
    # deployments)
    # if response is None:
    #     name = file["metadata"]["name"]
    # else:
    #     name = response.metadata.name

    app_client = kubernetes.client.AppsV1Api(client)
    while True:
        wait_print(f"Waiting for deployment {name} to deploy")
        try:
            # TODO: get namespace from response
            status = app_client.read_namespaced_deployment(
                name=name, namespace="default"
            ).status
        except kubernetes.client.ApiException:
            raise
        ready_replicas = status.ready_replicas
        if ready_replicas is not None and ready_replicas > 0:
            break
        time.sleep(1)
    print(f"\nDeployment {name} ready")
    return response


def create_service_from_yaml(
    client: kubernetes.client.ApiClient,
    file: str,
    repo: typing.Optional[str] = None,
    branch: typing.Optional[str] = None,
    wait: bool = False,
    ignore_if_exists: bool = False
):
    response = None
    with _get_file(file, repo, branch, ignore_if_exists) as file:
        with open(file, "r") as f:
            name = yaml.safe_load(f)["metadata"]["name"]
        response = kubernetes.utils.create_from_yaml(client, file)

    if not wait:
        return response

    # if response is None:
    #     name = file["metadata"]["name"]
    # else:
    #     name = response.metadata.name

    core_client = kubernetes.client.CoreV1Api(client)
    while True:
        wait_print(f"Waiting for service {name} to be ready")
        try:
            # TODO: get namespace from response
            status = core_client.read_namespaced_service(
                name=name, namespace="default"
            ).status
        except kubernetes.client.ApiException:
            raise
        try:
            ip = status.load_balancer.ingress[0].ip
            if ip is not None:
                break
        except:
            pass
    return ip


def t4_node_config(vcpus=8, gpus=1):
    OAUTH_SCOPES = [
        "https://www.googleapis.com/auth/devstorage.read_only",
        "https://www.googleapis.com/auth/logging.write",
        "https://www.googleapis.com/auth/monitoring",
        "https://www.googleapis.com/auth/service.management.readonly",
        "https://www.googleapis.com/auth/servicecontrol",
        "https://www.googleapis.com/auth/trace.append"
    ]
    return container.NodeConfig(
        machine_type=f"n1-standard-{vcpus}",
        oauth_scopes=OAUTH_SCOPES,
        accelerators=[container.AcceleratorConfig(
            accelerator_count=gpus,
            accelerator_type="nvidia-tesla-t4"
        )]
    )
