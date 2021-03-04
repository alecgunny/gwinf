import os
import re
import time
from contextlib import contextmanager
from functools import partial

from google.auth.transport.requests import Request as AuthRequest
from google.oauth2 import service_account
from google.cloud import storage
from google.cloud import container_v1 as container

from cloud_utils.k8s import K8sApiClient
from cloud_utils.utils import wait_for


def snakeify(name: str) -> str:
    return re.sub("(?<!^)(?=[A-Z])", "_", name).lower()


class ThrottledClient:
    def __init__(self, service_account_key_file, throttle_secs=1.0):
        self._client = container.ClusterManagerClient.from_service_account_file(
            filename=service_account_key_file
        )
        self.throttle_secs = throttle_secs
        self._last_request_time = time.time()

    def make_request(self, request, **kwargs):
        request_fn_name = snakeify(
            type(request).__name__.replace("Request", "")
        )
        request_fn = getattr(self._client, request_fn_name)
        while (time.time() - self._last_request_time) < self.throttle_secs:
            time.sleep(0.01)
        return request_fn(request=request, **kwargs)


class Resource:
    def __new__(cls, resource, *args, **kwargs):
        resource_type = type(resource).__name__
        if resource_type == "Cluster":
            cls = Cluster
        elif resource_type == "NodePool":
            cls = NodePool
        else:
            raise TypeError(f"Unknown GKE resource type {resource_type}")
        return object.__new__(cls)

    def __init__(self, resource, parent, **kwargs):
        self.resource = resource
        self.parent = parent.name
        self.client = parent.client

    @property
    def resource_type(self):
        return type(self.resource).__name__

    @property
    def name(self):
        resource_type = self.resource_type
        camel = resource_type[0].lower() + resource_type[1:]
        return self.parent + "/{}/{}".format(camel, self.resource.name)

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
        try:
            return self.client.make_request(create_request)
        except Exception as e:
            try:
                if e.code != 409:
                    raise
            except AttributeError:
                raise e

    def delete(self):
        delete_request_cls = getattr(
            container, f"Delete{self.resource_type}Request"
        )
        delete_request = delete_request_cls(name=self.name)
        return self.client.make_request(delete_request)

    def get(self, timeout=None):
        get_request_cls = getattr(
            container, f"Get{self.resource_type}Request"
        )
        get_request = get_request_cls(name=self.name)
        return self.client.make_request(get_request, timeout=timeout)


class NodePool(Resource):
    pass


class Cluster(Resource):
    def __init__(self, resource, parent, auth_token):
        super().__init__(resource, parent)
        self.auth_token = auth_token
        self._k8s_client = None

    def _make_k8s_client(self):
        if self._k8s_client is not None:
            raise ValueError(
                f"Already created kubernetes client for cluster {self.name}"
            )
        self._k8s_client = K8sApiClient(self)

    @property
    def k8s_client(self):
        if self._k8s_client is None:
            self._make_k8s_client()
        return self._k8s_client

    def create(self):
        response = super().create()
        if response is not None:
            self._make_k8s_client()
        return response

    def deploy(self, file: str):
        return self.k8s_client.create_from_yaml(file)

    def remove_deployment(self, name: str, namespace: str = "default"):
        return self.k8s_client.remove_deployment(name, namespace)


def resource_ready_callback(resource):
    try:
        status = resource.get(timeout=5).status
    except Exception:
        # TODO: something to catch here?
        raise
    if status == 2:
        return True
    elif status > 2:
        raise RuntimeError
    return False


def resource_delete_submit_callback(resource):
    # first try to submit the delete request,
    # possibly waiting for the resource to
    # become available to be deleted if we
    # need to
    try:
        resource.delete()
    except Exception as e:
        try:
            if e.code == 404:
                # resource is gone, we're good
                return True
            elif e.code != 400:
                # 400 means resource is tied up, so
                # wait and try again in a bit. Otherwise,
                # raise an error
                raise
            else:
                return False
        except AttributeError:
            # the exception didn't have a `.code`
            # attribute, so evidently something
            # else went wrong, raise it
            raise e
    else:
        # response went off ok, so we're good
        return True


def resource_delete_done_callback(resource):
    # now wait for the delete request to
    # be completed
    try:
        status = resource.get(timeout=5).status
    except Exception as e:
        try:
            if e.code == 404:
                # resource is gone, so we're good
                # to exit
                return True
            # some other error occured, raise it
            raise
        except AttributeError:
            # a non-HTTP error occurred, raise it
            raise e

    if status > 4:
        # something bad happened to the resource,
        # raise the issue
        raise RuntimeError(status)
    return False


class GKEClusterManager:
    def __init__(
        self,
        service_account_key_file: str,
        project: str,
        zone: str
    ):
        self.client = ThrottledClient(service_account_key_file)
        credentials = service_account.Credentials.from_service_account_file(
            service_account_key_file,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        credentials.refresh(AuthRequest())
        self.auth_token = credentials.token

        self.name = f"projects/{project}/locations/{zone}"
        self.resources = {}

    @contextmanager
    def manage_resource(self, resource, parent=None, keep=False):
        parent = parent or self
        resource = Resource(resource, parent, auth_token=self.auth_token)
        resource.create()

        resource_type = snakeify(resource.resource_type).replace("_", " ")
        resource_msg = resource_type + " " + resource.name

        wait_for(
            partial(resource_ready_callback, resource),
            f"Waiting for {resource_msg} to become ready",
            f"{resource_msg} ready"
        )

        def delete_resource(raised):
            if keep:
                return
            elif raised:
                print(f"Encountered error, removing {resource_msg}")

            wait_for(
                partial(resource_delete_submit_callback, resource),
                f"Waiting for {resource_msg} to become available to delete",
                f"{resource_msg} delete request submitted"
            )

            wait_for(
                partial(resource_delete_done_callback, resource),
                f"Waiting for {resource_type} {resource.name} to delete",
                f"{resource_type} {resource.name} deleted"
            )
            self.resources.pop(resource.name)

        self.resources[resource.name] = resource
        raised = False
        try:
            yield resource
        except Exception:
            raised = True
            raise
        finally:
            delete_resource(raised)


def t4_node_config(vcpus=8, gpus=1, **kwargs):
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
        )],
        **kwargs
    )


def make_credentials(service_account_key_file):
    # use GKE credentials to create Kubernetes
    # configuration for cluster
    credentials = service_account.Credentials.from_service_account_file(
        service_account_key_file,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    credentials.refresh(AuthRequest())
    return credentials


def copy_repo_to_bucket(
    repo_dir,
    bucket,
    service_account_key_file
):
    client = storage.client.Client(
        credentials=make_credentials(service_account_key_file)
    )

    try:
        bucket = client.get_bucket(bucket)
    except Exception as e:
        try:
            if e.code == 404:
                bucket = client.create_bucket(bucket)
            else:
                raise
        except AttributeError:
            raise e

    for blob in client.list_blobs(bucket):
        blob.delete()

    for root, _, files in os.walk(repo_dir):
        for f in files:
            path = os.path.join(root, f)

            # get rid of root level path and replace
            # path separaters in case we're on Windows
            blob_path = path.replace(os.path.join(repo_dir, ""), "").replace(
                "\\", "/"
            )
            print(f"Copying {path} to {blob_path}")

            blob = bucket.blob(blob_path)
            blob.upload_from_filename(path)
    return bucket


def update_model_configs(
    expt, bucket, service_account_key_file
):
    client = storage.client.Client(
        credentials=make_credentials(service_account_key_file)
    )

    try:
        bucket = client.get_bucket(bucket)
    except Exception as e:
        try:
            if e.code == 404:
                raise ValueError(
                    f"No bucket named {bucket}"
                )
            else:
                raise
        except AttributeError:
            raise e

    for blob in client.list_blobs(bucket):
        if (
            blob.name.startswith(f"kernel-stride-{expt.kernel_stride:0.3f}")
            and blob.name.endswith(".pbtxt")
        ):
            content = blob.download_as_bytes().decode("utf-8")
            content = re.sub("(?<=count: )[0-9]", str(expt.instances), content)
            blob.upload_from_string(
                content.encode("utf-8"), content_type="application/octet-stream"
            )
