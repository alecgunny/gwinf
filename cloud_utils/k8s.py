import os
import re
import requests
import typing
import yaml
from base64 import b64decode
from contextlib import contextmanager
from tempfile import NamedTemporaryFile

import kubernetes

from cloud_utils.utils import wait_for


class K8sApiClient:
    def __init__(self, cluster):
        try:
            response = cluster.get()
        except Exception as e:
            try:
                if e.code == 404:
                    raise RuntimeError(
                        f"Cluster {cluster.name} not currently deployed"
                    )
                raise
            except AttributeError:
                raise e

        # create configuration using bare minimum info
        configuration = kubernetes.client.Configuration()
        configuration.host = f"https://{response.endpoint}"
        with NamedTemporaryFile(delete=False) as ca_cert:
            ca_cert.write(
                b64decode(response.master_auth.cluster_ca_certificate)
            )
        configuration.ssl_ca_cert = ca_cert.name
        configuration.api_key_prefix["authorization"] = "Bearer"
        configuration.api_key["authorization"] = cluster.auth_token

        # return client instantiated with configuration
        self._client = kubernetes.client.ApiClient(configuration)

    def create_from_yaml(self, file: str):
        return kubernetes.utils.create_from_yaml(self._client, file)

    def remove_deployment(self, name: str, namespace: str = "default"):
        app_client = kubernetes.client.AppsV1Api(self._client)

        def _try_cmd(cmd):
            try:
                cmd(name=name, namespace=namespace)
            except Exception as e:
                try:
                    body = yaml.safe_load(e.body)
                    if body["code"] == 404:
                        return True
                    else:
                        return False
                except AttributeError:
                    raise e
                except Exception:
                    raise
            return False

        _try_cmd(app_client.delete_namespaced_deployment)

        def _deleted_callback():
            return _try_cmd(app_client.read_namespaced_deployment)

        wait_for(
            _deleted_callback,
            f"Waiting for deployment {name} to delete",
            f"Deployment {name} deleted"
        )

    def wait_for_deployment(self, name: str, namespace: str = "default"):
        app_client = kubernetes.client.AppsV1Api(self._client)

        def _ready_callback():
            try:
                response = app_client.read_namespaced_deployment_status(
                    name=name, namespace=namespace
                )
            except kubernetes.client.ApiException:
                raise RuntimeError(
                    f"Deployment {name} no longer exists!"
                )

            statuses = {
                i.type: eval(i.status) for i in response.status.conditions
            }
            try:
                if statuses["Available"]:
                    return True
            except KeyError:
                raise ValueError("Couldn't find readiness status")

            try:
                if not statuses["Progressing"]:
                    raise RuntimeError(
                        f"Deployment {name} stopped progressing"
                    )
            except KeyError:
                pass
            finally:
                return False

        wait_for(
            _ready_callback,
            f"Waiting for deployment {name} to deploy",
            f"Deployment {name} ready"
        )

    def wait_for_service(self, name: str, namespace: str = "default"):
        core_client = kubernetes.client.CoreV1Api(self._client)

        def _ready_callback():
            try:
                response = core_client.read_namespaced_service_status(
                    name=name, namespace=namespace
                )
            except kubernetes.client.ApiException:
                raise RuntimeError(
                    f"Service {name} no longer exists!"
                )

            ip = response.status.load_balancer.ingress[0].ip
            return ip or False

        return wait_for(
            _ready_callback,
            f"Waiting for service {name} to be ready",
            f"Service {name} ready"
        )

    def wait_for_daemon_set(self, name: str, namespace: str = "kube-system"):
        core_client = kubernetes.client.CoreV1Api(self._client)

        def _ready_callback():
            try:
                response = core_client.read_namespaced_daemon_set_status(
                    name=name, namespace=namespace
                )
            except kubernetes.client.ApiException:
                raise RuntimeError(
                    f"Daemon set {name} no longer exists!"
                )

            if (
                response.status.desired_number_scheduled ==
                response.status.number_ready
            ):
                return True
            return False


@contextmanager
def deploy_file(
    file: str,
    repo: typing.Optional[str] = None,
    branch: typing.Optional[str] = None,
    values: typing.Optional[typing.Dict[str, str]] = None,
    ignore_if_exists: bool = True
):
    if repo is not None:
        if branch is None:
            branches = ["main", "master"]
        else:
            branches = [branch]

        for branch in branches:
            url_header = "https://raw.githubusercontent.com"
            url = f"{url_header}/{repo}/{branch}/{file}"

            try:
                yaml_content = requests.get(url).content.decode("utf-8")
            except Exception:
                pass
            else:
                break
        else:
            raise ValueError(
                f"Couldn't find file {file} at github repo {repo}, ",
                "tried looking in branches {}".format(
                    ", ".join(branches)
                )
            )
    else:
        with open(file, "r") as f:
            yaml_content = f.read()

    values = values or {}
    values = values.copy()
    try:
        # try to load in values from file
        values_file = values.pop("_file")
    except KeyError:
        pass
    else:
        # use explicitly passed values to overwrite
        # values in file
        with open(values_file, "r") as f:
            values_map = yaml.safe_load(f)
        values_map.update(values)
        values = values_map

    # look for any Go variable indicators and try to
    # fill them in with their value from `values`
    def replace_fn(match):
        varname = re.search(
            "(?<={{ .Values.)[a-zA-Z0-9]+?(?= }})", match.group(0)
        ).group(0)
        try:
            return str(values[varname])
        except KeyError:
            raise ValueError(
                "No value provided for wildcard {}".format(
                    varname
                )
            )
    yaml_content = re.sub(
        "{{ .Values.[a-zA-Z0-9]+? }}", replace_fn, yaml_content
    )

    # write formatted yaml to temporary file
    with NamedTemporaryFile(mode="w", delete=False) as f:
        f.write(yaml_content)
        file = f.name

    try:
        try:
            yield file
        except Exception as e:
            if not ignore_if_exists:
                # doesn't matter what the issue was,
                # delete the temp files and raise
                raise

            try:
                # see if the exception included some yaml
                # formatted information about what went wrong
                info = str(e).split("(Conflict): ")[1]
                info = yaml.safe_load(info)
            except Exception:
                # expected formatting was different, raise
                # the initial error
                raise e

            if info["reason"] != "AlreadyExists":
                # if the reason was anything other
                # than that the objects in question
                # already exist, raise it
                raise
    finally:
        # remove the temporary file no matter
        # what happens
        os.remove(file)
