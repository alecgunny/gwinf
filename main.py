import os
import re
import typing

import pandas as pd

import cloud_utils as cloud
from expt_configs import expts, kernel_strides
from export.export import main as export_models
from client_benchmarking.client import main as run_experiment


_PROJECT = "gunny-multi-instance-dev"
CHANNELS, DATA_DIRS, FILE_PATTERNS = {}, {}, {}
for input_name in os.listdir("channels"):
    with open(os.path.join("channels", input_name), "r") as f:
        input_name = input_name.replace(".", "/")
        input_name = "kernel-stride-{:0.4f}_" + input_name
        CHANNELS[input_name] = [i for i in f.read().split("\n") if i]

    detector = re.search("[hl](?=.witness)", input_name)
    if detector is None:
        detector = "h"
    else:
        detector = detector.group(0)

    detector = detector.upper()
    DATA_DIRS[input_name] = f"/dev/shm/kafka/{detector}1_O2"
    FILE_PATTERNS[input_name] = f"{detector}-{detector}1_O2_llhoft-{{}}-1.gwf"


def format_for_expt(d, expt):
    return {k.format(expt.kernel_stride): v for k, v in d.items()}


def deploy_gpu_drivers(cluster):
    # deploy GPU drivers daemonset on to cluster
    with cloud.deploy_file(
        "nvidia-driver-installer/cos/daemonset-preloaded.yaml",
        repo="GoogleCloudPlatform/container-engine-accelerators",
        branch="master",
        ignore_if_exists=True
    ) as f:
        cluster.deploy(f)


def export_and_push(
    manager: cloud.GKEClusterManager,
    cluster: cloud.gke.Cluster,
    repo: cloud.GCSModelRepo,
    repo_dir: str,
    keep: bool
):
    node_pool = cloud.container.NodePool(
        name="converter-pool",
        initial_node_count=1,
        config=cloud.gke.t4_node_config(
            vcpus=4, gpus=1, labels={"trtconverter": "true"}
        )
    )
    with manager.manage_resource(node_pool, cluster, keep=keep) as node_pool:
        # make sure NVIDIA drivers got installed
        deploy_gpu_drivers(cluster)
        cluster.k8s_client.wait_for_daemon_set(name="nvidia-driver-installer")

        # now deploy TRT conversion app
        deploy_file = os.path.join("apps", "trt-conversion", "deploy.yaml")
        with cloud.deploy_file(deploy_file, ignore_if_exists=True) as f:
            cluster.deploy(f)
        cluster.k8s_client.wait_for_deployment("trt-converter")
        ip = cluster.k8s_client.wait_for_service("trt-converter")

        for kernel_stride in kernel_strides:
            export_models(
                platform=f"trt_fp16:http://{ip}:5000/onnx",
                gpus=1,
                count=1,
                base_name=f"kernel-stride-{kernel_stride:0.4f}",
                kernel_stride=kernel_stride,
                fs=4000,
                kernel_size=1.0,
                repo_dir=repo_dir
            )
    repo.export_repo(repo_dir, start_fresh=True, clear=True)


def run_inference_experiments(
    manager: cloud.GKEClusterManager,
    cluster: cloud.gke.Cluster,
    repo: cloud.GCSModelRepo,
    vcpus_per_gpu: int = 16,
    keep: bool = False,
    experiment_interval: float = 40.0,
):
    # configure the server node pool
    max_cpus = 4 * vcpus_per_gpu
    node_pool = cloud.container.NodePool(
        name="triton-t4-pool",
        initial_node_count=1,
        config=cloud.gke.t4_node_config(vcpus=max_cpus, gpus=4)
    )

    # spin up the node pool on the cluster
    with manager.manage_resource(node_pool, cluster, keep=keep) as node_pool:
        # make sure NVIDIA drivers got installed
        deploy_gpu_drivers(cluster)
        cluster.k8s_client.wait_for_daemon_set(name="nvidia-driver-installer")

        # set some values that we'll use to parse the deployment yaml
        helm_dir = os.path.join("apps", "server", "gw-triton")
        deploy_file = os.path.join(helm_dir, "templates", "deploy.yaml")
        deploy_values = {
            "_file": os.path.join(helm_dir, "values.yaml"),
            "repo": "gs://" + repo.bucket_name,
        }

        # iterate through our experiments and collect the results
        results = []
        current_instances, current_gpus = 0, 0
        for expt in expts:
            if current_instances != expt.instances or current_gpus != expt.gpus:
                # since Triton can't dynamically detect changes
                # to the instance group without explicit model
                # control, the simplest thing to do will be to
                # spin up a new server instance each time our
                # configuration changes

                # start by updating all the model configs if
                # the instances-per-gpu have changed
                if current_instances != expt.instances:
                    repo.update_model_configs_for_expt(expt)

                # now spin down the old deployment
                cluster.remove_deployment("tritonserver")

                # now add the new configuration details to our
                # yaml parsing values map
                num_cpus = min(vcpus_per_gpu * expt.gpus, max_cpus - 1)
                deploy_values.update({"numGPUs": expt.gpus, "cpu": num_cpus})

                # deploy this new yaml onto the cluster
                with cloud.deploy_file(
                    deploy_file, values=deploy_values, ignore_if_exists=True
                ) as f:
                    cluster.deploy(f)

                # wait for it to be ready
                cluster.k8s_client.wait_for_deployment("tritonserver")
                ip = cluster.k8s_client.wait_for_service("tritonserver")

                current_instances = expt.instances
                current_gpus = expt.gpus

            for n in [10, int(experiment_interval / expt.kernel_stride)]:
                df = run_experiment(
                    url=f"{ip}:8001",
                    model_name=f"kernel-stride-{expt.kernel_stride:0.4f}_gwe2e",
                    model_version=1,
                    sequence_id=1001,  # TODO: this will need to be random for real
                    kernel_stride=expt.kernel_stride,
                    channels=format_for_expt(CHANNELS, expt),
                    data_dirs=format_for_expt(DATA_DIRS, expt),
                    file_patterns=format_for_expt(FILE_PATTERNS, expt),
                    num_warm_ups=50,
                    num_iterations=n,
                )

            df["model"] = df["model"].str.split("_", n=1, expand=True)[1]
            df["kernel_stride"] = expt.kernel_stride
            df["instances"] = expt.instances
            df["gpus"] = expt.gpus
            print(df)

            results.append(df)

    results = pd.concat(results, axis=0, ignore_index=True)
    results.to_csv("results.csv", index=False)
    return results


def main(
    cluster_name: str,
    service_account_key_file: str,
    repo_dir: str = "repo",
    repo_bucket: typing.Optional[str] = None,
    zone: str = "us-west1-b",
    export: bool = False,
    keep: bool = False
):
    manager = cloud.GKEClusterManager(
        service_account_key_file, _PROJECT, zone
    )
    repo = cloud.GCSModelRepo(
        repo_bucket or cluster_name + "_model-repo", service_account_key_file
    )

    cluster = cloud.container.Cluster(
        name=cluster_name,
        node_pools=[cloud.container.NodePool(
            name="default-pool",
            initial_node_count=2,
            config=cloud.container.NodeConfig()
        )]
    )

    with manager.manage_resource(cluster, keep=keep) as cluster:
        if export:
            # build a node pool for deploying our TRT conversion app
            export_and_push(manager, cluster, repo, repo_dir, keep)

        results = run_inference_experiments(
            manager, cluster, repo, vcpus_per_gpu=16, keep=True
        )
        print(results)
    return manager


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cluster-name",
        type=str,
        default="gw-benchmarking"
    )
    parser.add_argument(
        "--service-account-key-file",
        type=str,
        required=True
    )
    parser.add_argument(
        "--export",
        action="store_true"
    )
    parser.add_argument(
        "--keep",
        action="store_true",
    )
    parser.add_argument(
        "--repo-dir",
        type=str,
        default="repo"
    )
    flags = parser.parse_args()

    if not os.path.exists(flags.repo_dir):
        print(f"Making model repo directory {flags.repo_dir}")
        os.makedirs(flags.repo_dir)
    elif len(os.listdir(flags.repo_dir)) > 0 and flags.export:
        print(f"Model repo director {flags.repo_dir} not empty")
        cloud.utils.clear_repo(flags.repo_dir)

    manager = main(**vars(flags))
