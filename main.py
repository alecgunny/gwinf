import os
import time

import cloud_utils as cloud
from expt_configs import expts, kernel_strides
# from export.export import main as export


def main(
    cluster_name,
    service_account_key_file,
    zone="us-west1-b",
    export=False,
    keep=False
):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_key_file
    manager = cloud.GKEClusterManager(
        service_account_key_file,
        "gunny-multi-instance-dev",
        zone
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
        # build a kubernetes client capable of communicating
        # with the new cluster
        k8s_app_client = cloud.make_k8s_app_client(cluster)
        if export:
            # build a node pool for deploying our TRT conversion app
            node_pool = cloud.container.NodePool(
                name="converter-pool",
                initial_node_count=1,
                config=cloud.t4_node_config(vcpus=4, gpus=1)
            )
            with manager.manage_resource(
                node_pool, cluster, keep=keep
            ) as node_pool:
                # deploy GPU drivers onto node pool
                response = cloud.create_deployment_from_yaml(
                    k8s_app_client,
                    "nvidia-driver-installer/cos/daemonset-preloaded.yaml",
                    repo="GoogleCloudPlatform/container-engine-accelerators",
                    branch="master",
                    ignore_if_exists=True
                )
                # TODO: do a real wait here
                time.sleep(10)

                # now deploy TRT conversion app
                response = cloud.create_deployment_from_yaml(
                    k8s_app_client,
                    "deployment.yaml",
                    repo="alecgunny/trt-converter-app",
                    branch="main",
                    wait=True,
                    ignore_if_exists=True
                )

                # expose it via loadbalancer
                ip = cloud.create_service_from_yaml(
                    k8s_app_client,
                    "converter-service.yaml",
                    wait=True,
                    ignore_if_exists=True
                )

                # url = "..."
                # for kernel_stride in kernel_strides:
                #     export(
                #         platform=f"trt_fp16:http://{ip}:5000",
                #         gpus=1,
                #         count=1,
                #         base_name=f"kernel-stride-{kernel_stride:0.3f}",
                #         kernel_stride=kernel_stride,
                #         fs=4000,
                #         kernel_size=1.0
                #     )
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
    flags = parser.parse_args()
    manager = main(**vars(flags))
