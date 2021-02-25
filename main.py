import cloud_utils as cloud
# from export.export import main as export



def main(
    cluster_name,
    service_account_key_file,
    zone="us-west1-b",
    export=False
):
    manager = cloud.GKEClusterManager(
        service_account_key_file,
        "gunny-multi-instance-dev",
        zone
    )

    cluster = cloud.container.Cluster(
        name=cluster_name,
        node_pools=[cloud.container.NodePool(
            name="default-pool",
            config=cloud.container.NodeConfig()
        )]
    )
    with manager.create_temporary_resource(cluster) as cluster:
        if export:
            node_pool = cloud.container.NodePool(
                name="converter-pool",
                initial_node_count=1,
                config=cloud.tf_node_config(vcpus=4, gpus=1)
            )
            with manager.create_temporary_resource(
                node_pool, cluster
            ) as node_pool:
                print("Pool ready!")


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
    flags = parser.parse_args()
    main(**vars(flags))
