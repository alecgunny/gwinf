#!/bin/bash -e
GPUS=4
gcloud container node-pools create big-t4-pool \
    --cluster gw-triton-dev \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-t4,count=$GPUS \
    --zone=us-west1-b \
    --num-nodes=2 \
    --project=gunny-multi-instance-dev

helm install --set numGPUs=$GPUS gwinf ./gw-triton

gcloud container node-pools delete big-t4-pool \
    --cluster=gw-triton-dev \
    --project=gunny-multi-instance-dev
