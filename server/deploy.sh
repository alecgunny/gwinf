#!/bin/bash -e
gcloud container node-pools create big-t4-pool \
    --cluster gw-triton-dev \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-t4,count=4 \
    --zone=us-west1-b \
    --num-nodes=2 \
    --project=gunny-multi-instance-dev

# INSERT POD AND SERVICE DEPLOYMENT

gcloud container node-pools delete big-t4-pool \
    --cluster=gw-triton-dev \
    --project=gunny-multi-instance-dev
