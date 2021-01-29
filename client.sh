#! /bin/bash -e

MODE=${1:-it}
MODELREPO=~/modelstores/gwe2e
TAG=20.11
docker run --rm -$MODE \
    -v $MODELREPO:/repo \
    -p 8000-8002:8000-8002 \
    --name tritonserver \
    --gpus all \
    nvcr.io/nvidia/tritonserver:20.11-py3 \
        bin/tritonserver --model-repository /repo

