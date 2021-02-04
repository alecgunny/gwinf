#! /bin/bash -e

COUNT=${1:-1}
STRIDE=${2:-0.002}
MODELREPO=~/modelstores/gwe2e
TAG=20.11

if [[ ! -d $MODELREPO ]]; then mkdir -p $MODELREPO; fi
if [[ ! -z $(ls $MODELREPO) ]]; then rm -rf $MODELREPO/*; fi

docker run --rm -it \
    -v $PWD/exportlib:/opt/exportlib \
    -v $PWD:/home/docker \
    -v $MODELREPO:/repo \
    --workdir /home/docker \
    --gpus all \
    -u $(id -u):$(id -g) \
    gwe2e/export:$TAG-dev \
        --count $COUNT \
        --platform onnx \
        --kernel-stride $STRIDE

# touch empty file in ensemble model version
# dir so that gsutil knows to copy it
touch $MODELREPO/gwe2e/1/model.empty
gsutil cp -r $MODELREPO/* gs://gwe2e/repo/
gsutil ls gs://gwe2e/repo
