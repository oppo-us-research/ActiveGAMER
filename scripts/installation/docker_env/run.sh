#!/bin/bash

### Arguments (personalized) ###
u=${1:-hyzhan}
docker_name=${2:-activegamer}
root_dir=$HOME
g=$(id -gn)
DOCKER_IMAGE=${u}/activegamer:1.0

### Run Docker ###
docker run --gpus all --ipc=host \
    --name ${docker_name} \
    --rm \
    -e ROOT_DIR=${root_dir} \
    -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    -v "/home/${u}/projects:/home/${u}/projects" \
    -v "/home/${u}/Data:/home/${u}/Data" \
    -it $DOCKER_IMAGE \
    /bin/bash \
    