#!/usr/bin/env bash

set -e 

# Check if CPU or GPU tools
if [ -z "$TIDL_TOOLS_TYPE" ];then
    echo "TIDL_TOOLS_TYPE unset, defaulting to CPU tools"
    tidl_gpu_tools=0
else
    if [ $TIDL_TOOLS_TYPE == GPU ];then
        tidl_gpu_tools=1
    else
        tidl_gpu_tools=0
    fi
fi

if [ "$tidl_gpu_tools" -eq 1 ]; then
    IMAGE_NAME="edgeai_tidl_tools_x86_ubuntu_22_gpu"
    echo "Running GPU Docker container..."
    docker run -w /home/root --gpus all -it --shm-size=4096m \
        --mount source="$(pwd)",target=/home/root,type=bind \
        "$IMAGE_NAME"
else
    IMAGE_NAME="edgeai_tidl_tools_x86_ubuntu_22"
    echo "Running CPU Docker container..."
    docker run -w /home/root -it --shm-size=4096m \
        --mount source="$(pwd)",target=/home/root,type=bind \
        "$IMAGE_NAME"
fi

############ script antigo ##############
#if [ $tidl_gpu_tools -eq 1 ];then
    #sudo docker run -w /home/root --gpus all -it --shm-size=4096m --mount source=$(pwd),target=/home/root,type=bind edgeai_tidl_tools_x86_ubuntu_22_gpu
#else
    #sudo docker run -w /home/root  -it --shm-size=4096m --mount source=$(pwd),target=/home/root,type=bind edgeai_tidl_tools_x86_ubuntu_22
#fi
