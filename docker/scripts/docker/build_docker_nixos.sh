#!/usr/bin/env bash

######################################################################
set -e 

script_dir=$(dirname -- ${BASH_SOURCE[0]})

#Check if CPU or GPU tools
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

if ping -c 1 -W 1 artifactory.itg.ti.com &>/dev/null; then
    echo "Dentro da rede TI — usando repositórios internos"
    REPO_LOCATION="artifactory.itg.ti.com/docker-public/library/"
    PROXY="http://webproxy.ext.ti.com:80"
else
    echo "Fora da rede TI — usando Docker Hub público"
    REPO_LOCATION="docker.io/library/"
    PROXY="none"
fi

# --- Mostra variáveis detectadas ---
echo "Using REPO_LOCATION: $REPO_LOCATION"
echo "Using PROXY: $PROXY"

# --- Escolhe Dockerfile e imagem ---
if [ "$tidl_gpu_tools" -eq 1 ]; then
    IMAGE_NAME="edgeai_tidl_tools_x86_ubuntu_22_gpu"
    DOCKERFILE="$script_dir/Dockerfile_GPU"
else
    IMAGE_NAME="edgeai_tidl_tools_x86_ubuntu_22"
    DOCKERFILE="$script_dir/Dockerfile"
fi

# --- Executa build ---
sudo docker build \
    --build-arg REPO_LOCATION="$REPO_LOCATION" \
    --build-arg PROXY="$PROXY" \
    -f "$DOCKERFILE" \
    -t "$IMAGE_NAME" \
    .

echo "Docker image '$IMAGE_NAME' built successfully!"

###### script antigo ####
#if [ -z "$REPO_LOCATION" ];then
    #echo "No REPO_LOCATION specified, using default"
#else
    #echo "Using REPO_LOCATION: $REPO_LOCATION"
#fi

#if [ -z "$PROXY" ];then
    #echo "No PROXY specified"
    #PROXY=none
#else
    #echo "Using PROXY: $PROXY"
#fi

#if [ $tidl_gpu_tools -eq 1 ];then
    #sudo docker build --build-arg REPO_LOCATION=$REPO_LOCATION --build-arg PROXY=$PROXY  -f $script_dir/Dockerfile_GPU -t edgeai_tidl_tools_x86_ubuntu_22_gpu .

#else
    #sudo docker build --build-arg REPO_LOCATION=$REPO_LOCATION --build-arg PROXY=$PROXY  -f $script_dir/Dockerfile -t edgeai_tidl_tools_x86_ubuntu_22 .
#fi 

