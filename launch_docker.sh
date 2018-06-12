#!/bin/bash
sudo nvidia-docker build -t $USER/pytorch:CUDA8-py36 .
sudo nvidia-docker run --rm -ti --volume=$(pwd):/app:rw --volume=/mnt/data/alex/data:/data:rw --workdir=/app --ipc=host $USER/pytorch:CUDA8-py36 /bin/bash
