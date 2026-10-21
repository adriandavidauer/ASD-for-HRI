#!/bin/bash
# Run from the repository root. The image is built with your uid/gid, so the results written to
# the data mount belong to you (the build is cached after the first run).
docker build -f docker/Dockerfile_GPU --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g) -t asd4hri .

## ['VVAD-LRS3-LSTM', 'CNN2Plus1D', 'CNN2Plus1D_Filters', 'CNN2Plus1D_Layers', 'CNN2Plus1D_Light', 'LipShape', 'FaceShape']


export MODEL=LipShape
docker run --gpus all -v /Data/data:/app/data asd4hri --data_dir /app/data --no_download --architecture $MODEL

export MODEL=CNN2Plus1D_Light
docker run --gpus all --name ${MODEL} -v /Data/data:/app/data asd4hri --data_dir /app/data --no_download --predictions_dir "/app/data/predictions_${MODEL}" --architecture $MODEL 

export MODEL=VVAD-LRS3-LSTM
docker run --gpus all --name ${MODEL} -v /Data/data:/app/data asd4hri --data_dir /app/data --no_download --predictions_dir "/app/data/predictions_${MODEL}" --architecture $MODEL 

export MODEL=CNN2Plus1D
docker run --gpus all --name ${MODEL} -v /Data/data:/app/data asd4hri --data_dir /app/data --no_download --predictions_dir "/app/data/predictions_${MODEL}" --architecture $MODEL 

export MODEL=CNN2Plus1D_Filters
docker run --gpus all --name ${MODEL} -v /Data/data:/app/data asd4hri --data_dir /app/data --no_download --predictions_dir "/app/data/predictions_${MODEL}" --architecture $MODEL 

export MODEL=CNN2Plus1D_Layers
docker run --gpus all --name ${MODEL} -v /Data/data:/app/data asd4hri --data_dir /app/data --no_download --predictions_dir "/app/data/predictions_${MODEL}" --architecture $MODEL 



## Not trained atm
# export MODEL=FaceShape
# docker run --gpus all --name ${MODEL} -v /Data/data:/app/data asd4hri --data_dir /app/data --no_download --predictions_dir "/app/data/predictions_${MODEL}" --architecture $MODEL 

export MODEL=TalkNet
docker run --gpus all -v "/Data/data:/app/data" --entrypoint "python" --name "${MODEL}" asd4hri src/TalkNet-ASD/unitalk_on_TalkNet.py --videoFolder /app/data/videos/val --predictions_dir "/app/data/predictions_${MODEL}"

export MODEL=LR-ASD
docker run --gpus all -v "/Data/data:/app/data" --entrypoint "python" --name "${MODEL}" asd4hri src/LR-ASD/unitalk_on_LRASD.py --videoFolder /app/data/videos/val --predictions_dir "/app/data/predictions_${MODEL}"