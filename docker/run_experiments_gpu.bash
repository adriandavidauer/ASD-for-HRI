#!/bin/bash
## ['VVAD-LRS3-LSTM', 'CNN2Plus1D', 'CNN2Plus1D_Filters', 'CNN2Plus1D_Layers', 'CNN2Plus1D_Light', 'LipShape', 'FaceShape']


export MODEL=LipShape
docker run --gpus all -v $HOME/ASD-for-HRI/data:/app/data asd4hri --data_dir /app/data --no_download --architecture $MODEL

export MODEL=CNN2Plus1D_Light
docker run --gpus all --name ${MODEL} -v $HOME/ASD-for-HRI/data:/app/data asd4hri --data_dir /app/data --no_download --predictions_dir "predictions_${MODEL}" --architecture $MODEL 

export MODEL=VVAD-LRS3-LSTM
docker run --gpus all --name ${MODEL} -v $HOME/ASD-for-HRI/data:/app/data asd4hri --data_dir /app/data --no_download --predictions_dir "predictions_${MODEL}" --architecture $MODEL 

export MODEL=CNN2Plus1D
docker run --gpus all --name ${MODEL} -v $HOME/ASD-for-HRI/data:/app/data asd4hri --data_dir /app/data --no_download --predictions_dir "predictions_${MODEL}" --architecture $MODEL 

export MODEL=CNN2Plus1D_Filters
docker run --gpus all --name ${MODEL} -v $HOME/ASD-for-HRI/data:/app/data asd4hri --data_dir /app/data --no_download --predictions_dir "predictions_${MODEL}" --architecture $MODEL 

export MODEL=CNN2Plus1D_Layers
docker run --gpus all --name ${MODEL} -v $HOME/ASD-for-HRI/data:/app/data asd4hri --data_dir /app/data --no_download --predictions_dir "predictions_${MODEL}" --architecture $MODEL 



## Not trained atm
# export MODEL=FaceShape
# docker run --gpus all --name ${MODEL} -v $HOME/ASD-for-HRI/data:/app/data asd4hri --data_dir /app/data --no_download --predictions_dir "predictions_${MODEL}" --architecture $MODEL 

export MODEL=TalkNet

docker run --gpus all -v "/Data/data:/app/data" --entrypoint "python" --name "${MODEL}" asd4hri src/TalkNet-ASD/unitalk_on_TalkNet.py --videoFolder /app/data/videos/val --predictions_dir "/app/data/predictions_${MODEL}"