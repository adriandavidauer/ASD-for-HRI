#!/bin/bash
## ['VVAD-LRS3-LSTM', 'CNN2Plus1D', 'CNN2Plus1D_Filters', 'CNN2Plus1D_Layers', 'CNN2Plus1D_Light', 'LipShape', 'FaceShape']
# Experiment 1
export MODEL=LipShape
docker run --gpus all -v $HOME/ASD-for-HRI/data:/app/data asd4hri --data_dir /app/data --no_download --architecture $MODEL
# move the results to a specific destination not to be overwritten by the next experiment
mv $HOME/ASD-for-HRI/src/data/results $HOME/ASD-for-HRI/src/data/${MODEL}_results

export MODEL=CNN2Plus1D_Light
docker run --gpus all -v $HOME/ASD-for-HRI/data:/app/data asd4hri --data_dir /app/data --no_download --architecture $MODEL
# move the results to a specific destination not to be overwritten by the next experiment
mv $HOME/ASD-for-HRI/src/data/results $HOME/ASD-for-HRI/src/data/${MODEL}_results

export MODEL=VVAD-LRS3-LSTM
docker run --gpus all -v $HOME/ASD-for-HRI/data:/app/data asd4hri --data_dir /app/data --no_download --architecture $MODEL
# move the results to a specific destination not to be overwritten by the next experiment
mv $HOME/ASD-for-HRI/src/data/results $HOME/ASD-for-HRI/src/data/${MODEL}_results

export MODEL=CNN2Plus1D
docker run --gpus all -v $HOME/ASD-for-HRI/data:/app/data asd4hri --data_dir /app/data --no_download --architecture $MODEL
# move the results to a specific destination not to be overwritten by the next experiment
mv $HOME/ASD-for-HRI/src/data/results $HOME/ASD-for-HRI/src/data/${MODEL}_results

export MODEL=CNN2Plus1D_Filters
docker run --gpus all -v $HOME/ASD-for-HRI/data:/app/data asd4hri --data_dir /app/data --no_download --architecture $MODEL
# move the results to a specific destination not to be overwritten by the next experiment
mv $HOME/ASD-for-HRI/src/data/results $HOME/ASD-for-HRI/src/data/${MODEL}_results

export MODEL=CNN2Plus1D_Layers
docker run --gpus all -v $HOME/ASD-for-HRI/data:/app/data asd4hri --data_dir /app/data --no_download --architecture $MODEL
# move the results to a specific destination not to be overwritten by the next experiment
mv $HOME/ASD-for-HRI/src/data/results $HOME/ASD-for-HRI/src/data/${MODEL}_results



## Not trained atm
# export MODEL=FaceShape
# docker run --gpus all -v $HOME/ASD-for-HRI/data:/app/data asd4hri --data_dir /app/data --no_download --architecture $MODEL
# # move the results to a specific destination not to be overwritten by the next experiment
# mv $HOME/ASD-for-HRI/src/data/results $HOME/ASD-for-HRI/src/data/${MODEL}_results