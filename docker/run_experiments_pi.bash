#!/bin/bash
## ['VVAD-LRS3-LSTM', 'CNN2Plus1D', 'CNN2Plus1D_Filters', 'CNN2Plus1D_Layers', 'CNN2Plus1D_Light', 'LipShape', 'FaceShape']
# Experiment 1
export MODEL=LipShape
docker run -v $HOME/ASD-for-HRI/data:/app/data asd4hri --data_dir /app/data --no_download ----architecture $MODEL
# move the results to a specific destination not to be overwritten by the next experiment
mv $HOME/ASD-for-HRI/data/results $HOME/ASD-for-HRI/data/${MODEL}_results

export MODEL=CNN2Plus1D_Light
docker run -v $HOME/ASD-for-HRI/data:/app/data asd4hri --data_dir /app/data --no_download ----architecture $MODEL
# move the results to a specific destination not to be overwritten by the next experiment
mv $HOME/ASD-for-HRI/data/results $HOME/ASD-for-HRI/data/${MODEL}_results

export MODEL=VVAD-LRS3-LSTM
docker run -v $HOME/ASD-for-HRI/data:/app/data asd4hri --data_dir /app/data --no_download ----architecture $MODEL
# move the results to a specific destination not to be overwritten by the next experiment
mv $HOME/ASD-for-HRI/data/results $HOME/ASD-for-HRI/data/${MODEL}_results

export MODEL=CNN2Plus1D
docker run -v $HOME/ASD-for-HRI/data:/app/data asd4hri --data_dir /app/data --no_download ----architecture $MODEL
# move the results to a specific destination not to be overwritten by the next experiment
mv $HOME/ASD-for-HRI/data/results $HOME/ASD-for-HRI/data/${MODEL}_results

export MODEL=CNN2Plus1D_Filters
docker run -v $HOME/ASD-for-HRI/data:/app/data asd4hri --data_dir /app/data --no_download ----architecture $MODEL
# move the results to a specific destination not to be overwritten by the next experiment
mv $HOME/ASD-for-HRI/data/results $HOME/ASD-for-HRI/data/${MODEL}_results

export MODEL=CNN2Plus1D_Layers
docker run -v $HOME/ASD-for-HRI/data:/app/data asd4hri --data_dir /app/data --no_download ----architecture $MODEL
# move the results to a specific destination not to be overwritten by the next experiment
mv $HOME/ASD-for-HRI/data/results $HOME/ASD-for-HRI/data/${MODEL}_results



## Not trained atm
# export MODEL=FaceShape
# docker run -v $HOME/ASD-for-HRI/data:/app/data asd4hri --data_dir /app/data --no_download ----architecture $MODEL
# # move the results to a specific destination not to be overwritten by the next experiment
# mv $HOME/ASD-for-HRI/data/results $HOME/ASD-for-HRI/data/${MODEL}_results