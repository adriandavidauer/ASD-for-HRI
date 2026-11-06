# Evaluation

The evaluation runs the ASD pipeline over the UniTalk or AVA ActiveSpeaker videos and scores the
predictions against the ground truth.

--8<-- "README.md:evaluation"

## Individual steps

The prediction and the stats phase can also be run on their own, and the stats phase can run
outside the Docker image. Both, and the files the stats phase writes, are described in the
[README](https://github.com/adriandavidauer/ASD-for-HRI#individual-steps).
