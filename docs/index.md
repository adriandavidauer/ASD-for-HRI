# ASD-for-HRI

`asd4hri` is a scalable Active Speaker Detection (ASD) for Human-Robot Interaction. For every video
frame it detects the faces and classifies each of them as speaking or not speaking from the last
frames of that face. It is built on the [PAZ](https://github.com/oarriaga/paz) framework with
Keras 3 and TensorFlow, and is evaluated on the UniTalk and AVA ActiveSpeaker datasets.

## How it works

Everything is a chain of PAZ processors:

1. [`ASD`][asd4hri.asd.ASD] is the per-frame entry point. It detects the faces (YuNet by default,
   or RetinaFace or a Haar cascade), squares, clips and crops their boxes, and writes a
   `class_name` and a `score` into each box.
2. [`ClassifyVVAD`][asd4hri.asd.ClassifyVVAD] is the temporal part. It preprocesses every face
   (resizing for the image models, dlib landmarks for `LipShape` and `FaceShape`), buffers the
   last 38 frames of every face with [`BufferFeatures`][asd4hri.processors.BufferFeatures], runs
   the model every `stride` frames and averages the last predictions.

Until enough frames of a face are buffered, its box is labelled `'No Prediction yet'` with a score
of -1. Faces are identified by their position in the detection batch: there is no face tracking
yet, the boxes are only sorted by position
([`FaceTracker`][asd4hri.processors.FaceTracker]).

## Contents

- [Installation and usage](usage.md): install the package and run it on a camera or video.
- [Evaluation](evaluation.md): run the evaluation in the Docker image.
- API reference: generated from the docstrings of [`asd4hri.asd`](api/asd.md),
  [`asd4hri.processors`](api/processors.md), [`asd4hri.retinaface_onnx`](api/retinaface_onnx.md)
  and [`asd4hri.models`](api/models.md).
