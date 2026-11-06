# Installation and usage

## Installation

`asd4hri` needs Python 3.10 or newer. dlib is compiled from source during the installation, so
CMake and a C++ compiler must be available (e.g. `apt install build-essential cmake`).

```bash
git clone https://github.com/adriandavidauer/ASD-for-HRI.git
cd ASD-for-HRI
pip install -e .            # the library
pip install -e '.[dev]'     # plus tests, linters and the experiment dependencies
pip install -e '.[docs]'    # plus the tools to build this documentation
```

On x86_64 the GPU build of ONNX Runtime is installed; it falls back to the CPU when no GPU is
available. The model weights are downloaded on first use. To download all of them at once, run
`python src/initialize_models.py`.

For development there is a devcontainer based on `docker/Dockerfile_GPU`; see the
[README](https://github.com/adriandavidauer/ASD-for-HRI#devcontainer).

## Usage

[`ASD`][asd4hri.asd.ASD] processes one frame per call and returns the image and the face boxes
with their predictions:

```python
import cv2
from asd4hri.asd import ASD

pipeline = ASD(architecture='CNN2Plus1D_Light')
video = cv2.VideoCapture('video.mp4')
while True:
    ok, frame = video.read()
    if not ok:
        break
    for box in pipeline(frame)['boxes2D']:
        print(box.coordinates, box.class_name, box.score)
```

The architectures are listed in `Architecture_Options`: `VVAD-LRS3-LSTM`, `CNN2Plus1D`,
`CNN2Plus1D_Filters`, `CNN2Plus1D_Layers`, `CNN2Plus1D_Light`, `LipShape` and `FaceShape`.
`stride` sets how often the model runs, `averaging_window_size` and `weighted` how the last
predictions are smoothed; see the [API reference](api/asd.md) for all options.

To run it live on a camera with PAZ:

```python
from paz.backend.camera import Camera, VideoPlayer
from asd4hri.asd import ASD

pipeline = ASD(architecture='LipShape', annotate_output=True)
player = VideoPlayer((640, 480), pipeline, Camera(0))
player.run()
```
