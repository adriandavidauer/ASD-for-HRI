# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`asd4hri`: Active Speaker Detection (ASD) for Human-Robot Interaction. It is built on the PAZ framework (`pypaz`, installed from the fork `adriandavidauer/paz`) with Keras 3 / TensorFlow. Experiments evaluate it on the UniTalk and AVA ActiveSpeaker datasets.

## Commands

```bash
pip install -e '.[dev]'                       # dev install (devcontainer postCreateCommand)
pytest                                        # all tests (testpaths=tests, pythonpath=src)
pytest tests/test_processors.py::test_name    # single test
ruff check .                                  # docstring lint (rules D + DOC, preview, google convention)
interrogate -c pyproject.toml -v              # docstring coverage
python src/initialize_models.py               # pre-download all model weights
```

Tests load real models, so the first run downloads weights.

### Quality CI (`.github/workflows/quality.yml`)
CI runs only ruff and interrogate, not pytest. It is a **ratchet** against `.github/quality-baseline.json`: the job fails if the ruff violation count goes up or docstring coverage goes down. When the numbers improve, update the baseline file. The ruff version is pinned to `0.16.9` in both `pyproject.toml` and the workflow; keep them in sync because preview rules change between releases. CI also rewrites the badges in `README.md` (between `<!-- badges:start/end -->`) and commits them, so don't edit those lines by hand.

Write new docstrings in Google style with `Args:`/`Returns:`/`Raises:` sections that match the signature (the DOC rules check this).

## Architecture

### Library: `src/asd4hri/` (the only installed package)
Everything is a PAZ `Processor`/`SequentialProcessor` chain:

- `asd.py`
  - `ASD(Processor)` is the per-frame entry point. It detects faces (YuNet by default, or RetinaFace or HaarCascade), squares, clips and crops the boxes, then calls `ClassifyVVAD` on the batch of crops and writes `class_name`/`score` into each `Box2D`. Until enough frames have been buffered, boxes are labelled `'No Prediction yet'` with score -1.
  - `ClassifyVVAD` is the temporal part: preprocess (image resize, or dlib landmarks for `*Shape` architectures), then `BufferFeatures` (a per-entity sliding window of 38 frames with a `stride`), then the model through `PredictWithNones`, then a second `BufferFeatures` plus `AveragePredictions` to smooth the output.
  - **Entities are identified by their index in the detection batch.** There is no real face tracking: `FaceTracker` (`face_tracking_approach`, default `'sort'`) only sorts the boxes by position, so the buffers assume faces keep a stable left-to-right order across frames. `'face_recognition'` is not implemented yet.
  - `Architecture_Options` lists the valid architectures. `CNN2Plus1D*` and `VVAD-LRS3-LSTM` come from PAZ. `LipShape` and `FaceShape` are Keras files downloaded into `src/models/`.
  - PAZ's class order is (speaking, not-speaking); `ASD` swaps it on purpose.
  - `DetectVVAD` is deprecated.
- `processors.py`: the custom processors (buffering, averaging, shape features, the face detectors). The ONNX face detector weights live in `src/models/`.
- `retinaface_onnx.py`: prior boxes, decoding and NMS for the ONNX RetinaFace model.
- `models.py`: rebuilds the landmark LSTM in Keras 3 format.

### Experiments: `src/experiments/` (not packaged; run as modules from the repo root)
- `run_evaluation.py` is the Docker entrypoint (`python -m src.experiments.run_evaluation`). It runs the prediction phase from `run_full_pipeline.py`, then the scoring phase from `stats.py`.
- `run_full_pipeline.py` fetches the videos (UniTalk via `download_uni_talk.py`/yt-dlp, AVA via `ava_dataset.py`) and calls `run_asd_on_video.py` for each video. That writes `<predictions_dir>/<video_id>.csv` with boxes normalised to [0, 1], plus `aggregate_time.csv`.
- `stats.py` is deliberately standalone: it only needs pandas and numpy and imports nothing from the repo, so it can run outside the image. It matches predictions to ground truth by IoU and timestamp tolerance. It requires `aggregate_time.csv` in the predictions folder.
- `--predictions_dir`, `--stats_dir` and `--groundtruth_csv` are resolved **relative to `--data_dir`**.

### Other
- `src/TalkNet-ASD` and `src/LR-ASD` are git submodules (forks of baseline ASD models) and are excluded from ruff and interrogate. `src/eval/` holds the official AVA evaluation script. `src/model_training/` holds the training scripts.
- Docker: `docker/Dockerfile_GPU` is both the devcontainer image and the evaluation image. Build it from the repo root: `docker build -f docker/Dockerfile_GPU -t unitalk-gpu_buffer .`. Arguments passed to `docker run` replace `CMD`, so you must repeat `--data_dir /app/data`. The README documents all evaluation flags.
- The devcontainer does not persist data. The host home directory is mounted at `/home/developer/host-home`.
