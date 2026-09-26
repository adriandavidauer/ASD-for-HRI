# ASD-for-HRI
[![Quality](https://github.com/adriandavidauer/ASD-for-HRI/actions/workflows/quality.yml/badge.svg)](https://github.com/adriandavidauer/ASD-for-HRI/actions/workflows/quality.yml)
[![Lint](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/arunimaCh29/274393500a8a3dd25c33fb1591d7a156/raw/lint.json)](https://github.com/adriandavidauer/ASD-for-HRI/actions/workflows/quality.yml)
[![Docstring coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/arunimaCh29/274393500a8a3dd25c33fb1591d7a156/raw/docs.json)](https://github.com/adriandavidauer/ASD-for-HRI/actions/workflows/quality.yml)

A scalable Active Speaker Detection for Human-Robot Interaction

## Install
In your Env run 
```bash
pip install -r requirements.txt
```
## Running Experiments

Evaluation script runs inside the GPU Docker image, which runs Predictions and stats calculation. The individual steps can also be run on their own (see *Individual steps* below). All
commands are issued from the repository root.

### 1. Predictions - Docker (GPU)

Build relative to the build context that is provided in the build command, in this case it is '.' (repository root) — the Dockerfile lives in `docker/` but its `COPY` paths are relative:

```bash
docker build -f docker/Dockerfile_GPU -t unitalk-gpu_buffer .
```

### Full evaluation - `src/experiments/run_evaluation.py`

```bash
docker run -d --gpus all -v /Data/data:/app/data unitalk-gpu_buffer \
    --data_dir /app/data --no_download \
    --predictions_dir predictions_cnn1d_scores --architecture CNN2Plus1D
```

- `-v /Data/data:/app/data` mounts the host data directory where the pipeline expects it.
- Follow a detached run with `docker logs -f <container>`.

Everything after the image name is forwarded to the entrypoint
(`python -m src.experiments.run_evaluation`) and *replaces* the image's default `CMD`, so repeat
`--data_dir`. Prediction flags:

| Flag | Meaning |
| --- | --- |
| `--data_dir` | Root data directory, i.e. the mount point (default `data`) |
| `--dataset {unitalk,ava}` | Dataset to run over (default `unitalk`) |
| `--split` | Dataset split, UniTalk only (default `val`) |
| `--no_download` | Never fetch videos; skip the ones missing on disk |
| `--video <id>` | Run a single video only |
| `--predictions_dir` | Output folder, resolved **relative to `--data_dir`** (default `predictions`) |
| `--architecture` | Model to use (default `CNN2Plus1D_Light`) |
| `--stride` | Frames between predictions |
| `-v` | Also log INFO to the console |

Stats flags:

| Flag | Meaning |
| --- | --- |
| `--groundtruth_csv` | Ground-truth CSV, relative to `--data_dir` (default `csv/val_orig.csv`) |
| `--stats_dir` | Stats output folder, relative to `--data_dir` (default `stats/<predictions_dir>`) |
| `--iou_threshold` | Minimum IoU for a box match (default `0.5`) |
| `--timestamp_tolerance_ms` | Max prediction/GT frame offset in ms (default `20`) |
| `--workers` | Parallel scoring processes (default: all CPUs) |
| `--skip_stats` | Stop after predictions |

Architectures: `VVAD-LRS3-LSTM`, `CNN2Plus1D`, `CNN2Plus1D_Filters`, `CNN2Plus1D_Layers`,
`CNN2Plus1D_Light`, `LipShape`, `FaceShape`.

UniTalk videos are downloaded from YouTube one at a time during the run; AVA videos are fetched all at once. With `--no_download`, videos missing from `--data_dir` are skipped with a warning.

The run above writes to the host:

- `/Data/data/predictions_cnn1d_scores/`: one `<video_id>.csv` per video plus `aggregate_time.csv`
  (frames processed and elapsed time).
- `/Data/data/stats/predictions_cnn1d_scores/`: the stats output described below.
- `/Data/data/logs_stats/`: stats log files.

## Individual steps

The two phases of `run_evaluation.py` can also be run separately.

### 1. Predictions - `src/experiments/run_full_pipeline.py`

Accepts the prediction flags above. To run it inside the image, override the entrypoint:

```bash
docker run -d --gpus all -v /Data/data:/app/data \
    --entrypoint python unitalk-gpu_buffer -m src.experiments.run_full_pipeline \
    --data_dir /app/data --no_download \
    --predictions_dir predictions_cnn1d_scores --architecture CNN2Plus1D
```

or equivalently use the default entrypoint with `--skip_stats`.

### 2. Stats - `src/experiments/run_stats_batch.sh`

Stats calculation runs outside the image. The file `stats.py` has no imports from the rest of the repository, only needs pandas + numpy. The batch script runs it over one or more
prediction folders:

```bash
bash src/experiments/run_stats_batch.sh
```

Configure it by editing
the block at the top of the file: add the folders produced above to `INPUT_FOLDERS` and adjust
`INPUT_BASE` (default `/Data/data`), `OUTPUT_BASE` (default `/Data/data/stats`) and
`GROUNDTRUTH_CSV` (default `/Data/data/csv/val_orig.csv`).

Results land in `$OUTPUT_BASE/<folder>/`: a `<video_id>_detail.csv` per video (one row per
ground-truth box) and `aggregate_results.csv` with a summary row per video plus a final
`micro_average` row. Logs go to `logs_stats/`.

Note:

- `aggregate_time.csv` must exist in the prediction folder; the summary writer reads timing for
  every video unconditionally.

To calculate stats for a single folder, call
[`stats.py`](src/experiments/stats.py) using flags:
(`--predictions_dir`, `--groundtruth_csv`, `--result_dir`, `--workers`) it accepts `--video <id>`,
`--iou_threshold` (default `0.5`), `--timestamp_tolerance_ms` (default `20`, max prediction/GT frame
offset) and `-v`.

# Contributing

## Devcontainer
We provide a Devcontainer for the development of ASD4HRI.
### Prerequisites
You need to [install Docker](https://docs.docker.com/engine/install/) and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.14.1/install-guide.html) if you whish to use GPU support.

### Usage of Data in the Container
The Container does not safe any data. If you want to use Data for Experiments, you can use `/home/vscode/host-home/` which is mounted from your local home directory.
