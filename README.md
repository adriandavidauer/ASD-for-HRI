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

Predictions run inside the GPU Docker image; stats calculation runs after it. All commands are
issued from the repository root.

### 1. Predictions - Docker (GPU)

Build from the repository root — the Dockerfile lives in `docker/` but its `COPY` paths are
root-relative:

```bash
docker build -f docker/Dockerfile_GPU -t unitalk-gpu_buffer .
```

Run:

```bash
docker run -d --gpus all -v /Data/data:/app/data unitalk-gpu_buffer \
    --data_dir /app/data --no_download \
    --predictions_dir predictions_cnn1d_scores --architecture CNN2Plus1D
```

- `-v /Data/data:/app/data` mounts the host data directory where the pipeline expects it.
- Follow a detached run with `docker logs -f <container>`.

Everything after the image name is forwarded to the entrypoint
(`python -m src.experiments.run_full_pipeline`) and *replaces* the image's default `CMD`

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

Architectures: `VVAD-LRS3-LSTM`, `CNN2Plus1D`, `CNN2Plus1D_Filters`, `CNN2Plus1D_Layers`,
`CNN2Plus1D_Light`, `LipShape`, `FaceShape`.

The run above writes to the host under `/Data/data/predictions_cnn1d_scores/`: one `<video_id>.csv`
per video plus `aggregate_time.csv` (frames processed and elapsed time, consumed by the scoring
step).

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
