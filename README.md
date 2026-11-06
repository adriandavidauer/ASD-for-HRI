# ASD-for-HRI
[![Quality](https://github.com/adriandavidauer/ASD-for-HRI/actions/workflows/quality.yml/badge.svg)](https://github.com/adriandavidauer/ASD-for-HRI/actions/workflows/quality.yml)
<!-- badges:start -->
[![Ruff linting issues](https://img.shields.io/badge/Ruff%20linting%20issues-70%20issues-yellow)](https://github.com/adriandavidauer/ASD-for-HRI/actions/runs/42231640515)
[![Docstring coverage](https://img.shields.io/badge/Docstring%20coverage-86.3%25-green)](https://github.com/adriandavidauer/ASD-for-HRI/actions/runs/42231640515)
[![Test coverage](https://img.shields.io/badge/Test%20coverage-94%25-brightgreen)](https://github.com/adriandavidauer/ASD-for-HRI/actions/runs/42231640690)
<!-- badges:end -->

A scalable Active Speaker Detection for Human-Robot Interaction

Documentation, including the API reference: [adriandavidauer.github.io/ASD-for-HRI](https://adriandavidauer.github.io/ASD-for-HRI/)

## Install
In your Env run 
```bash
pip install -r requirements.txt
```
## Running Experiments

<!-- --8<-- [start:evaluation] -->
Evaluation script runs inside the GPU Docker image, which runs Predictions and stats calculation. The individual steps can also be run on their own (see *Individual steps* below). All
commands are issued from the repository root.

### 1. Predictions - Docker (GPU)

Build relative to the build context that is provided in the build command, in this case it is '.' (repository root) — the Dockerfile lives in `docker/` but its `COPY` paths are relative:

```bash
docker build -f docker/Dockerfile_GPU --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g) \
    -t unitalk-gpu_buffer .
```

The container runs as a non-root user with the uid/gid given at build time, so the results it
writes to the mounted data directory belong to you. Without the build args it uses 1000:1000. The
mounted directory must be writable by that user.

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
| `--boxes {detector,gt}` | Classify the detected faces (end-to-end, default) or the ground-truth face boxes (literature protocol); see *Comparable metrics* |
| `-v` | Also log INFO to the console |

Stats flags:

| Flag | Meaning |
| --- | --- |
| `--groundtruth_csv` | Ground-truth CSV, relative to `--data_dir` (default `csv/val_orig.csv` for UniTalk, `ava/annotations/ava_val_groundtruth.csv` for AVA) |
| `--stats_dir` | Stats output folder, relative to `--data_dir` (default `stats/<predictions_dir>`) |
| `--iou_threshold` | Minimum IoU for a box match (default `0.5`) |
| `--timestamp_tolerance_ms` | Max prediction/GT frame offset in ms (default `20`) |
| `--workers` | Parallel scoring processes (default: all CPUs) |
| `--skip_stats` | Stop after predictions |

Architectures: `VVAD-LRS3-LSTM`, `CNN2Plus1D`, `CNN2Plus1D_Filters`, `CNN2Plus1D_Layers`,
`CNN2Plus1D_Light`, `LipShape`, `FaceShape`.

UniTalk videos are downloaded from YouTube one at a time during the run; AVA videos are fetched all at once. The ground-truth CSV for either dataset is downloaded and built automatically if missing. With `--no_download`, nothing is downloaded and videos missing from `--data_dir` are skipped with a warning.

The run above writes to the host:

- `/Data/data/predictions_cnn1d_scores/`: one `<video_id>.csv` per video plus `aggregate_time.csv`
  (frames processed and elapsed time).
- `/Data/data/stats/predictions_cnn1d_scores/`: the stats output described below.
- `/Data/data/logs_stats/`: stats log files.

#### Comparable metrics

Every run is scored with the **official AVA script** (`src/eval/get_ava_active_speaker_performance.py`)
in the protocol that matches `--boxes`:

| `--boxes` | Protocol | Comparable with |
| --- | --- | --- |
| `gt` | The classifier runs on the ground-truth face tracks; the detector is skipped. mAP over all GT boxes. | The mAP that AVA / UniTalk / WASD papers report |
| `detector` | **End-to-end mAP@0.5**: detections are matched one-to-one to GT boxes at IoU ≥ 0.5; detections without a GT box count as false positives. | Ego4D's mAP@0.5; the whole pipeline (detector + tracking + classifier) |

In both protocols, a GT box without a score (missed by the detector, or no prediction yet while
a buffer fills) gets score 0, because the official script needs a score for every GT box.

`<stats_dir>/protocol_results.csv` has one row per video and a `pooled` row (all GT boxes of the
run ranked together, as papers report it):

| Column | Meaning |
| --- | --- |
| `map` | Official AVA mAP in the run's protocol |
| `auc` | Area under the ROC curve |
| `precision`, `recall`, `f1`, `balanced_accuracy`, `tp`/`fp`/`fn`/`tn` | At the decision threshold 0.5 |
| `coverage` | Share of GT boxes that got a score |
| `det_recall`, `dropped_faces`, `det_precision` | Face detector: share of GT faces found at IoU ≥ 0.5, number missed, share of detections that are GT faces (`detector` runs only) |
| `n_<bin>`, `map_<bin>`, `det_recall_<bin>` | The same per face width: `small` < 64 px, `medium` 64–128 px, `large` > 128 px |

`summary.json` holds the pooled numbers together with the run configuration (dataset,
architecture, boxes, stride, host, CPU architecture, git commit) and the speed. For `gt` runs,
`ava_groundtruth.csv` and `ava_predictions.csv` can be re-scored with the official script itself:

```bash
python src/eval/get_ava_active_speaker_performance.py \
    -g stats/<run>/ava_groundtruth.csv -p stats/<run>/ava_predictions.csv
```

`aggregate_results.csv` keeps the older metrics (threshold metrics and AP over *matched* boxes only)
for comparison with earlier runs.
<!-- --8<-- [end:evaluation] -->

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

To calculate stats for a single folder, call
[`stats.py`](src/experiments/stats.py) using flags:
(`--predictions_dir`, `--groundtruth_csv`, `--result_dir`, `--workers`) it accepts `--video <id>`,
`--iou_threshold` (default `0.5`), `--timestamp_tolerance_ms` (default `20`, max prediction/GT frame
offset) and `-v`.

## Publishing results

Evaluation results are attached to a GitHub release by hand. For `v0.0.0`:

1. Run the experiments for every architecture on each machine (GPU cluster, Raspberry Pi).
2. On each machine, zip the predictions and stats of each run. The script needs only the Python
   standard library, so it runs outside the image:

   ```bash
   python src/experiments/package_results.py --data_dir /Data/data \
       --predictions_dir predictions_cnn1d_scores [--predictions_dir ...] \
       --out_dir release_assets --label gpu
   ```

   Pass `--all` instead of `--predictions_dir` to package every folder in `--data_dir` that has a
   matching `stats/` folder. Each run becomes `results_<label>_<predictions_dir>.zip`. It holds
   `<predictions_dir>/` (per-video CSVs and `aggregate_time.csv`) and `stats/<predictions_dir>/`
   (detail CSVs and `aggregate_results.csv`) under the same paths as in `--data_dir`. Videos,
   audio and logs are left out. The script stops if a folder, `aggregate_time.csv` or
   `aggregate_results.csv` is missing, and it only overwrites existing zips with `--force`.
3. Tag `main` and push the tag: `git tag v0.0.0 && git push origin v0.0.0`.
4. Create the release for `v0.0.0` in the GitHub web UI (*Releases → Draft a new release*).
5. Upload the zips from every machine as release assets, then publish the release.

# Contributing

## Devcontainer
We provide a Devcontainer for the development of ASD4HRI.
### Prerequisites
You need to [install Docker](https://docs.docker.com/engine/install/) and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.14.1/install-guide.html) if you whish to use GPU support.

### Usage of Data in the Container
The container does not save any data: everything outside the mounted folders is lost when the container is rebuilt. To use data for experiments, or to keep results, use `~/host-home/` (`/home/developer/host-home/`). It is a bind mount of your local home directory (`$HOME` on the host), so files written there end up directly on your machine.

The container runs as the non-root user `developer` (created in `docker/Dockerfile_GPU`), so the mount is placed in that user's home. If you change the user in the Dockerfile, update the mount `target` in `.devcontainer/devcontainer.json` to match.
