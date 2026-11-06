# ASD-for-HRI
[![Quality](https://github.com/adriandavidauer/ASD-for-HRI/actions/workflows/quality.yml/badge.svg)](https://github.com/adriandavidauer/ASD-for-HRI/actions/workflows/quality.yml)
<!-- badges:start -->
[![Ruff linting issues](https://img.shields.io/badge/Ruff%20linting%20issues-41%20issues-yellow)](https://github.com/adriandavidauer/ASD-for-HRI/actions/runs/42310798165)
[![Docstring coverage](https://img.shields.io/badge/Docstring%20coverage-92.8%25-brightgreen)](https://github.com/adriandavidauer/ASD-for-HRI/actions/runs/42310798165)
[![Test coverage](https://img.shields.io/badge/Test%20coverage-97%25-brightgreen)](https://github.com/adriandavidauer/ASD-for-HRI/actions/runs/42251771587)
<!-- badges:end -->

A scalable Active Speaker Detection for Human-Robot Interaction

Documentation, including the API reference: [adriandavidauer.github.io/ASD-for-HRI](https://adriandavidauer.github.io/ASD-for-HRI/)

## Install
In your Env run 
```bash
pip install -r requirements.txt
```
## Docker image (cluster and Raspberry Pi)

One multi-arch image is built by CI (`.github/workflows/docker.yml`) from `docker/Dockerfile`:
`linux/amd64` with CUDA (TensorFlow, ONNX Runtime and PyTorch from pip wheels; the host only
needs the NVIDIA driver and container toolkit) and `linux/arm64` on CPU (Raspberry Pi 4/5 with
a 64-bit OS). `docker pull` picks the right one:

```bash
docker pull ghcr.io/adriandavidauer/asd-for-hri:main
# cluster (GPU)
docker run --rm --gpus all --user $(id -u):$(id -g) -v /Data/data:/app/data \
    ghcr.io/adriandavidauer/asd-for-hri:main --data_dir /app/data --no_download
# Raspberry Pi: the same without --gpus all
```

`--user $(id -u):$(id -g)` makes the results on the data mount yours; the model weights live
outside `$HOME`, so any uid works. Build locally with
`git submodule update --init && docker build -f docker/Dockerfile -t asd4hri .`
(targets: `base` = asd4hri only, `experiments` = default, adds the evaluation and PyTorch deps).
`docker/Dockerfile_GPU` (also the devcontainer) and `docker/Dockerfile_PI` are kept for now.

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
| `--dataset {unitalk,ava,wasd,easycom}` | Dataset to run over (default `unitalk`) |
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
`CNN2Plus1D_Light`, `LipShape`, `FaceShape`, `LipShapeCausal`, `FaceShapeCausal`.

`LipShapeCausal` / `FaceShapeCausal` (#45) predict **every frame from that frame and the frames
before it** (zero look-ahead), from the first frame of a face on. `--causal_mode streaming`
(default) keeps one LSTM state per face and costs one step per frame; `--causal_mode window`
re-runs the model on the last `--window` frames (default 38). Retrain them with
`python -m src.model_training.train_causal_shape /Data/lipFeatures.h5 -f LipShape -o out/`.

Audio-visual baselines (#40, #80; `pip install -e '.[av]'`, needs the git submodules):
`AV:LR-ASD` (0.84 M parameters) and `AV:TalkNet` (15.7 M), both with their AVA weights. They
use the video's own audio track; every face is scored for its newest frame from the last
`--av_window` frames (default 25) and the audio of the same span. Without audio,
`--audio_fallback noise` (default) feeds low-level noise and `visual` uses the model's
visual-only head; `--no_audio` forces the fallback.

UniTalk videos are downloaded from YouTube one at a time during the run; AVA videos are fetched all at once. The ground-truth CSV for either dataset is downloaded and built automatically if missing. With `--no_download`, nothing is downloaded and videos missing from `--data_dir` are skipped with a warning.

The run above writes to the host:

- `/Data/data/predictions_cnn1d_scores/`: one `<video_id>.csv` per video plus `aggregate_time.csv`
  (frames processed and elapsed time).
- `/Data/data/stats/predictions_cnn1d_scores/`: the stats output described below.
- `/Data/data/logs_stats/`: stats log files.

#### Benchmark: all models x datasets x protocols

```bash
python -m src.experiments.run_benchmark --data_dir /app/data --no_download --tag gpu \
    --architectures LipShapeCausal CNN2Plus1D_Light --datasets ava unitalk wasd easycom \
    --boxes gt detector [-- <extra run_evaluation flags>]
```

Every combination is one `run_evaluation` run into `<data_dir>/benchmark/<tag>_<dataset>_<arch>_<boxes>/`
(predictions + stats). Runs that already have a `stats/summary.json` are skipped, so an interrupted
benchmark resumes, and `--tag` keeps cluster and Pi results apart. At the end all summaries are
collected into `benchmark/benchmark.csv` and a booktabs `benchmark/benchmark.tex`.
`--dry_run` prints the commands, `--collect_only` only rebuilds the table.

#### WASD

`--dataset wasd` evaluates on the WASD val split (Roxo et al.; 751 annotated 30-s sub-clips of
164 YouTube videos, CC-BY-SA-4.0). Without `--no_download`, the CSVs (0.3 GB) and the source
videos (28.9 GB) are fetched with gdown from the official repository's Google Drive links into
`<data_dir>/wasd`, and the sub-clips are cut exactly like WASD's own `prepare_setup.py` does.
`protocol_results.csv` then also has one `category:<code>` row per WASD category: OC (optimal
conditions), SI (speech impairment), FO (face occlusion), HVN (human voice noise) and SS
(surveillance settings), taken from WASD's `wasd_metadata.csv`.

#### EasyCom

`--dataset easycom --split test` (sessions 10, 11; `val` = 4, 12) evaluates on EasyCom
(Donley et al. 2021; AR glasses at a table, 20 fps, CC-BY-NC-4.0), the dataset closest to a
stationary social robot. EasyCom has no official ASD protocol; `easycom_dataset.py` defines one:
a face counts as speaking in the frames covered by that participant's speech transcription
segments, and the face boxes become AVA-style rows (`video_id` = `Session_<s>_<file>`). The
videos carry a participant-photo banner on the left; it is cropped away before detection and
the boxes are converted to the cropped image. Download (~36 GB of the 70 GB release, only the
needed folders) with the command in the module docstring into `<data_dir>/easycom`.

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

#### Where the time goes

`ASD` records the time of every pipeline stage (switch off with `ASD(profile=False)`):

| Stage | What it covers |
| --- | --- |
| `read` | Decoding the next video frame |
| `detect`, `track` | Face detector, ordering/tracking of the boxes |
| `crop` | Squaring, clipping and cropping the faces |
| `preprocess` | Resize (image models) or dlib landmarks (`*Shape` models) |
| `buffer`, `normalize` | Per-face temporal buffers, landmark normalisation |
| `model` | The classifier's forward pass, including the GPU sync |
| `average` | Smoothing the scores over the last predictions |
| `audio_decode`, `audio` | Audio-visual models: decoding the track, MFCCs per window |
| `frame` | Everything per frame, from `detect` to `average` |

Per video, `<predictions_dir>/timing/<video_id>.csv` has calls, total, mean/median/p95/max ms and
ms per face for each stage. Over the run, `<stats_dir>/stage_times.csv` gives ms per frame and
the share of the frame time, and `summary.json` has `stage_ms_per_frame`.

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
