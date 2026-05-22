# ── UniTalk VVAD full evaluation pipeline ─────────────────────────────────────
#
# Runs run_full_pipeline.py which:
#   1. Downloads the UniTalk val split (CSV, clip videos, clip audio)
#   2. Evaluates DetectVVAD on every val video
#   3. Writes per-video result files → /app/data/results/<video_id>_results.txt
#   4. Writes aggregate results     → /app/data/results/aggregate_results.txt
#
# Build:
#   docker build -t unitalk-vvad .
#
# Run (download + evaluate, data persisted in a named volume):
#   docker run --rm -v unitalk-data:/app/data unitalk-vvad
#
# Skip download if dataset is already present:
#   docker run --rm -v unitalk-data:/app/data unitalk-vvad --skip_download
#
# Resume an interrupted run:
#   docker run --rm -v unitalk-data:/app/data unitalk-vvad --skip_download --resume
#
# Single video (useful for testing):
#   docker run --rm -v unitalk-data:/app/data unitalk-vvad \
#       --skip_download --video qv3-HaaxGUc
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

LABEL org.opencontainers.image.title="UniTalk VVAD Pipeline" \
      org.opencontainers.image.description="Download + evaluate DetectVVAD on UniTalk val split"

# ── System packages ────────────────────────────────────────────────────────────
# ffmpeg       : H264 video encoding used by FFmpegWriter
# curl         : dataset download (dowload_uni_talk.py uses curl subprocess)
# git          : required to pip-install pypaz directly from GitHub
# libgl1       : OpenCV runtime dependency (libGL.so.1)
# libglib2.0-0 : OpenCV runtime dependency (libgthread-2.0.so.0)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        curl \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ────────────────────────────────────────────────────────
# TensorFlow is installed first (large, slow to resolve) so Docker can cache
# this layer independently from the lighter packages in requirements.txt.
# pypaz lists tensorflow as a dependency and will reuse the pinned version.
RUN pip install --no-cache-dir \
    tensorflow==2.21.0 \
    numpy==2.4.4

COPY requirements.txt .
# requirements.txt installs: pypaz (from GitHub), pandas, yt-dlp, wget
RUN pip install --no-cache-dir -r requirements.txt

# ── Application source ────────────────────────────────────────────────────────
COPY src/           src/
COPY video_list/    video_list/
COPY run_full_pipeline.py .

# ── Data volume ───────────────────────────────────────────────────────────────
# Mount a volume here to persist downloaded data and evaluation outputs across
# container runs.  Expected sub-directories created by the pipeline:
#   data/csv/              annotation CSVs (val_orig.csv)
#   data/videos/val/       full YouTube videos (if --download_videos is set)
#   data/clips_videos/val/ short clip videos (downloaded by default)
#   data/clips_audios/val/ short clip audio  (downloaded by default)
#   data/output_videos/val annotated output MP4s
#   data/results/          per-video and aggregate result text files
VOLUME ["/app/data"]

# ── Entry point ───────────────────────────────────────────────────────────────
# Default behaviour: download the dataset then evaluate all val videos.
# Any extra flags passed to `docker run` are appended to CMD and forwarded
# to run_full_pipeline.py (e.g. --skip_download, --resume, --video ID).
ENTRYPOINT ["python", "run_full_pipeline.py"]
CMD ["--data_dir", "/app/data"]
