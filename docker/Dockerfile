# syntax=docker/dockerfile:1.7
# ── asd4hri: one multi-arch image for the GPU cluster and the Raspberry Pi (#78) ──────────────
#
# linux/amd64 : TensorFlow + ONNX Runtime + PyTorch with CUDA (pip wheels; the host only needs
#               the NVIDIA driver and container toolkit, run with --gpus all)
# linux/arm64 : the same code on CPU (Raspberry Pi 4/5 with a 64-bit OS)
#
# Targets:
#   base         asd4hri only (library + runtime deps, model weights)
#   experiments  + evaluation deps (yt-dlp, the audio-visual models' torch stack) = default
#
# Published by .github/workflows/docker.yml as ghcr.io/adriandavidauer/asd-for-hri:<tag>,
# `docker pull` picks the right architecture. Local build (current architecture):
#   docker build -f docker/Dockerfile -t asd4hri .
# Run as yourself so that results on the data mount belong to you:
#   docker run --rm --gpus all --user $(id -u):$(id -g) -v /Data/data:/app/data \
#       ghcr.io/adriandavidauer/asd-for-hri:main --data_dir /app/data --no_download
# (on the Pi: drop --gpus all)

ARG PYTHON_VERSION=3.10

# ── base ─────────────────────────────────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim-bookworm AS base
ARG TARGETARCH

LABEL org.opencontainers.image.title="asd4hri" \
      org.opencontainers.image.description="Active Speaker Detection for HRI (amd64+CUDA / arm64 CPU)" \
      org.opencontainers.image.source="https://github.com/adriandavidauer/ASD-for-HRI"

# ffmpeg: video/audio decoding; git: pip installs pypaz from GitHub; libgl1/libglib2.0-0: OpenCV;
# build-essential/cmake/libopenblas-dev: dlib is compiled from source (no wheels, no arm64 wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg git curl libgl1 libglib2.0-0 build-essential cmake libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=0 \
    PIP_ROOT_USER_ACTION=ignore \
    PYTHONUNBUFFERED=1 \
    # weights live outside $HOME, so the image runs with any --user uid:gid
    KERAS_HOME=/opt/keras \
    TF_FORCE_GPU_ALLOW_GROWTH=true

# TensorFlow: CUDA wheels on amd64, the official aarch64 CPU wheel on arm64. Same version on both.
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$TARGETARCH" = "amd64" ]; then pip install "tensorflow[and-cuda]==2.18.0"; \
    else pip install "tensorflow==2.18.0"; fi

# The CUDA libraries from the pip wheels, for ONNX Runtime (TF finds them itself). Harmless on arm64.
ENV NV=/usr/local/lib/python3.10/site-packages/nvidia
ENV LD_LIBRARY_PATH=${NV}/cudnn/lib:${NV}/cublas/lib:${NV}/cuda_runtime/lib:${NV}/cuda_nvrtc/lib:${NV}/cufft/lib:${NV}/curand/lib:${NV}/cusolver/lib:${NV}/cusparse/lib:${NV}/nvjitlink/lib

WORKDIR /app
# dependencies first (cached unless pyproject.toml changes); onnxruntime(-gpu) per arch via markers
COPY pyproject.toml README.md ./
RUN mkdir -p src/asd4hri && touch src/asd4hri/__init__.py \
    && pip install . && pip uninstall -y asd4hri

COPY src/ src/
RUN pip install --no-deps -e . \
    && python src/initialize_models.py \
    && chmod -R a+rX /opt/keras src/models

# ── experiments ──────────────────────────────────────────────────────────────────────────────
FROM base AS experiments
ARG TARGETARCH
# torch on amd64: the CUDA 12.6 build, because the default PyPI wheel (CUDA 13) only runs on
# compute capability >= 7.5 and the cluster's Pascal (6.x) and V100 (7.0) GPUs need older kernels
# (#93). torchvision (needed by pyasd) has to come from the same index, or its ops do not match
# torch. On arm64 (Pi) the default PyPI wheels are the CPU builds.
# The [av] extra adds pyasd (TalkNet, LR-ASD, Light-ASD, LRX-ASD) at a pinned commit.
ENV ASD4HRI_WEIGHTS=/opt/asd4hri/weights
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$TARGETARCH" = "amd64" ]; then \
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126; \
    fi \
    && pip install -e ".[av]" yt-dlp pandas h5py scikit-learn wget pytest \
    && python -c "from asd4hri.av import AV_MODELS, weights_path; [weights_path(n) for n in AV_MODELS]" \
    && chmod -R a+rX /opt/asd4hri

# smoke test on the target architecture: the pipeline builds, every AV model loads its weights
RUN python -c "from asd4hri.asd import ASD; ASD(architecture='LipShape'); \
from asd4hri.av import AV_MODELS, AVModel; [AVModel(n, device='cpu') for n in AV_MODELS]"

RUN mkdir -p /app/data && chmod 777 /app/data
VOLUME ["/app/data"]
ENTRYPOINT ["python", "-m", "src.experiments.run_evaluation"]
CMD ["--data_dir", "/app/data"]
