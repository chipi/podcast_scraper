# DGX-hosted pyannote.audio diarization service (#926).
#
# Built from NVIDIA's PyTorch image so CUDA + cuDNN are present without us
# pinning versions. Pinned to pyannote 3.x (NOT 4.x) because the NGC
# 25.04-py3 base ships torch 2.7.0a0+nv25.4 — pyannote 4.x requires
# torch>=2.8.0 which is not available in NGC releases compatible with GB10's
# CUDA stack at the time of this commit. pyannote 3.x is also what runs in
# the in-process LOCAL fallback (providers/ml/diarization/), so DGX-side and
# fallback execute the same library version, keeping behavior consistent.
#
# Build:    docker build -t podcast-pyannote:0.1.0 infra/dgx/pyannote-server/
# Run:      docker run --rm --gpus all -p 8001:8001 --env-file ~/.env \
#               podcast-pyannote:0.1.0
#
# In prod we ship via the compose file dropped by pyinfra deploy.py at
# /opt/pyannote-server/docker-compose.yml.

FROM nvcr.io/nvidia/pytorch:25.04-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pin pyannote to the 3.x line — see comment at the top of this Dockerfile
# for the torch-version reason. app.py is version-compat for both 3.x and
# 4.x via getattr-on-DiarizeOutput, so the bump to 4.x when we move base
# images is non-breaking. torchcodec is intentionally NOT installed:
# app.py loads audio via torchaudio (in the base image), and torchcodec is
# a pyannote-4-only dependency.
#
# CRITICAL: torchaudio is pinned to 2.7.x to match the NGC base image's
# torch 2.7 + CUDA 12 stack. Without an explicit pin, pyannote.audio's
# unbounded `torchaudio` dep resolves to the latest PyPI wheel (2.11.x at
# time of writing), which is built against CUDA 13 and fails to load at
# import time with `libcudart.so.13: cannot open shared object file`. When
# we bump the base image to a CUDA-13 NGC release, bump this pin in lockstep.
#
# We install torchaudio first with --no-deps because PyPI torchaudio wheels
# strict-pin `torch==2.7.0` (exact), but NVIDIA's NGC image ships
# `torch==2.7.0a0+79aa17489c.nv25.4` (custom dev build). The version strings
# differ → pip ResolutionImpossible. --no-deps skips the strict pin; ABIs
# are compatible at the major.minor level (both 2.7).
#
# IMPORTANT: pyannote.audio + speechbrain + torch-audiomentations transitively
# depend on torchaudio (unbounded), so any normal `pip install` for them
# pulls in the latest PyPI wheel (2.11.x, CUDA-13). The strategy:
#
# 1. Install pyannote.audio fully (incl. transitive deps) — this WILL pull
#    the bad torchaudio 2.11.0.
# 2. Install our app's own deps (fastapi etc.).
# 3. LAST: force-reinstall torchaudio with --no-deps at the version that
#    matches the base image's torch 2.7 + CUDA 12. The --force-reinstall
#    overwrites the bad 2.11.0; --no-deps skips torchaudio's strict
#    `torch==2.7.0` exact pin (which would conflict with NVIDIA's
#    `2.7.0a0+nv25.4` dev build). ABIs match at the major.minor level.
#
# When the base image bumps to a CUDA-13 NGC release, drop the final step
# and let pyannote pull its native torchaudio version cleanly.
RUN pip install \
        "fastapi>=0.115,<1.0" \
        "uvicorn[standard]>=0.30,<1.0" \
        "python-multipart>=0.0.9" \
        "pyannote.audio>=3.3,<4.0" \
        "huggingface_hub>=0.25,<1.0" \
 && pip install --force-reinstall --no-deps "torchaudio>=2.7,<2.8"
# huggingface_hub MUST be <1.0 because pyannote 3.x internally calls
# `hf_hub_download(use_auth_token=...)`, and HF Hub 1.x removed that
# parameter (renamed to `token=`). Without this pin, the model load at
# startup raises `TypeError: hf_hub_download() got an unexpected keyword
# argument 'use_auth_token'`. When we bump to pyannote 4.x, the HF Hub 1.x
# pin can be dropped (pyannote 4 uses the new `token=` name internally).

COPY app.py /app/app.py

EXPOSE 8001

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
