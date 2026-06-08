# DGX-hosted pyannote.audio diarization service (#926).
#
# Built from NVIDIA's PyTorch image so CUDA + cuDNN are present without us
# pinning versions. pyannote 4.x + torchaudio + torchcodec follow.
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

# Pin pyannote + transformers to a known-good window. Same versions as the
# operator's laptop venv (after the wave-2 audio migration: pyannote 4 +
# numpy 2 + torch >=2.11). torchaudio is installed in the base image; pin
# to its version range only if a future Dockerfile bump causes drift.
RUN pip install \
        "fastapi>=0.115,<1.0" \
        "uvicorn[standard]>=0.30,<1.0" \
        "python-multipart>=0.0.9" \
        "pyannote.audio>=4.0,<5.0" \
        "torchcodec>=0.14,<1.0" \
        "huggingface_hub>=0.25"

COPY app.py /app/app.py

EXPOSE 8001

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
