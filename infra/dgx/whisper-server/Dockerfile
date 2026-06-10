# DGX-hosted OpenAI Whisper transcription service (#952 prerequisite).
#
# Built from NVIDIA's PyTorch image so CUDA + cuDNN are already present at
# versions matching the GB10 stack. The openai-whisper library wraps these
# without us having to pin torch ourselves — same base image as
# pyannote-server, same versioning concerns.
#
# Build:    docker build -t podcast-whisper:0.1.0 infra/dgx/whisper-server/
# Run:      docker run --rm --gpus all -p 8002:8002 \
#               -v /opt/llm-models/whisper-cache:/root/.cache/whisper \
#               podcast-whisper:0.1.0
#
# In prod we ship via the compose file dropped by pyinfra deploy.py at
# /opt/whisper-server/docker-compose.yml.

FROM nvcr.io/nvidia/pytorch:25.04-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ffmpeg is required by openai-whisper to decode audio files (mp3, m4a, etc.).
# The NGC base image doesn't ship it. libsndfile1 is a runtime dep for
# soundfile, which openai-whisper uses indirectly via numba audio loaders.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# openai-whisper installs with --no-deps because its setup.py declares a
# strict torch version range that conflicts with NVIDIA's NGC dev-build
# torch (``2.7.0a0+79aa17489c.nv25.4`` doesn't satisfy ``torch>=2.0,<2.8``
# even though the ABI is compatible — same logic as pyannote-server's
# torchaudio dance). We install openai-whisper's runtime deps explicitly
# below; the NGC base already has torch + numpy at compatible versions.
#
# Deps openai-whisper actually uses at runtime (from its source code):
#   - torch              (already in NGC base)
#   - numpy              (already in NGC base)
#   - numba              (jit compile + audio loading)
#   - tqdm               (progress bars; we won't see them but the lib uses them)
#   - more-itertools     (small utility)
#   - tiktoken           (BPE tokenizer for Whisper)
#   - triton             (GPU kernels; included via torch)
RUN pip install \
        "fastapi>=0.115,<1.0" \
        "uvicorn[standard]>=0.30,<1.0" \
        "python-multipart>=0.0.9" \
        "numba>=0.59" \
        "tqdm>=4.66" \
        "more-itertools>=10.0" \
        "tiktoken>=0.5" \
 && pip install --no-deps "openai-whisper>=20231117"

COPY app.py /app/app.py

EXPOSE 8002

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]
