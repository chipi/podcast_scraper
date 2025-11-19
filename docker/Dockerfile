# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    XDG_CACHE_HOME=/opt/whisper-cache

ARG WHISPER_PRELOAD_MODELS=base.en
ENV WHISPER_PRELOAD_MODELS=${WHISPER_PRELOAD_MODELS}

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p "${XDG_CACHE_HOME}"

WORKDIR /tmp/build

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# Install torch CPU-only (much smaller than CUDA version, ~150MB vs 4GB+)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && rm -rf /root/.cache/pip

# Install remaining ML dependencies (without torch, already installed above)
RUN pip install --no-cache-dir \
    openai-whisper>=20231117 \
    spacy>=3.7.0 \
    transformers>=4.30.0 \
    sentencepiece>=0.1.99 \
    accelerate>=0.20.0 \
    protobuf>=3.20.0 \
    && rm -rf /root/.cache/pip

# Install the podcast_scraper package itself (without reinstalling ML deps)
RUN pip install --no-cache-dir --no-deps .

# hadolint ignore=SC2261
RUN python - <<'PY'
import os
import whisper

model_csv = os.environ.get("WHISPER_PRELOAD_MODELS", "").strip()
if model_csv:
    models = [m.strip() for m in model_csv.split(",") if m.strip()]
else:
    models = []
if not models:
    models = ["base.en"]

for name in models:
    print(f"Preloading Whisper model: {name}")
    whisper.load_model(name)
PY

RUN mkdir -p /opt/podcast_scraper/examples \
    && cp examples/config.example.* /opt/podcast_scraper/examples/

RUN mkdir -p /app

WORKDIR /app

# Clean up build directory and pip cache
RUN rm -rf /tmp/build /root/.cache/pip /root/.cache/torch

ENTRYPOINT ["python", "-m", "podcast_scraper.service"]
