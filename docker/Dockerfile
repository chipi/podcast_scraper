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

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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

COPY . .

RUN pip install --no-cache-dir .

RUN mkdir -p /opt/podcast_scraper/examples \
    && cp examples/config.example.* /opt/podcast_scraper/examples/

RUN mkdir -p /app

WORKDIR /app

RUN rm -rf /tmp/build

ENTRYPOINT ["python", "-m", "podcast_scraper.service"]
