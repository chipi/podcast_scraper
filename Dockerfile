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

# Copy minimal files needed for dependency installation
COPY pyproject.toml .
COPY README.md .
# Copy Python source files and package subdirectories
# Package is now in src/podcast_scraper/ directory
COPY src/ ./src/

# Install torch CPU-only first (optimization: smaller than CUDA version, ~150MB vs 4GB+)
# This will be replaced by the version from pyproject.toml if needed, but CPU version is preferred
# Use BuildKit cache mount for pip cache (faster rebuilds)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install all dependencies from pyproject.toml (core + ML extras)
# All versions come from pyproject.toml - no hardcoded versions
# Use BuildKit cache mount for pip cache (faster rebuilds)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir .[ml] && \
    pip uninstall -y podcast-scraper

# Ensure torch CPU-only version is used (reinstall CPU version to override any CUDA version)
# Use BuildKit cache mount for pip cache (faster rebuilds)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --force-reinstall --no-deps torch --index-url https://download.pytorch.org/whl/cpu

# Copy all remaining files and install the podcast_scraper package itself (without reinstalling deps)
COPY . .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir --no-deps .

# hadolint ignore=SC2261
# Use BuildKit cache mount for Whisper model cache (faster rebuilds when models already downloaded)
RUN --mount=type=cache,target=/opt/whisper-cache \
    python - <<'PY'
import os
import whisper

model_csv = os.environ.get("WHISPER_PRELOAD_MODELS", "").strip()
if model_csv:
    models = [m.strip() for m in model_csv.split(",") if m.strip()]
else:
    models = []
# Only preload if models are specified (skip for fast builds when WHISPER_PRELOAD_MODELS is empty)
if models:
    for name in models:
        print(f"Preloading Whisper model: {name}")
        whisper.load_model(name)
else:
    print("Skipping model preloading (WHISPER_PRELOAD_MODELS is empty)")
PY

RUN mkdir -p /opt/podcast_scraper/examples \
    && cp examples/config.example.* /opt/podcast_scraper/examples/

RUN mkdir -p /app

WORKDIR /app

# Clean up build directory (pip cache is handled by cache mounts, no need to clean)
RUN rm -rf /tmp/build /root/.cache/torch

ENTRYPOINT ["python", "-m", "podcast_scraper.service"]
