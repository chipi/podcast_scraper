# syntax=docker/dockerfile:1
FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    XDG_CACHE_HOME=/opt/whisper-cache

# Build args for ML model preloading
# PRELOAD_ML_MODELS: Set to "true" to preload all models, "false" to skip (default: "true")
# WHISPER_MODELS: Comma-separated list of Whisper models to preload.
#                 Default in script is "tiny.en" (for local dev speed), but Docker uses "base.en" (production quality).
#                 Override to use different models (e.g., "tiny.en" for faster builds, "base.en,tiny.en" for multiple).
# TRANSFORMERS_MODELS: Comma-separated list of Transformers models to preload.
#                      If empty (default), script preloads all 4 models (bart-base, bart-large-cnn, distilbart, led-base-16384).
#                      Specify to override (e.g., "facebook/bart-base" for faster builds).
# SKIP_TRANSFORMERS: Set to "1" to skip Transformers preloading entirely (for fast builds)
ARG PRELOAD_ML_MODELS=true
ARG WHISPER_MODELS=base.en
ARG TRANSFORMERS_MODELS=
ARG SKIP_TRANSFORMERS=
ENV PRELOAD_ML_MODELS=${PRELOAD_ML_MODELS}
ENV WHISPER_MODELS=${WHISPER_MODELS}
ENV TRANSFORMERS_MODELS=${TRANSFORMERS_MODELS}
ENV SKIP_TRANSFORMERS=${SKIP_TRANSFORMERS}

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
# Preload ML models using unified script
# Use BuildKit cache mounts for model caches (faster rebuilds when models already downloaded)
# Cache locations:
# - Whisper: /opt/whisper-cache (via XDG_CACHE_HOME)
# - Transformers: /root/.cache/huggingface (via default cache location)
# - spaCy: Installed as dependency, no separate cache needed
RUN --mount=type=cache,target=/opt/whisper-cache \
    --mount=type=cache,target=/root/.cache/huggingface \
    bash -c 'if [ "$PRELOAD_ML_MODELS" = "true" ]; then \
        python scripts/cache/preload_ml_models.py; \
    else \
        echo "Skipping ML model preloading (PRELOAD_ML_MODELS=false)"; \
    fi'

RUN mkdir -p /opt/podcast_scraper/examples \
    && cp examples/config.example.* /opt/podcast_scraper/examples/

RUN mkdir -p /app

WORKDIR /app

# Clean up build directory (pip cache is handled by cache mounts, no need to clean)
RUN rm -rf /tmp/build /root/.cache/torch

ENTRYPOINT ["python", "-m", "podcast_scraper.service"]
