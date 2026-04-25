# syntax=docker/dockerfile:1

# ============================================================================
# Build Arguments
# ============================================================================
# INSTALL_EXTRAS: Which optional dependencies to install
#   - "" (empty) = core only (minimal, no ML / no cloud SDK bundle)
#   - "llm" = core + API SDK extras only (``pip install '.[llm]'`` — no torch/spaCy/FAISS)
#   - "ml" = core + ML dependencies (~1-3GB with models)
#   Default: "ml" for backwards compatibility
ARG INSTALL_EXTRAS=ml

# PRELOAD_ML_MODELS: Set to "true" to bake in the full ML cache (default), "false" to skip (e.g. docker-build-fast).
# When true and INSTALL_EXTRAS=ml, runs `scripts/cache/preload_ml_models.py --production` — same bundle as CI
# `make preload-ml-models-production` (Whisper tiny.en + base.en, prod+test HF models, hybrid LongT5/FLAN, GIL evidence).
ARG PRELOAD_ML_MODELS=true

# ============================================================================
# Build Stage: Install dependencies and build the package
# ============================================================================
FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONOPTIMIZE=1

# Accept build arguments
ARG INSTALL_EXTRAS
ARG PRELOAD_ML_MODELS

# Validate build arguments
RUN if [ -n "$INSTALL_EXTRAS" ] && [ "$INSTALL_EXTRAS" != "ml" ] && [ "$INSTALL_EXTRAS" != "llm" ] && [ "$INSTALL_EXTRAS" != "" ]; then \
        echo "Error: INSTALL_EXTRAS must be 'ml', 'llm', or '' (empty). Got: '$INSTALL_EXTRAS'" >&2; \
        exit 1; \
    fi && \
    if [ -n "$PRELOAD_ML_MODELS" ] && [ "$PRELOAD_ML_MODELS" != "true" ] && [ "$PRELOAD_ML_MODELS" != "false" ]; then \
        echo "Error: PRELOAD_ML_MODELS must be 'true' or 'false'. Got: '$PRELOAD_ML_MODELS'" >&2; \
        exit 1; \
    fi

ENV PRELOAD_ML_MODELS=${PRELOAD_ML_MODELS}

# Install build dependencies (only if ML extras are requested)
# Build deps are needed for compiling some packages from source if wheels aren't available
# Python 3.12 has good wheel support, so most packages won't need compilation
RUN if [ -n "$INSTALL_EXTRAS" ] && { [ "$INSTALL_EXTRAS" = "ml" ] || [ "$INSTALL_EXTRAS" = "llm" ]; }; then \
        apt-get update && \
        apt-get upgrade -y --no-install-recommends && \
        apt-get install -y --no-install-recommends \
            gcc \
            g++ \
            make \
            python3-dev \
            libc6-dev && \
        rm -rf /var/lib/apt/lists/*; \
    fi

WORKDIR /tmp/build

# Copy minimal files needed for dependency installation
COPY pyproject.toml .
COPY README.md .
# Copy Python source files and package subdirectories
# Package is now in src/podcast_scraper/ directory
COPY src/ ./src/

# Install dependencies based on INSTALL_EXTRAS
# If INSTALL_EXTRAS is empty, install core only (LLM-only)
# If INSTALL_EXTRAS is "ml", install core + ML dependencies
# Use BuildKit cache mount for pip cache to speed up rebuilds
# Note: We use pip cache during install, then purge it at the end to keep image size small
RUN --mount=type=cache,target=/root/.cache/pip \
    export PIP_CACHE_DIR=/root/.cache/pip && \
    if [ -n "$INSTALL_EXTRAS" ] && [ "$INSTALL_EXTRAS" = "ml" ]; then \
        # Install torch CPU-only first (optimization: smaller than CUDA version, ~150MB vs 4GB+)
        # This will be replaced by the version from pyproject.toml if needed, but CPU version is preferred
        pip install torch --index-url https://download.pytorch.org/whl/cpu && \
        # Install core + ML dependencies
        pip install .[ml] && \
        pip uninstall -y podcast-scraper && \
        # Ensure torch CPU-only version is used (reinstall CPU version to override any CUDA version)
        pip install --force-reinstall --no-deps torch --index-url https://download.pytorch.org/whl/cpu; \
    elif [ "$INSTALL_EXTRAS" = "llm" ]; then \
        pip install '.[llm]' && \
        pip uninstall -y podcast-scraper; \
    else \
        # Install core dependencies only (minimal)
        pip install . && \
        pip uninstall -y podcast-scraper; \
    fi && \
    # Cleanup: Remove pip cache from final image to reduce size (cache is preserved in BuildKit cache mount)
    pip cache purge || true && \
    rm -rf /tmp/pip-* /tmp/build-* /tmp/*.whl /tmp/*.tar.gz || true && \
    find /tmp -type f -name "*.pyc" -delete 2>/dev/null || true && \
    find /tmp -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Copy all remaining files and install the podcast_scraper package itself (without reinstalling deps)
# Note: Dependencies are already installed in previous step, so --no-deps is safe
COPY . .
RUN --mount=type=cache,target=/root/.cache/pip \
    export PIP_CACHE_DIR=/root/.cache/pip && \
    pip install --no-deps . && \
    # Verify critical dependencies are available (yaml module)
    python -c "import yaml; print('PyYAML available')" && \
    # Clean up temporary build files
    rm -rf /tmp/pip-* /tmp/build-* /tmp/*.whl /tmp/*.tar.gz || true

# hadolint ignore=SC2261
# Preload ML models (production bundle) when INSTALL_EXTRAS=ml and PRELOAD_ML_MODELS=true.
# Do NOT use BuildKit cache mounts for Whisper/HF paths here: cache mounts are not persisted into
# image layers, so the runtime image would ship without model files. Pip uses its own cache mounts
# on earlier RUN steps only.
# Paths match podcast_scraper.cache.directories (WHISPER_CACHE_DIR, HF_HUB_CACHE).
RUN bash -c 'set -e; \
    mkdir -p /opt/whisper-cache /root/.cache/huggingface/hub; \
    if [ -n "$INSTALL_EXTRAS" ] && [ "$INSTALL_EXTRAS" = "ml" ] && [ "$PRELOAD_ML_MODELS" = "true" ]; then \
        export WHISPER_CACHE_DIR=/opt/whisper-cache; \
        export HF_HUB_CACHE=/root/.cache/huggingface/hub; \
        echo "Preloading ML models (production bundle, same as make preload-ml-models-production)..."; \
        echo "Working directory: $(pwd)"; \
        python scripts/cache/preload_ml_models.py --production || { \
            echo ""; \
            echo "ERROR: ML model preloading failed with exit code $?"; \
            echo "This may be due to network, disk space, or missing dependencies."; \
            echo "To skip: rebuild with --build-arg PRELOAD_ML_MODELS=false (see make docker-build-fast)."; \
            exit 1; \
        }; \
    else \
        echo "Skipping ML model preloading (INSTALL_EXTRAS=$INSTALL_EXTRAS, PRELOAD_ML_MODELS=$PRELOAD_ML_MODELS)"; \
    fi'

# ============================================================================
# Runtime Stage: Minimal runtime image with only necessary files
# ============================================================================
FROM python:3.12-slim

# Version can be set via build arg: --build-arg VERSION=2.5.0
ARG VERSION=latest

# Image metadata labels (OCI standard)
LABEL org.opencontainers.image.title="Podcast Scraper" \
      org.opencontainers.image.description="Download podcast transcripts from RSS feeds with optional Whisper fallback" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.source="https://github.com/chipi/podcast_scraper" \
      org.opencontainers.image.documentation="https://chipi.github.io/podcast_scraper" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.vendor="Podcast Scraper Maintainers" \
      org.opencontainers.image.authors="Podcast Scraper Maintainers"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONOPTIMIZE=1 \
    XDG_CACHE_HOME=/opt/whisper-cache \
    WHISPER_CACHE_DIR=/opt/whisper-cache \
    HF_HUB_CACHE=/home/podcast/.cache/huggingface/hub \
    PODCAST_SCRAPER_CONFIG=/app/config.yaml \
    PODCAST_SCRAPER_WORK_DIR=/app \
    PATH="/home/podcast/.local/bin:$PATH" \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

# Install runtime dependencies. ffmpeg is needed in BOTH modes:
#   - ml: local Whisper transcription decodes mp3/m4a → 16k mono.
#   - llm/cloud-thin: ``preprocessing_enabled: true`` in the cloud
#     profiles still runs client-side audio normalization (loudness +
#     silence trim + bitrate) before uploading to the cloud STT API,
#     so ffmpeg is required there too. Earlier versions skipped it for
#     LLM-only and stack-test cloud-thin runs failed with ``ffmpeg is
#     not installed or not in PATH``.
ARG INSTALL_EXTRAS
RUN apt-get update && \
    apt-get upgrade -y --no-install-recommends && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        supervisor && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user for security
# UID 1000 is a common convention for application users
RUN useradd -m -u 1000 -s /bin/bash podcast && \
    # Create necessary directories with proper ownership
    # /app/output: pre-created so a fresh `corpus_data` named volume
    # inherits podcast:podcast ownership on first mount (#666 follow-up
    # — without this, running `make stack-test-run` before
    # `make stack-test-up` left the mountpoint root-owned and the
    # pipeline failed with PermissionError on /app/output/run_*).
    # Mirrors docker/api/Dockerfile's mkdir + chown of /app/output.
    mkdir -p /app/output \
             /opt/podcast_scraper/examples \
             /opt/whisper-cache \
             /etc/supervisor/conf.d \
             /var/log/supervisor \
             /var/log/podcast_scraper \
             /var/run/supervisor \
             /home/podcast/.cache/huggingface && \
    # Set ownership for application directories
    chown -R podcast:podcast /app \
                             /opt/podcast_scraper \
                             /opt/whisper-cache \
                             /home/podcast/.cache \
                             /var/log/supervisor \
                             /var/log/podcast_scraper \
                             /var/run/supervisor \
                             /etc/supervisor/conf.d

# Copy installed packages from builder stage
# Copy Python packages from builder's site-packages
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Bake ML caches from builder (empty dirs when PRELOAD_ML_MODELS=false or INSTALL_EXTRAS="").
# Paths must match WHISPER_CACHE_DIR and HF_HUB_CACHE above.
COPY --from=builder --chown=podcast:podcast /opt/whisper-cache /opt/whisper-cache
COPY --from=builder --chown=podcast:podcast /root/.cache/huggingface /home/podcast/.cache/huggingface

# Copy application files and set permissions
COPY --chown=podcast:podcast config/examples/config.example.* /opt/podcast_scraper/examples/
# Ship packaged pipeline profiles at the cwd-relative path
# ``Config._resolve_profile`` and ``server.profile_presets`` look up
# first. Without this the pipeline can't resolve ``profile:
# airgapped_thin`` etc. on disk inside the container.
COPY --chown=podcast:podcast config/profiles/ /app/config/profiles/
COPY --chown=podcast:podcast docker/pipeline/supervisord.conf /etc/supervisor/supervisord.conf
COPY --chown=podcast:podcast docker/pipeline/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

WORKDIR /app

# ============================================================================
# Recommended Volume Mounts
# ============================================================================
# For production deployments, consider mounting these directories as volumes:
#
# 1. Config file (required):
#    -v /host/config.yaml:/app/config.yaml
#
# 2. Output directory (recommended for persistence):
#    -v /host/output:/app/output
#    (or mount to location specified in config file)
#
# 3. Model cache directories (recommended for ML variant, speeds up restarts):
#    -v /host/whisper-cache:/opt/whisper-cache
#    -v /host/huggingface-cache:/home/podcast/.cache/huggingface
#    Benefits: Models persist across container restarts, faster startup
#
# 4. Log directories (optional, for supervisor mode):
#    -v /host/logs:/var/log/podcast_scraper
#    -v /host/supervisor-logs:/var/log/supervisor
#
# 5. Supervisor config (optional, for supervisor mode):
#    -v /host/supervisor.conf:/etc/supervisor/conf.d/podcast_scraper.conf
#
# Example docker run command:
#   docker run -v ./config.yaml:/app/config.yaml \
#              -v ./output:/app/output \
#              -v ./whisper-cache:/opt/whisper-cache \
#              -v ./huggingface-cache:/home/podcast/.cache/huggingface \
#              podcast-scraper:latest

# Switch to non-root user
# Note: Supervisor may need root for some operations, but we'll run the service as non-root
USER podcast

# Health check: Verify Python and podcast_scraper module are accessible
# This is a lightweight check that doesn't require config file or network access
# Interval: check every 30 seconds
# Timeout: 10 seconds per check
# Start period: 40 seconds grace period on startup
# Retries: 3 consecutive failures before marking unhealthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import podcast_scraper; import sys; sys.exit(0)" || exit 1

# Use entrypoint script for service-oriented execution
ENTRYPOINT ["/app/entrypoint.sh"]
