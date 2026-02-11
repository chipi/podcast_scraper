# syntax=docker/dockerfile:1

# ============================================================================
# Build Arguments
# ============================================================================
# INSTALL_EXTRAS: Which optional dependencies to install
#   - "" (empty) = core only (LLM-only, ~50MB, no ML deps)
#   - "ml" = core + ML dependencies (~1-3GB with models)
#   Default: "ml" for backwards compatibility
ARG INSTALL_EXTRAS=ml

# PRELOAD_ML_MODELS: Set to "true" to preload all models, "false" to skip (default: "true")
# Only used when INSTALL_EXTRAS includes "ml"
# WHISPER_MODELS: Comma-separated list of Whisper models to preload.
#                 Default in script is "tiny.en" (for local dev speed), but Docker uses "base.en" (production quality).
#                 Override to use different models (e.g., "tiny.en" for faster builds, "base.en,tiny.en" for multiple).
# TRANSFORMERS_MODELS: Comma-separated list of Transformers models to preload.
#                      If empty (default), script preloads all 4 models (bart-base, bart-large-cnn, distilbart, led-base-16384).
#                      Specify to override (e.g., "facebook/bart-base" for faster builds).
# SKIP_TRANSFORMERS: Set to "1" to skip Transformers preloading entirely (for fast builds, default in Docker)
ARG PRELOAD_ML_MODELS=true
ARG WHISPER_MODELS=base.en
ARG TRANSFORMERS_MODELS=
ARG SKIP_TRANSFORMERS=1

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
ARG WHISPER_MODELS
ARG TRANSFORMERS_MODELS
ARG SKIP_TRANSFORMERS

# Validate build arguments
RUN if [ -n "$INSTALL_EXTRAS" ] && [ "$INSTALL_EXTRAS" != "ml" ] && [ "$INSTALL_EXTRAS" != "" ]; then \
        echo "Error: INSTALL_EXTRAS must be 'ml' or '' (empty). Got: '$INSTALL_EXTRAS'" >&2; \
        exit 1; \
    fi && \
    if [ -n "$PRELOAD_ML_MODELS" ] && [ "$PRELOAD_ML_MODELS" != "true" ] && [ "$PRELOAD_ML_MODELS" != "false" ]; then \
        echo "Error: PRELOAD_ML_MODELS must be 'true' or 'false'. Got: '$PRELOAD_ML_MODELS'" >&2; \
        exit 1; \
    fi

ENV PRELOAD_ML_MODELS=${PRELOAD_ML_MODELS}
ENV WHISPER_MODELS=${WHISPER_MODELS}
ENV TRANSFORMERS_MODELS=${TRANSFORMERS_MODELS}
ENV SKIP_TRANSFORMERS=${SKIP_TRANSFORMERS}

# Install build dependencies (only if ML extras are requested)
# Build deps are needed for compiling some packages from source if wheels aren't available
# Python 3.12 has good wheel support, so most packages won't need compilation
RUN if [ -n "$INSTALL_EXTRAS" ] && [ "$INSTALL_EXTRAS" = "ml" ]; then \
        apt-get update && \
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
    else \
        # Install core dependencies only (LLM-only, no ML)
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
# Preload ML models using unified script (only if ML extras are installed)
# Use BuildKit cache mounts for model caches (faster rebuilds when models already downloaded)
# Cache locations:
# - Whisper: /opt/whisper-cache (via XDG_CACHE_HOME)
# - Transformers: /root/.cache/huggingface (via default cache location)
# - spaCy: Installed as dependency, no separate cache needed
RUN --mount=type=cache,target=/opt/whisper-cache \
    --mount=type=cache,target=/root/.cache/huggingface \
    bash -c 'set -e; \
    if [ -n "$INSTALL_EXTRAS" ] && [ "$INSTALL_EXTRAS" = "ml" ] && [ "$PRELOAD_ML_MODELS" = "true" ]; then \
        echo "Preloading ML models..."; \
        echo "Working directory: $(pwd)"; \
        echo "Python path: $(which python)"; \
        echo "Python version: $(python --version)"; \
        echo "Script exists: $(test -f scripts/cache/preload_ml_models.py && echo yes || echo no)"; \
        python -c "import sys; print(f\"Python executable: {sys.executable}\"); print(f\"Python path: {sys.path[:3]}\")" || true; \
        python scripts/cache/preload_ml_models.py 2>&1 || { \
            echo ""; \
            echo "ERROR: ML model preloading failed with exit code $?"; \
            echo "This may be due to:"; \
            echo "  - Network issues downloading models"; \
            echo "  - Disk space constraints"; \
            echo "  - Missing dependencies (check imports above)"; \
            echo ""; \
            echo "To skip model preloading, rebuild with: --build-arg PRELOAD_ML_MODELS=false"; \
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
    PODCAST_SCRAPER_CONFIG=/app/config.yaml \
    PODCAST_SCRAPER_WORK_DIR=/app \
    PATH="/home/podcast/.local/bin:$PATH" \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

# Install only runtime dependencies
# ffmpeg is only needed for local Whisper transcription (ML mode)
# If using LLM-only mode, ffmpeg is not required
ARG INSTALL_EXTRAS
RUN if [ -n "$INSTALL_EXTRAS" ] && [ "$INSTALL_EXTRAS" = "ml" ]; then \
        apt-get update && \
        apt-get install -y --no-install-recommends \
            ffmpeg \
            supervisor && \
        apt-get clean && \
        rm -rf /var/lib/apt/lists/*; \
    else \
        apt-get update && \
        apt-get install -y --no-install-recommends \
            supervisor && \
        apt-get clean && \
        rm -rf /var/lib/apt/lists/*; \
    fi

# Create non-root user for security
# UID 1000 is a common convention for application users
RUN useradd -m -u 1000 -s /bin/bash podcast && \
    # Create necessary directories with proper ownership
    mkdir -p /app \
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

# Copy application files and set permissions
COPY --chown=podcast:podcast config/examples/config.example.* /opt/podcast_scraper/examples/
COPY --chown=podcast:podcast docker/supervisord.conf /etc/supervisor/supervisord.conf
COPY --chown=podcast:podcast docker/entrypoint.sh /app/entrypoint.sh
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
