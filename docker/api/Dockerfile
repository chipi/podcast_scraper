# RFC-079 / GitHub #659: FastAPI viewer API (no bundled SPA — Nginx serves static + proxies /api/*).
#
# Preflight: ``.[server]`` alone is not enough for ``create_app`` (routers import FAISS / embeddings).
# This image installs ``.[server]`` plus CPU torch, FAISS, and sentence-transformers (semantic search)
# without the full ``.[ml]`` stack (Whisper/spaCy/llama-cpp). See docs/guides/DOCKER_SERVICE_GUIDE.md.
#
# Build from repository root: docker build -f docker/api/Dockerfile .

# syntax=docker/dockerfile:1

FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PODCAST_STACK_OUTPUT_DIR=/app/output

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Docker CLI + Compose v2 for ``PODCAST_PIPELINE_EXEC_MODE=docker`` (mount ``/var/run/docker.sock`` at runtime).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl gnupg \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc \
    && chmod a+r /etc/apt/keyrings/docker.asc \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian bookworm stable" > /etc/apt/sources.list.d/docker.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends docker-ce-cli docker-compose-plugin \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install ".[server]" \
        "numpy>=1.24.0" \
        "faiss-cpu>=1.9.0,<2.0.0" \
        "sentence-transformers>=5.4.1,<6.0.0" \
        "transformers>=4.41.0,<5.0.0" \
    && pip install --no-deps . \
    && python -c "from podcast_scraper.server.app import create_app; print('create_app import ok')"

COPY docker/api/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Run as root so writes work on a fresh named volume at /app/output (see compose + guide).

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:8000/api/health" >/dev/null || exit 1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-m", "podcast_scraper.cli", "serve", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--no-static", \
     "--output-dir", "/app/output", \
     "--config-file", "/app/config.yaml", \
     "--enable-feeds-api", \
     "--enable-operator-config-api", \
     "--enable-jobs-api"]
