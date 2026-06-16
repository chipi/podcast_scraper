# GitHub #659: FastAPI viewer API (no bundled SPA — Nginx serves static + proxies /api/*).
#
# Preflight: the HTTP stack alone is not enough for ``create_app`` (routers import LanceDB / embeddings).
# This image installs the same FastAPI/uvicorn pins as ``[dev]`` in pyproject.toml (no full ``.[dev]`` —
# avoids pytest/black in production) plus ``.[search]`` (CPU torch, lancedb, sentence-transformers)
# without the full ``.[ml]`` stack (Whisper/spaCy/llama-cpp). See docs/guides/DOCKER_SERVICE_GUIDE.md.
#
# Build from repository root: docker build -f docker/api/Dockerfile .

# syntax=docker/dockerfile:1

FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PODCAST_STACK_OUTPUT_DIR=/app/output

# Pull latest Debian security patches on top of whatever the base image
# was built with. Closes the lag between python:3.12-slim-bookworm's last
# rebuild and the latest Debian advisories. Won't fix CVEs Debian hasn't
# patched yet (those are dismissed separately via reachability analysis;
# see docs/ci/CODEQL_DISMISSALS.md Type 2). Adds ~30s on cold builds.
RUN apt-get update \
    && apt-get upgrade -y --no-install-recommends \
    && apt-get install -y --no-install-recommends curl util-linux \
    && apt-get clean \
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
    && pip install \
         "fastapi>=0.115.0,<1.0.0" \
         "uvicorn[standard]>=0.32.0,<1.0.0" \
         "prometheus-fastapi-instrumentator>=7.0.0,<8.0.0" \
         "apscheduler>=3.10.0,<4.0.0" \
         ".[search]" "numpy>=1.24.0" \
    && pip install --force-reinstall --no-deps torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-deps . \
    && python -c "from podcast_scraper.server.app import create_app; print('create_app import ok')"

# Preload the sentence-transformers embedding model used by ``GET
# /api/search``. The endpoint calls ``embedding_loader.encode(...,
# allow_download=False)`` so the model MUST be cached on disk before
# runtime — otherwise queries fail with ``embed_failed`` even though
# the LanceDB index is present. ``HF_HOME`` is shared across users via
# ``/opt/podcast_hf``; the entrypoint chowns it for the runtime user.
ENV HF_HOME=/opt/podcast_hf
RUN mkdir -p "$HF_HOME" \
    && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Ship packaged pipeline profiles so the operator-config endpoint can
# enumerate them via ``GET /api/operator-config``. Profiles live
# outside the Python package (``config/profiles/``) so they can't ride
# along in the wheel — copy them into the working dir at the cwd-
# relative path the resolver checks first
# (``server.profile_presets._profile_directories``).
COPY config/profiles/ ./config/profiles/

# Ship the packaged ``viewer_operator.example.yaml`` so the operator-config
# auto-seed can populate ``viewer_operator.yaml`` on a fresh corpus. Same
# story as profiles — the file lives under ``config/examples/`` (outside
# the Python package), so it can't ride along in the wheel.
COPY config/examples/ ./config/examples/

# Ship the search backend/serving config so hybrid retrieval (RFC-090 Phase 2) is
# togglable in the container via ``serving.hybrid_enabled`` — it lives outside the
# wheel like the profiles above, and ``search.hybrid_search`` reads it at this
# cwd-relative path (or via the PODCAST_SEARCH_CONFIG / PODCAST_HYBRID_SEARCH envs).
COPY config/search.yaml ./config/search.yaml

COPY docker/api/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# #666 review item #5: run as a non-root user, consistent with
# ``docker/pipeline/Dockerfile``. The entrypoint chowns ``/app/output``
# so the user can write corpus artifacts and LanceDB lock files on a
# fresh named volume (root-owned on first mount). When
# ``PODCAST_PIPELINE_EXEC_MODE=docker`` is enabled, the entrypoint also
# adds ``podcast`` to the group that owns the host-mounted
# ``/var/run/docker.sock`` so ``docker compose`` works without root.
ARG PODCAST_UID=1000
ARG PODCAST_GID=1000
RUN groupadd -g "${PODCAST_GID}" podcast \
    && useradd -u "${PODCAST_UID}" -g "${PODCAST_GID}" -m -s /bin/bash podcast \
    && mkdir -p /app/output \
    && chown -R podcast:podcast /app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:8000/api/health" >/dev/null || exit 1

ENTRYPOINT ["/entrypoint.sh"]
# Default to per-corpus operator-config mode (the API writes
# ``<corpus>/viewer_operator.yaml`` for each corpus root, no static
# ``--config-file``). Operators that want single-config mode can
# override the CMD to add ``--config-file <path>`` and bind-mount their
# YAML on top.
CMD ["python", "-m", "podcast_scraper.cli", "serve", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--no-static", \
     "--output-dir", "/app/output", \
     "--enable-feeds-api", \
     "--enable-operator-config-api", \
     "--enable-jobs-api"]
