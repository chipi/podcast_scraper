"""DGX Spark convergent install: Speaches Whisper server via Docker (#814).

Installs the Whisper transcription service that prod
(``cloud_with_dgx_primary.yaml``) targets via the tailnet. Speaches
is the renamed upstream of ``faster-whisper-server`` — same OpenAI-compatible
API surface, distributed as a CUDA Docker image (the pip package was abandoned
at v0.0.2).

What this lays down on DGX:

- A ``docker-compose.yml`` at ``/opt/faster-whisper/docker-compose.yml`` that
  pulls ``ghcr.io/speaches-ai/speaches:latest-cuda`` and exposes :8000.
- The container is configured with ``restart: unless-stopped`` so it survives
  reboots without a systemd wrapper. Docker daemon already starts at boot.
- A model preload is NOT done in this recipe — Speaches downloads on first
  request (~3 GB for ``Systran/faster-whisper-large-v3``). First transcribe is
  slow; subsequent are fast.

What this deliberately does NOT do:

- Install Docker. The DGX has it from #810 bring-up (operator runs vLLM
  experiments via Docker).
- Wire systemd. Compose's ``restart: unless-stopped`` is enough; one fewer
  moving part. Crash-restart hardening (OOM detection, restart-rate-limit)
  stays out of scope — see #910.

Idempotent: re-running ``make dgx-deploy`` re-renders the compose YAML +
``docker compose up -d``. Container is recreated only when image / config
changes; otherwise no-op.

Operator manual steps still pending (per #810):

- NVIDIA driver + CUDA runtime install.
- HuggingFace cache pre-warm if you want to avoid the ~3 GB first-request
  download latency.

Verify with ``make dgx-verify``.
"""

from __future__ import annotations

from pyinfra.operations import files, server

# Knobs the downstream code (the provider client) expects:
#
# - PORT 8000 — added to the dgx-llm-host ACL alongside :11434 in
#   tailscale/policy.hujson. Don't change without updating the ACL.
# - MODEL ID — Hugging Face repo ID. ``Systran/faster-whisper-large-v3`` is
#   the upstream-recommended unquantized large-v3 weights; Speaches fetches
#   on first request.
SERVICE_NAME = "faster-whisper"
INSTALL_ROOT = "/opt/faster-whisper"
COMPOSE_FILE = f"{INSTALL_ROOT}/docker-compose.yml"
BUILD_CTX = f"{INSTALL_ROOT}/build"
# Derived image built locally on the DGX from
# ``infra/dgx/faster-whisper-server/Dockerfile`` (FROM the pinned
# upstream Speaches image below, plus the #968 Thread B
# temperature-fallback patch baked in at build time).
#
# Pinned to ``v0.9.0-rc.3-cuda`` (2025-12-27, marked non-prerelease
# upstream) per #920. Upgrade procedure:
# 1. Drop a new entry under ``infra/dgx/speaches/decisions/`` with
#    bench results from the candidate version (mirror the homelab
#    autoresearch vLLM pattern).
# 2. Edit BASE_IMAGE below to the new tag.
# 3. ``make dgx-deploy`` — pre-warm + healthcheck below will fail
#    loudly if the new image is broken.
IMAGE = "podcast-speaches:0.1.0"
BASE_IMAGE = "ghcr.io/speaches-ai/speaches:v0.9.0-rc.3-cuda"
PORT = 8000
MODEL = "Systran/faster-whisper-large-v3"

from pathlib import Path as _FasterWhisperPath  # noqa: E402

_FASTER_WHISPER_SRC = _FasterWhisperPath(__file__).resolve().parents[1] / "speaches-gb10"

# HF config lives in the operator's ``~/.env`` on DGX (single source of truth
# for HF_TOKEN / HF_HOME / HF_HUB_CACHE / HF_DATASETS_CACHE). Compose injects it
# via ``env_file:``. The model cache itself is shared with vLLM (also bind-mounts
# /opt/llm-models/huggingface) so weights aren't duplicated.
OPERATOR_ENV_FILE = "/home/markodragoljevic/.env"
HF_CACHE_HOST = "/opt/llm-models/huggingface"

# 1. Install root only — no separate HF cache directory needed; the operator's
# centralized cache at /opt/llm-models/huggingface is the bind-mount target.
files.directory(
    name="dir: /opt/faster-whisper (install root)",
    path=INSTALL_ROOT,
    mode="755",
    present=True,
    _sudo=True,
)

# 1a. HF cache ownership (#1046 — multi-model on-demand install).
#
# The cache dir was originally created root-owned (vLLM compose runs as root
# in its container). Speaches runs as uid 1000 inside its container, so when
# we POST /v1/models/<id> on-demand the speaches process must be able to
# write under HF_HUB_CACHE. Without this chown the install fails with
# PermissionError: [Errno 13] Permission denied: '/opt/llm-models/huggingface/.locks/...'.
#
# Idempotent: if already 1000:1000, this no-ops at the recursive layer
# (pyinfra's `user`/`group` ops detect drift). Pre-existing models inside
# the cache (large-v3 baked in at first deploy) keep working — uid 1000 +
# root group means both speaches (uid 1000) and root-owned containers
# retain read access.
files.directory(
    name=f"chown 1000:1000 {HF_CACHE_HOST} (#1046 — speaches on-demand install)",
    path=HF_CACHE_HOST,
    user="1000",
    group="1000",
    recursive=True,
    present=True,
    _sudo=True,
)

# 1b. Build context for the derived image (mirrors whisper-server / pyannote
# pattern). Holds the Dockerfile that applies the #968 Thread B patch on top
# of the upstream speaches:latest-cuda base.
files.directory(
    name="dir: /opt/faster-whisper/build (Docker build context)",
    path=BUILD_CTX,
    mode="755",
    present=True,
    _sudo=True,
)

files.put(
    name="ship: faster-whisper-server/Dockerfile (#968 Thread B patch)",
    src=str(_FASTER_WHISPER_SRC / "Dockerfile"),
    dest=f"{BUILD_CTX}/Dockerfile",
    mode="644",
    create_remote_dir=False,
    _sudo=True,
)

# 2. docker-compose.yml.
#
# `network_mode: host` so the container binds :8000 directly on the host
# (matching Ollama's :11434 pattern, which is how the tailnet ACL boundary
# becomes the real network gate). Without this we'd need explicit -p 8000:8000
# and the tailscale ACL would still only see the host-side socket — host mode
# is cleaner.
COMPOSE_CONTENT = f"""# Auto-generated by infra/dgx/converge/deploy.py. Edit there, not here.
# Re-run ``make dgx-deploy`` from the laptop to redeploy.

services:
  {SERVICE_NAME}:
    build: {BUILD_CTX}
    image: {IMAGE}
    container_name: {SERVICE_NAME}
    restart: unless-stopped
    network_mode: host
    runtime: nvidia
    # HF_TOKEN / HF_HOME / HF_HUB_CACHE / HF_DATASETS_CACHE all come from the
    # operator's centralized ~/.env on DGX. Update there to update everywhere.
    env_file:
      - {OPERATOR_ENV_FILE}
    environment:
      # Compose-specific (NOT in ~/.env): Speaches-only knobs.
      - WHISPER__MODEL={MODEL}
      - WHISPER__DEVICE=cuda
      # Pinned to int8 per #957, empirically re-confirmed in #948 against
      # the v0.9.0-rc.3-cuda image (CTranslate2 4.8.0).
      # See: infra/dgx/speaches/decisions/2026-06-15-compute-type-int8.md
      # for the full benchmark table. Headline (33-min episode, idle box):
      #   - int8:        407s  (4.89× realtime) ← chosen
      #   - int8_float16:423s  (4.71× realtime) — ties int8 within noise
      #   - float32:     735s  (2.71× realtime)
      #   - float16:    1178s  (1.69× realtime) — slower than fp32
      #   - bfloat16:   1506s  (1.32× realtime) — slowest
      # All compute types now LOAD and produce coherent output on this
      # image (the historical "fp16/bf16 hard-error" claim was specific
      # to the pre-#920 :latest-cuda image, which has since been pinned).
      # fp16/bf16 kernels exist in CTranslate2 4.8.0 for Blackwell sm_120
      # but are sub-optimal — int8 is the fastest path by a wide margin.
      - WHISPER__COMPUTE_TYPE=int8
      - LOG_LEVEL=INFO
      - ENABLE_UI=false
      - UVICORN_HOST=0.0.0.0
      - UVICORN_PORT={PORT}
    volumes:
      # Mount HF cache at the SAME path inside the container as outside, so
      # HF_HOME=/opt/llm-models/huggingface (from ~/.env) resolves correctly
      # from inside. Shared with vLLM's compose so weights aren't duplicated.
      - {HF_CACHE_HOST}:{HF_CACHE_HOST}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    # Healthcheck (#920): without this, ``docker ps`` says "running" even
    # if Speaches is dead inside or still loading the model. The
    # ``/v1/models`` endpoint is what verify.py already curls, so reusing
    # it keeps the health signal aligned. ``start_period`` is generous
    # because cold-start with model download can take minutes; the
    # pre-warm step run after ``compose up -d`` makes the post-deploy
    # health-converge much faster on subsequent restarts.
    healthcheck:
      test: ["CMD", "curl", "-fsS", "http://127.0.0.1:{PORT}/v1/models"]
      interval: 30s
      timeout: 5s
      retries: 6
      start_period: 600s
""".replace("{PORT}", str(PORT))

server.shell(
    name="compose: write /opt/faster-whisper/docker-compose.yml",
    commands=[
        f"cat > {COMPOSE_FILE} <<'EOF'\n{COMPOSE_CONTENT}EOF",
        f"chmod 644 {COMPOSE_FILE}",
    ],
    _sudo=True,
)

# 3. Pull image + start. Idempotent: docker compose only recreates the
# container if the YAML config-hash or image digest changed.
#
# The pull is wrapped in `|| true` so a GHCR slowdown/outage doesn't bail
# the whole deploy. We observed this on 2026-06-08: ghcr.io took >30s to
# respond, pyinfra dropped to "Error: executed 0 commands" + "No hosts
# remaining", and the pyannote install steps further down never ran. With
# the pull tolerated, `docker compose up -d` below uses the locally-cached
# image — fine for the common case where Speaches is already running on
# DGX and we're only redeploying pyannote. The trade-off: a deploy could
# silently miss a NEW upstream image. We accept that since (a) the timing
# message still surfaces success vs warn-only-fall-through, (b) the next
# successful deploy still picks up changes, and (c) image bumps are rare
# enough that catching them via an explicit `dgx-refresh` target later is
# a better tool for the job than gating every deploy on GHCR reachability.
server.shell(
    name="compose: pull base image (tolerated; falls back to cached on GHCR blip)",
    commands=[
        f"docker pull {BASE_IMAGE} "
        f"|| echo '::warning::GHCR pull of {BASE_IMAGE} failed; will use cached base if present'"
    ],
    _sudo=True,
)

# Build the derived image on every deploy. Re-applying the #968 Thread B
# sed patch is cheap (no model download — just a layer over the base) and
# this guarantees the latest pull is patched. The grep-guard in the
# Dockerfile fails the deploy loudly if speaches' upstream shape ever
# changes the temperature pattern; better to know at deploy time than at
# first-bad-output time.
server.shell(
    name=f"compose: build derived image ({IMAGE}, #968 Thread B patch baked in)",
    commands=[f"cd {INSTALL_ROOT} && docker compose build"],
    _sudo=True,
)

server.shell(
    name="compose: up -d (start / restart on config change)",
    commands=[f"cd {INSTALL_ROOT} && docker compose up -d"],
    _sudo=True,
)

# #920 — pre-warm the Whisper model so the first real
# ``/v1/audio/transcriptions`` call doesn't pay the ~3 GB HF download
# cost interactively. The model lives in the shared HF cache
# (``/opt/llm-models/huggingface``, bind-mounted into the container)
# so this is at most a one-time cost on a fresh DGX. We:
#
# 1. Wait until ``/v1/models`` responds (or the healthcheck's
#    start_period expires — same threshold).
# 2. Trigger an instantiation of ``Systran/faster-whisper-large-v3``
#    inside the container via the bundled ``faster_whisper`` package.
#    Successful load pulls the weights into the cache and confirms
#    GB10 + the pinned int8 compute type are working together
#    end-to-end.
#
# Wrapped in ``|| true`` so a one-time pre-warm failure doesn't fail
# the whole deploy — the healthcheck below will still surface the
# problem and the next call will trigger the download anyway. Logs
# stay visible via ``docker logs faster-whisper``.
server.shell(
    name="compose: pre-warm whisper-large-v3 in the model cache (#920)",
    commands=[
        # Wait for /v1/models to respond, up to ~90s, then attempt prewarm.
        f"for i in $(seq 1 30); do "
        f"  if curl -fsS --max-time 3 http://127.0.0.1:{PORT}/v1/models "
        f">/dev/null 2>&1; then break; fi; "
        f"  sleep 3; "
        f"done; "
        f"docker exec {SERVICE_NAME} python3 -c "
        f'"from faster_whisper import WhisperModel; '
        f"WhisperModel('{MODEL}', device='cuda', compute_type='int8'); "
        f'print(\\"prewarmed\\")" '
        f'|| echo "::warning::prewarm step did not complete; '
        f'first transcribe call will pull the model"',
    ],
    _sudo=True,
)


# ---------------------------------------------------------------------------
# #926 — pyannote diarization service. Same shape as Speaches install above:
# Docker-managed, restart: unless-stopped, env_file shared with the operator's
# centralized ~/.env (HF_TOKEN comes from there). Listens on :8001 — the
# legacy embedding-shim slot, already in the tailnet ACL.
# ---------------------------------------------------------------------------

PYANNOTE_INSTALL_ROOT = "/opt/pyannote-server"
PYANNOTE_COMPOSE_FILE = f"{PYANNOTE_INSTALL_ROOT}/docker-compose.yml"
PYANNOTE_BUILD_CTX = f"{PYANNOTE_INSTALL_ROOT}/build"
PYANNOTE_IMAGE = "podcast-pyannote:0.1.0"
PYANNOTE_PORT = 8001
PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"
# We BUILD the image on DGX (no public registry yet) — sources for the
# Dockerfile + app live in the repo at infra/dgx/pyannote-server/. The
# deploy reads those files and ships their contents up.
from pathlib import Path as _Path

_PYANNOTE_SRC = _Path(__file__).resolve().parents[1] / "pyannote-server"

# Install root for the service compose + build context.
files.directory(
    name="dir: /opt/pyannote-server (install root)",
    path=PYANNOTE_INSTALL_ROOT,
    mode="755",
    present=True,
    _sudo=True,
)

files.directory(
    name="dir: /opt/pyannote-server/build (Docker build context)",
    path=PYANNOTE_BUILD_CTX,
    mode="755",
    present=True,
    _sudo=True,
)

# Push Dockerfile + app.py to the build context on DGX.
files.put(
    name="ship: pyannote-server/Dockerfile",
    src=str(_PYANNOTE_SRC / "Dockerfile"),
    dest=f"{PYANNOTE_BUILD_CTX}/Dockerfile",
    mode="644",
    create_remote_dir=False,
    _sudo=True,
)

files.put(
    name="ship: pyannote-server/app.py",
    src=str(_PYANNOTE_SRC / "app.py"),
    dest=f"{PYANNOTE_BUILD_CTX}/app.py",
    mode="644",
    create_remote_dir=False,
    _sudo=True,
)

# docker-compose.yml. Same env_file pattern as Speaches → HF_TOKEN flows
# from the operator's centralized ~/.env. HF cache shared with Speaches +
# vLLM at /opt/llm-models/huggingface so we don't re-download the model.
PYANNOTE_COMPOSE_CONTENT = f"""# Auto-generated. Edit infra/dgx/converge/deploy.py instead.
# Re-run ``make dgx-deploy`` from the laptop to redeploy.

services:
  pyannote:
    build: {PYANNOTE_BUILD_CTX}
    image: {PYANNOTE_IMAGE}
    container_name: pyannote
    restart: unless-stopped
    network_mode: host
    runtime: nvidia
    env_file:
      - {OPERATOR_ENV_FILE}
    environment:
      - PYANNOTE_MODEL={PYANNOTE_MODEL}
      - PYANNOTE_DEVICE=cuda
      - LOG_LEVEL=INFO
      - UVICORN_HOST=0.0.0.0
      - UVICORN_PORT={PYANNOTE_PORT}
    volumes:
      - {HF_CACHE_HOST}:{HF_CACHE_HOST}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
"""

server.shell(
    name="compose: write /opt/pyannote-server/docker-compose.yml",
    commands=[
        f"cat > {PYANNOTE_COMPOSE_FILE} <<'EOF'\n{PYANNOTE_COMPOSE_CONTENT}EOF",
        f"chmod 644 {PYANNOTE_COMPOSE_FILE}",
    ],
    _sudo=True,
)

server.shell(
    name="build: pyannote image (one-time + on Dockerfile/app changes)",
    commands=[f"cd {PYANNOTE_INSTALL_ROOT} && docker compose build"],
    _sudo=True,
)

server.shell(
    name="compose: up -d (start / restart pyannote service)",
    commands=[f"cd {PYANNOTE_INSTALL_ROOT} && docker compose up -d"],
    _sudo=True,
)


# ---------------------------------------------------------------------------
# #953 — openai-whisper transcription service. Runs alongside the speaches
# faster-whisper container on a different port (8002). Reason: speaches'
# bundled ctranslate2 is a community reimplementation of Whisper inference
# (#948 surfaced this). Until #952 validates that faster-whisper's WER
# matches openai-whisper on real podcasts, we want a first-party path
# available. Both services run side-by-side; the consumer picks via URL.
# Same Docker-based shape as Speaches + pyannote.
# ---------------------------------------------------------------------------

WHISPER_INSTALL_ROOT = "/opt/whisper-server"
WHISPER_COMPOSE_FILE = f"{WHISPER_INSTALL_ROOT}/docker-compose.yml"
WHISPER_BUILD_CTX = f"{WHISPER_INSTALL_ROOT}/build"
WHISPER_IMAGE = "podcast-whisper:0.1.0"
WHISPER_PORT = 8002
WHISPER_MODEL = "large-v3"
# Persistent cache for openai-whisper model weights (~3GB for large-v3).
# Separate from the HF cache because openai-whisper uses its own cache layout
# (URL-hashed filenames, not HF's snapshots/blobs structure).
WHISPER_CACHE_HOST = "/opt/llm-models/whisper-cache"

_WHISPER_SRC = _Path(__file__).resolve().parents[1] / "whisper-server"

files.directory(
    name="dir: /opt/whisper-server (install root)",
    path=WHISPER_INSTALL_ROOT,
    mode="755",
    present=True,
    _sudo=True,
)

files.directory(
    name="dir: /opt/whisper-server/build (Docker build context)",
    path=WHISPER_BUILD_CTX,
    mode="755",
    present=True,
    _sudo=True,
)

files.directory(
    name="dir: /opt/llm-models/whisper-cache (model weights persistence)",
    path=WHISPER_CACHE_HOST,
    mode="755",
    present=True,
    _sudo=True,
)

files.put(
    name="ship: whisper-server/Dockerfile",
    src=str(_WHISPER_SRC / "Dockerfile"),
    dest=f"{WHISPER_BUILD_CTX}/Dockerfile",
    mode="644",
    create_remote_dir=False,
    _sudo=True,
)

files.put(
    name="ship: whisper-server/app.py",
    src=str(_WHISPER_SRC / "app.py"),
    dest=f"{WHISPER_BUILD_CTX}/app.py",
    mode="644",
    create_remote_dir=False,
    _sudo=True,
)

# docker-compose.yml. Same env_file + GPU passthrough as the other two
# services. Whisper cache is OUTSIDE the operator's HF cache because
# openai-whisper writes its own URL-hashed layout, not HF's snapshots.
WHISPER_COMPOSE_CONTENT = f"""# Auto-generated. Edit infra/dgx/converge/deploy.py instead.
# Re-run ``make dgx-deploy`` from the laptop to redeploy.

services:
  whisper-openai:
    build: {WHISPER_BUILD_CTX}
    image: {WHISPER_IMAGE}
    container_name: whisper-openai
    restart: unless-stopped
    network_mode: host
    runtime: nvidia
    env_file:
      - {OPERATOR_ENV_FILE}
    environment:
      - WHISPER_MODEL={WHISPER_MODEL}
      - WHISPER_DEVICE=cuda
      - WHISPER_CACHE_DIR=/root/.cache/whisper
      - LOG_LEVEL=INFO
      - UVICORN_HOST=0.0.0.0
      - UVICORN_PORT={WHISPER_PORT}
    volumes:
      # Persistent model cache so the ~3GB large-v3 download survives
      # container restarts. openai-whisper writes to /root/.cache/whisper.
      - {WHISPER_CACHE_HOST}:/root/.cache/whisper
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
"""

server.shell(
    name="compose: write /opt/whisper-server/docker-compose.yml",
    commands=[
        f"cat > {WHISPER_COMPOSE_FILE} <<'EOF'\n{WHISPER_COMPOSE_CONTENT}EOF",
        f"chmod 644 {WHISPER_COMPOSE_FILE}",
    ],
    _sudo=True,
)

server.shell(
    name="build: whisper-openai image (one-time + on Dockerfile/app changes)",
    commands=[f"cd {WHISPER_INSTALL_ROOT} && docker compose build"],
    _sudo=True,
)

server.shell(
    name="compose: up -d (start / restart whisper-openai service)",
    commands=[f"cd {WHISPER_INSTALL_ROOT} && docker compose up -d"],
    _sudo=True,
)

# ----------------------------------------------------------------------
# 4. Observability stack (#943): DCGM exporter + node-exporter + cAdvisor.
#
# All upstream images; no local Dockerfile build. The compose lives at
# infra/dgx/observability/docker-compose.yml in this repo and is shipped
# verbatim. Tailscale ACL opens :9100 / :9400 / :8080 on tag:dgx-llm-host
# (tailscale/policy.hujson, same commit).
# ----------------------------------------------------------------------

OBS_INSTALL_ROOT = "/opt/observability"
OBS_COMPOSE_FILE = f"{OBS_INSTALL_ROOT}/docker-compose.yml"
_OBS_COMPOSE_SRC = (
    _FasterWhisperPath(__file__).resolve().parents[1] / "observability" / "docker-compose.yml"
)

files.directory(
    name="dir: /opt/observability (DGX exporters install root)",
    path=OBS_INSTALL_ROOT,
    mode="755",
    present=True,
    _sudo=True,
)

files.put(
    name="ship: observability/docker-compose.yml (#943)",
    src=str(_OBS_COMPOSE_SRC),
    dest=OBS_COMPOSE_FILE,
    mode="644",
    _sudo=True,
)

server.shell(
    name="compose: up -d (start / restart DCGM + node-exporter + cAdvisor)",
    commands=[f"cd {OBS_INSTALL_ROOT} && docker compose up -d"],
    _sudo=True,
)
