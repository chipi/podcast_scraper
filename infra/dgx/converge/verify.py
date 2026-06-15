"""DGX Spark read-only verify (RFC-089 / ADR-098 / #814).

Runs assertions against the live DGX. Exits non-zero on any drift; never
modifies remote state. Use ``make dgx-verify``.

Scope is intentionally check-only — Tailscale + Ollama + NVIDIA driver bring-up
stays operator-manual per #810. The faster-whisper-server service (#814) IS
convergent — installed via ``deploy.py`` — but verify still only asserts.
"""

from __future__ import annotations

from pyinfra import host
from pyinfra.operations import server

BASELINE_MODELS = [
    "llama3.3:70b-instruct",
    "qwen2.5:72b-instruct",
    "gemma2:27b-instruct",
    "whisper-large-v3",
]

EXPECTED_TAG = "tag:dgx-llm-host"

# 1. GPU
server.shell(
    name="assert: nvidia-smi reports GB10",
    commands=[
        "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader "
        "| grep -qiE 'GB10|Spark'",
    ],
)

# 1a. CDI spec present + enumerates the GB10 (per #948). Without this,
# nvidia-container-toolkit 1.19.1's mode=auto silently falls back to a
# degraded legacy injection that breaks ctranslate2's CUDA enumeration.
server.shell(
    name="assert: /etc/cdi/nvidia.yaml present + nvidia-ctk lists GPU devices",
    commands=[
        "test -f /etc/cdi/nvidia.yaml",
        "nvidia-ctk cdi list 2>&1 | grep -q 'nvidia.com/gpu=all'",
    ],
)

# 2. Tailscale identity matches expected FQDN + has required tag.
# Use jq instead of inline python -c — quoting nests cleaner and jq is already
# a hard dep of scripts/ops/resolve_dgx_tailnet_host.sh, so we know it's installed.
_want_fqdn = host.data.dgx_fqdn.lower()
server.shell(
    name="assert: tailscale Self matches DGX_TAILNET_FQDN and is tagged dgx-llm-host",
    commands=[
        # Online + DNSName match
        f"j=$(tailscale status --json); "
        f'[ "$(echo "$j" | jq -r .Self.Online)" = true ] '
        f"|| {{ echo 'ERR: tailscale Self not Online' >&2; exit 1; }}; "
        f"name=$(echo \"$j\" | jq -r .Self.DNSName | sed 's/\\.$//' | tr A-Z a-z); "
        f'[ "$name" = "{_want_fqdn}" ] '
        f'|| {{ echo "ERR: tailscale DNSName $name != {_want_fqdn}" >&2; exit 1; }}; '
        f'echo "$j" | jq -e \'.Self.Tags // [] | index("{EXPECTED_TAG}")\' >/dev/null '
        f'|| {{ echo "ERR: missing tag {EXPECTED_TAG}" >&2; exit 1; }}',
    ],
)

# 3. Ollama daemon.
server.shell(
    name="assert: ollama systemd unit active + enabled",
    commands=[
        "systemctl is-active --quiet ollama",
        "systemctl is-enabled --quiet ollama",
    ],
)

server.shell(
    name="assert: ollama API responsive on :11434",
    commands=["curl -fsS --max-time 5 http://127.0.0.1:11434/api/tags >/dev/null"],
)

# 4. Baseline models — warn-only; not a hard failure (pulls are overnight).
server.shell(
    name="warn: report missing baseline Ollama models",
    commands=[
        "set -e; tags=$(curl -fsS http://127.0.0.1:11434/api/tags); "
        + "; ".join(
            f'echo "$tags" | grep -q \'"{m}"\' || echo "::warning::missing Ollama model: {m}"'
            for m in BASELINE_MODELS
        ),
    ],
)

# 5. faster-whisper-server / Speaches (#814) — installed by deploy.py as a
# docker-compose stack (no systemd wrapper; restart: unless-stopped in compose).
server.shell(
    name="assert: faster-whisper container is up",
    commands=[
        "docker ps --filter name=^faster-whisper$ --filter status=running --format '{{.Names}}' "
        "| grep -q '^faster-whisper$'",
    ],
)

# 5a. #920 — assert the compose-level healthcheck is reporting healthy.
# Distinct from "container is up" (above) and "API responds" (below) —
# this catches the middle state where the container is up and the API
# returns but the healthcheck has flagged repeated failures. ``starting``
# is treated as a soft pass because the pre-warm step is generous; if
# we're still ``starting`` after the assert above + the API curl below
# succeed, the model is being loaded and we'll be ``healthy`` shortly.
server.shell(
    name="assert: faster-whisper compose healthcheck not in unhealthy state (#920)",
    commands=[
        "state=$(docker inspect --format '{{.State.Health.Status}}' "
        "faster-whisper 2>/dev/null || echo none); "
        'echo "healthcheck state: $state"; '
        '[ "$state" = "healthy" ] || [ "$state" = "starting" ] || [ "$state" = "none" ]',
    ],
)

# 6. Speaches API responsive on the loopback port. Provider client from laptop
# reaches this via the tailnet ACL on tag:dgx-llm-host:8000; verify runs
# locally on the DGX, so it hits 127.0.0.1.
server.shell(
    name="assert: faster-whisper API responsive on :8000",
    commands=[
        "curl -fsS --max-time 10 http://127.0.0.1:8000/v1/models >/dev/null",
    ],
)

# 6a. Per #948 — the actual GPU readiness signal. Without this check the
# container can be "up" and the API responsive while quietly transcribing
# on CPU at ~3% of GPU speed. nvidia-smi util% is unreliable on GB10
# (unified memory; reports 0% mid-transcription), so we use ctranslate2's
# own enumeration. ``cuda_device_count > 0`` means the custom build's
# CUDA path is reachable; anything else means we're back to the pre-fix
# CPU silent-fallback state and need to rebuild the image.
server.shell(
    name="assert: ctranslate2 inside faster-whisper sees GB10 on cuda",
    commands=[
        "docker exec faster-whisper /home/ubuntu/speaches/.venv/bin/python -c "
        '"import ctranslate2 as c; n=c.get_cuda_device_count(); '
        "print('cuda_device_count=', n); assert n >= 1, 'no CUDA device — see #948'\"",
    ],
)

# 7. pyannote diarize service (#926) — installed by deploy.py alongside
# Speaches. Same Docker pattern + loopback check.
server.shell(
    name="assert: pyannote container is up",
    commands=[
        "docker ps --filter name=^pyannote$ --filter status=running --format '{{.Names}}' "
        "| grep -q '^pyannote$'",
    ],
)

server.shell(
    name="assert: pyannote API responsive on :8001",
    commands=[
        # /health returns 200 only after the pyannote model has finished
        # loading at startup. First boot can take ~30-60s for the model to
        # warm; subsequent restarts are quick.
        "curl -fsS --max-time 30 http://127.0.0.1:8001/health >/dev/null",
        "curl -fsS --max-time 10 http://127.0.0.1:8001/v1/models >/dev/null",
    ],
)

# 8. openai-whisper service (#953) — parallel to speaches on :8002.
# First-party OpenAI Whisper inference code. Runs alongside the
# faster-whisper container until #952 picks the winner.
server.shell(
    name="assert: whisper-openai container is up",
    commands=[
        "docker ps --filter name=^whisper-openai$ --filter status=running --format '{{.Names}}' "
        "| grep -q '^whisper-openai$'",
    ],
)

server.shell(
    name="assert: whisper-openai API responsive on :8002",
    commands=[
        # First boot can take 2-5 minutes — the ~3GB large-v3 model downloads
        # on first startup. Subsequent restarts are quick (cache persists at
        # /opt/llm-models/whisper-cache).
        "curl -fsS --max-time 300 http://127.0.0.1:8002/health >/dev/null",
        "curl -fsS --max-time 10 http://127.0.0.1:8002/v1/models >/dev/null",
    ],
)

# 9. vllm-autoresearch reachability (#928 / 2026-06-12 relocation).
# vllm-autoresearch is provisioned by github.com/chipi/agentic-ai-homelab
# (moved out of podcast_scraper on 2026-06-12). podcast_scraper is a CLIENT
# of the :8003 endpoint, not its provisioner — the only thing we still
# verify here is that the autoresearch sweeps will have something to talk
# to. Container existence + model-matches-compose probes moved to the
# homelab repo's verify script.
server.shell(
    name="assert: vllm-autoresearch API responsive on :8003",
    commands=[
        # vLLM model load on cold cache + large model can take 5-15 min,
        # which is why the compose's healthcheck has a 600s start_period.
        # We give verify the same patience — a fresh DGX boot can hit it.
        "curl -fsS --max-time 900 http://127.0.0.1:8003/health >/dev/null",
        "curl -fsS --max-time 10 http://127.0.0.1:8003/v1/models >/dev/null",
    ],
)
