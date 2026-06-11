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

# 6. Speaches API responsive on the loopback port. Provider client from laptop
# reaches this via the tailnet ACL on tag:dgx-llm-host:8000; verify runs
# locally on the DGX, so it hits 127.0.0.1.
server.shell(
    name="assert: faster-whisper API responsive on :8000",
    commands=[
        "curl -fsS --max-time 10 http://127.0.0.1:8000/v1/models >/dev/null",
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

# 9. vllm-autoresearch service (#928) — NVIDIA-prebuilt vLLM serving an
# open-weight LLM on :8003 for autoresearch summary/GI/KG scoring.
server.shell(
    name="assert: vllm-autoresearch container is up",
    commands=[
        "docker ps --filter name=^vllm-autoresearch$ --filter status=running --format '{{.Names}}' "
        "| grep -q '^vllm-autoresearch$'",
    ],
)

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

server.shell(
    name="assert: vllm-autoresearch serves the model its compose declares",
    commands=[
        # Extract the served model id from the live /v1/models endpoint
        # and the model id declared in the compose, then assert they
        # match. Catches drift if vLLM started against a different model
        # than the compose was written for (e.g., an interrupted swap).
        "served=$(curl -fsS --max-time 10 http://127.0.0.1:8003/v1/models "
        '| python3 -c \'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])\') && '
        # Compose declares the model on the line right after ``- serve``;
        # the value is the next ``- <model>`` entry. awk picks the line
        # that comes one after ``- serve`` and strips list/quote chrome.
        "declared=$(awk '/^      - serve$/{getline; gsub(/^[[:space:]-]+/,\"\"); print; exit}' "
        "/opt/vllm-autoresearch/docker-compose.yml) && "
        'test "$served" = "$declared" '
        '|| { echo "drift: vLLM serves $served but compose declares $declared"; exit 1; }',
    ],
)
