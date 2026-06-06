"""DGX Spark read-only verify (RFC-089 / ADR-098).

Runs assertions against the live DGX. Exits non-zero on any drift; never
modifies remote state. Use ``make dgx-verify``.

Scope is intentionally check-only — Tailscale + Ollama + NVIDIA driver bring-up
stays operator-manual per #810. Post-ADR-098 there is no convergent work
(the FastAPI embedding shim was removed), so no `deploy.py`.
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
