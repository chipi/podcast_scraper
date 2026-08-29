#!/usr/bin/env bash
# infra/deploy/deploy-litellm.sh — bring up the PROD LiteLLM gateway (#1357, ADR-142).
#
# Runs ON the VPS as the deploy user (invoked by deploy-litellm.yml after the runner joins
# the tailnet, or by hand). Standalone `-p litellm` compose project — isolated from the app /
# operator / player stacks. The gateway is a LOCAL inference proxy (loopback-bound, never
# public); ALL telemetry ships to the homelab pane over the tailnet.
#
# Required env (staged by the workflow from GH Actions secrets, or set for a manual run):
#   LITELLM_PG_PASSWORD    postgres password (compose-internal + loopback :5433)
#   LITELLM_MASTER_KEY     proxy admin master key (mints virtual keys; never a consumer's)
#   OPENROUTER_API_KEY     the prod gateway's OWN upstream OpenRouter key
#   LANGFUSE_PUBLIC_KEY    } litellm-vps Langfuse project (telemetry → homelab:4000)
#   LANGFUSE_SECRET_KEY    }
#   SENTRY_DSN             litellm-vps GlitchTip project (errors → homelab:8090)
#
# Exit: 0 ok / 1 compose up failed / 3 health failed / 4 smoke failed / 5 secrets missing
set -euo pipefail

REPO_DIR=/srv/podcast-scraper
cd "$REPO_DIR/infra/litellm"

LITELLM_ENV="$REPO_DIR/infra/litellm/.env"
umask 077

# Build .env from the environment when not already staged (the workflow scp's a pre-staged
# .env; a manual run assembles it here). Provider/telemetry keys are REQUIRED — refuse to
# boot a gateway that can't authenticate upstream or would silently drop all telemetry.
if [ ! -f "$LITELLM_ENV" ]; then
  : "${LITELLM_PG_PASSWORD:?set LITELLM_PG_PASSWORD}"
  : "${LITELLM_MASTER_KEY:?set LITELLM_MASTER_KEY}"
  : "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY - the prod gateway upstream key}"
  {
    echo "LITELLM_PG_PASSWORD=${LITELLM_PG_PASSWORD}"
    echo "LITELLM_MASTER_KEY=${LITELLM_MASTER_KEY}"
    echo "OPENROUTER_API_KEY=${OPENROUTER_API_KEY}"
    echo "LANGFUSE_HOST=http://homelab:4000"
    echo "LANGFUSE_PUBLIC_KEY=${LANGFUSE_PUBLIC_KEY:-}"
    echo "LANGFUSE_SECRET_KEY=${LANGFUSE_SECRET_KEY:-}"
    echo "SENTRY_DSN=${SENTRY_DSN:-}"
    echo "HOMELAB_TAILNET_IP="
  } >"$LITELLM_ENV"
fi
chmod 600 "$LITELLM_ENV"

# Resolve the homelab tailnet IP FRESH for extra_hosts so the container can reach the
# remote Langfuse/GlitchTip (Docker's bridge can't resolve MagicDNS `homelab`). Mirrors
# deploy.sh / deploy-operator.sh. NON-fatal: on failure the compose loopback fallback keeps
# the gateway serving; only telemetry drops until it re-resolves.
HL_IP="$(tailscale ip -4 homelab 2>/dev/null | head -1 || true)"
if [ -n "$HL_IP" ]; then
  if grep -qE '^HOMELAB_TAILNET_IP=' "$LITELLM_ENV"; then
    sed -i "s#^HOMELAB_TAILNET_IP=.*#HOMELAB_TAILNET_IP=$HL_IP#" "$LITELLM_ENV"
  else
    echo "HOMELAB_TAILNET_IP=$HL_IP" >>"$LITELLM_ENV"
  fi
  echo "[$(date -u +%FT%TZ)] resolved homelab tailnet IP for telemetry extra_hosts: $HL_IP"
else
  echo "WARN: could not resolve 'homelab' tailnet IP; gateway telemetry (Langfuse/GlitchTip) will not ship until it re-resolves" >&2
fi

# Pin the image tag when the workflow supplies one (repo convention: pin after first boot);
# otherwise the compose default (main-stable) applies. Never fail the deploy on an unset tag.
if [ -n "${LITELLM_IMAGE_TAG:-}" ]; then
  export LITELLM_IMAGE_TAG
  echo "[$(date -u +%FT%TZ)] LITELLM_IMAGE_TAG=${LITELLM_IMAGE_TAG} (pinned)"
fi

COMPOSE=(docker compose -p litellm --env-file "$LITELLM_ENV" -f docker-compose.litellm.yml)

# Also publish the admin UI on the box's OWN tailnet IP so it's reachable from a laptop/phone
# (http://<tailnet-ip>:4001/ui) — gated by the ACL (autogroup:admin -> tag:prod:4001). Loopback
# (base compose) always stays, so the app reaches the gateway on 127.0.0.1 and the gateway never
# depends on tailscale being up. If the self IP can't resolve, deploy loopback-only + warn.
SELF_TS_IP="$(tailscale ip -4 2>/dev/null | head -1 || true)"
if [ -n "$SELF_TS_IP" ]; then
  if grep -qE '^LITELLM_TAILNET_IP=' "$LITELLM_ENV"; then
    sed -i "s#^LITELLM_TAILNET_IP=.*#LITELLM_TAILNET_IP=$SELF_TS_IP#" "$LITELLM_ENV"
  else
    echo "LITELLM_TAILNET_IP=$SELF_TS_IP" >>"$LITELLM_ENV"
  fi
  COMPOSE+=(-f docker-compose.litellm-tailnet.yml)
  echo "[$(date -u +%FT%TZ)] admin UI also on the tailnet: http://$SELF_TS_IP:4001/ui"
else
  echo "WARN: could not resolve the box's tailnet IP — gateway on 127.0.0.1 only (UI via ssh tunnel)" >&2
fi

echo "[$(date -u +%FT%TZ)] pulling + starting litellm gateway..."
# --force-recreate is LOAD-BEARING, not belt-and-braces. config.yaml is bind-mounted as a single
# file, so the container holds the file's INODE from creation time; the repo checkout replaces the
# file atomically (new inode) and a plain `up -d` sees no compose-level change and recreates
# nothing. 2026-08-29: a config-only deploy shipped a new model alias, every step reported success,
# and the gateway kept serving the two-week-old config — the deploy deployed nothing.
if ! "${COMPOSE[@]}" up -d --force-recreate --remove-orphans; then
  echo "ERROR: docker compose up failed" >&2
  exit 1
fi

# Health-gate: LiteLLM's liveliness endpoint via the HOST-published loopback port
# (127.0.0.1:4001 -> container :4000). Host-side curl, NOT `docker exec ... curl`: the
# litellm image is wolfi-minimal and ships no curl, so an in-container curl false-negatives
# even when uvicorn is up (2026-08-02: gateway healthy, gate failed exit 3). Mirrors the
# canonical host-curl gate in deploy.sh / deploy-operator.sh. 45x2s = 90s covers a cold
# start's prisma migrations (~35-40s) plus headroom.
echo "[$(date -u +%FT%TZ)] waiting for gateway health (host curl 127.0.0.1:4001)..."
ok=0
for _ in $(seq 1 45); do
  if curl -fsS http://127.0.0.1:4001/health/liveliness >/dev/null 2>&1; then
    ok=1; break
  fi
  sleep 2
done
if [ "$ok" != 1 ]; then
  echo "ERROR: litellm gateway did not report healthy within 90s" >&2
  "${COMPOSE[@]}" logs --tail=50 litellm >&2 || true
  exit 3
fi

# Ship the gateway's container logs to VictoriaLogs via the shared node Alloy (ADR-121/130):
# drop litellm.alloy into the deploy-writable config.d + hot-reload. NON-fatal — a logging
# hiccup must not fail the gateway deploy. Langfuse carries the LLM *calls*; this carries the
# gateway's container stdout (startup, config reloads, provider failures, budget refusals).
ALLOY_DIR=/opt/vps-observability/config.d
if [ -d "$ALLOY_DIR" ] && [ -f "$REPO_DIR/infra/observability/litellm.alloy" ]; then
  echo "[$(date -u +%FT%TZ)] installing litellm.alloy log rules + reloading Alloy..."
  cp "$REPO_DIR/infra/observability/litellm.alloy" "$ALLOY_DIR/litellm.alloy"
  chmod 0644 "$ALLOY_DIR/litellm.alloy"
  docker kill -s HUP alloy >/dev/null 2>&1 \
    || echo "WARN: could not HUP alloy — gateway logs may lag until its next reload" >&2
fi

# Post-deploy smoke — the gateway-focused analogue of the app deploy's live smokes. Liveliness
# only proves uvicorn answers; it said "healthy" while the gateway served a two-week-stale config.
# Two gates, both FATAL:
#   1. CONFIG IS LIVE — every alias in the shipped config.yaml appears in the served /v1/models.
#      Catches the stale-inode class above, a config the gateway rejected, or a partial load.
#   2. E2E THROUGH THE ROUTE — a 1-token completion through each podcast-* chat alias. Catches an
#      expired/revoked upstream key (OpenRouter), a dead route, or a bad model id — the failure
#      modes that otherwise surface mid-corpus as fail-open degradation. Costs a fraction of a
#      cent. Scoped to podcast-* because those are the pipeline's contract; homelab-*/groq-* ride
#      along in /v1/models check only (groq-whisper is audio_transcription — no chat endpoint).
echo "[$(date -u +%FT%TZ)] post-deploy smoke: served-config parity + e2e completion..."
MASTER_KEY="$(grep -E '^LITELLM_MASTER_KEY=' "$LITELLM_ENV" | head -1 | cut -d= -f2-)"
if [ -z "$MASTER_KEY" ]; then
  echo "ERROR: LITELLM_MASTER_KEY not found in $LITELLM_ENV — cannot run the smoke" >&2
  exit 5
fi
if ! SMOKE_MASTER_KEY="$MASTER_KEY" SMOKE_CONFIG="$REPO_DIR/infra/litellm/config.yaml" \
     python3 - <<'PYSMOKE'
import json
import os
import sys
import urllib.request

import yaml

base = "http://127.0.0.1:4001"
hdr = {
    "Authorization": "Bearer " + os.environ["SMOKE_MASTER_KEY"],
    "Content-Type": "application/json",
}

shipped = [m["model_name"] for m in yaml.safe_load(open(os.environ["SMOKE_CONFIG"]))["model_list"]]

req = urllib.request.Request(base + "/v1/models", headers=hdr)
served = {m["id"] for m in json.load(urllib.request.urlopen(req, timeout=15))["data"]}

missing = [a for a in shipped if a not in served]
if missing:
    print(f"SMOKE FAIL: shipped aliases not served (stale container or rejected config): {missing}",
          file=sys.stderr)
    sys.exit(1)
print(f"smoke 1/2 OK: all {len(shipped)} shipped aliases served")

failures = []
for alias in (a for a in shipped if a.startswith("podcast-")):
    body = json.dumps({
        "model": alias,
        "max_tokens": 5,
        "messages": [{"role": "user", "content": "Say OK"}],
    }).encode()
    try:
        r = urllib.request.Request(base + "/v1/chat/completions", data=body, headers=hdr)
        resp = json.load(urllib.request.urlopen(r, timeout=90))
        content = resp["choices"][0]["message"]["content"]
        print(f"smoke 2/2: {alias} -> completion OK ({resp.get('usage', {}).get('total_tokens')} tok)")
    except Exception as exc:  # noqa: BLE001 — every failure mode here means the same thing: not e2e
        failures.append(f"{alias}: {type(exc).__name__}: {exc}")
if failures:
    print("SMOKE FAIL: alias(es) not working end-to-end (expired upstream key? dead route?):",
          file=sys.stderr)
    for f in failures:
        print(f"  {f}", file=sys.stderr)
    sys.exit(1)
print("smoke 2/2 OK: every podcast-* alias completes end-to-end")
PYSMOKE
then
  echo "ERROR: post-deploy smoke failed — the gateway is NOT serving what this deploy shipped" >&2
  "${COMPOSE[@]}" logs --tail=50 litellm >&2 || true
  exit 4
fi

echo "[$(date -u +%FT%TZ)] litellm gateway healthy on 127.0.0.1:4001 (project=litellm)."
echo "  next: mint the app virtual key with the master key —"
echo "  curl -s http://127.0.0.1:4001/key/generate -H \"Authorization: Bearer \$LITELLM_MASTER_KEY\" \\"
echo "    -H 'Content-Type: application/json' -d '{\"key_alias\":\"proj-podcast-prod\",\"max_budget\":25.0}'"
