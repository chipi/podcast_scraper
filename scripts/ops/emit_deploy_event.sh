#!/usr/bin/env bash
# Push a structured deploy_event to Grafana Cloud Loki (#803 D1).
#
# The GHA runner's logs don't reach Loki (grafana-agent only scrapes the VPS containers), so
# deploy-prod.yml pushes the deploy event directly here. Kept as a reusable script so it can be
# exercised locally (against a `--env test` label) without a real prod deploy.
#
# Env (required): LOKI_PUSH_URL (…/loki/api/v1/push), LOKI_USER, LOKI_TOKEN (logs:write).
# Args: --status S --sha SHA --duration-ms N --triggered-by WHO [--env ENV (default prod)]
set -euo pipefail

status=""
sha=""
duration_ms="0"
triggered_by=""
env_label="prod"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --status) status="$2"; shift 2 ;;
    --sha) sha="$2"; shift 2 ;;
    --duration-ms) duration_ms="$2"; shift 2 ;;
    --triggered-by) triggered_by="$2"; shift 2 ;;
    --env) env_label="$2"; shift 2 ;;
    *) echo "emit_deploy_event: unknown arg: $1" >&2; exit 2 ;;
  esac
done

: "${LOKI_PUSH_URL:?LOKI_PUSH_URL is required}"
: "${LOKI_USER:?LOKI_USER is required}"
: "${LOKI_TOKEN:?LOKI_TOKEN is required}"

# Normalise to the push endpoint — callers may pass the base host or a query URL.
push_url="${LOKI_PUSH_URL%/}"
push_url="${push_url%/loki/api/v1/push}"
push_url="${push_url%/loki/api/v1/query_range}"
push_url="${push_url%/loki/api/v1/query}"
push_url="${push_url}/loki/api/v1/push"

ts_ns="$(date +%s)000000000"

# Build the Loki push payload with python so the JSON log line is escaped correctly inside the
# stream value. Labels are low-cardinality ({app, env, event_type}); volatile fields (sha,
# duration, status) live in the JSON line so the panel parses them with `| json`.
payload="$(python3 - "$env_label" "$ts_ns" "$status" "$sha" "$duration_ms" "$triggered_by" <<'PY'
import json
import sys

env, ts, status, sha, duration_ms, triggered_by = sys.argv[1:7]
try:
    duration = int(duration_ms)
except ValueError:
    duration = 0
line = json.dumps(
    {
        "event_type": "deploy_event",
        "status": status,
        "sha": sha,
        "duration_ms": duration,
        "triggered_by": triggered_by,
    }
)
print(
    json.dumps(
        {
            "streams": [
                {
                    "stream": {"app": "podcast_scraper", "env": env, "event_type": "deploy_event"},
                    "values": [[ts, line]],
                }
            ]
        }
    )
)
PY
)"

curl -fsS -u "$LOKI_USER:$LOKI_TOKEN" -H 'Content-Type: application/json' \
  -X POST "$push_url" --data-binary "$payload"
echo "emit_deploy_event: pushed status=$status sha=$sha duration_ms=$duration_ms env=$env_label"
