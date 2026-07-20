#!/usr/bin/env bash
# Push the VPS/podcast-owned dashboards into the SHARED Grafana on the DGX
# (folder "VPS — Podcast"). Dashboards-as-code: the JSON in git is the source of
# truth; this re-pushes them idempotently (stable uid + overwrite).
#
# Ownership split (see agentic-ai-homelab docs/wip/observability-vps-dashboards-handover.md):
#   homelab repo  -> shared infra dashboards (host/GPU/containers/logs)
#   THIS repo     -> podcast/VPS app + edge dashboards, pushed here on deploy.
#
# Secrets come from the gitignored root .env (never committed):
#   GRAFANA_URL           e.g. http://dgx-llm-1:3000  (DGX, tailnet)
#   GRAFANA_DEPLOY_TOKEN  glsa_... service-account token (podcast-deploy, Editor)
#   GRAFANA_FOLDER_UID    vps-podcast
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DASH_DIR="${DASH_DIR:-$REPO_ROOT/config/grafana/dashboards/vps}"
ENV_FILE="${ENV_FILE:-$REPO_ROOT/.env}"

# shellcheck disable=SC1090
[ -f "$ENV_FILE" ] && set -a && . "$ENV_FILE" && set +a

: "${GRAFANA_URL:?set GRAFANA_URL in .env}"
: "${GRAFANA_DEPLOY_TOKEN:?set GRAFANA_DEPLOY_TOKEN in .env}"
FOLDER_UID="${GRAFANA_FOLDER_UID:-vps-podcast}"

fail=0
shopt -s nullglob
for f in "$DASH_DIR"/*.json; do
  name="$(basename "$f")"
  if ! jq empty "$f" 2>/dev/null; then
    echo "INVALID JSON  $name — skipped"; fail=1; continue
  fi
  resp="$(jq -c --arg fuid "$FOLDER_UID" \
            '{dashboard: (. + {id:null}), folderUid: $fuid, overwrite: true}' "$f" \
          | curl -sS -H "Authorization: Bearer $GRAFANA_DEPLOY_TOKEN" \
                 -H 'Content-Type: application/json' \
                 "$GRAFANA_URL/api/dashboards/db" -d @-)"
  status="$(printf '%s' "$resp" | jq -r '.status // "error"')"
  if [ "$status" = "success" ]; then
    printf 'OK    %-22s -> %s%s\n' "$name" "$GRAFANA_URL" "$(printf '%s' "$resp" | jq -r '.url')"
  else
    printf 'FAIL  %-22s : %s\n' "$name" "$(printf '%s' "$resp" | jq -r '.message // .')"; fail=1
  fi
done

exit "$fail"
