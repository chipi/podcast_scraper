#!/usr/bin/env bash
# Delete orphan Hetzner Cloud resources in the *drill* project after a partial
# ``tofu apply`` (same logical names as ``infra/terraform/main.tf``; drill
# differs by API token / project only). Idempotent if resources are already gone.
#
# Requires ``HCLOUD_TOKEN_DRILL`` (never use prod ``HCLOUD_TOKEN`` here).
#
# Usage:
#   export HCLOUD_TOKEN_DRILL='…'   # token scoped to the drill Hetzner project
#   ./scripts/ops/delete_drill_hetzner_orphans.sh
#
# Optional env (defaults match ``terraform.drill.ci.tfvars`` / variables.tf):
#   DRILL_TAILNET_HOSTNAME — default dr-podcast
#   DRILL_SSH_KEY_NAME     — default operator-laptop

set -euo pipefail

if [[ -z "${HCLOUD_TOKEN_DRILL:-}" ]]; then
  echo "ERROR: Set HCLOUD_TOKEN_DRILL to the drill-project Hetzner API token." >&2
  echo "  (This script refuses HCLOUD_TOKEN to avoid accidental prod deletes.)" >&2
  exit 1
fi

API="https://api.hetzner.cloud/v1"
AUTH_HEADER="Authorization: Bearer ${HCLOUD_TOKEN_DRILL}"

SSH_KEY_NAME="${DRILL_SSH_KEY_NAME:-operator-laptop}"
FIREWALL_NAME="podcast-scraper-prod"
NETWORK_NAME="podcast-scraper-prod"
SERVER_NAME="${DRILL_TAILNET_HOSTNAME:-dr-podcast}"
VOLUME_NAME="podcast-scraper-corpus"

delete_by_name() {
  local kind="$1"
  local name="$2"
  local resp
  resp=$(curl -fsS -H "$AUTH_HEADER" "$API/$kind?name=$name") || {
    echo "  WARN: list $kind name=$name failed; skipping." >&2
    return 0
  }
  local ids
  ids=$(echo "$resp" | jq -r ".$kind[].id" 2>/dev/null || true)
  if [[ -z "$ids" ]]; then
    echo "  $kind '$name': not found"
    return 0
  fi
  while read -r id; do
    [[ -z "$id" ]] && continue
    if curl -fsS -X DELETE -H "$AUTH_HEADER" "$API/$kind/$id" >/dev/null 2>&1; then
      echo "  $kind '$name' id=$id: DELETED"
    else
      echo "  $kind '$name' id=$id: DELETE FAILED (may have attachments; retry after dependents)" >&2
    fi
  done <<< "$ids"
}

echo "==> Deleting drill Hetzner orphans (server → volume → firewall → network → ssh_key)..."
delete_by_name servers "$SERVER_NAME"
delete_by_name volumes "$VOLUME_NAME"
delete_by_name firewalls "$FIREWALL_NAME"
delete_by_name networks "$NETWORK_NAME"
delete_by_name ssh_keys "$SSH_KEY_NAME"
echo "==> Done."
