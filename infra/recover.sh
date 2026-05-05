#!/usr/bin/env bash
# infra/recover.sh — Idempotent recovery from a partial/failed `tofu apply`.
#
# A failed apply can leave Hetzner resources created without corresponding
# terraform state, blocking subsequent apply attempts on name-collision
# errors. This script:
#   1. Sources infra/.env.local (so HCLOUD_TOKEN is in env).
#   2. Looks up + deletes any Hetzner resources by the project's expected
#      names (idempotent — silently skips if not found).
#   3. Removes any stale local terraform state.
#   4. Runs `make infra-apply`. If the apply errors on tailscale_acl with
#      "overwrite a non-default policy file", auto-imports the live ACL
#      into terraform state and retries.
#
# Run via: `make infra-recover` (preferred) or `bash infra/recover.sh`.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f infra/.env.local ]]; then
  echo "ERROR: infra/.env.local missing." >&2
  exit 1
fi
# shellcheck disable=SC1091
set -a; . ./infra/.env.local; set +a

if [[ -z "${HCLOUD_TOKEN:-}" ]]; then
  echo "ERROR: HCLOUD_TOKEN not in env after sourcing infra/.env.local." >&2
  exit 1
fi

API="https://api.hetzner.cloud/v1"
AUTH_HEADER="Authorization: Bearer $HCLOUD_TOKEN"

SSH_KEY_NAME="${TF_VAR_ssh_public_key_name:-operator-laptop}"
FIREWALL_NAME="podcast-scraper-prod"
NETWORK_NAME="podcast-scraper-prod"
SERVER_NAME="${TF_VAR_tailnet_hostname:-prod-podcast}"
VOLUME_NAME="podcast-scraper-corpus"

# delete_by_name <kind-plural> <name>
# Looks up Hetzner resources matching the name and deletes them. Idempotent.
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
      echo "  $kind '$name' id=$id: DELETE FAILED (may have attachments; retry after dependents)"
    fi
  done <<< "$ids"
}

echo "==> [recover] Deleting any orphan Hetzner resources by name..."
# Order: dependents first so attachments don't block parent deletion.
delete_by_name servers   "$SERVER_NAME"
delete_by_name volumes   "$VOLUME_NAME"
delete_by_name firewalls "$FIREWALL_NAME"
delete_by_name networks  "$NETWORK_NAME"
delete_by_name ssh_keys  "$SSH_KEY_NAME"

echo ""
echo "==> [recover] Cleaning up local terraform state..."
rm -f infra/terraform/terraform.tfstate \
      infra/terraform/terraform.tfstate.backup \
      infra/terraform/terraform.tfstate.enc \
      infra/terraform/terraform.tfstate.enc.new
echo "  state files removed."

echo ""
echo "==> [recover] First apply attempt (-auto-approve, since you invoked recover explicitly)..."
APPLY_LOG=$(mktemp -t recover-apply.XXXXXX.log)
trap 'rm -f "$APPLY_LOG"' EXIT

# Bypass `make infra-apply` because that target uses an interactive y/n prompt
# which fails inside a non-TTY subprocess. Recover is explicitly hands-off:
# tofu apply -auto-approve is fine here.
run_tofu_apply() {
  ( cd infra && ./tofu init -input=false && ./tofu apply -auto-approve -input=false )
}

if run_tofu_apply 2>&1 | tee "$APPLY_LOG"; then
  echo ""
  echo "==> [recover] Apply succeeded on first attempt — recovery complete."
  exit 0
fi

# Auto-handle the known ACL-conflict mode: import the live ACL, retry.
if grep -q "overwrite a non-default policy file" "$APPLY_LOG"; then
  echo ""
  echo "==> [recover] ACL conflict detected — importing existing ACL..."
  ( cd infra && ./tofu import tailscale_acl.main acl )
  echo ""
  echo "==> [recover] Re-running apply after ACL import..."
  run_tofu_apply
  exit $?
fi

echo ""
echo "==> [recover] Apply failed for a non-ACL reason; review log above." >&2
exit 1
