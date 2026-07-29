#!/usr/bin/env bash
# deploy_config_caddy.sh — install staged Caddy vhosts + validate + restart-if-valid.
#
# Runs ON THE BOX as deploy@ (config.d/sites are deploy-writable; `systemctl restart caddy`
# is the narrow allowlisted grant). Invoked by .github/workflows/deploy-config.yml, which
# scp's this script + the templated vhosts into /tmp/deploy-config-staging/ first.
#
# Contract:
#   $1..     = vhost basenames to install (e.g. "player player-telemetry")
#   env DRY_RUN=true  → stage + `caddy validate` only; roll the files back, no restart.
#   Staged files:   /tmp/deploy-config-staging/<name>.caddy
#   Installed to:   /etc/caddy/sites/<name>.caddy
#
# Safety: Caddy only picks up changes on RESTART (admin off / T-02 → restart, not reload).
# So an invalid new vhost on disk never affects the running engine — we `caddy validate`
# first and roll the files back before it could. Rollback restores the prior vhosts.
set -euo pipefail

VHOSTS=("$@")
[ "${#VHOSTS[@]}" -gt 0 ] || { echo "usage: deploy_config_caddy.sh <vhost>..." >&2; exit 2; }

SITES=/etc/caddy/sites
STG=/tmp/deploy-config-staging
BK="$(mktemp -d)"
cleanup() { rm -rf "$BK" "$STG"; }
trap cleanup EXIT

changed=0
for v in "${VHOSTS[@]}"; do
  new="$STG/${v}.caddy"
  cur="$SITES/${v}.caddy"
  [ -f "$new" ] || { echo "::error::staged vhost missing: $new"; exit 1; }
  if [ -f "$cur" ] && cmp -s "$new" "$cur"; then
    echo "  ${v}.caddy unchanged"
    continue
  fi
  [ -f "$cur" ] && cp -a "$cur" "$BK/${v}.caddy"
  cp "$new" "$cur"
  chmod 0644 "$cur"
  changed=1
  echo "  ${v}.caddy staged"
done

_restore() {
  for v in "${VHOSTS[@]}"; do
    [ -f "$BK/${v}.caddy" ] && cp "$BK/${v}.caddy" "$SITES/${v}.caddy"
  done
  # Always succeed: with nothing to restore (e.g. dry-run, no vhost changed) the loop's
  # final `[ -f ] && cp` is false, and under `set -e` that would abort before the caller's
  # `exit 0`. The restore itself is best-effort, so its own status must never fail the run.
  return 0
}

# Validate the WHOLE config (base Caddyfile `import`s sites/*). Source the systemd env the
# base Caddyfile interpolates ({$CADDY_BIND_ADDRS} / {$GLITCHTIP_UPSTREAM}) so it resolves.
#
# We gate on `caddy adapt`, NOT `caddy validate`: caddy runs as user `caddy` and owns
# /var/log/caddy/*.log (0640). `validate` *provisions* the full config — which opens those
# log writers — and this script runs as the deploy user (not in the caddy group), so
# `validate` dies with "open /var/log/caddy/access.log: permission denied" on EVERY run.
# `adapt` does the Caddyfile→JSON parse only (no log provisioning): it runs fine as deploy
# and still catches syntax errors, unknown directives, unresolved placeholders, and
# duplicate/ambiguous site addresses (`Error: ambiguous site definition: <host>`) — which is
# the real breakage a vhost deploy can introduce. Provision-time issues (unreachable upstream)
# don't fail `caddy start` anyway; the post-restart `is-active` check below is the backstop.
BIND="$(sed -n 's/^Environment=CADDY_BIND_ADDRS=//p' /etc/systemd/system/caddy.service.d/10-public-bind.conf 2>/dev/null || true)"
GT="$(sed -n 's/^Environment=GLITCHTIP_UPSTREAM=//p' /etc/systemd/system/caddy.service.d/20-glitchtip-upstream.conf 2>/dev/null || true)"
_validate() {
  CADDY_BIND_ADDRS="$BIND" GLITCHTIP_UPSTREAM="$GT" \
    caddy adapt --config /etc/caddy/Caddyfile --adapter caddyfile >/dev/null 2>&1
}

if _validate; then
  echo "  caddy config valid"
else
  echo "::error::caddy config INVALID after staging — rolling back vhosts, NOT restarting"
  _restore
  # Re-run to surface WHY (the adapt error goes to stderr; drop the JSON on stdout).
  CADDY_BIND_ADDRS="$BIND" GLITCHTIP_UPSTREAM="$GT" \
    caddy adapt --config /etc/caddy/Caddyfile --adapter caddyfile 2>&1 >/dev/null | head -20 >&2 || true
  exit 1
fi

if [ "${DRY_RUN:-false}" = "true" ]; then
  echo "  DRY-RUN: valid; rolling back staged files, not restarting"
  _restore
  exit 0
fi

if [ "$changed" = 0 ]; then
  echo "  no vhost changed — caddy not restarted"
  exit 0
fi

if sudo -n /usr/bin/systemctl restart caddy && systemctl is-active --quiet caddy; then
  echo "  caddy restarted"
else
  echo "::error::caddy failed to restart — rolling back + restarting to last-good"
  _restore
  sudo -n /usr/bin/systemctl restart caddy || true
  exit 1
fi
