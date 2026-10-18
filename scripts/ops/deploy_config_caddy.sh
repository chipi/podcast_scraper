#!/usr/bin/env bash
# deploy_config_caddy.sh — install staged Caddy vhosts + validate + reload-if-valid.
#
# Runs ON THE BOX as deploy@ (config.d/sites are deploy-writable; `systemctl reload|restart
# caddy` is the narrow allowlisted grant). Invoked by .github/workflows/deploy-config.yml,
# which scp's this script + the templated vhosts into /tmp/deploy-config-staging/ first.
#
# Contract:
#   $1..     = vhost basenames to install (e.g. "player player-telemetry")
#   env DRY_RUN=true  → stage + `caddy adapt` only; roll the files back, no reload.
#   Staged files:   /tmp/deploy-config-staging/<name>.caddy
#   Installed to:   /etc/caddy/sites/<name>.caddy
#
# Safety: Caddy picks up vhost files only when it loads its config (reload or restart), so an
# invalid new vhost on disk never affects the running engine — we `caddy adapt` first and roll
# the files back before it could. A refused reload also leaves the running config in place.
# Rollback restores the prior vhosts.
set -euo pipefail

VHOSTS=("$@")
[ "${#VHOSTS[@]}" -gt 0 ] || { echo "usage: deploy_config_caddy.sh <vhost>..." >&2; exit 2; }

SITES=/etc/caddy/sites
STG=/tmp/deploy-config-staging
BK="$(mktemp -d)"
cleanup() { rm -rf "$BK" "$STG"; }
trap cleanup EXIT

# One lock for every tenant that edits /etc/caddy/sites (orrery, websites), held until exit, so
# two install -> adapt -> reload sequences never interleave. Opened read-only when it exists:
# fs.protected_regular=2 refuses an O_CREAT open of another user's file in sticky /tmp, and
# flock needs no write access.
CADDY_SITES_LOCK=/tmp/caddy-sites.lock
if [ -e "$CADDY_SITES_LOCK" ]; then exec 9<"$CADDY_SITES_LOCK"; else exec 9>>"$CADDY_SITES_LOCK"; fi
flock -w 300 9 || { echo "::error::could not take $CADDY_SITES_LOCK within 5 minutes; another tenant is editing the edge"; exit 1; }

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
# The bind line is quoted (`Environment="CADDY_BIND_ADDRS=<v4> <v6>"`): unquoted, systemd keeps
# only the first address. Accept the old unquoted form too.
BIND="$(sed -n -E 's/^Environment="?CADDY_BIND_ADDRS=([^"]*)"?[[:space:]]*$/\1/p' /etc/systemd/system/caddy.service.d/10-public-bind.conf 2>/dev/null || true)"
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
  echo "  no vhost changed — caddy not reloaded"
  exit 0
fi

# RELOAD, graceful (admin = 0600 unix socket, T-02); RESTART only when the reload is refused
# (a box whose base Caddyfile still says `admin off`). A restart drops every tenant's traffic
# for seconds — on 2026-10-15 it failed the player smoke with a 521.
if sudo -n /usr/bin/systemctl reload caddy && systemctl is-active --quiet caddy; then
  echo "  caddy reloaded"
elif sudo -n /usr/bin/systemctl restart caddy && systemctl is-active --quiet caddy; then
  echo "  caddy restarted (reload refused — is the base Caddyfile's admin socket live? apply-edge)"
else
  echo "::error::caddy failed to load the vhosts — rolling back + restarting to last-good"
  _restore
  sudo -n /usr/bin/systemctl restart caddy || true
  exit 1
fi
