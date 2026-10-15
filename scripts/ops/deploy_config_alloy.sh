#!/usr/bin/env bash
# deploy_config_alloy.sh — validate staged Alloy drop-ins as the SET Alloy will load, install,
# hot-reload, and prove the reload took.
#
# Runs ON THE BOX as deploy@ (config.d is deploy-writable). Invoked by
# .github/workflows/deploy-config.yml, which scp's this script + the drop-ins into
# /tmp/deploy-config-alloy-staging/ first.
#
# Contract:
#   $1..     = drop-in basenames to install (e.g. "player.alloy base.alloy")
#   env DRY_RUN=true  → validate only; nothing installed, no HUP.
#   Staged files:   /tmp/deploy-config-alloy-staging/<name>.alloy
#   Installed to:   /opt/vps-observability/config.d/<name>.alloy
#
# Why validate the whole set: Alloy loads config.d as ONE configuration, so a drop-in is only
# valid together with the files it references (every pipeline forwards to base.alloy's
# loki.relabel.naming) and with the drop-ins other repos own (orrery.alloy). And a rejected HUP
# keeps the OLD config running while logging only "failed to reload config" — on 2026-10-15
# operator.alloy referenced a stage the box's base.alloy lacked and the reload was refused
# silently for 12 minutes. So: validate before installing, read Alloy's verdict after the HUP
# (its alloy_config_last_load_* metrics), and roll back if it refused or gave none.
set -euo pipefail

FILES=("$@")
[ "${#FILES[@]}" -gt 0 ] || { echo "usage: deploy_config_alloy.sh <name>.alloy..." >&2; exit 2; }

DIR=/opt/vps-observability/config.d
STG=/tmp/deploy-config-alloy-staging
# The alloy container sees the host root read-only at /rootfs, so a candidate set built under
# the host's /tmp is visible to `alloy validate` without touching the live config.d.
CAND="$(mktemp -d /tmp/alloy-candidate.XXXXXX)"
BK="$(mktemp -d)"
cleanup() { rm -rf "$CAND" "$BK" "$STG"; }
trap cleanup EXIT

cp "$DIR"/*.alloy "$CAND"/
changed=()
for f in "${FILES[@]}"; do
  new="$STG/$f"
  [ -f "$new" ] || { echo "::error::staged drop-in missing: $new"; exit 1; }
  cp "$new" "$CAND/$f"
  if [ -f "$DIR/$f" ] && cmp -s "$new" "$DIR/$f"; then
    echo "  $f unchanged"
  else
    changed+=("$f")
    echo "  $f changed"
  fi
done
chmod 0755 "$CAND"
chmod 0644 "$CAND"/*.alloy

if docker exec alloy alloy validate "/rootfs$CAND" >/dev/null 2>&1; then
  echo "  alloy set valid ($(find "$CAND" -name '*.alloy' | wc -l) files)"
else
  echo "::error::alloy set INVALID with the staged drop-ins — nothing installed, no HUP"
  docker exec alloy alloy validate "/rootfs$CAND" 2>&1 | head -30 >&2 || true
  exit 1
fi

# Alloy's verdict on a reload comes from its own metrics (host-network listener): a successful
# load moves alloy_config_last_load_success_timestamp_seconds and sets
# alloy_config_last_load_successful 1; a refused one sets it 0 and leaves the timestamp.
# Metric names do not drift with log wording, and need no host/container clock agreement.
# awk reads to the end (no early `exit`): closing the pipe under curl makes it fail with 23,
# which pipefail + set -e turned into an exit between install and HUP on 2026-10-15.
_metric() {
  curl -fsS --max-time 5 http://127.0.0.1:12345/metrics 2>/dev/null \
    | awk -v m="$1" '$1 == m { v = $2 } END { if (v != "") print v }' || true
}
before="$(_metric alloy_config_last_load_success_timestamp_seconds)"

if [ "${DRY_RUN:-false}" = "true" ]; then
  echo "  DRY-RUN: valid; nothing installed, no HUP"
  exit 0
fi
[ -n "$before" ] || { echo "::error::cannot read alloy's metrics — nothing installed, no HUP"; exit 1; }

# A drop-in on disk that is newer than Alloy's last successful load is not live yet (a file
# placed without a reload, e.g. by an interrupted run). Reload for it even when nothing changed.
newest="$(stat -c %Y "$DIR"/*.alloy | sort -n | tail -1)"
stale="$(awk -v n="$newest" -v b="$before" 'BEGIN { print (n + 0 > b + 0) ? 1 : 0 }')"
if [ "${#changed[@]}" -eq 0 ] && [ "$stale" = 0 ]; then
  echo "  no drop-in changed and all are loaded — alloy not reloaded"
  exit 0
fi
[ "$stale" = 1 ] && echo "  a drop-in on disk is newer than alloy's last load — reloading"

# Stage-then-rename into config.d, never a direct copy onto the destination: a `*.alloy`
# half-written during a concurrent reload would be loaded, and a root-owned destination (the
# hand-edited base.alloy of old) refuses an in-place write while `mv` in this deploy-owned
# directory replaces it. The staging name does not end in `.alloy`, so Alloy ignores it.
for f in "${changed[@]}"; do
  if [ -f "$DIR/$f" ]; then cp -a "$DIR/$f" "$BK/$f"; fi
  cp "$STG/$f" "$DIR/.$f.incoming"
  chmod 0644 "$DIR/.$f.incoming"
  mv -f "$DIR/.$f.incoming" "$DIR/$f"
done

_restore() {
  for f in ${changed[@]+"${changed[@]}"}; do
    if [ -f "$BK/$f" ]; then
      cp "$BK/$f" "$DIR/.$f.incoming" && mv -f "$DIR/.$f.incoming" "$DIR/$f"
    else
      rm -f "$DIR/$f"
    fi
  done
  return 0
}

docker kill -s HUP alloy >/dev/null || { echo "::error::could not HUP alloy — rolling back"; _restore; exit 1; }
verdict=""
for _ in $(seq 1 15); do
  sleep 2
  ok="$(_metric alloy_config_last_load_successful)"
  ts="$(_metric alloy_config_last_load_success_timestamp_seconds)"
  if [ "$ok" = "0" ]; then verdict=failed; break; fi
  if [ "$ok" = "1" ] && [ -n "$ts" ] && [ "$ts" != "$before" ]; then verdict=ok; break; fi
done
case "$verdict" in
  ok)
    echo "  alloy hot-reloaded: ${changed[*]}" ;;
  failed)
    echo "::error::alloy REFUSED the reload — rolling back ${changed[*]} and reloading last-good"
    docker logs alloy --since 2m 2>&1 | grep 'failed to reload config' | tail -3 >&2 || true
    _restore
    docker kill -s HUP alloy >/dev/null || true
    exit 1 ;;
  *)
    echo "::error::alloy logged no reload verdict within 30s — rolling back ${changed[*]}"
    _restore
    docker kill -s HUP alloy >/dev/null || true
    exit 1 ;;
esac
