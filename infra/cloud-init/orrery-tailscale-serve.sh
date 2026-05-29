#!/bin/sh
# Publishes orrery on the tailnet at :8443 -> 127.0.0.1:8090.
# Sibling of podcast-tailscale-serve.sh; co-tenant on prod-podcast-1.
# Phase 1 (chipi/orrery#260): no systemd unit, invoked by cloud-init runcmd
# at first boot + by orrery's deploy workflow via narrow sudo allowlist.
#
# Co-tenant rule (#845): never ``tailscale serve reset`` — wipes ALL
# co-tenants' publishes. Use port-scoped ``--https=8443 off`` if a clear
# is needed for backend rotation; ``--bg`` is already idempotent for the
# same port so additive-only is fine when the backend stays put.
# podcast-tailscale-serve.sh uses the same pattern (per #845).
#
# POSIX ``/bin/sh`` (dash on Ubuntu) — cloud-init ``write_files`` and
# ``sudo sh`` paths must not hit bash-only syntax.
set -eu
PORT=8090
i=1
while [ "$i" -le 60 ]; do
  if curl -fsS "http://127.0.0.1:${PORT}/" >/dev/null 2>&1; then
    break
  fi
  sleep 2
  i=$((i + 1))
done
exec /usr/bin/tailscale serve --bg --https=8443 "$PORT"
