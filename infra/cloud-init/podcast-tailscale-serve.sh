#!/bin/sh
# Expose the viewer stack (host VIEWER_PORT, default 8080) via ``tailscale serve`` HTTPS :443.
#
# Must be POSIX ``/bin/sh`` (dash on Ubuntu): cloud-init ``write_files``, systemd
# ``ExecStartPost``, and ``sudo sh`` paths must not hit bash-only syntax; first-boot
# Terraform/cloud-init edge cases have produced ``(`` parse errors on broken copies.
set -eu
PORT=8080
if [ -f /srv/podcast-scraper/.env ]; then
  line=$(grep -E '^VIEWER_PORT=' /srv/podcast-scraper/.env | tail -1 || true)
  if [ -n "$line" ]; then
    v=$(echo "$line" | cut -d= -f2- | tr -d ' \t"' | tr -d "'")
    if [ -n "$v" ]; then PORT="$v"; fi
  fi
fi
i=1
while [ "$i" -le 60 ]; do
  if curl -fsS "http://127.0.0.1:${PORT}/api/health" >/dev/null 2>&1; then
    break
  fi
  sleep 2
  i=$((i + 1))
done
# Port-scoped clear (#845): ``tailscale serve reset`` wipes the entire
# serve config — including other co-tenant apps' publishes (orrery on
# :8443, etc.). Using ``--https=443 off`` clears only this app's :443
# entry, leaving co-tenants' ports untouched. ``tailscale serve --bg``
# below is itself idempotent for the same port, so the ``off`` is only
# needed for the ``$PORT`` rotation case (8080 → 8081 etc.) where the
# backend mapping changes between deploys.
/usr/bin/tailscale serve --https=443 off || true
exec /usr/bin/tailscale serve --bg "$PORT"
