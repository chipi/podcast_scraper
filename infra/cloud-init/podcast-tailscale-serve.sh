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
/usr/bin/tailscale serve reset || true
exec /usr/bin/tailscale serve --bg "$PORT"
