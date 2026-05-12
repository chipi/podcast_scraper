#!/bin/bash
# Expose the viewer stack (host VIEWER_PORT, default 8080) via ``tailscale serve`` HTTPS :443.
# Kept as a plain file (not Terraform-interpolated) so ``$`` / ``((`` are never mangled on disk.
set -euo pipefail
PORT=8080
if [ -f /srv/podcast-scraper/.env ]; then
  line=$(grep -E '^VIEWER_PORT=' /srv/podcast-scraper/.env | tail -1 || true)
  if [ -n "$line" ]; then
    v=$(echo "$line" | cut -d= -f2- | tr -d ' \t"' | tr -d "'")
    if [ -n "$v" ]; then PORT="$v"; fi
  fi
fi
# Brace range avoids ``$((n+1))``, which is easy to mis-render when this script is embedded in templates.
for _ in {1..60}; do
  if curl -fsS "http://127.0.0.1:${PORT}/api/health" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done
/usr/bin/tailscale serve reset || true
exec /usr/bin/tailscale serve --bg "$PORT"
