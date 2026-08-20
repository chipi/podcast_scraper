#!/usr/bin/env bash
# Validate the shared Caddy edge config the way the prod deploy does — BEFORE it can merge.
#
# The vhost drop-ins in this dir are not standalone: they `import hardened` from the base
# Caddyfile (infra/cloud-init/Caddyfile) and are combined at runtime under /etc/caddy/sites/.
# So a faithful check renders every drop-in's placeholders (mirroring the sed substitutions in
# deploy-player.sh / deploy-operator.sh), points a copy of the base Caddyfile at them, and runs
# `caddy adapt` on the MERGED config.
#
# `caddy adapt` (Caddyfile -> JSON), NOT `caddy validate`: validate also PROVISIONS (opens the
# caddy-owned access.log) and false-fails with "permission denied" even on a valid config — the
# exact prod incident (2026-07-23) that deploy-player.sh calls out. adapt only parses structure,
# which is what we want to gate on.
#
# Runs in CI (.github/workflows/caddy-validate.yml) on any infra/caddy/** change, and locally.
# Uses a local `caddy` binary if present, else the official caddy:2 Docker image (no host install).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BASE="$ROOT/infra/cloud-init/Caddyfile"
SITES_SRC="$ROOT/infra/caddy"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/sites"

# Render every drop-in with dummy-but-valid values. Real DNS / secrets are irrelevant to adapt
# (it only parses structure); we substitute the deploy-time placeholders so nothing is left as a
# raw token that could mask a real syntax error. Mirrors deploy-player.sh + deploy-operator.sh.
for f in "$SITES_SRC"/*.caddy; do
  sed -e 's|__PREVIEW_COOKIE__|dummy-preview-cookie|g' \
      -e 's|__OPERATOR_PREVIEW_COOKIE__|dummy-operator-cookie|g' \
      -e 's|__TAILNET__|example-tailnet.ts.net|g' \
      "$f" >"$TMP/sites/$(basename "$f")"
done

# Base Caddyfile, with its import repointed from the absolute prod path to our temp sites dir
# (relative — resolved against the working dir we run caddy from below).
sed 's#import /etc/caddy/sites/\*\.caddy#import sites/*.caddy#' "$BASE" >"$TMP/Caddyfile"

# The base + drop-ins reference these {$ENV} placeholders; caddy substitutes them at adapt time.
# Empty values would break adapt (e.g. a reverse_proxy with no upstream), so pin safe dummies.
export CADDY_BIND_ADDRS="${CADDY_BIND_ADDRS:-0.0.0.0}"
export GLITCHTIP_UPSTREAM="${GLITCHTIP_UPSTREAM:-127.0.0.1:8080}"

n_sites="$(find "$SITES_SRC" -maxdepth 1 -name '*.caddy' | wc -l | tr -d ' ')"
echo "Validating $n_sites Caddy vhost drop-in(s) against the base Caddyfile via 'caddy adapt'..."

if command -v caddy >/dev/null 2>&1; then
  ( cd "$TMP" && caddy adapt --config Caddyfile --adapter caddyfile >/dev/null )
else
  echo "(no local caddy binary — using the official caddy:2 Docker image)"
  docker run --rm -w /work \
    -e CADDY_BIND_ADDRS -e GLITCHTIP_UPSTREAM \
    -v "$TMP:/work:ro" \
    caddy:2 caddy adapt --config Caddyfile --adapter caddyfile >/dev/null
fi

echo "OK — Caddy edge config is valid ($n_sites vhost drop-in(s), merged with the base Caddyfile)."
