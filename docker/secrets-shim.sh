#!/bin/sh
# ADR-115 C1 secret shim. Exports file-mounted secrets as env vars, then hands
# off to the image's real entrypoint.
#
# Runs as the container's initial user (root for api, podcast for pipeline);
# exported vars survive the real entrypoint's su-exec/setpriv drop. Each file in
# ${SECRETS_DIR} (default /run/secrets, where compose ``secrets:`` mounts them)
# becomes the UPPERCASE env var the app already reads:
#   /run/secrets/openai_api_key -> OPENAI_API_KEY
#
# Tolerant by design: a missing/empty secrets dir is fine — the shim just execs
# the real entrypoint, so the image still boots pre-cutover (secrets still in env)
# and this change is non-breaking.
set -eu

SECRETS_DIR="${SECRETS_DIR:-/run/secrets}"
if [ -d "$SECRETS_DIR" ]; then
  for f in "$SECRETS_DIR"/*; do
    [ -f "$f" ] || continue   # also handles the no-match glob (dir empty)
    name="$(basename "$f" | tr '[:lower:]' '[:upper:]')"
    export "$name=$(cat "$f")"
  done
fi

exec "${WRAPPED_ENTRYPOINT:-/entrypoint.sh}" "$@"
