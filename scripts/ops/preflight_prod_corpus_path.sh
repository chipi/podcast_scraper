#!/usr/bin/env bash
# Pre-deploy check: host PODCAST_CORPUS_HOST_PATH is set and is a directory (#808).
#
# Usage (from GHA or laptop with SSH to prod/drill):
#   scripts/ops/preflight_prod_corpus_path.sh <deploy-user@tailnet-fqdn>
#
# Reads /srv/podcast-scraper/.env on the host (same path as deploy-prod.yml).
# Exit 0 when OK; exit 1 with stderr message on failure.

set -euo pipefail

SSH_TARGET="${1:-}"
if [ -z "$SSH_TARGET" ]; then
  echo "usage: $0 <deploy-user@tailnet-fqdn>" >&2
  exit 1
fi

SSH_IDENTITY="${SSH_PROD_IDENTITY:-${SSH_DRILL_IDENTITY:-}}"
SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o BatchMode=yes)
if [ -n "$SSH_IDENTITY" ]; then
  SSH_OPTS=(-i "$SSH_IDENTITY" -o IdentitiesOnly=yes "${SSH_OPTS[@]}")
fi

HOST_CORPUS=$(ssh "${SSH_OPTS[@]}" "$SSH_TARGET" \
  "grep '^PODCAST_CORPUS_HOST_PATH=' /srv/podcast-scraper/.env 2>/dev/null | cut -d= -f2-" || true)

if [ -z "${HOST_CORPUS:-}" ]; then
  echo "::error::PODCAST_CORPUS_HOST_PATH missing from host /srv/podcast-scraper/.env" >&2
  exit 1
fi

if ! ssh "${SSH_OPTS[@]}" "$SSH_TARGET" "test -d $(printf '%q' "$HOST_CORPUS")"; then
  echo "::error::PODCAST_CORPUS_HOST_PATH is not a directory on host: $HOST_CORPUS" >&2
  exit 1
fi

echo "preflight corpus path OK: $HOST_CORPUS"
