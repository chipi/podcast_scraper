#!/usr/bin/env bash
# Restore corpus from snapshot.tgz on a VPS (prod or DR drill).
#
# Runs on the host as deploy@ after the GHA runner uploads the tarball
# (and this script copy) over Tailscale SSH. Overwrites /srv/podcast-scraper/corpus,
# recreates api + viewer, smoke-checks /api/health inside the api container (GH-745).
#
# Usage: restore_corpus_from_tarball_host.sh <tarball_path>
#
# Env (optional): PODCAST_REPO_DIR — default /srv/podcast-scraper
#   RESTORE_EXTRACT_ONLY=1 — extract + corpus/ check only (CI / local rehearsal)

set -euo pipefail

TARBALL="${1:?usage: restore_corpus_from_tarball_host.sh <tarball_path>}"
REPO_DIR="${PODCAST_REPO_DIR:-/srv/podcast-scraper}"

cd "$REPO_DIR"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
if [ -d corpus ]; then
  mv corpus "corpus.bak.$STAMP"
fi
tar -xzf "$TARBALL" -C "$REPO_DIR"
if [ ! -d corpus ]; then
  echo "ERROR: expected top-level corpus/ after prod layout extract under $REPO_DIR" >&2
  exit 1
fi
if [ "${RESTORE_EXTRACT_ONLY:-}" = "1" ]; then
  echo "Restore extract OK under $REPO_DIR/corpus"
  exit 0
fi
chown -R deploy:deploy corpus
rm -f "$TARBALL"

COMPOSE=(
  docker compose --env-file .env
  -f compose/docker-compose.stack.yml
  -f compose/docker-compose.prod.yml
  -f compose/docker-compose.vps-prod.yml
)
"${COMPOSE[@]}" up -d --force-recreate api viewer
sleep 8
"${COMPOSE[@]}" exec -T api curl -fsS http://127.0.0.1:8000/api/health | head -c 200
echo
echo "Restore complete on host"
