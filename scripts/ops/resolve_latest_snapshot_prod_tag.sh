#!/usr/bin/env bash
# Resolve backup release tag for prod snapshot restore workflows.
#
# Env:
#   BACKUP_REPO (optional) — default chipi/podcast_scraper-backup
#   BACKUP_TAG (optional) — pinned tag; when empty, newest compatible
#     snapshot-prod-* with sibling snapshot.manifest.json (ADR-092)
#   BACKUP_REPO_TOKEN (optional) — when set, used as GH_TOKEN for gh
#   GITHUB_OUTPUT (optional) — when set, writes tag= and repo= outputs
#
# stdout: tag and repo on separate lines when GITHUB_OUTPUT is unset.
# exit 0 on success, non-zero on failure.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -n "${BACKUP_REPO_TOKEN:-}" ]; then
  export GH_TOKEN="$BACKUP_REPO_TOKEN"
else
  export GH_TOKEN="${GITHUB_TOKEN:-${GH_TOKEN:?Set GH_TOKEN, GITHUB_TOKEN, or BACKUP_REPO_TOKEN for gh}}"
fi

REPO="${BACKUP_REPO:-chipi/podcast_scraper-backup}"
export BACKUP_REPO="$REPO"
export PODCAST_BACKUP_TAG="${BACKUP_TAG:-}"
export TAG_REGEX='^snapshot-prod-[0-9]{8}$'

TAG="$(bash "$SCRIPT_DIR/corpus_snapshot/select_release_tag.sh")"

if [ -n "${GITHUB_OUTPUT:-}" ]; then
  {
    echo "tag=$TAG"
    echo "repo=$REPO"
  } >>"$GITHUB_OUTPUT"
else
  printf '%s\n%s\n' "$TAG" "$REPO"
fi
