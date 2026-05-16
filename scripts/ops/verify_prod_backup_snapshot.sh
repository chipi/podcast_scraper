#!/usr/bin/env bash
# Download and inspect a prod corpus backup (snapshot-prod-*) from the backup
# GitHub repo (same asset ``backup-corpus-prod.yml`` uploads). For operator
# validation; see docs/guides/PROD_RUNBOOK.md (corpus backup / restore).
#
# Requires: gh auth with read access to the backup repo.
#
# Usage:
#   ./scripts/ops/verify_prod_backup_snapshot.sh
#   ./scripts/ops/verify_prod_backup_snapshot.sh snapshot-prod-20260511
#   ./scripts/ops/verify_prod_backup_snapshot.sh --no-extract
#   PODCAST_BACKUP_REPO=owner/other-backup ./scripts/ops/verify_prod_backup_snapshot.sh
#
# Env:
#   PODCAST_BACKUP_REPO (optional) — default chipi/podcast_scraper-backup
#   PODCAST_BACKUP_TAG (optional) — explicit tag; overrides auto newest-compatible selection
#
# Output: under repo root ``.tmp_backup_verify/`` (gitignored): snapshot.tgz
# and, unless ``--no-extract``, unpacked ``corpus/`` tree.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

BACKUP_REPO="${PODCAST_BACKUP_REPO:-chipi/podcast_scraper-backup}"
NO_EXTRACT=0
TAG=""

usage() {
  cat <<'EOF'
Download and inspect snapshot.tgz for a prod backup release.

Usage:
  ./scripts/ops/verify_prod_backup_snapshot.sh [options] [TAG]

Options:
  --tag TAG        Use this release tag (same as positional TAG)
  --no-extract     Only list tarball; do not unpack
  -h, --help       This help

Environment:
  PODCAST_BACKUP_REPO   Default: chipi/podcast_scraper-backup
  PODCAST_BACKUP_TAG    Default: latest snapshot-prod-YYYYMMDD on that repo

Examples:
  ./scripts/ops/verify_prod_backup_snapshot.sh
  ./scripts/ops/verify_prod_backup_snapshot.sh snapshot-prod-20260511
  ./scripts/ops/verify_prod_backup_snapshot.sh --no-extract
EOF
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h | --help)
      usage 0
      ;;
    --no-extract)
      NO_EXTRACT=1
      shift
      ;;
    --tag)
      TAG="${2:?--tag requires a value}"
      shift 2
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage 1
      ;;
    *)
      if [[ -n "$TAG" ]]; then
        echo "Unexpected extra argument: $1" >&2
        exit 1
      fi
      TAG="$1"
      shift
      ;;
  esac
done

if [[ -z "$TAG" ]]; then
  TAG="${PODCAST_BACKUP_TAG:-}"
fi

if [[ -z "$TAG" ]]; then
  export BACKUP_REPO="$BACKUP_REPO"
  export TAG_REGEX='^snapshot-prod-[0-9]{8}$'
  TAG="$(bash "$SCRIPT_DIR/corpus_snapshot/select_release_tag.sh")"
fi

if [[ -z "$TAG" ]]; then
  echo "No compatible snapshot-prod-YYYYMMDD tag found in $BACKUP_REPO (and none passed)." >&2
  echo "Set PODCAST_BACKUP_TAG or pass the tag as the first argument." >&2
  exit 1
fi

WORKDIR="${REPO_ROOT}/.tmp_backup_verify"
rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

echo "Backup repo: $BACKUP_REPO"
echo "Release tag: $TAG"
echo "Work dir:    $WORKDIR"

bash "$SCRIPT_DIR/corpus_snapshot/download_and_verify_snapshot.sh" \
  --tag "$TAG" --output-dir "$WORKDIR"

ls -lh snapshot.tgz

echo "--- tar listing (first 30 paths) ---"
tar -tzf snapshot.tgz | head -30

total_paths="$(tar -tzf snapshot.tgz | wc -l | tr -d ' ')"
gi_json="$(tar -tzf snapshot.tgz | grep -c '\.gi\.json$' || true)"
echo "--- counts ---"
echo "total_paths $total_paths"
echo "gi_json     $gi_json"

if [[ "$NO_EXTRACT" -eq 1 ]]; then
  echo "OK (--no-extract: skipped unpack). Artifacts: $WORKDIR/snapshot.tgz"
  exit 0
fi

mkdir -p unpacked
tar -xzf snapshot.tgz -C unpacked

if [[ ! -d unpacked/corpus ]]; then
  echo "ERROR: expected top-level corpus/ after unpack" >&2
  exit 1
fi

echo "--- sample paths under corpus/ (maxdepth 3, first 25) ---"
find unpacked/corpus -maxdepth 3 \( -type d -o -type f \) | head -25

echo "OK. Unpacked tree: $WORKDIR/unpacked/corpus"
