#!/usr/bin/env bash
# Select (optional) and restore a corpus snapshot release into a workspace directory.
# See RFC-084 / ADR-092.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Restore corpus snapshot from backup repo release assets.

Usage:
  restore_corpus_release.sh --layout codespace|prod

Environment:
  PODCAST_BACKUP_REPO / BACKUP_REPO
  PODCAST_BACKUP_TAG (optional pin)
  TAG_REGEX (required when tag unset; set by caller for layout)
  WORKSPACE_DIR (extract parent; default PWD)
  GH_TOKEN
EOF
  exit "${1:-0}"
}

LAYOUT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h | --help)
      usage 0
      ;;
    --layout)
      LAYOUT="${2:?--layout requires codespace or prod}"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage 1
      ;;
  esac
done

if [[ "$LAYOUT" != "codespace" && "$LAYOUT" != "prod" ]]; then
  echo "ERROR: --layout must be codespace or prod" >&2
  exit 1
fi

WORKSPACE_DIR="${WORKSPACE_DIR:-$PWD}"
BACKUP_REPO="${PODCAST_BACKUP_REPO:-${BACKUP_REPO:-chipi/podcast_scraper-backup}}"
export BACKUP_REPO PODCAST_BACKUP_TAG

if [[ -z "${PODCAST_BACKUP_TAG:-}" && -z "${TAG_REGEX:-}" ]]; then
  echo "ERROR: TAG_REGEX is required when PODCAST_BACKUP_TAG is unset" >&2
  exit 1
fi

TAG="$(bash "$SCRIPT_DIR/select_release_tag.sh")"
echo "Restoring corpus snapshot $TAG from $BACKUP_REPO (layout=$LAYOUT) ..."

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

bash "$SCRIPT_DIR/download_and_verify_snapshot.sh" --tag "$TAG" --output-dir "$STAGE"

# Refuse to clobber an existing corpus tree (may be live data) — mirrors the
# guard in import_local_snapshot.sh that this path lacked (review low/restore-overwrite).
if [[ "$LAYOUT" == "codespace" ]]; then
  if [[ -e "$WORKSPACE_DIR/.codespace_corpus" ]]; then
    echo "ERROR: $WORKSPACE_DIR/.codespace_corpus already exists; refusing to overwrite (may be live data)." >&2
    exit 1
  fi
  mkdir -p "$WORKSPACE_DIR/.codespace_corpus"
  tar -xzf "$STAGE/snapshot.tgz" -C "$WORKSPACE_DIR" --strip-components=0
  echo "OK: corpus restored from $TAG into $WORKSPACE_DIR/.codespace_corpus/"
else
  if [[ -e "$WORKSPACE_DIR/corpus" ]]; then
    echo "ERROR: $WORKSPACE_DIR/corpus already exists; refusing to overwrite (may be live data)." >&2
    exit 1
  fi
  mkdir -p "$WORKSPACE_DIR"
  tar -xzf "$STAGE/snapshot.tgz" -C "$WORKSPACE_DIR" --strip-components=0
  if [[ ! -d "$WORKSPACE_DIR/corpus" ]]; then
    echo "ERROR: expected top-level corpus/ after prod layout extract under $WORKSPACE_DIR" >&2
    exit 1
  fi
  echo "OK: corpus restored from $TAG into $WORKSPACE_DIR/corpus/"
fi
