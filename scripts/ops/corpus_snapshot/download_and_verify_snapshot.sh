#!/usr/bin/env bash
# Download snapshot.manifest.json + snapshot.tgz for a release; validate manifest and
# optional archive.sha256. See RFC-084 / ADR-092.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

usage() {
  cat <<'EOF'
Download and verify corpus snapshot release assets.

Usage:
  download_and_verify_snapshot.sh --tag TAG --output-dir DIR

Environment:
  BACKUP_REPO / PODCAST_BACKUP_REPO   Default chipi/podcast_scraper-backup
  GH_TOKEN                            gh auth token
  CORPUS_SNAPSHOT_SKIP_SHA256_VERIFY=1  Skip tarball digest check
EOF
  exit "${1:-0}"
}

TAG=""
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h | --help)
      usage 0
      ;;
    --tag)
      TAG="${2:?--tag requires a value}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:?--output-dir requires a value}"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage 1
      ;;
  esac
done

if [[ -z "$TAG" || -z "$OUTPUT_DIR" ]]; then
  echo "ERROR: --tag and --output-dir are required" >&2
  usage 1
fi

BACKUP_REPO="${BACKUP_REPO:-${PODCAST_BACKUP_REPO:-chipi/podcast_scraper-backup}}"
corpus_snapshot_require_cmd gh
corpus_snapshot_require_cmd jq

mkdir -p "$OUTPUT_DIR"
MANIFEST="$OUTPUT_DIR/snapshot.manifest.json"
TARBALL="$OUTPUT_DIR/snapshot.tgz"

if ! gh release download "$TAG" --repo "$BACKUP_REPO" --pattern snapshot.manifest.json \
  --output "$MANIFEST" 2>/dev/null; then
  echo "WARNING: release $TAG has no snapshot.manifest.json sibling asset" >&2
else
  bash "$SCRIPT_DIR/validate_snapshot_manifest.sh" "$MANIFEST"
  read -r MIN_VER MAX_VER < <(corpus_snapshot_read_reader_range)
  FMT="$(jq -r '.corpus_format_version' "$MANIFEST")"
  if ! corpus_snapshot_version_supported "$FMT" "$MIN_VER" "$MAX_VER"; then
    echo "ERROR: corpus_format_version=$FMT outside reader range [$MIN_VER,$MAX_VER]" >&2
    exit 3
  fi
fi

gh release download "$TAG" --repo "$BACKUP_REPO" --pattern snapshot.tgz --output "$TARBALL"

if [[ -f "$MANIFEST" && "${CORPUS_SNAPSHOT_SKIP_SHA256_VERIFY:-0}" != "1" ]]; then
  EXPECTED="$(jq -r '.archive.sha256 // empty' "$MANIFEST")"
  if [[ -n "$EXPECTED" ]]; then
    if command -v sha256sum >/dev/null 2>&1; then
      ACTUAL="$(sha256sum "$TARBALL" | awk '{print $1}')"
    elif command -v shasum >/dev/null 2>&1; then
      ACTUAL="$(shasum -a 256 "$TARBALL" | awk '{print $1}')"
    else
      echo "ERROR: need sha256sum or shasum to verify archive.sha256" >&2
      exit 1
    fi
    if [[ "$ACTUAL" != "$EXPECTED" ]]; then
      echo "ERROR: snapshot.tgz sha256 mismatch for $TAG (expected $EXPECTED, got $ACTUAL)" >&2
      exit 4
    fi
    echo "OK: snapshot.tgz sha256 matches manifest"
  fi
fi

echo "OK: downloaded $TAG assets under $OUTPUT_DIR"
