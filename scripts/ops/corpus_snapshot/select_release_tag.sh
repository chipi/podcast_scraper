#!/usr/bin/env bash
# Select newest compatible backup release tag via sibling snapshot.manifest.json.
# See RFC-084 / ADR-092.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

usage() {
  cat <<'EOF'
Select a corpus snapshot release tag (newest compatible by default).

Usage:
  select_release_tag.sh

Environment:
  BACKUP_REPO              GitHub owner/repo (default: chipi/podcast_scraper-backup)
  PODCAST_BACKUP_TAG       Pinned tag; skips newest-compatible scan when set
  TAG_REGEX                ERE for candidate tags (required when unpinned)
  GH_TOKEN                 gh auth token for release API
  CORPUS_SNAPSHOT_REPO_ROOT
  CORPUS_SNAPSHOT_SKIP_COMPAT_CHECK=1  With pin: skip reader range check
EOF
  exit "${1:-0}"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage 0
fi

BACKUP_REPO="${BACKUP_REPO:-${PODCAST_BACKUP_REPO:-chipi/podcast_scraper-backup}}"
PINNED="${PODCAST_BACKUP_TAG:-${BACKUP_TAG:-}}"
TAG_REGEX="${TAG_REGEX:-}"

corpus_snapshot_require_cmd jq
corpus_snapshot_require_cmd gh

read -r MIN_VER MAX_VER < <(corpus_snapshot_read_reader_range)

check_manifest_compatible() {
  local manifest="$1" tag="$2"
  bash "$SCRIPT_DIR/validate_snapshot_manifest.sh" "$manifest"
  local fmt
  fmt="$(jq -r '.corpus_format_version' "$manifest")"
  if corpus_snapshot_version_supported "$fmt" "$MIN_VER" "$MAX_VER"; then
    return 0
  fi
  echo "skip: $tag corpus_format_version=$fmt outside reader range [$MIN_VER,$MAX_VER]" >&2
  return 1
}

if [[ -n "$PINNED" ]]; then
  if ! gh release view "$PINNED" --repo "$BACKUP_REPO" >/dev/null 2>&1; then
    echo "ERROR: release $PINNED not found on $BACKUP_REPO" >&2
    exit 2
  fi
  if [[ "${CORPUS_SNAPSHOT_SKIP_COMPAT_CHECK:-0}" != "1" ]]; then
    tmp="$(mktemp -d)"
    if gh release download "$PINNED" --repo "$BACKUP_REPO" -p snapshot.manifest.json -D "$tmp" 2>/dev/null; then
      if ! check_manifest_compatible "$tmp/snapshot.manifest.json" "$PINNED"; then
        rm -rf "$tmp"
        echo "ERROR: pinned release $PINNED is not compatible with reader [$MIN_VER,$MAX_VER]" >&2
        exit 3
      fi
    else
      echo "WARNING: pinned release $PINNED has no snapshot.manifest.json; proceeding without compatibility check" >&2
    fi
    rm -rf "$tmp"
  fi
  printf '%s\n' "$PINNED"
  exit 0
fi

if [[ -z "$TAG_REGEX" ]]; then
  echo "ERROR: TAG_REGEX is required when PODCAST_BACKUP_TAG / BACKUP_TAG is unset" >&2
  exit 1
fi

TAGS=()
while IFS= read -r _tag; do
  [[ -n "$_tag" ]] && TAGS+=("$_tag")
done < <(
  gh release list --repo "$BACKUP_REPO" --limit 200 --json tagName,publishedAt \
    | jq -r --arg re "$TAG_REGEX" \
      '[.[] | select(.tagName | test($re))] | sort_by(.publishedAt) | reverse | .[].tagName'
)

if [[ ${#TAGS[@]} -eq 0 ]]; then
  echo "ERROR: no releases matching $TAG_REGEX on $BACKUP_REPO" >&2
  exit 2
fi

for tag in "${TAGS[@]}"; do
  tmp="$(mktemp -d)"
  if ! gh release download "$tag" --repo "$BACKUP_REPO" -p snapshot.manifest.json -D "$tmp" 2>/dev/null; then
    echo "skip: $tag (no snapshot.manifest.json sibling asset)" >&2
    rm -rf "$tmp"
    continue
  fi
  if check_manifest_compatible "$tmp/snapshot.manifest.json" "$tag"; then
    rm -rf "$tmp"
    printf '%s\n' "$tag"
    exit 0
  fi
  rm -rf "$tmp"
done

echo "ERROR: no compatible snapshot with snapshot.manifest.json on $BACKUP_REPO (regex $TAG_REGEX)" >&2
echo "       Pin PODCAST_BACKUP_TAG / backup_tag or publish a backup with a supported corpus_format_version." >&2
exit 3
