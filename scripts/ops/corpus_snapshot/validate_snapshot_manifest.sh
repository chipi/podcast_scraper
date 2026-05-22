#!/usr/bin/env bash
# Validate snapshot.manifest.json (schema v1). See RFC-084 / ADR-092.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

usage() {
  cat <<'EOF'
Validate snapshot.manifest.json (schema v1).

Usage:
  validate_snapshot_manifest.sh PATH

Environment:
  CORPUS_SNAPSHOT_REQUIRE_SHA256=1  Fail when archive.sha256 is missing
EOF
  exit "${1:-0}"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage 0
fi

if [[ $# -ne 1 ]]; then
  echo "ERROR: expected exactly one manifest path" >&2
  usage 1
fi

MANIFEST="$1"
if [[ ! -f "$MANIFEST" ]]; then
  echo "ERROR: manifest not found: $MANIFEST" >&2
  exit 1
fi

corpus_snapshot_require_cmd jq

if ! jq -e . "$MANIFEST" >/dev/null 2>&1; then
  echo "ERROR: manifest is not valid JSON: $MANIFEST" >&2
  exit 1
fi

schema="$(jq -r '.schema_version // empty' "$MANIFEST")"
if [[ -z "$schema" || "$schema" != "1" ]]; then
  echo "ERROR: unsupported or missing schema_version (expected 1): $MANIFEST" >&2
  exit 1
fi

for field in corpus_format_version created_at; do
  if ! jq -e --arg f "$field" '.[$f] != null' "$MANIFEST" >/dev/null; then
    echo "ERROR: missing required field $field in $MANIFEST" >&2
    exit 1
  fi
done

if ! jq -e '.producer.git_sha // .producer.image_digest' "$MANIFEST" >/dev/null; then
  echo "ERROR: producer must include git_sha and/or image_digest" >&2
  exit 1
fi

if ! jq -e '.archive.relative_path | strings' "$MANIFEST" >/dev/null; then
  echo "ERROR: archive.relative_path must be a string" >&2
  exit 1
fi

if [[ "${CORPUS_SNAPSHOT_REQUIRE_SHA256:-0}" == "1" ]]; then
  if ! jq -e '.archive.sha256 | strings' "$MANIFEST" >/dev/null; then
    echo "ERROR: archive.sha256 required but missing in $MANIFEST" >&2
    exit 1
  fi
fi

corpus_fmt="$(jq -r '.corpus_format_version' "$MANIFEST")"
if [[ ! "$corpus_fmt" =~ ^[0-9]+$ ]]; then
  echo "ERROR: corpus_format_version must be a non-negative integer" >&2
  exit 1
fi

git_sha="$(jq -r '.producer.git_sha // empty' "$MANIFEST")"
if [[ -n "$git_sha" && ! "$git_sha" =~ ^[0-9a-fA-F]{40}$ ]]; then
  echo "ERROR: producer.git_sha must be 40 hex characters when set" >&2
  exit 1
fi

echo "OK: $MANIFEST" >&2
