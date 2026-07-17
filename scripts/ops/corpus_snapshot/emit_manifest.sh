#!/usr/bin/env bash
# Write snapshot.manifest.json (schema v1). See RFC-084 / ADR-092.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

usage() {
  cat <<'EOF'
Write snapshot.manifest.json for a corpus snapshot release.

Usage:
  emit_manifest.sh --output PATH [options]

Options:
  --output PATH           Destination file (required)
  --git-sha SHA           Producer git SHA (40 hex)
  --image-digest DIGEST   Optional producer image digest (sha256:...)
  --archive-path PATH     archive.relative_path (default: snapshot.tgz)
  --archive-sha256 HEX    Optional tarball sha256 hex
  --created-at RFC3339    UTC timestamp (default: now)
  --workflow-name NAME    Optional backup_workflow.name
  --workflow-run-id ID    Optional backup_workflow.run_id
  --workflow-attempt N    Optional backup_workflow.attempt (default: 1)
  -h, --help              This help
EOF
  exit "${1:-0}"
}

OUTPUT=""
GIT_SHA=""
IMAGE_DIGEST=""
ARCHIVE_PATH="snapshot.tgz"
ARCHIVE_SHA256=""
CREATED_AT=""
WORKFLOW_NAME=""
WORKFLOW_RUN_ID=""
WORKFLOW_ATTEMPT="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h | --help)
      usage 0
      ;;
    --output)
      OUTPUT="${2:?--output requires a path}"
      shift 2
      ;;
    --git-sha)
      GIT_SHA="${2:?--git-sha requires a value}"
      shift 2
      ;;
    --image-digest)
      IMAGE_DIGEST="${2:?--image-digest requires a value}"
      shift 2
      ;;
    --archive-path)
      ARCHIVE_PATH="${2:?--archive-path requires a value}"
      shift 2
      ;;
    --archive-sha256)
      ARCHIVE_SHA256="${2:?--archive-sha256 requires a value}"
      shift 2
      ;;
    --created-at)
      CREATED_AT="${2:?--created-at requires a value}"
      shift 2
      ;;
    --workflow-name)
      WORKFLOW_NAME="${2:?--workflow-name requires a value}"
      shift 2
      ;;
    --workflow-run-id)
      WORKFLOW_RUN_ID="${2:?--workflow-run-id requires a value}"
      shift 2
      ;;
    --workflow-attempt)
      WORKFLOW_ATTEMPT="${2:?--workflow-attempt requires a value}"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage 1
      ;;
  esac
done

if [[ -z "$OUTPUT" ]]; then
  echo "ERROR: --output is required" >&2
  usage 1
fi

if [[ -z "$GIT_SHA" && -z "$IMAGE_DIGEST" ]]; then
  echo "ERROR: at least one of --git-sha or --image-digest is required" >&2
  exit 1
fi

corpus_snapshot_require_cmd jq
read -r SCHEMA_VERSION CORPUS_FORMAT_VERSION < <(corpus_snapshot_read_producer_format)

if [[ -z "$CREATED_AT" ]]; then
  CREATED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
fi

producer_json='{}'
if [[ -n "$GIT_SHA" ]]; then
  producer_json="$(jq -n --arg sha "$GIT_SHA" '{git_sha: $sha}')"
fi
if [[ -n "$IMAGE_DIGEST" ]]; then
  producer_json="$(jq -n --argjson base "$producer_json" --arg digest "$IMAGE_DIGEST" \
    '$base + {image_digest: $digest}')"
fi

archive_json="$(jq -n --arg path "$ARCHIVE_PATH" '{relative_path: $path}')"
if [[ -n "$ARCHIVE_SHA256" ]]; then
  archive_json="$(jq -n --arg path "$ARCHIVE_PATH" --arg sha "$ARCHIVE_SHA256" \
    '{relative_path: $path, sha256: $sha}')"
fi

doc="$(jq -n \
  --argjson schema_version "$SCHEMA_VERSION" \
  --argjson corpus_format_version "$CORPUS_FORMAT_VERSION" \
  --arg created_at "$CREATED_AT" \
  --argjson producer "$producer_json" \
  --argjson archive "$archive_json" \
  '{
    schema_version: $schema_version,
    corpus_format_version: $corpus_format_version,
    created_at: $created_at,
    producer: $producer,
    archive: $archive
  }')"

if [[ -n "$WORKFLOW_NAME" && -n "$WORKFLOW_RUN_ID" ]]; then
  # jq --argjson requires valid JSON; a non-integer attempt aborts jq with a
  # cryptic error. Coerce anything non-numeric to 1 (review low/emit-manifest).
  [[ "$WORKFLOW_ATTEMPT" =~ ^[0-9]+$ ]] || WORKFLOW_ATTEMPT=1
  doc="$(jq -n --argjson base "$doc" \
    --arg name "$WORKFLOW_NAME" \
    --arg run_id "$WORKFLOW_RUN_ID" \
    --argjson attempt "$WORKFLOW_ATTEMPT" \
    '$base + {backup_workflow: {name: $name, run_id: $run_id, attempt: $attempt}}')"
fi

printf '%s\n' "$doc" | jq '.' >"$OUTPUT"
