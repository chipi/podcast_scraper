#!/usr/bin/env bash
# Inject snapshot.manifest.json into snapshot.tgz and write sibling manifest.
# See RFC-084 / ADR-092.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

usage() {
  cat <<'EOF'
Finalize a corpus snapshot tarball with snapshot.manifest.json at archive root.

Usage:
  finalize_backup_bundle.sh TARBALL_PATH

Environment:
  CORPUS_SNAPSHOT_REPO_ROOT   Repo root (default: auto)
  GIT_SHA / GITHUB_SHA        Producer git SHA (required unless IMAGE_DIGEST set)
  IMAGE_DIGEST                Optional producer image digest
  BACKUP_WORKFLOW_NAME        Optional workflow name for manifest
  BACKUP_WORKFLOW_RUN_ID      Optional workflow run id
  BACKUP_WORKFLOW_ATTEMPT     Optional attempt (default: 1)
EOF
  exit "${1:-0}"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage 0
fi

if [[ $# -ne 1 ]]; then
  echo "ERROR: expected tarball path argument" >&2
  usage 1
fi

TARBALL="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
if [[ ! -f "$TARBALL" ]]; then
  echo "ERROR: tarball not found: $TARBALL" >&2
  exit 1
fi

GIT_SHA="${GIT_SHA:-${GITHUB_SHA:-}}"
IMAGE_DIGEST="${IMAGE_DIGEST:-}"
WORKFLOW_NAME="${BACKUP_WORKFLOW_NAME:-}"
WORKFLOW_RUN_ID="${BACKUP_WORKFLOW_RUN_ID:-}"
WORKFLOW_ATTEMPT="${BACKUP_WORKFLOW_ATTEMPT:-1}"

if [[ -z "$GIT_SHA" && -z "$IMAGE_DIGEST" ]]; then
  echo "ERROR: set GIT_SHA or GITHUB_SHA (or IMAGE_DIGEST) for producer identity" >&2
  exit 1
fi

corpus_snapshot_require_cmd tar
corpus_snapshot_require_cmd jq

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

tar -xzf "$TARBALL" -C "$WORKDIR"

INNER_MANIFEST="$WORKDIR/snapshot.manifest.json"
emit_args=(--output "$INNER_MANIFEST" --archive-path snapshot.tgz)
if [[ -n "$GIT_SHA" ]]; then
  emit_args+=(--git-sha "$GIT_SHA")
fi
if [[ -n "$IMAGE_DIGEST" ]]; then
  emit_args+=(--image-digest "$IMAGE_DIGEST")
fi
if [[ -n "$WORKFLOW_NAME" && -n "$WORKFLOW_RUN_ID" ]]; then
  emit_args+=(--workflow-name "$WORKFLOW_NAME" --workflow-run-id "$WORKFLOW_RUN_ID" \
    --workflow-attempt "$WORKFLOW_ATTEMPT")
fi

bash "$SCRIPT_DIR/emit_manifest.sh" "${emit_args[@]}"
bash "$SCRIPT_DIR/validate_snapshot_manifest.sh" "$INNER_MANIFEST"

# Pack outside $WORKDIR so we never archive snapshot.tgz while it is being written
# (GNU/BSD tar: "file changed as we read it" when output lives under ".").
# Template must end in XXXXXX (BSD mktemp); do not append .tgz after Xs (parallel mkstemp races).
NEW_TARBALL="$(mktemp "${TMPDIR:-/tmp}/snapshot-finalize.XXXXXX")"
tar -czf "$NEW_TARBALL" -C "$WORKDIR" .

if command -v sha256sum >/dev/null 2>&1; then
  ARCHIVE_SHA256="$(sha256sum "$NEW_TARBALL" | awk '{print $1}')"
elif command -v shasum >/dev/null 2>&1; then
  ARCHIVE_SHA256="$(shasum -a 256 "$NEW_TARBALL" | awk '{print $1}')"
else
  echo "ERROR: need sha256sum or shasum" >&2
  exit 1
fi

SIBLING_MANIFEST="$(dirname "$TARBALL")/snapshot.manifest.json"
sibling_args=(--output "$SIBLING_MANIFEST" --archive-path snapshot.tgz --archive-sha256 "$ARCHIVE_SHA256")
if [[ -n "$GIT_SHA" ]]; then
  sibling_args+=(--git-sha "$GIT_SHA")
fi
if [[ -n "$IMAGE_DIGEST" ]]; then
  sibling_args+=(--image-digest "$IMAGE_DIGEST")
fi
if [[ -n "$WORKFLOW_NAME" && -n "$WORKFLOW_RUN_ID" ]]; then
  sibling_args+=(--workflow-name "$WORKFLOW_NAME" --workflow-run-id "$WORKFLOW_RUN_ID" \
    --workflow-attempt "$WORKFLOW_ATTEMPT")
fi

bash "$SCRIPT_DIR/emit_manifest.sh" "${sibling_args[@]}"
CORPUS_SNAPSHOT_REQUIRE_SHA256=1 bash "$SCRIPT_DIR/validate_snapshot_manifest.sh" "$SIBLING_MANIFEST"

mv "$NEW_TARBALL" "$TARBALL"
echo "OK: finalized $TARBALL and wrote $(dirname "$TARBALL")/snapshot.manifest.json"
