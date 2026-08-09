#!/usr/bin/env bash
# Pack a live corpus tree into a canonical snapshot.tgz + sibling manifest,
# WITHOUT going through the CI backup workflow. Bit-format identical to what
# ``backup-corpus.yml`` / ``backup-corpus-prod.yml`` produce, so the resulting
# tarball is consumable by the CI-restore path (and vice versa).
#
# See RFC-084 / ADR-092 and docs/guides/CORPUS_SNAPSHOT_MANIFEST_AND_RESTORE.md.
# Instance-to-instance transport (scp / USB / etc.) is the operator's choice —
# this script only prepares the artifact.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

usage() {
  cat <<'EOF'
Pack a local corpus into a portable snapshot.tgz + snapshot.manifest.json.

Usage:
  pack_corpus_local.sh --corpus-dir DIR --out FILE [--layout codespace|prod]

Options:
  --corpus-dir DIR   Corpus root (the parent that contains feeds.spec.yaml)
  --out FILE         Destination tarball path (e.g. /tmp/snapshot.tgz)
  --layout NAME      Archive layout: codespace (default) or prod
  -h, --help         This help

Environment:
  GIT_SHA / GITHUB_SHA   Producer git SHA. Falls back to ``git rev-parse HEAD``
                         when unset. One of these MUST resolve to a value.
  IMAGE_DIGEST           Optional producer image digest.
  CORPUS_SNAPSHOT_MIN_TARBALL_BYTES
                         Minimum accepted tarball size (default 1024). Test-only
                         override; keep the 1 KiB floor in production use.

Sanity checks (refuse to pack when any fails):
  - --corpus-dir contains feeds.spec.yaml
  - tree contains at least one *.gi.json (mirrors backup-corpus.yml)
  - resulting tarball is at least 1 KiB

Layout matrix:
  codespace  →  archive root is .codespace_corpus/  (default)
  prod       →  archive root is corpus/
EOF
  exit "${1:-0}"
}

CORPUS_DIR=""
OUT=""
LAYOUT="codespace"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h | --help)
      usage 0
      ;;
    --corpus-dir)
      CORPUS_DIR="${2:?--corpus-dir requires a value}"
      shift 2
      ;;
    --out)
      OUT="${2:?--out requires a value}"
      shift 2
      ;;
    --layout)
      LAYOUT="${2:?--layout requires codespace or prod}"
      shift 2
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage 1
      ;;
  esac
done

if [[ -z "$CORPUS_DIR" || -z "$OUT" ]]; then
  echo "ERROR: --corpus-dir and --out are required" >&2
  usage 1
fi

case "$LAYOUT" in
  codespace) LAYOUT_ROOT=".codespace_corpus" ;;
  prod)      LAYOUT_ROOT="corpus" ;;
  *)
    echo "ERROR: --layout must be codespace or prod (got $LAYOUT)" >&2
    exit 1
    ;;
esac

if [[ ! -d "$CORPUS_DIR" ]]; then
  echo "ERROR: corpus dir not found: $CORPUS_DIR" >&2
  exit 1
fi
if [[ ! -f "$CORPUS_DIR/feeds.spec.yaml" ]]; then
  echo "ERROR: $CORPUS_DIR/feeds.spec.yaml missing — is this a corpus root?" >&2
  exit 1
fi

# DR-8: a REAL count (not -print -quit, which is only ever 0/1), excluding macOS AppleDouble
# ._* companions so the number is the true gi.json count — catches under-capture AND the ._*
# doubling (#14: a 105-ep corpus read as 210). Assert a floor when CORPUS_SNAPSHOT_MIN_GI is set.
GI_COUNT="$(find "$CORPUS_DIR" -name '*.gi.json' ! -name '._*' 2>/dev/null | wc -l | tr -d ' ')"
echo "gi.json artifacts: $GI_COUNT"
if [[ "$GI_COUNT" -eq 0 ]]; then
  echo "ERROR: $CORPUS_DIR has no *.gi.json artifacts — corpus looks empty." >&2
  echo "       (backup-corpus.yml applies the same guard; see #699.)" >&2
  exit 1
fi
MIN_GI="${CORPUS_SNAPSHOT_MIN_GI:-0}"
if [[ "$MIN_GI" -gt 0 && "$GI_COUNT" -lt "$MIN_GI" ]]; then
  echo "ERROR: $CORPUS_DIR has $GI_COUNT *.gi.json < expected floor $MIN_GI — under-capture?" >&2
  exit 1
fi

corpus_snapshot_require_cmd tar
corpus_snapshot_require_cmd jq

# Resolve producer identity. finalize_backup_bundle.sh accepts GIT_SHA or
# GITHUB_SHA (env, either); locally we fall back to `git rev-parse HEAD` so
# the manifest still carries producer identity without CI running.
if [[ -z "${GIT_SHA:-}" && -z "${GITHUB_SHA:-}" ]]; then
  if command -v git >/dev/null 2>&1 && git -C "$(corpus_snapshot_repo_root)" rev-parse HEAD >/dev/null 2>&1; then
    GIT_SHA="$(git -C "$(corpus_snapshot_repo_root)" rev-parse HEAD)"
    export GIT_SHA
    echo "INFO: GIT_SHA fell back to \`git rev-parse HEAD\` → $GIT_SHA"
  else
    echo "ERROR: no GIT_SHA / GITHUB_SHA env and \`git rev-parse HEAD\` failed." >&2
    echo "       Set GIT_SHA=<sha> or IMAGE_DIGEST=<digest> and retry." >&2
    exit 1
  fi
fi

OUT_DIR="$(cd "$(dirname "$OUT")" && pwd)"
OUT_ABS="$OUT_DIR/$(basename "$OUT")"
mkdir -p "$OUT_DIR"

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

echo "INFO: packing $CORPUS_DIR → $OUT_ABS (layout=$LAYOUT, archive root=$LAYOUT_ROOT/)"

# Stage the corpus tree under the layout root. Using ``cp -a`` preserves modes
# and mtimes so the tarball is deterministic across replays of the same tree.
mkdir "$WORKDIR/$LAYOUT_ROOT"
# The `.` shell-glob in the source path copies contents, not the parent dir
# itself; this gives us `<WORKDIR>/<LAYOUT_ROOT>/<contents>` — matching what
# the CI tar step in backup-corpus.yml produces (`tar -czf snapshot.tgz .codespace_corpus`).
# DR-5: COPYFILE_DISABLE=1 stops macOS from writing AppleDouble (._*) companions during cp/tar,
# and the find-delete sweeps any that slipped in — otherwise they ride into the tarball and land
# on the Linux box as ._*.gi.json etc. (the #14 105->210 gi.json inflation).
COPYFILE_DISABLE=1 cp -a "$CORPUS_DIR/." "$WORKDIR/$LAYOUT_ROOT/"
find "$WORKDIR/$LAYOUT_ROOT" -name '._*' -delete 2>/dev/null || true

COPYFILE_DISABLE=1 tar -czf "$OUT_ABS" -C "$WORKDIR" "$LAYOUT_ROOT"

MIN_BYTES="${CORPUS_SNAPSHOT_MIN_TARBALL_BYTES:-1024}"
SIZE_BYTES="$(wc -c < "$OUT_ABS" | tr -d ' ')"
if [[ "$SIZE_BYTES" -lt "$MIN_BYTES" ]]; then
  echo "ERROR: tarball is suspiciously small ($SIZE_BYTES bytes, floor=$MIN_BYTES); aborting." >&2
  rm -f "$OUT_ABS"
  exit 1
fi

# Manifest injection + sibling manifest + sha256. finalize_backup_bundle.sh
# re-tars the archive so the inner snapshot.manifest.json lands at archive root.
bash "$SCRIPT_DIR/finalize_backup_bundle.sh" "$OUT_ABS"

echo ""
echo "OK: $OUT_ABS"
echo "OK: $OUT_DIR/snapshot.manifest.json  (sibling manifest with archive.sha256)"
echo ""
echo "Transport the tarball + sibling manifest together to the target host,"
echo "then run: make import-corpus FILE=$(basename "$OUT_ABS") WORKSPACE_DIR=<parent> LAYOUT=$LAYOUT"
