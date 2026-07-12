#!/usr/bin/env bash
# Validate + extract a local snapshot.tgz into a workspace, WITHOUT going through
# the CI restore workflow (no gh, no network). Consumes any snapshot.tgz that
# ``backup-corpus.yml`` / ``pack_corpus_local.sh`` produced — see RFC-084 /
# ADR-092 and docs/guides/CORPUS_SNAPSHOT_MANIFEST_AND_RESTORE.md.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

usage() {
  cat <<'EOF'
Import a local corpus snapshot.tgz into a workspace directory. No network calls.

Usage:
  import_local_snapshot.sh --file FILE --workspace-dir DIR [--layout codespace|prod]

Options:
  --file FILE          Path to snapshot.tgz on disk
  --workspace-dir DIR  Parent directory the layout root will be extracted into
  --layout NAME        Expected archive layout: codespace (default) or prod.
                       Import refuses to extract if the tarball's root does
                       not match this layout.
  -h, --help           This help

Environment:
  CORPUS_SNAPSHOT_SKIP_SHA256_VERIFY=1   Skip archive.sha256 verification
                                          even when the sibling manifest carries one.

Behavior:
  1. Prefer sibling snapshot.manifest.json next to FILE. If none, extract the
     inner one from the archive root and validate that.
  2. Run validate_snapshot_manifest.sh on the chosen manifest.
  3. Reader-range check against config/corpus_snapshot_reader_support.json
     (same helper download_and_verify_snapshot.sh uses after a GH download).
  4. Verify archive.sha256 when sibling manifest supplies one.
  5. Verify the archive layout root matches --layout. Refuse mismatch.
  6. Extract into --workspace-dir. Result: <workspace-dir>/.codespace_corpus/
     or <workspace-dir>/corpus/, depending on layout.
EOF
  exit "${1:-0}"
}

FILE=""
WORKSPACE_DIR=""
LAYOUT="codespace"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h | --help)
      usage 0
      ;;
    --file)
      FILE="${2:?--file requires a value}"
      shift 2
      ;;
    --workspace-dir)
      WORKSPACE_DIR="${2:?--workspace-dir requires a value}"
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

if [[ -z "$FILE" || -z "$WORKSPACE_DIR" ]]; then
  echo "ERROR: --file and --workspace-dir are required" >&2
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

if [[ ! -f "$FILE" ]]; then
  echo "ERROR: tarball not found: $FILE" >&2
  exit 1
fi

corpus_snapshot_require_cmd tar
corpus_snapshot_require_cmd jq

FILE_ABS="$(cd "$(dirname "$FILE")" && pwd)/$(basename "$FILE")"
FILE_DIR="$(dirname "$FILE_ABS")"

SIBLING_MANIFEST="$FILE_DIR/snapshot.manifest.json"
WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

# Pick manifest: prefer sibling next to tarball (that's the one finalize
# writes with archive.sha256), fall back to the inner one at archive root.
CHOSEN_MANIFEST=""
MANIFEST_SOURCE=""
if [[ -f "$SIBLING_MANIFEST" ]]; then
  CHOSEN_MANIFEST="$SIBLING_MANIFEST"
  MANIFEST_SOURCE="sibling"
else
  # tar --extract with -C into workdir; look for snapshot.manifest.json at root.
  tar -xzf "$FILE_ABS" -C "$WORKDIR" ./snapshot.manifest.json 2>/dev/null \
    || tar -xzf "$FILE_ABS" -C "$WORKDIR" snapshot.manifest.json 2>/dev/null \
    || true
  if [[ -f "$WORKDIR/snapshot.manifest.json" ]]; then
    CHOSEN_MANIFEST="$WORKDIR/snapshot.manifest.json"
    MANIFEST_SOURCE="inner"
  fi
fi

if [[ -z "$CHOSEN_MANIFEST" ]]; then
  echo "ERROR: no snapshot.manifest.json next to $FILE and none inside the archive." >&2
  echo "       This tarball is not RFC-084 conformant." >&2
  exit 1
fi

echo "INFO: using $MANIFEST_SOURCE manifest → $CHOSEN_MANIFEST"

bash "$SCRIPT_DIR/validate_snapshot_manifest.sh" "$CHOSEN_MANIFEST"

read -r MIN_VER MAX_VER < <(corpus_snapshot_read_reader_range)
FMT="$(jq -r '.corpus_format_version' "$CHOSEN_MANIFEST")"
if ! corpus_snapshot_version_supported "$FMT" "$MIN_VER" "$MAX_VER"; then
  echo "ERROR: corpus_format_version=$FMT outside reader range [$MIN_VER,$MAX_VER]" >&2
  exit 3
fi
echo "INFO: reader-range check passed (fmt=$FMT ∈ [$MIN_VER,$MAX_VER])"

# sha256 verification when sibling manifest carries one.
if [[ "$MANIFEST_SOURCE" == "sibling" && "${CORPUS_SNAPSHOT_SKIP_SHA256_VERIFY:-0}" != "1" ]]; then
  EXPECTED="$(jq -r '.archive.sha256 // empty' "$CHOSEN_MANIFEST")"
  if [[ -n "$EXPECTED" ]]; then
    if command -v sha256sum >/dev/null 2>&1; then
      ACTUAL="$(sha256sum "$FILE_ABS" | awk '{print $1}')"
    elif command -v shasum >/dev/null 2>&1; then
      ACTUAL="$(shasum -a 256 "$FILE_ABS" | awk '{print $1}')"
    else
      echo "ERROR: need sha256sum or shasum for archive digest check" >&2
      exit 1
    fi
    if [[ "$ACTUAL" != "$EXPECTED" ]]; then
      echo "ERROR: archive.sha256 mismatch — tarball corrupted or tampered." >&2
      echo "       expected=$EXPECTED" >&2
      echo "       actual  =$ACTUAL" >&2
      exit 1
    fi
    echo "INFO: archive.sha256 verified"
  else
    echo "WARN: sibling manifest carries no archive.sha256 — skipping digest check"
  fi
fi

# Layout guard: peek at the archive's top-level entries and refuse mismatch.
# The `sed 's|^\./||'` strips the leading `./` that GNU tar emits when packed
# with `-C dir .` (see finalize_backup_bundle.sh); without it, every entry
# collapses to root-`.` under `awk -F/`, hiding the real top-level names.
TAR_ROOTS="$(tar -tzf "$FILE_ABS" | sed 's|^\./||' | awk -F/ '{print $1}' | sort -u | grep -Ev '^\.?$' || true)"
if ! grep -qxF "$LAYOUT_ROOT" <<<"$TAR_ROOTS"; then
  echo "ERROR: archive top-level does not contain '$LAYOUT_ROOT/' — layout mismatch." >&2
  echo "       expected layout=$LAYOUT (root=$LAYOUT_ROOT/); archive roots: $(echo "$TAR_ROOTS" | tr '\n' ' ')" >&2
  echo "       Retry with the correct --layout." >&2
  exit 1
fi

mkdir -p "$WORKSPACE_DIR"
WORKSPACE_ABS="$(cd "$WORKSPACE_DIR" && pwd)"

if [[ -e "$WORKSPACE_ABS/$LAYOUT_ROOT" ]]; then
  echo "ERROR: $WORKSPACE_ABS/$LAYOUT_ROOT already exists. Move or remove it first;" >&2
  echo "       refusing to overwrite an existing corpus (this may be live data)." >&2
  exit 1
fi

echo "INFO: extracting into $WORKSPACE_ABS/ (layout root=$LAYOUT_ROOT/)"
tar -xzf "$FILE_ABS" -C "$WORKSPACE_ABS" "$LAYOUT_ROOT"

echo ""
echo "OK: $WORKSPACE_ABS/$LAYOUT_ROOT/"
echo ""
echo "Compat check (informational — will NOT fail import):"
echo "  make corpus-compat-check CORPUS_DIR=$WORKSPACE_ABS/$LAYOUT_ROOT"
