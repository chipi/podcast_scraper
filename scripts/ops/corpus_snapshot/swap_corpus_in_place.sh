#!/usr/bin/env bash
# swap_corpus_in_place.sh — replace a corpus/ dir's CONTENTS while preserving its inode.
#
# Why in-place (DR-2 / #14): the corpus is a bind-backed docker volume
# (device=/srv/podcast-scraper/corpus) shared across the compose/operator/player stacks. `mv
# corpus corpus.bak` swaps the DIR INODE — the bind keeps resolving to the moved .bak inode, so
# containers serve the OLD corpus while /api/health reports green (the trap the restore script's
# own comment admitted, and the task-#14 swap re-hit). Emptying the dir and extracting INTO it
# keeps the inode, so every bind re-resolves to the new contents with no volume-rm (→ no
# public-player outage).
#
# Pure file swap — NO docker. The caller (restore_corpus_from_tarball_host.sh /
# cutover_corpus_inplace.sh) recreates the consumers afterward. Kept docker-free so the
# inode-preservation property is unit-testable (tests/.../test_corpus_swap_inode.py, DR-7).
#
# Usage: swap_corpus_in_place.sh <tarball> <corpus_dir> [backup_dir]
#   Extracts <tarball> (prod layout: top-level corpus/, or bare contents) into <corpus_dir>,
#   preserving <corpus_dir>'s inode. Prior contents are moved (not copied) to <backup_dir>
#   (default <corpus_dir>.bak.<UTC stamp>) — a rename within the same fs, so no double-disk.
set -euo pipefail

TARBALL="${1:?usage: swap_corpus_in_place.sh <tarball> <corpus_dir> [backup_dir]}"
CORPUS_DIR="${2:?corpus_dir required}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP_DIR="${3:-${CORPUS_DIR%/}.bak.${STAMP}}"

[ -f "$TARBALL" ] || { echo "ERROR: tarball not found: $TARBALL" >&2; exit 1; }
mkdir -p "$CORPUS_DIR"
CORPUS_DIR="$(cd "$CORPUS_DIR" && pwd)"
# STAGE lives INSIDE corpus/ (a hidden dir), NOT as a sibling. The corpus may be a separate
# Hetzner volume (terraform hcloud_volume.corpus, mounted at the corpus path) — a sibling STAGE
# would then be on a DIFFERENT filesystem, making the install move a cross-fs copy (double-disk +
# slow). Inside corpus/, the install move is always a same-fs rename regardless of the mount layout.
STAGE="$CORPUS_DIR/.corpus-incoming.${STAMP}"

cleanup_stage() { rm -rf "$STAGE" 2>/dev/null || true; }
trap cleanup_stage EXIT

# Portable dotfile-safe "move every child of $1 into $2/" (BSD + GNU; no `mv -t`).
_move_children() {
  find "$1" -mindepth 1 -maxdepth 1 -exec mv {} "$2"/ \;
}
# Same, but never touch the staging dir (used when emptying the live corpus, which contains STAGE).
_move_children_except_stage() {
  find "$1" -mindepth 1 -maxdepth 1 ! -path "$STAGE" -exec mv {} "$2"/ \;
}

# 1. Extract + validate into STAGE. corpus/'s real contents are UNTOUCHED until the new corpus is
#    proven good, so a bad tarball / disk-full leaves the live corpus in place (rollback = nothing).
rm -rf "$STAGE"; mkdir -p "$STAGE"
tar -xzf "$TARBALL" -C "$STAGE"
if [ -d "$STAGE/corpus" ]; then NEW="$STAGE/corpus"; else NEW="$STAGE"; fi
find "$NEW" -name '._*' -delete 2>/dev/null || true   # DR-5: strip macOS AppleDouble
if [ -z "$(find "$NEW" -name '*.gi.json' ! -name '._*' -print -quit 2>/dev/null)" ]; then
  echo "ERROR: staged corpus has no *.gi.json — refusing to swap in an empty corpus" >&2
  exit 1
fi

# 2. In-place swap preserving the $CORPUS_DIR inode. STAGE→corpus is a same-fs rename (both on the
#    corpus filesystem); the corpus/ dir inode is never touched so the shared bind keeps resolving.
#    Only the OLD-contents backup may cross fs when the corpus is a separate volume — a bounded,
#    one-time cost capped by the DR-6 prune (pass BACKUP_DIR on the corpus fs to avoid even that).
mkdir -p "$BACKUP_DIR"
_move_children_except_stage "$CORPUS_DIR" "$BACKUP_DIR"   # empty the live dir (keeps its inode)

install_ok=1
if [ "${SWAP_TEST_FAIL_INSTALL:-}" = "1" ]; then
  install_ok=0                               # test seam (DR-7): exercise the rollback branch
else
  _move_children "$NEW" "$CORPUS_DIR" || install_ok=0
fi

if [ "$install_ok" -eq 0 ]; then
  # Rollback must be as rigorous as the happy path (advisor H1): guard the restore and, if it
  # ALSO fails (e.g. genuine disk-full), fail LOUD with the recovery location — never a silent
  # green that leaves a corpus split across two dirs.
  echo "ERROR: install failed — rolling back prior corpus from $BACKUP_DIR" >&2
  find "$CORPUS_DIR" -mindepth 1 -maxdepth 1 ! -path "$STAGE" -exec rm -rf {} + 2>/dev/null || true
  if _move_children "$BACKUP_DIR" "$CORPUS_DIR"; then
    rmdir "$BACKUP_DIR" 2>/dev/null || true
    echo "rolled back: prior corpus restored to $CORPUS_DIR (inode preserved)" >&2
  else
    echo "FATAL: rollback ALSO failed — MANUAL RECOVERY NEEDED." >&2
    echo "       Prior corpus is in $BACKUP_DIR; partial contents (if any) in $CORPUS_DIR." >&2
    echo "       Do NOT start/keep consumers on $CORPUS_DIR until restored by hand." >&2
  fi
  exit 1
fi

echo "swapped corpus in place: ${CORPUS_DIR} (inode preserved) — backup: ${BACKUP_DIR}"
