#!/usr/bin/env bash
# prune_corpus_backups.sh — retention for corpus.bak.* dirs (DR-6).
#
# Each restore/cutover leaves a full corpus copy as corpus.bak.<UTC stamp> under the device path.
# On a single VPS these accumulate → disk-full during a later restore is the very failure the
# rollback can't recover from. Keep the newest N, delete the rest.
#
# Usage: prune_corpus_backups.sh <dir> [keep]
#   dir  — parent holding corpus.bak.* (e.g. /srv/podcast-scraper)
#   keep — how many newest to retain (default 2; 0 disables pruning)
#
# Pure shell (no docker) so it's unit-testable; restore_corpus_from_tarball_host.sh and
# cutover_corpus_inplace.sh both call it.
set -euo pipefail

DIR="${1:?usage: prune_corpus_backups.sh <dir> [keep]}"
KEEP="${2:-2}"

case "$KEEP" in
  ''|*[!0-9]*) echo "ERROR: keep must be a non-negative integer, got '$KEEP'" >&2; exit 1 ;;
esac
[ "$KEEP" -eq 0 ] && exit 0   # 0 = disabled

# corpus.bak.<UTC stamp> names sort lexically by age; newest first, drop everything past KEEP.
ls -1d "$DIR"/corpus.bak.* 2>/dev/null | sort -r | tail -n "+$((KEEP + 1))" | while read -r old; do
  echo "pruning old corpus backup: $old"
  rm -rf "$old"
done
