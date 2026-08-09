#!/usr/bin/env bash
# cutover_corpus_inplace.sh — the ONE sanctioned prod corpus swap (DR-3).
#
# Composes the existing pieces so a corpus cutover can't re-hit the class of bugs the task-#14
# swap surfaced (bind-volume inode orphaning, VIA_FILES secrets, missing topic_clusters.json,
# stale-corpus-reports-green). Run on the VPS host as deploy@.
#
#   1. validate the tarball's manifest      (corpus_snapshot/validate_snapshot_manifest.sh)
#   2. swap IN PLACE, inode-preserving      (corpus_snapshot/swap_corpus_in_place.sh — also strips
#                                            macOS AppleDouble ._* and refuses an empty tarball)
#   3. recreate ALL control-plane consumers (up -d --force-recreate — they cached the old corpus)
#   4. generate search/topic_clusters.json  (query-time-read; prep/pipeline never made it → #14
#                                            404; built inside the api container which has the pkg)
#   5. recreate the public player           (separate compose project — reload its corpus too)
#   6. IDENTITY smoke                        (post_deploy_smoke.sh --expect-corpus-produced-at —
#                                            fails red if the SERVED corpus != the one we shipped)
#
# Usage: cutover_corpus_inplace.sh <tarball_path>
# Env:   PODCAST_REPO_DIR (default /srv/podcast-scraper)
#        EXPECT_CORPUS_PRODUCED_AT — the new corpus manifest's produced_at (drives the #6 smoke)
#        TOPIC_CLUSTER_THRESHOLD  (default 0.75 — the cloud_balanced profile value, NOT 0.35)
#        PLAYER_COMPOSE           (default compose/docker-compose.player-public.yml; "" to skip)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARBALL="${1:?usage: cutover_corpus_inplace.sh <tarball_path>}"
REPO_DIR="${PODCAST_REPO_DIR:-/srv/podcast-scraper}"
THRESHOLD="${TOPIC_CLUSTER_THRESHOLD:-0.75}"
PLAYER_COMPOSE="${PLAYER_COMPOSE-compose/docker-compose.player-public.yml}"
cd "$REPO_DIR"

log() { printf '[cutover %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

# 1. Validate the tarball manifest (extract just the manifest to a temp dir).
log "validating snapshot manifest"
_mtmp="$(mktemp -d)"
trap 'rm -rf "$_mtmp"' EXIT
tar -xzf "$TARBALL" -C "$_mtmp" --wildcards '*snapshot.manifest.json' 2>/dev/null || true
_manifest="$(find "$_mtmp" -name 'snapshot.manifest.json' -print -quit)"
if [ -n "$_manifest" ]; then
  bash "$SCRIPT_DIR/corpus_snapshot/validate_snapshot_manifest.sh" "$_manifest"
elif [ "${ALLOW_UNMANIFESTED:-}" = "1" ]; then
  log "WARN: no snapshot.manifest.json — proceeding (ALLOW_UNMANIFESTED=1)"
else
  # This is the ONE sanctioned prod swap — refuse to cut over an unverifiable tarball. The whole
  # point of the arc is 'verify before swap'. Override with ALLOW_UNMANIFESTED=1 if intentional.
  echo "ERROR: no snapshot.manifest.json in tarball — refusing unverified cutover." >&2
  echo "       (set ALLOW_UNMANIFESTED=1 to override.)" >&2
  exit 1
fi

# 2. In-place, inode-preserving swap (the linchpin — see swap_corpus_in_place.sh).
log "swapping corpus in place"
bash "$SCRIPT_DIR/corpus_snapshot/swap_corpus_in_place.sh" \
  "$TARBALL" "$REPO_DIR/corpus" "$REPO_DIR/corpus.bak.$(date -u +%Y%m%dT%H%M%SZ)"
chown -R deploy:deploy corpus 2>/dev/null || true

CONTROL=(
  docker compose --env-file .env
  -f compose/docker-compose.stack.yml
  -f compose/docker-compose.prod.yml
  -f compose/docker-compose.vps-prod.yml
)

# 3. Recreate control-plane consumers (all — each cached the old corpus at boot).
log "recreating control-plane consumers"
"${CONTROL[@]}" up -d --force-recreate

# 4. Generate topic_clusters.json inside the api container (has the package + corpus at /app/output).
log "generating topic_clusters.json (threshold=$THRESHOLD)"
"${CONTROL[@]}" exec -T api \
  python -m podcast_scraper.cli topic-clusters --output-dir /app/output --threshold "$THRESHOLD"

# 5. Recreate the public player so it reloads the new corpus too (separate compose project).
if [ -n "$PLAYER_COMPOSE" ] && [ -f "$PLAYER_COMPOSE" ]; then
  log "recreating public player"
  docker compose --env-file .env -f "$PLAYER_COMPOSE" up -d --force-recreate || \
    log "WARN: player recreate failed — recreate it via deploy-player"
fi

# 6. Identity smoke — RED if the served corpus is not the one we intended (DR-1).
log "running identity smoke"
_smoke_args=(--base-url http://127.0.0.1:8090 --corpus-path /app/output)
[ -n "${EXPECT_CORPUS_PRODUCED_AT:-}" ] && _smoke_args+=(--expect-corpus-produced-at "$EXPECT_CORPUS_PRODUCED_AT")
bash "$SCRIPT_DIR/post_deploy_smoke.sh" "${_smoke_args[@]}"

# DR-6: retention for the corpus.bak.* left by the swap.
bash "$SCRIPT_DIR/corpus_snapshot/prune_corpus_backups.sh" "$REPO_DIR" "${RESTORE_BACKUP_KEEP:-2}"

log "cutover complete"
