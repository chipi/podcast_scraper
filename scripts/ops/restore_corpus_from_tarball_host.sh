#!/usr/bin/env bash
# Restore corpus from snapshot.tgz on a VPS (prod or DR drill).
#
# Runs on the host as deploy@ after the GHA runner uploads the tarball
# (and this script copy) over Tailscale SSH. Overwrites /srv/podcast-scraper/corpus,
# recreates api + viewer, smoke-checks /api/health inside the api container (GH-745).
#
# Restore is FAITHFUL (restore-as-is); corpus migrations are decoupled and OFF by
# default. A manual restore may opt into a smart, conditional upgrade via
# RESTORE_UPGRADE_MODE (see the block after the smoke check).
#
# Usage: restore_corpus_from_tarball_host.sh <tarball_path>
#
# Env (optional): PODCAST_REPO_DIR — default /srv/podcast-scraper
#   RESTORE_EXTRACT_ONLY=1 — extract + corpus/ check only (CI / local rehearsal)
#   RESTORE_UPGRADE_MODE=skip|auto|force — default skip (DR = restore-as-is).
#     auto  = apply migrations only if 'upgrade status' reports pending (exit 2).
#     force = always apply pending migrations. Runs in the live api container.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARBALL="${1:?usage: restore_corpus_from_tarball_host.sh <tarball_path>}"
REPO_DIR="${PODCAST_REPO_DIR:-/srv/podcast-scraper}"

cd "$REPO_DIR"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
BAK="$REPO_DIR/corpus.bak.$STAMP"

# DR-2: swap the corpus IN PLACE (preserving the corpus/ inode) instead of `mv corpus corpus.bak`.
# The corpus is a bind-backed volume (device=.../corpus) shared across the compose/operator/player
# stacks; mv-ing the dir orphaned every bind onto the stale .bak inode while /api/health reported
# green (the trap this script's old comment admitted + the #14 swap re-hit). swap_corpus_in_place.sh
# extracts+validates into a temp dir, then empties+refills corpus/ keeping its inode, backing up the
# old contents to $BAK (a rename within the fs — no double-disk). It refuses an empty tarball, so a
# bad/corrupt archive leaves the live corpus untouched.
bash "$SCRIPT_DIR/corpus_snapshot/swap_corpus_in_place.sh" "$TARBALL" "$REPO_DIR/corpus" "$BAK"

if [ "${RESTORE_EXTRACT_ONLY:-}" = "1" ]; then
  echo "Restore swap OK under $REPO_DIR/corpus (RESTORE_EXTRACT_ONLY — skipping docker recreate)"
  exit 0
fi
chown -R deploy:deploy corpus
rm -f "$TARBALL"

COMPOSE=(
  docker compose --env-file .env
  -f compose/docker-compose.stack.yml
  -f compose/docker-compose.prod.yml
  -f compose/docker-compose.vps-prod.yml
)

# DR-2: recreate ALL corpus consumers in this compose (not just `api viewer`). In-place swap keeps
# the bind resolved, but each consumer loaded the OLD corpus at boot, so every service must be
# recreated to re-read it. (The PUBLIC player is a separate compose project — recreate it via
# deploy-player / the blessed cutover_corpus_inplace.sh, which also runs the identity smoke.)
# Restore is FAITHFUL (restore-as-is); corpus migrations stay DECOUPLED (#862/#1176).
"${COMPOSE[@]}" up -d --force-recreate
sleep 8
"${COMPOSE[@]}" exec -T api curl -fsS http://127.0.0.1:8000/api/health | head -c 200
echo

# DR-6: prune old corpus.bak.* so full-corpus copies don't accumulate on the single-VPS device
# path (disk-full-during-restore is the very failure the rollback can't recover from). Keep the
# newest RESTORE_BACKUP_KEEP (default 2); set 0 to disable. Factored into a testable helper.
bash "$SCRIPT_DIR/corpus_snapshot/prune_corpus_backups.sh" "$REPO_DIR" "${RESTORE_BACKUP_KEEP:-2}"
echo "Restore complete on host"

# Optional, decoupled corpus upgrade for a MANUAL restore — off by default
# (DR drill / backup-restore leave it unset = restore-as-is). It runs in the LIVE
# post-boot api container (correct /app/output mount, unlike the old pre-boot
# one-off), and is SMART: ``auto`` applies only when ``upgrade status`` reports
# pending migrations (exit 2 = the corpus's ledger is behind the deployed
# software's registry — i.e. backup-version < deploy-version).
UP_CLI=(python -m podcast_scraper.cli upgrade)
case "${RESTORE_UPGRADE_MODE:-skip}" in
  skip)
    echo "Corpus upgrade: skipped (restore-as-is; set RESTORE_UPGRADE_MODE=auto|force to opt in)"
    ;;
  auto)
    set +e
    "${COMPOSE[@]}" exec -T api "${UP_CLI[@]}" status --corpus-dir /app/output
    st=$?
    set -e
    if [ "$st" -eq 2 ]; then
      echo "Corpus upgrade: pending migrations — applying (backup is behind deployed version)"
      "${COMPOSE[@]}" exec -T api "${UP_CLI[@]}" run --corpus-dir /app/output --yes
      "${COMPOSE[@]}" restart api
    elif [ "$st" -eq 0 ]; then
      echo "Corpus upgrade: already current — nothing to apply"
    else
      echo "WARN: 'upgrade status' errored (exit $st); leaving corpus as-restored — investigate." >&2
    fi
    ;;
  force)
    echo "Corpus upgrade: forced — applying all pending"
    "${COMPOSE[@]}" exec -T api "${UP_CLI[@]}" run --corpus-dir /app/output --yes
    "${COMPOSE[@]}" restart api
    ;;
  *)
    echo "WARN: unknown RESTORE_UPGRADE_MODE='${RESTORE_UPGRADE_MODE}' — treating as skip." >&2
    ;;
esac
