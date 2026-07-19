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

TARBALL="${1:?usage: restore_corpus_from_tarball_host.sh <tarball_path>}"
REPO_DIR="${PODCAST_REPO_DIR:-/srv/podcast-scraper}"

cd "$REPO_DIR"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
if [ -d corpus ]; then
  mv corpus "corpus.bak.$STAMP"
fi
# Roll the timestamped backup back on any extract failure — otherwise a corrupt
# tarball / disk-full / bad layout leaves the host with NO corpus/ and the api
# can't boot (review 2026-07-17 H7).
_restore_backup_and_die() {
  echo "ERROR: $1 — restoring prior corpus." >&2
  rm -rf corpus 2>/dev/null || true
  [ -d "corpus.bak.$STAMP" ] && mv "corpus.bak.$STAMP" corpus
  exit 1
}
if ! tar -xzf "$TARBALL" -C "$REPO_DIR"; then
  _restore_backup_and_die "tar extraction failed"
fi
if [ ! -d corpus ]; then
  _restore_backup_and_die "expected top-level corpus/ after prod layout extract under $REPO_DIR"
fi
if [ "${RESTORE_EXTRACT_ONLY:-}" = "1" ]; then
  echo "Restore extract OK under $REPO_DIR/corpus"
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

# Restore is FAITHFUL: the corpus is restored exactly as it was backed up. Corpus
# migrations are DECOUPLED from restore (they used to run here — #862/#1176 — but
# a) the api tolerates an un-migrated corpus [corpus_version compat is a WARNING,
# not a gate, down to MIN_SUPPORTED], and b) that pre-boot ``run`` container saw a
# stale/empty ``/app/output`` after the mv+re-extract swapped the corpus_data bind
# dir, silently no-op'ing every migration. DR restore never needs it.
"${COMPOSE[@]}" up -d --force-recreate api viewer
sleep 8
"${COMPOSE[@]}" exec -T api curl -fsS http://127.0.0.1:8000/api/health | head -c 200
echo
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
