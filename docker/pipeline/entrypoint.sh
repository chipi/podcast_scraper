#!/bin/bash
set -e

# ============================================================================
# Pipeline image entrypoint
#
# Pipeline runs as a one-shot container spawned by the API job factory
# (``PODCAST_PIPELINE_EXEC_MODE=docker``):
#
#   docker compose run --rm pipeline python -m podcast_scraper.cli \
#       --output-dir /app/output --profile <name> \
#       --config /app/output/viewer_operator.yaml \
#       --feeds-spec /app/output/feeds.spec.yaml
#
# The CLI reads its configuration from the operator-driven flags above,
# so there is no longer a static ``/app/config.yaml`` mount — the older
# single-config path was retired with the move to the operator-config API.
#
# The supervisor fallback below stays for legacy ``compose up pipeline``
# launches (no CMD given), where the image runs as a long-lived service
# instead of a one-shot CLI. Production deployments that want that mode
# bring their own supervisor config in ``/etc/supervisor/conf.d/``.
# ============================================================================

WORK_DIR="${PODCAST_SCRAPER_WORK_DIR:-/app}"

# Logging helper function with timestamps
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" >&2
}

log "Starting podcast_scraper container entrypoint..."

# If a CMD was passed (e.g. the API job factory's ``python -m
# podcast_scraper.cli ...`` invocation), honour it. ``--help`` /
# ``--version`` fall through here too; the CLI handles them itself.
if [ "$#" -gt 0 ]; then
    log "Executing CMD: $*"
    cd "$WORK_DIR"
    exec "$@"
fi

# No CMD provided — fall through to a long-running service. Used by
# legacy ``compose up pipeline`` style launches; the API-driven flow
# never reaches this branch.
if [ -f "/etc/supervisor/conf.d/podcast_scraper.conf" ]; then
    log "Supervisor config detected — starting supervisord"
    exec supervisord -c /etc/supervisor/supervisord.conf
fi

log "No supervisor config — starting podcast_scraper service directly"
cd "$WORK_DIR"
exec python -m podcast_scraper.service
