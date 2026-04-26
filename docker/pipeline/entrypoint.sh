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

# Special-case ``--help`` / ``-h`` / ``--version``: bash's ``exec``
# builtin treats ``--help`` as its own flag, so a bare
# ``docker run podcast-scraper:llm --help`` would print exec's help
# text and exit non-zero. Route these to the CLI explicitly. The
# docker-build-fast workflow asserts ``docker run … --help`` succeeds.
case "$1" in
    --help|-h|--version)
        cd "$WORK_DIR"
        exec python -m podcast_scraper.cli "$1"
        ;;
esac

# If a CMD was passed (e.g. the API job factory's ``python -m
# podcast_scraper.cli ...`` invocation), honour it.
if [ "$#" -gt 0 ]; then
    log "Executing CMD: $*"
    cd "$WORK_DIR"
    exec "$@"
fi

# No CMD provided — fall through to a long-running service. Used by
# legacy ``compose up pipeline`` / ``docker run podcast-scraper`` style
# launches; the API-driven flow never reaches this branch.
if [ -f "/etc/supervisor/conf.d/podcast_scraper.conf" ]; then
    log "Supervisor config detected — starting supervisord"
    exec supervisord -c /etc/supervisor/supervisord.conf
fi

# Legacy single-container service fallback reads ``$CONFIG_FILE``
# (default ``/app/config.yaml``). Validate it exists with a clear
# operator-friendly error before exec-ing the service — the underlying
# Python ValueError is less actionable for someone trying out
# ``docker run`` for the first time.
CONFIG_FILE="${PODCAST_SCRAPER_CONFIG:-/app/config.yaml}"
if [ ! -f "$CONFIG_FILE" ]; then
    log "ERROR: Config file not found: $CONFIG_FILE"
    log "Mount one with -v <host-config>:/app/config.yaml or set PODCAST_SCRAPER_CONFIG."
    log "For the recommended end-to-end stack (no per-image config) see"
    log "docs/guides/DOCKER_COMPOSE_GUIDE.md."
    exit 1
fi

log "Starting podcast_scraper service directly (config: $CONFIG_FILE)"
cd "$WORK_DIR"
exec python -m podcast_scraper.service
