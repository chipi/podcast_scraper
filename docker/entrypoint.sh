#!/bin/bash
set -e

# ============================================================================
# Entrypoint script for podcast_scraper Docker container
# Handles service startup, validation, and supervisor integration
# ============================================================================

# Default config path
CONFIG_FILE="${PODCAST_SCRAPER_CONFIG:-/app/config.yaml}"
WORK_DIR="${PODCAST_SCRAPER_WORK_DIR:-/app}"

# Logging helper function with timestamps
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" >&2
}

log "Starting podcast_scraper container entrypoint..."

# Handle --help and --version flags (should work without config file)
if [ "$1" = "--help" ] || [ "$1" = "-h" ] || [ "$1" = "--version" ]; then
    python -m podcast_scraper.service "$1"
    exit 0
fi

# Validate config file exists and is readable
if [ ! -f "$CONFIG_FILE" ]; then
    log "ERROR: Config file not found: $CONFIG_FILE"
    log "Please mount a config file or set PODCAST_SCRAPER_CONFIG environment variable"
    exit 1
fi

if [ ! -r "$CONFIG_FILE" ]; then
    log "ERROR: Config file is not readable: $CONFIG_FILE"
    log "Please check file permissions"
    exit 1
fi

# Basic config file format validation (check if it's JSON or YAML)
# This is a lightweight check - full validation happens in the service
log "Validating config file format..."
if ! python3 -c "
import json
import yaml
import sys
try:
    with open('$CONFIG_FILE', 'r') as f:
        content = f.read()
    # Try JSON first
    try:
        json.loads(content)
        sys.exit(0)
    except json.JSONDecodeError:
        pass
    # Try YAML
    try:
        yaml.safe_load(content)
        sys.exit(0)
    except yaml.YAMLError:
        pass
    sys.exit(1)
except Exception:
    sys.exit(1)
" 2>/dev/null; then
    log "WARNING: Config file format validation failed (may not be valid JSON/YAML)"
    log "Continuing anyway - service will perform full validation on startup"
else
    log "Config file format validation passed"
fi

# Check if supervisor config exists (advanced usage)
# Supervisor can run as non-root if directories have proper permissions
if [ -f "/etc/supervisor/conf.d/podcast_scraper.conf" ]; then
    log "Supervisor config detected - starting supervisor mode"
    log "Supervisor config: /etc/supervisor/conf.d/podcast_scraper.conf"
    log "Service will be managed by supervisor"
    # exec replaces this process with supervisor
    # Supervisor handles its own signal handling
    exec supervisord -c /etc/supervisor/supervisord.conf
else
    log "Starting podcast_scraper service directly (no supervisor)"
    log "Config file: $CONFIG_FILE"
    log "Work directory: $WORK_DIR"
    cd "$WORK_DIR"
    # exec replaces this process with the service
    # Python service handles its own signal handling
    exec python -m podcast_scraper.service
fi
