#!/usr/bin/env bash
# Freeze scheduled_jobs on the prod failover spare. Called via SSH from
# .github/workflows/prod-failover-stand-up.yml after corpus restore and
# before validation.
#
# Why: the in-process APScheduler (src/podcast_scraper/server/scheduler.py)
# starts only when ``corpus/viewer_operator.yaml`` has any enabled
# ``scheduled_jobs`` entry. Strip the key so the spare cannot fire ingestion
# jobs while prod still owns the corpus (dual-writer guard, RFC-083 §6).
#
# Idempotent: a second invocation finds no scheduled_jobs key and exits 0.

set -euo pipefail

REPO_DIR="${PODCAST_REPO_DIR:-/srv/podcast-scraper}"
OPERATOR_YAML="${REPO_DIR}/corpus/viewer_operator.yaml"
COMPOSE_FILE="${REPO_DIR}/compose/docker-compose.prod.yml"

if [ ! -f "$OPERATOR_YAML" ]; then
  echo "freeze-ingestion: $OPERATOR_YAML missing — scheduler cannot start without it; nothing to freeze."
  exit 0
fi

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
cp -p "$OPERATOR_YAML" "${OPERATOR_YAML}.preserved-by-failover.${STAMP}"

python3 - "$OPERATOR_YAML" <<'PY'
import sys
import yaml

path = sys.argv[1]
with open(path) as fh:
    data = yaml.safe_load(fh) or {}
removed = data.pop("scheduled_jobs", None)
with open(path, "w") as fh:
    yaml.safe_dump(data, fh, sort_keys=False)
count = len(removed) if isinstance(removed, list) else (1 if removed else 0)
print(f"freeze-ingestion: removed {count} scheduled_jobs entries from {path}")
PY

if [ -f "$COMPOSE_FILE" ]; then
  docker compose -f "$COMPOSE_FILE" restart api 2>&1 | tail -10
else
  echo "freeze-ingestion: $COMPOSE_FILE missing — skipping api restart (deploy step should have created it)."
fi
