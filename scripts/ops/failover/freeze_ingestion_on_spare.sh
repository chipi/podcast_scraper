#!/usr/bin/env bash
# Freeze scheduled_jobs on the prod failover spare. Called via SSH from
# .github/workflows/prod-failover-stand-up.yml after corpus restore and
# before validation.
#
# Why: the in-process APScheduler (src/podcast_scraper/server/scheduler.py)
# reads ``corpus/viewer_operator.yaml`` at api startup and only enables jobs
# present under ``scheduled_jobs``. Strip the key on the spare so that when
# the operator runs the manual cutover (PROD_RUNBOOK § failover), the spare
# starts with zero scheduled jobs and cannot dual-write against prod's
# corpus (RFC-083 §6 dual-writer guard).
#
# The api is NOT restarted here. The spare is not serving traffic during
# preprod stand-up validation; the api will re-read the (now key-less)
# yaml on its next natural restart — manual cutover, host reboot, or full
# compose redeploy. Restarting it here only couples the freeze step to the
# full prod compose overlay stack (env-file, PODCAST_CORPUS_HOST_PATH, all
# three overlay files) for zero validation benefit.
#
# Idempotent: a second invocation finds no scheduled_jobs key and exits 0.

set -euo pipefail

REPO_DIR="${PODCAST_REPO_DIR:-/srv/podcast-scraper}"
OPERATOR_YAML="${REPO_DIR}/corpus/viewer_operator.yaml"

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

# Read-back verification — confirm the key is gone from the on-disk yaml.
# The spare api will pick this up on its next start (cutover time, manual).
python3 - "$OPERATOR_YAML" <<'PY'
import sys
import yaml

path = sys.argv[1]
with open(path) as fh:
    data = yaml.safe_load(fh) or {}
if "scheduled_jobs" in data:
    print(f"freeze-ingestion: ERROR scheduled_jobs still present in {path}", file=sys.stderr)
    sys.exit(1)
print(f"freeze-ingestion: verified scheduled_jobs absent from {path}")
PY
