#!/usr/bin/env bash
# scripts/ops/prod_ops_health_scheduled.sh — the on-box daily run of prod_ops_health.sh (#1877).
#
# Started by infra/systemd/prod-ops-health.timer as the deploy user. GitHub's cron fired the old
# schedule hours late (06:00 UTC slot, runs started 09:52-13:09 on 2026-10-13..17) and disables
# scheduled workflows after 60 days without a commit; the box is already inside the trust boundary
# and pushes the metrics itself, so it schedules itself too.
#
# A RED run (exit 1) or missing prerequisites (exit 5) raises a GlitchTip error event. It is sent
# from inside compose-api-1 with that container's own PODCAST_SENTRY_DSN_API and sentry_sdk, so no
# DSN is copied onto the host. If that container is down the event cannot go out; the run still
# pushes prod_ops_health_aggregate 0 and exits non-zero (`systemctl status prod-ops-health`).
#
# Exit: the health script's own exit code.
set -uo pipefail

HEALTH="$(dirname "${BASH_SOURCE[0]}")/prod_ops_health.sh"

"$HEALTH"
rc=$?
[ "$rc" = 0 ] && exit 0

host="$(hostname)"
if docker exec -i -e RC="$rc" -e HOST="$host" compose-api-1 python - <<'PY'; then
import os

import sentry_sdk

sentry_sdk.init(dsn=os.environ["PODCAST_SENTRY_DSN_API"])
sentry_sdk.capture_message(
    f"prod-ops-health FAILED (exit {os.environ['RC']}) on {os.environ['HOST']}: a between-deploys "
    "check is red. Details: journalctl -u prod-ops-health on the box.",
    level="error",
)
sentry_sdk.flush()
PY
  echo "prod-ops-health: GlitchTip alert sent (exit $rc)" >&2
else
  echo "prod-ops-health: GlitchTip alert NOT sent (compose-api-1 unavailable?)" >&2
fi
exit "$rc"
