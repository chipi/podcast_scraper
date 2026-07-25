#!/usr/bin/env bash
# Push a canonical ops event to the self-hosted VictoriaLogs sink (ADR-119).
#
# Tier-1 CI/ops observability (#803 follow-up): GHA runners don't ship their logs
# to the homelab, so discrete ops events (deploy / backup / drill / drift) are
# pushed directly here. Vendor-neutral by ADR-119 — the sink is VictoriaLogs
# :9428 `/insert/jsonline` over the tailnet (no token; tailnet-gated). Swapping the
# sink is a URL change, not a code change.
#
# Envelope (ADR-119): {_time, schema, event_type, app, env, ...fields, _msg}.
# Low-cardinality stream fields = {app, env, event_type}; volatile values (status,
# sha, duration_ms, …) are searchable JSON fields via `--field k=v`.
#
# Env (required): VICTORIALOGS_URL  (e.g. http://homelab:9428)
# Args: --event-type T (required) [--app A] [--env E] [--msg M] [--field k=v ...]
set -euo pipefail

event_type=""
app="podcast_scraper"
env_label="prod"
msg=""
fields=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --event-type) event_type="$2"; shift 2 ;;
    --app)        app="$2";        shift 2 ;;
    --env)        env_label="$2";  shift 2 ;;
    --msg)        msg="$2";        shift 2 ;;
    --field)      fields+=("$2");  shift 2 ;;
    *) echo "emit_ops_event: unknown arg: $1" >&2; exit 2 ;;
  esac
done

: "${VICTORIALOGS_URL:?VICTORIALOGS_URL is required (e.g. http://homelab:9428)}"
[ -n "$event_type" ] || { echo "emit_ops_event: --event-type is required" >&2; exit 2; }

base="${VICTORIALOGS_URL%/}"
# Normalise if a caller passed the full ingest path already.
base="${base%/insert/jsonline}"
ingest="${base}/insert/jsonline?_stream_fields=app,env,event_type&_time_field=_time&_msg_field=_msg"

# Build the JSONL line with python: safe escaping + best-effort scalar coercion so
# duration_ms/attempt land as numbers (queryable with >/< in LogsQL), not strings.
line="$(python3 - "$app" "$env_label" "$event_type" "$msg" ${fields[@]+"${fields[@]}"} <<'PY'
import json
import sys
from datetime import datetime, timezone

app, env, event_type, msg = sys.argv[1:5]
raw_fields = sys.argv[5:]

obj = {
    "_time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "schema": "ops_event/v1",
    "event_type": event_type,
    "app": app,
    "env": env,
}
for kv in raw_fields:
    if "=" not in kv:
        continue
    k, v = kv.split("=", 1)
    if v.lower() in ("true", "false"):
        obj[k] = v.lower() == "true"
    else:
        try:
            obj[k] = int(v)
        except ValueError:
            try:
                obj[k] = float(v)
            except ValueError:
                obj[k] = v
# _msg: explicit summary, else a compact "event_type status" line.
obj["_msg"] = msg or " ".join(x for x in (event_type, str(obj.get("status", ""))) if x).strip()
print(json.dumps(obj))
PY
)"

curl -fsS -H 'Content-Type: application/stream+json' \
  -X POST "$ingest" --data-binary "$line"
echo "emit_ops_event: pushed event_type=$event_type app=$app env=$env_label -> $base"
