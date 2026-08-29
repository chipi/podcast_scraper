#!/usr/bin/env bash
# scripts/ops/prod_ops_health.sh — daily between-deploys prod health check (#1876).
#
# Runs ON the prod VPS as the deploy user (invoked by prod-ops-health.yml on a schedule, or by
# hand). Deploy-time gates only prove health AT deploy time; this proves it stays true between
# deploys. Born from two same-day incidents (2026-08-29): a green deploy whose gateway silently
# served a two-week-old config, and the realisation that an upstream key expiring between deploys
# surfaces only as silent fail-open degradation mid-corpus.
#
# TRAFFIC-LIGHT MODEL (operator-specified): every check declares its CUSTOMER IMPACT up front.
#   GREEN  (1)   check passed
#   ORANGE (0.5) check failed but customers are not impacted — the app serves; something behind
#                it needs fixing (o11y ingestion, alerting). Run stays green with ::warning.
#   RED    (0)   check failed and the failure reaches customers (pipeline stops ingesting, gate
#                degrades output quality). Run goes red, GlitchTip fires.
# The aggregate is min() across checks — one red makes the app red, one orange makes it orange.
#
# THE O11Y CHECKS PROVE FLOW, NOT UPTIME. "VictoriaLogs answers" is worthless if the podcast
# stopped shipping logs to it — an o11y outage otherwise looks identical to "nothing happened".
# Each check queries for the podcast's OWN recent signals in the store. Endpoints + query shapes
# were live-verified from this box on 2026-08-29; notably the raw homelab ports for VL/GlitchTip
# are ACL-blocked from prod — those two go via their tailnet HTTPS FQDNs.
#
# METRICS pushed to the homelab VictoriaMetrics (the same VM the homelab home page queries via
# its /vm/ proxy, so the page renders one light per app + per-check drill-down):
#   prod_ops_health_check{app="podcast",check="<name>"}  1 / 0.5 / 0
#   prod_ops_health_aggregate{app="podcast"}             min over checks
#   prod_ops_health_last_run_timestamp{app="podcast"}    unix seconds
# The timestamp is load-bearing: "all green but stale > 26h" must render as a DEAD health check,
# never as healthy. `app` is a label (not a name prefix) so other production apps reuse the page
# template by swapping one label value.
#
# Exit: 0 = no RED (oranges allowed, annotated) / 1 = at least one RED / 5 = prereqs missing
set -uo pipefail

REPO_DIR=/srv/podcast-scraper
LITELLM_ENV="$REPO_DIR/infra/litellm/.env"
GATEWAY_BASE="http://127.0.0.1:4001"

# Homelab o11y control plane. VM/VT are reachable on raw ports over the tailnet; VL and
# GlitchTip only via their HTTPS FQDNs (ACL) — verified live 2026-08-29.
HL_IP="$(tailscale ip -4 homelab 2>/dev/null | head -1 || true)"
VM_URL="http://${HL_IP}:8428"
VT_URL="http://${HL_IP}:10428"
VL_URL="https://vlogs.tail6d0ed4.ts.net"
GT_URL="https://glitchtip.tail6d0ed4.ts.net"

declare -A RESULT   # check name -> 1 / 0.5 / 0
ANY_RED=0

run_check() {
  # run_check <name> <impact-on-failure: red|orange> <fn>
  # Never lets one check's failure stop the rest — the summary must always be complete.
  local name="$1" impact="$2" fn="$3"
  if "$fn"; then
    RESULT[$name]=1
    echo "CHECK $name: GREEN"
  elif [ "$impact" = "red" ]; then
    RESULT[$name]=0
    ANY_RED=1
    echo "CHECK $name: RED — customer-impacting, see lines above" >&2
  else
    RESULT[$name]=0.5
    echo "::warning::CHECK $name: ORANGE — not customer-impacting, but fix it (see log)"
    echo "CHECK $name: ORANGE"
  fi
}

# --- gateway (RED: no gateway -> no summarisation -> ingest stops -> customer-visible) --------
# The SHARED smoke (infra/deploy/litellm_smoke.py) — identical code to the fatal deploy gate, so
# this daily check can never pass where the deploy would fail. Config parity + a 1-token
# completion through each podcast-* alias with the gateway's own staged key.
check_gateway() {
  local master_key
  master_key="$(grep -E '^LITELLM_MASTER_KEY=' "$LITELLM_ENV" 2>/dev/null | head -1 | cut -d= -f2-)"
  if [ -z "$master_key" ]; then
    echo "gateway: LITELLM_MASTER_KEY not found in $LITELLM_ENV" >&2
    return 1
  fi
  SMOKE_MASTER_KEY="$master_key" \
  SMOKE_CONFIG="$REPO_DIR/infra/litellm/config.yaml" \
  SMOKE_BASE="$GATEWAY_BASE" \
    python3 "$REPO_DIR/infra/deploy/litellm_smoke.py"
}

# --- o11y: logs (ORANGE) ---------------------------------------------------------------------
# The podcast's OWN emit_event stream must show a pipeline_stage event within 26h. The nightly
# runs 03:00 UTC daily, so 26h always spans one; the filter is `_msg:"pipeline_stage"` because
# a field filter (`event_type:`) live-returns 0 against real pushed data (see
# src/podcast_obs/sources/victoria.py). Quiet ingestion is indistinguishable from a dead nightly
# AND from broken shipping — either way someone must look.
check_o11y_logs() {
  local n
  n=$(curl -fsS --max-time 10 "$VL_URL/select/logsql/query" \
        --data-urlencode 'query=_msg:"pipeline_stage" AND _time:26h' \
        --data-urlencode "limit=1" | wc -c)
  if [ "${n:-0}" -gt 2 ]; then return 0; fi
  echo "o11y_logs: no pipeline_stage events in VictoriaLogs for 26h — nightly dead OR log shipping broken" >&2
  return 1
}

# --- o11y: metrics (ORANGE) ------------------------------------------------------------------
# The api's scrape job must have live series in the homelab VM. This proves box-alloy ->
# remote_write -> VM end to end, not merely that VM answers.
check_o11y_metrics() {
  local n
  n=$(curl -fsS --max-time 10 "$VM_URL/api/v1/query" \
        --data-urlencode 'query=count({job="api"})' \
      | python3 -c 'import json,sys; r=json.load(sys.stdin)["data"]["result"]; print(int(float(r[0]["value"][1])) if r else 0)' 2>/dev/null)
  if [ "${n:-0}" -gt 0 ]; then return 0; fi
  echo "o11y_metrics: no live series for job=\"api\" in VictoriaMetrics — scrape or remote_write broken" >&2
  return 1
}

# --- o11y: traces (ORANGE) -------------------------------------------------------------------
# The podcast services must be present in VictoriaTraces' service list (direct OTLP, not via
# Alloy). Presence is retention-window bounded, so this proves recent spans, not history.
check_o11y_traces() {
  if curl -fsS --max-time 10 "$VT_URL/select/jaeger/api/services" \
      | python3 -c 'import json,sys; svcs=json.load(sys.stdin).get("data") or []; sys.exit(0 if any(s in svcs for s in ("podcast-api","podcast-pipeline","pipeline")) else 1)'; then
    return 0
  fi
  echo "o11y_traces: no podcast service in VictoriaTraces — OTLP export broken or VT unreachable" >&2
  return 1
}

# --- o11y: glitchtip (ORANGE) ----------------------------------------------------------------
# Error-tracking reachability on the DSN host prod actually uses (tmpfs-staged DSNs point at the
# tailnet FQDN). Deliberately NO test event is injected — reachability of the ingest host is the
# agreed check; a synthetic-event round-trip can be added later if this ever proves too weak.
check_o11y_glitchtip() {
  local code
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$GT_URL/")
  case "$code" in
    2*|3*) return 0 ;;
  esac
  echo "o11y_glitchtip: $GT_URL returned HTTP ${code:-none} — error tracking unreachable from prod" >&2
  return 1
}

# --- run all checks --------------------------------------------------------------------------
echo "[$(date -u +%FT%TZ)] prod-ops-health starting (app=podcast)"
if [ -z "$HL_IP" ]; then
  echo "WARN: cannot resolve homelab tailnet IP — o11y checks will fail as orange" >&2
fi
run_check gateway         red    check_gateway
run_check o11y_logs       orange check_o11y_logs
run_check o11y_metrics    orange check_o11y_metrics
run_check o11y_traces     orange check_o11y_traces
run_check o11y_glitchtip  orange check_o11y_glitchtip

# --- aggregate + metrics push ----------------------------------------------------------------
AGG=1
for name in "${!RESULT[@]}"; do
  # min() in shell over 1/0.5/0
  case "${RESULT[$name]}" in
    0)   AGG=0 ;;
    0.5) [ "$AGG" = 1 ] && AGG=0.5 ;;
  esac
done

# Push NON-fatally: a homelab hiccup must not flip a healthy prod run to red — but it must be
# visible, so staleness of last_run_timestamp is the page's guard against silent non-running.
if [ -n "$HL_IP" ]; then
  {
    for name in "${!RESULT[@]}"; do
      echo "prod_ops_health_check{app=\"podcast\",check=\"$name\"} ${RESULT[$name]}"
    done
    echo "prod_ops_health_aggregate{app=\"podcast\"} $AGG"
    echo "prod_ops_health_last_run_timestamp{app=\"podcast\"} $(date +%s)"
  } | curl -fsS --max-time 10 --data-binary @- "$VM_URL/api/v1/import/prometheus" \
    && echo "[$(date -u +%FT%TZ)] metrics pushed to homelab VM" \
    || echo "WARN: metrics push failed — home page will show this run as stale" >&2
fi

# --- summary ---------------------------------------------------------------------------------
echo "----------------------------------------"
for name in "${!RESULT[@]}"; do
  case "${RESULT[$name]}" in
    1)   echo "  $name: GREEN" ;;
    0.5) echo "  $name: ORANGE" ;;
    0)   echo "  $name: RED" ;;
  esac
done
echo "  aggregate: $AGG"
if [ "$ANY_RED" = 1 ]; then
  echo "prod-ops-health: RED — customer-impacting failure, see above" >&2
  exit 1
fi
echo "prod-ops-health: no red (oranges, if any, are annotated)"
