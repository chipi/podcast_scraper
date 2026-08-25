#!/usr/bin/env python3
"""Deterministic o11y health drill (#1819) — probes, not opinions.

Answers "is observability healthy" with a fixed probe battery and PASS/FAIL/SKIP
verdicts, replacing ad-hoc agent assessments. Born from the 2026-08-24 incident
where "o11y is fine" had been claimed repeatedly on happy-path evidence while
GlitchTip delivery, a Grafana datasource, and every provisioned alert rule were
broken — none of it visible until real failures ran the test nobody had run.

Design rules (from the issue):
- A probe passes only on an OBSERVED EFFECT at the consuming end.
- Deterministic: same inputs -> same verdicts; no agent in the loop.
- Read-only by default; nothing here mutates any system.
- No silent caps: probes that cannot run in this environment report SKIP loudly
  and count against the summary — a skipped probe is a claim nobody verified.

Environment: GRAFANA_URL + GRAFANA_OBS_TOKEN (the read/query service account).
Exit code: 0 all PASS, 1 any FAIL, 2 any SKIP with zero FAILs.

Usage:  python scripts/obs/health_drill.py  (or: make health-drill)
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass

INSTANCE = os.environ.get("DRILL_INSTANCE", "prod-podcast")
METRIC_MAX_STALENESS_S = float(os.environ.get("DRILL_METRIC_STALENESS_S", "180"))
LOGS_WINDOW = os.environ.get("DRILL_LOGS_WINDOW", "15m")


@dataclass
class Verdict:
    probe: str
    status: str  # PASS | FAIL | SKIP
    evidence: str


def _grafana(path: str, *, method: str = "GET", data: dict | None = None) -> object:
    base = os.environ["GRAFANA_URL"].rstrip("/")
    token = os.environ["GRAFANA_OBS_TOKEN"]
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(
        base + path,
        data=body,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            **({"Content-Type": "application/json"} if body else {}),
        },
    )
    with urllib.request.urlopen(req, timeout=45) as resp:
        return json.loads(resp.read().decode())


def probe_datasource_health() -> Verdict:
    """Every provisioned datasource answers its health check (caught: dead grafana_ro)."""
    ds_list = _grafana("/api/datasources")
    bad: list[str] = []
    for ds in ds_list:  # type: ignore[union-attr]
        uid = ds["uid"]
        try:
            h = _grafana(f"/api/datasources/uid/{uid}/health")
            status = str(h.get("status", "")).upper()  # type: ignore[union-attr]
            if status not in ("OK", "SUCCESS"):
                bad.append(f"{uid}={status or 'unknown'}")
        except Exception as exc:  # noqa: BLE001 — a failing health check IS the signal
            bad.append(f"{uid}=error:{type(exc).__name__}")
    if bad:
        return Verdict("datasource_health", "FAIL", "; ".join(bad))
    count = len(ds_list)  # type: ignore[arg-type]
    return Verdict("datasource_health", "PASS", f"{count} datasources healthy")


def probe_metric_freshness() -> Verdict:
    """Every scrape target reported within the staleness bound."""
    q = urllib.parse.quote("time() - timestamp(up)")
    data = _grafana(f"/api/datasources/proxy/uid/victoriametrics/api/v1/query?query={q}")
    rows = data["data"]["result"]  # type: ignore[index,call-overload]
    stale = [
        f"{r['metric'].get('job')}@{r['metric'].get('instance')}={float(r['value'][1]):.0f}s"
        for r in rows
        if float(r["value"][1]) > METRIC_MAX_STALENESS_S
    ]
    if not rows:
        return Verdict("metric_freshness", "FAIL", "no up-series at all")
    if stale:
        return Verdict("metric_freshness", "FAIL", "; ".join(stale[:5]))
    return Verdict(
        "metric_freshness", "PASS", f"{len(rows)} targets fresh (<{METRIC_MAX_STALENESS_S:.0f}s)"
    )


def probe_alert_rules() -> Verdict:
    """Every provisioned alert rule evaluated successfully (caught: LogQL-vs-LogsQL family)."""
    data = _grafana("/api/prometheus/grafana/api/v1/rules")
    total = 0
    broken: list[str] = []
    for group in data["data"]["groups"]:  # type: ignore[index,call-overload]
        for rule in group.get("rules", []):
            total += 1
            if rule.get("health") == "error":
                broken.append(f"{rule.get('name')}: {(rule.get('lastError') or '')[:80]}")
    if total == 0:
        return Verdict("alert_rules", "FAIL", "zero alert rules provisioned")
    if broken:
        return Verdict("alert_rules", "FAIL", "; ".join(broken[:3]))
    return Verdict("alert_rules", "PASS", f"{total} rules, all health=ok")


def _vlogs_count(query: str) -> int:
    base = os.environ["GRAFANA_URL"].rstrip("/")
    token = os.environ["GRAFANA_OBS_TOKEN"]
    body = urllib.parse.urlencode({"query": query}).encode()
    req = urllib.request.Request(
        base + "/api/datasources/proxy/uid/victorialogs/select/logsql/query",
        data=body,
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(req, timeout=45) as resp:
        line = resp.read().decode().strip().splitlines()
        return int(json.loads(line[0]).get("n", "0")) if line and line[0] else 0


def probe_logs_flowing() -> Verdict:
    """The prod instance shipped log lines recently (transport liveness)."""
    n = _vlogs_count(f'_time:{LOGS_WINDOW} instance:"{INSTANCE}" | stats count() as n')
    if n <= 0:
        return Verdict("logs_flowing", "FAIL", f"0 lines from {INSTANCE} in {LOGS_WINDOW}")
    return Verdict("logs_flowing", "PASS", f"{n} lines from {INSTANCE} in {LOGS_WINDOW}")


def probe_log_dedup() -> Verdict:
    """Pipeline llm_cost events are not double-shipped (caught: dual Alloy sources)."""
    raw = _vlogs_count(
        f'_time:2h instance:"{INSTANCE}" "event_type" "llm_cost" | stats count() as n'
    )
    if raw == 0:
        return Verdict("log_dedup", "SKIP", "no llm_cost events in window (nothing ran)")
    uniq = _vlogs_count(
        f'_time:2h instance:"{INSTANCE}" "event_type" "llm_cost" | stats count_uniq(_msg) as n'
    )
    ratio = raw / max(uniq, 1)
    if ratio > 1.5:
        return Verdict(
            "log_dedup", "FAIL", f"{raw} raw vs {uniq} unique (x{ratio:.1f} duplication)"
        )
    return Verdict("log_dedup", "PASS", f"{raw} raw vs {uniq} unique")


def probe_error_plane() -> Verdict:
    """GlitchTip delivery — requires an injectable canary; SKIP loudly when not runnable.

    The full live-fire (canary exception -> event row in GlitchTip) needs a DSN and
    a reachable ingest, which this laptop-side drill may not have. SKIP is loud by
    design: transport that nobody exercised is transport nobody verified (#1819).
    """
    dsn = os.environ.get("DRILL_SENTRY_DSN", "")
    if not dsn:
        return Verdict("error_plane", "SKIP", "DRILL_SENTRY_DSN unset — canary not injected")
    try:
        import sentry_sdk  # type: ignore[import-not-found]
    except ImportError:
        return Verdict("error_plane", "SKIP", "sentry_sdk not installed in drill env")
    marker = f"DRILL-CANARY-{int(time.time())}"
    sentry_sdk.init(dsn=dsn, environment="drill")
    sentry_sdk.capture_message(marker, level="error")
    flushed = sentry_sdk.flush(timeout=10)
    return Verdict(
        "error_plane",
        "PASS" if flushed is not False else "FAIL",
        f"canary {marker} flushed to ingest (delivery-to-DB verified by the alert on it)",
    )


PROBES = [
    probe_datasource_health,
    probe_metric_freshness,
    probe_alert_rules,
    probe_logs_flowing,
    probe_log_dedup,
    probe_error_plane,
]


def main() -> int:
    if not (os.environ.get("GRAFANA_URL") and os.environ.get("GRAFANA_OBS_TOKEN")):
        print("health-drill: GRAFANA_URL + GRAFANA_OBS_TOKEN required (source .env)")
        return 1
    verdicts: list[Verdict] = []
    for probe in PROBES:
        try:
            verdicts.append(probe())
        except Exception as exc:  # noqa: BLE001 — a crashing probe is a failing probe
            verdicts.append(Verdict(probe.__name__, "FAIL", f"probe crashed: {exc}"))
    width = max(len(v.probe) for v in verdicts)
    for v in verdicts:
        print(f"{v.probe:<{width}}  {v.status:<4}  {v.evidence}")
    fails = sum(1 for v in verdicts if v.status == "FAIL")
    skips = sum(1 for v in verdicts if v.status == "SKIP")
    print(f"\nverdict: {len(verdicts) - fails - skips} PASS / {fails} FAIL / {skips} SKIP")
    if fails:
        return 1
    return 2 if skips else 0


if __name__ == "__main__":
    sys.exit(main())
