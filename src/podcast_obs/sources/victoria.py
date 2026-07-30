"""Current-stack sources: self-hosted Victoria* on the homelab box (o11y re-alignment gap #1/#2/#4).

Three backends the legacy sources don't speak:

- **VictoriaLogs** (LogsQL, ``/select/logsql/query``) — the ``emit_event`` stream (``llm_cost`` /
  ``pipeline_stage`` / ``search_query``) + container logs; a streaming newline-delimited-JSON query
  API. The self-hosted logs backend for this deployment (there is no Grafana-Cloud Loki).
- **VictoriaMetrics** (PromQL, ``/api/v1/query``) — RED / resource metrics. No source existed.
- **VictoriaTraces** (Jaeger API, ``/select/jaeger/api/*``) — general request/span traces for the
  API + pipeline services (vs Langfuse, which only sees LLM calls).

Auth is an optional bearer (``victoria_token``) if the tailnet endpoints are fronted by one.
All functions return the shared ``ok``/``err`` envelope and degrade gracefully when unconfigured.
"""

from __future__ import annotations

import json
import time
from typing import Optional

from .._http import get_json, get_ndjson
from ..config import TargetConfig
from ..result import err, ok

_LOGS = "victorialogs.events"
_LOGSRAW = "victorialogs.logs"
_METRICS = "victoriametrics.query"
_TRACES = "victoriatraces.traces"
_TRACE = "victoriatraces.trace"

_ERROR_RE = "(?i)(error|critical|exception|traceback|fatal)"
_WINDOW_MULT = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def _headers(target: TargetConfig) -> dict:
    return {"Authorization": f"Bearer {target.victoria_token}"} if target.victoria_token else {}


def _window_seconds(window: str, default: int = 3600) -> int:
    try:
        return int(window[:-1]) * _WINDOW_MULT.get(window[-1], 0) or default
    except (ValueError, IndexError):
        return default


def _q(value: str) -> str:
    """LogsQL quoted phrase — escape backslashes then double-quotes."""
    escaped = value.replace(chr(92), chr(92) * 2).replace(chr(34), chr(92) + chr(34))
    return f'"{escaped}"'


# --- VictoriaLogs (LogsQL) ---------------------------------------------------------


def events(
    target: TargetConfig,
    event_type: str,
    *,
    surface: Optional[str] = None,
    run_id: Optional[str] = None,
    episode_id: Optional[str] = None,
    window: str = "1h",
    limit: int = 50,
) -> dict:
    """Query the canonical ``emit_event`` stream by ``event_type`` (llm_cost / pipeline_stage / …).

    Optional ``surface`` (api/pipeline) matches either the ``surface`` or ``component`` label; and
    ``run_id`` / ``episode_id`` scope to one run/episode for correlation.
    """
    base = target.victorialogs_url
    if not base:
        return err(_LOGS, "victorialogs_url not configured", configured=False)
    # emit_event ships with `_msg_field: event_type` (dev_push.py), so VictoriaLogs stores the
    # event type as the built-in `_msg` message field and keeps NO `event_type` field to filter on.
    # In prod (Alloy tails raw stdout) `_msg` is the whole JSON line, which still contains the type
    # as a phrase — so `_msg:"pipeline_stage"` matches both shipping paths. Live-verified: an
    # `event_type:` filter returns 0 against real pushed data.
    filters = [f"_msg:{_q(event_type)}", f"_time:{window}"]
    if surface:
        filters.append(f"(surface:{_q(surface)} OR component:{_q(surface)})")
    if run_id:
        filters.append(f"run_id:{_q(run_id)}")
    if episode_id:
        filters.append(f"episode_id:{_q(episode_id)}")
    query = " AND ".join(filters)
    url = f"{base.rstrip('/')}/select/logsql/query"
    try:
        rows = get_ndjson(
            url,
            params={"query": query, "limit": max(limit, 1)},
            headers=_headers(target),
            timeout=target.timeout,
        )
    except Exception as exc:  # noqa: BLE001
        return err(_LOGS, f"victorialogs query failed: {exc}")
    return ok(
        _LOGS,
        {
            "event_type": event_type,
            "surface": surface,
            "window": window,
            "count": len(rows),
            "events": rows,
        },
    )


def recent_logs(
    target: TargetConfig,
    *,
    surface: Optional[str] = None,
    level: str = "error",
    window: str = "1h",
    limit: int = 50,
    contains: Optional[str] = None,
) -> dict:
    """Recent container log lines (error-ish by default), optionally scoped to a surface."""
    base = target.victorialogs_url
    if not base:
        return err(_LOGSRAW, "victorialogs_url not configured", configured=False)
    filters = [f"_time:{window}"]
    if surface:
        filters.append(f"(surface:{_q(surface)} OR component:{_q(surface)})")
    if level and level.lower() == "error":
        filters.append(f"_msg:~{_q(_ERROR_RE)}")
    if contains:
        filters.append(f"_msg:{_q(contains)}")
    query = " AND ".join(filters)
    url = f"{base.rstrip('/')}/select/logsql/query"
    try:
        rows = get_ndjson(
            url,
            params={"query": query, "limit": max(limit, 1)},
            headers=_headers(target),
            timeout=target.timeout,
        )
    except Exception as exc:  # noqa: BLE001
        return err(_LOGSRAW, f"victorialogs query failed: {exc}")
    return ok(
        _LOGSRAW,
        {"surface": surface, "level": level, "window": window, "count": len(rows), "lines": rows},
    )


# --- VictoriaMetrics (PromQL) ------------------------------------------------------


def metrics_instant(target: TargetConfig, query: str) -> dict:
    """Run one PromQL instant query and return the vector result."""
    base = target.victoriametrics_url
    if not base:
        return err(_METRICS, "victoriametrics_url not configured", configured=False)
    url = f"{base.rstrip('/')}/api/v1/query"
    try:
        data = get_json(
            url, params={"query": query}, headers=_headers(target), timeout=target.timeout
        )
    except Exception as exc:  # noqa: BLE001
        return err(_METRICS, f"victoriametrics query failed: {exc}")
    series = []
    try:
        for item in data["data"]["result"]:  # type: ignore[index]
            series.append({"metric": item.get("metric", {}), "value": item.get("value")})
    except (KeyError, TypeError):
        pass
    return ok(_METRICS, {"query": query, "series": series})


def red_metrics(target: TargetConfig, job: str, *, window: str = "5m") -> dict:
    """RED-ish snapshot for a surface: request rate, 5xx error rate, p95 latency.

    Filtered by the Prometheus ``job`` label (live-verified: the instrumentator series
    ``http_requests_total`` / ``http_request_duration_seconds_bucket`` carry ``job="api"`` etc.,
    not a ``service`` label). A surface with no HTTP metrics (the pipeline subprocess) returns empty
    series — honest, not an error. Each sub-query degrades independently.
    """
    if not target.victoriametrics_url:
        return err(_METRICS, "victoriametrics_url not configured", configured=False)
    j = job.replace(chr(92), "").replace('"', "")
    q_rate = f'sum(rate(http_requests_total{{job="{j}"}}[{window}]))'
    q_err = f'sum(rate(http_requests_total{{job="{j}",status=~"5.."}}[{window}]))'
    q_p95 = (
        "histogram_quantile(0.95, sum(rate("
        f'http_request_duration_seconds_bucket{{job="{j}"}}[{window}])) by (le))'
    )
    return ok(
        _METRICS,
        {
            "job": j,
            "window": window,
            "request_rate": metrics_instant(target, q_rate).get("data"),
            "error_rate_5xx": metrics_instant(target, q_err).get("data"),
            "latency_p95_s": metrics_instant(target, q_p95).get("data"),
        },
    )


# --- VictoriaTraces (Jaeger API) ---------------------------------------------------


def traces_recent(
    target: TargetConfig, service: str, *, window: str = "1h", limit: int = 10
) -> dict:
    """Recent traces for a Jaeger ``service.name`` (e.g. podcast-api / podcast-pipeline)."""
    base = target.victoriatraces_url
    if not base:
        return err(_TRACES, "victoriatraces_url not configured", configured=False)
    end_us = int(time.time() * 1_000_000)
    start_us = end_us - _window_seconds(window) * 1_000_000
    url = f"{base.rstrip('/')}/select/jaeger/api/traces"
    params = {"service": service, "limit": max(limit, 1), "start": start_us, "end": end_us}
    try:
        data = get_json(url, params=params, headers=_headers(target), timeout=target.timeout)
    except Exception as exc:  # noqa: BLE001
        return err(_TRACES, f"victoriatraces query failed: {exc}")
    traces = data.get("data") if isinstance(data, dict) else None
    return ok(
        _TRACES,
        {"service": service, "window": window, "count": len(traces or []), "traces": traces or []},
    )


def traces_by_run(
    target: TargetConfig,
    run_id: str,
    *,
    service: str = "pipeline",
    window: str = "24h",
    limit: int = 20,
) -> dict:
    """Traces for a ``run_id`` — filters the Jaeger API by the ``run_id`` span tag.

    The pipeline's per-episode ``episode.process`` root span stamps ``run_id`` / ``episode_id`` /
    ``feed_id`` as attributes (``otel_init.episode_span``), so this is the run→trace pivot on the
    current stack: an agent investigating a run gets its episode spans without needing Langfuse.
    """
    base = target.victoriatraces_url
    if not base:
        return err(_TRACES, "victoriatraces_url not configured", configured=False)
    end_us = int(time.time() * 1_000_000)
    start_us = end_us - _window_seconds(window) * 1_000_000
    url = f"{base.rstrip('/')}/select/jaeger/api/traces"
    params = {
        "service": service,
        "limit": max(limit, 1),
        "start": start_us,
        "end": end_us,
        "tags": json.dumps({"run_id": run_id}),
    }
    try:
        data = get_json(url, params=params, headers=_headers(target), timeout=target.timeout)
    except Exception as exc:  # noqa: BLE001
        return err(_TRACES, f"victoriatraces query failed: {exc}")
    traces = data.get("data") if isinstance(data, dict) else None
    return ok(
        _TRACES,
        {"run_id": run_id, "service": service, "count": len(traces or []), "traces": traces or []},
    )


def trace_by_id(target: TargetConfig, trace_id: str) -> dict:
    """Full span tree for one ``trace_id`` — the request-level drill-down."""
    base = target.victoriatraces_url
    if not base:
        return err(_TRACE, "victoriatraces_url not configured", configured=False)
    safe = trace_id.replace("/", "").strip()
    url = f"{base.rstrip('/')}/select/jaeger/api/traces/{safe}"
    try:
        data = get_json(url, headers=_headers(target), timeout=target.timeout)
    except Exception as exc:  # noqa: BLE001
        return err(_TRACE, f"victoriatraces trace fetch failed: {exc}")
    spans = data.get("data") if isinstance(data, dict) else None
    return ok(_TRACE, {"trace_id": trace_id, "trace": spans})
