"""Loki-backed probes (Grafana Cloud Loki query API, HTTP Basic auth).

- ``cost_today`` — sums the #804 ``estimated_cost_usd`` of ``event_type="llm_cost"`` log events
  over the last 24h (the ``{app="podcast_scraper", env=…}`` stream).
- ``recent_logs`` — raw container logs (error-ish by default). This is the signal Sentry misses:
  stderr tracebacks from pipeline subprocesses, ``ERROR``/``WARNING`` lines, crash output — things
  the Sentry SDK never wrapped. Filterable by service / level / free text / window.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Optional

from .._http import get_json
from ..config import TargetConfig
from ..result import err, ok

_COST = "loki.cost"
_COST_RUN = "loki.cost_run"
_LOGS = "loki.logs"
_ERROR_FILTER = r'|~ "(?i)(error|critical|exception|traceback|fatal)"'
_PATH_SUFFIXES = ("/loki/api/v1/push", "/loki/api/v1/query_range", "/loki/api/v1/query")
_NOT_CONFIGURED = "loki not configured (loki_url + loki_user + loki_token [logs:read])"


def _query_base(loki_url: Optional[str]) -> Optional[str]:
    """Normalise a push/query URL to the Loki API base (strip the known endpoint suffix)."""
    if not loki_url:
        return None
    base = loki_url.rstrip("/")
    for suffix in _PATH_SUFFIXES:
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base.rstrip("/") or None


def _creds(target: TargetConfig) -> Optional[tuple[str, str, str]]:
    # Loki's data endpoint wants a Cloud access-policy token (logs:read), distinct from the
    # Grafana service-account token used for the alerting API. Fall back to grafana_token for
    # self-hosted setups where one token serves both.
    base = _query_base(target.loki_url)
    token = target.loki_token or target.grafana_token
    if not base or not target.loki_user or not token:
        return None
    return base, target.loki_user, token


def _parse_window_seconds(window: str, default: int = 3600) -> int:
    try:
        value, unit = int(window[:-1]), window[-1]
    except (ValueError, IndexError):
        return default
    mult = {"s": 1, "m": 60, "h": 3600, "d": 86400}.get(unit)
    return value * mult if mult else default


def _ns_to_iso(ts_ns: str) -> Optional[str]:
    try:
        return datetime.fromtimestamp(int(ts_ns) / 1e9, tz=timezone.utc).isoformat()
    except (ValueError, TypeError, OSError):
        return None


def cost_today(target: TargetConfig) -> dict:
    """Sum of estimated LLM spend over the last 24h for the target's ``env``."""
    creds = _creds(target)
    if not creds:
        return err(_COST, _NOT_CONFIGURED, configured=False)
    base, user, token = creds
    selector = f'{{app="podcast_scraper", env="{target.env_label}"}}'
    query = (
        f"sum(sum_over_time({selector} | json "
        f'| event_type="llm_cost" | unwrap estimated_cost_usd [24h]))'
    )
    url = f"{base}/loki/api/v1/query"
    try:
        data = get_json(url, params={"query": query}, auth=(user, token), timeout=target.timeout)
    except Exception as exc:  # noqa: BLE001
        return err(_COST, f"loki query failed: {exc}")
    return ok(
        _COST,
        {
            "window": "24h",
            "environment": target.env_label,
            "estimated_cost_usd": _vector_value(data),
        },
    )


def _vector_value(data: object) -> Optional[float]:
    try:
        result = data["data"]["result"]  # type: ignore[index]
    except (KeyError, TypeError):
        return None
    if not result:
        return 0.0  # no cost events in window is a real zero, not an error
    try:
        return round(float(result[0]["value"][1]), 6)
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def recent_logs(
    target: TargetConfig,
    *,
    level: str = "error",
    service: Optional[str] = None,
    window: str = "1h",
    limit: int = 50,
    contains: Optional[str] = None,
) -> dict:
    """Recent container log lines (error-ish by default) — what Sentry didn't capture."""
    creds = _creds(target)
    if not creds:
        return err(_LOGS, _NOT_CONFIGURED, configured=False)
    base, user, token = creds
    labels = ['app="podcast_scraper"', f'env="{target.env_label}"']
    if service:
        labels.append(f'service="{service}"')
    pipeline = _ERROR_FILTER if level and level.lower() == "error" else ""
    if contains:
        # Escape backslashes BEFORE quotes so a trailing "\" can't break out of the LogQL string.
        escaped = contains.replace(chr(92), chr(92) + chr(92)).replace(chr(34), chr(92) + chr(34))
        pipeline += f' |= "{escaped}"'
    query = "{" + ", ".join(labels) + "}" + pipeline
    end_ns = time.time_ns()
    start_ns = end_ns - _parse_window_seconds(window) * 1_000_000_000
    url = f"{base}/loki/api/v1/query_range"
    params = {
        "query": query,
        "start": str(start_ns),
        "end": str(end_ns),
        "limit": str(max(limit, 1)),
        "direction": "backward",
    }
    try:
        data = get_json(url, params=params, auth=(user, token), timeout=target.timeout)
    except Exception as exc:  # noqa: BLE001
        return err(_LOGS, f"loki query_range failed: {exc}")
    lines = _flatten_streams(data, limit)
    return ok(
        _LOGS,
        {
            "window": window,
            "environment": target.env_label,
            "level": level,
            "service": service,
            "count": len(lines),
            "lines": lines,
        },
    )


def cost_for_run(
    target: TargetConfig, run_id: str, *, window: str = "24h", limit: int = 200
) -> dict:
    """All ``llm_cost`` events for one ``run_id`` (#1053) — per-call cost for correlation."""
    creds = _creds(target)
    if not creds:
        return err(_COST_RUN, _NOT_CONFIGURED, configured=False)
    base, user, token = creds
    # run_id is our own resolved id (safe charset), but escape defensively for LogQL.
    safe = run_id.replace(chr(92), chr(92) + chr(92)).replace(chr(34), chr(92) + chr(34))
    selector = f'{{app="podcast_scraper", env="{target.env_label}"}}'
    query = f'{selector} | json | event_type="llm_cost" | run_id="{safe}"'
    end_ns = time.time_ns()
    start_ns = end_ns - _parse_window_seconds(window) * 1_000_000_000
    url = f"{base}/loki/api/v1/query_range"
    params = {
        "query": query,
        "start": str(start_ns),
        "end": str(end_ns),
        "limit": str(max(limit, 1)),
        "direction": "backward",
    }
    try:
        data = get_json(url, params=params, auth=(user, token), timeout=target.timeout)
    except Exception as exc:  # noqa: BLE001
        return err(_COST_RUN, f"loki query_range failed: {exc}")
    events: list[dict] = []
    total = 0.0
    for entry in _flatten_streams(data, limit):
        try:
            event = json.loads(entry["line"])
        except (ValueError, TypeError):
            continue
        cost = event.get("estimated_cost_usd")
        if isinstance(cost, (int, float)):
            total += float(cost)
        events.append(
            {
                key: event.get(key)
                for key in (
                    "provider",
                    "stage",
                    "model",
                    "estimated_cost_usd",
                    "prompt_tokens",
                    "completion_tokens",
                )
            }
        )
    return ok(
        _COST_RUN,
        {
            "run_id": run_id,
            "window": window,
            "count": len(events),
            "total_cost_usd": round(total, 6),
            "events": events,
        },
    )


def _flatten_streams(data: object, limit: int) -> list[dict]:
    try:
        result = data["data"]["result"]  # type: ignore[index]
    except (KeyError, TypeError):
        return []
    entries: list[dict] = []
    for stream in result:
        svc = (stream.get("stream") or {}).get("service")
        for ts_ns, line in stream.get("values", []):
            entries.append({"ts": _ns_to_iso(ts_ns), "ts_ns": ts_ns, "service": svc, "line": line})
    entries.sort(key=lambda entry: entry["ts_ns"], reverse=True)
    return entries[: max(limit, 0)]
