"""Tiny HTTP helper. ``httpx`` is imported lazily so the package imports without it."""

from __future__ import annotations

from typing import Any, Mapping, Optional


def get_json(
    url: str,
    *,
    headers: Optional[Mapping[str, str]] = None,
    params: Optional[Mapping[str, Any]] = None,
    auth: Optional[tuple[str, str]] = None,
    timeout: float = 10.0,
) -> Any:
    """GET *url* and return parsed JSON; raise on a non-2xx response or transport error.

    ``auth`` is a ``(username, password)`` pair for HTTP Basic (Grafana Cloud Loki query API);
    bearer-token APIs (GitHub/Sentry/Grafana alerting) pass an ``Authorization`` header instead.
    """
    import httpx

    resp = httpx.get(
        url,
        headers=dict(headers or {}),
        params=dict(params or {}),
        auth=auth,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def post_json(
    url: str,
    *,
    headers: Optional[Mapping[str, str]] = None,
    params: Optional[Mapping[str, Any]] = None,
    json: Optional[Mapping[str, Any]] = None,
    auth: Optional[tuple[str, str]] = None,
    timeout: float = 10.0,
) -> Any:
    """POST *url* with optional JSON body; return parsed JSON; raise on transport/HTTP error."""
    import httpx

    resp = httpx.post(
        url,
        headers=dict(headers or {}),
        params=dict(params or {}),
        json=dict(json) if json is not None else None,
        auth=auth,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()
