"""Umami (self-hosted, cookieless) — the user-action lens for the control plane (ADR-126).

Where VictoriaLogs/Traces show what the BACKEND did, Umami shows what USERS did on the operator
viewer / player: page views, visitors, and the typed custom events (search / explore / graph-handoff
…) the frontends register. Reading needs admin auth — a bearer token, or a username+password we
exchange for one via ``/api/auth/login``. Without creds the source degrades to ``configured=False``
like every other external source, so ``surface operator`` still returns its other signals.
"""

from __future__ import annotations

import time
from typing import Optional

from .._http import get_json, post_json
from ..config import TargetConfig
from ..result import err, ok

_STATS = "umami.stats"
_EVENTS = "umami.events"
_ACTIVE = "umami.active"
_NOT_CONFIGURED = (
    "umami not configured (umami_url + umami_website_id + umami_token OR umami_username/password)"
)
_WINDOW_MULT = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def _window_ms(window: str, default: int = 86400) -> int:
    """Window string (e.g. ``24h``) → milliseconds; Umami's stats/metrics API takes epoch-ms."""
    try:
        seconds = int(window[:-1]) * _WINDOW_MULT.get(window[-1], 0) or default
    except (ValueError, IndexError):
        seconds = default
    return seconds * 1000


def _bearer(target: TargetConfig) -> Optional[str]:
    """The bearer token: the configured one, else exchange username+password via /api/auth/login.

    Raises on a failed login so the caller can report it (vs a silent not-configured).
    """
    if target.umami_token:
        return target.umami_token
    if target.umami_username and target.umami_password and target.umami_url:
        url = f"{target.umami_url.rstrip('/')}/api/auth/login"
        data = post_json(
            url,
            json={"username": target.umami_username, "password": target.umami_password},
            timeout=target.timeout,
        )
        if isinstance(data, dict):
            return data.get("token")
    return None


def _prepare(target: TargetConfig, source: str) -> tuple[Optional[dict], Optional[dict]]:
    """Return ``(headers, error_envelope)`` — exactly one is non-None.

    Degrades to ``configured=False`` when url / website / creds are missing; surfaces a login
    failure as a real error so a misconfigured token doesn't masquerade as "not wired".
    """
    if not (target.umami_url and target.umami_website_id):
        return None, err(source, _NOT_CONFIGURED, configured=False)
    try:
        token = _bearer(target)
    except Exception as exc:  # noqa: BLE001
        return None, err(source, f"umami login failed: {exc}")
    if not token:
        return None, err(source, _NOT_CONFIGURED, configured=False)
    return {"Authorization": f"Bearer {token}"}, None


def _api(target: TargetConfig, suffix: str) -> str:
    # Callers reach here only past _prepare (url + website confirmed); the `or ""` satisfies mypy.
    base = (target.umami_url or "").rstrip("/")
    return f"{base}/api/websites/{target.umami_website_id}{suffix}"


def stats(target: TargetConfig, *, window: str = "24h") -> dict:
    """Aggregate site stats over the window — pageviews / visitors / visits / bounces."""
    headers, error = _prepare(target, _STATS)
    if error:
        return error
    end = int(time.time() * 1000)
    params = {"startAt": end - _window_ms(window), "endAt": end}
    try:
        data = get_json(
            _api(target, "/stats"), headers=headers, params=params, timeout=target.timeout
        )
    except Exception as exc:  # noqa: BLE001
        return err(_STATS, f"umami stats failed: {exc}")
    return ok(_STATS, {"website_id": target.umami_website_id, "window": window, "stats": data})


def events(target: TargetConfig, *, window: str = "24h") -> dict:
    """User-action counts by event name — the operator "what did users DO" view (ADR-126 events)."""
    headers, error = _prepare(target, _EVENTS)
    if error:
        return error
    end = int(time.time() * 1000)
    params = {"type": "event", "startAt": end - _window_ms(window), "endAt": end}
    try:
        data = get_json(
            _api(target, "/metrics"), headers=headers, params=params, timeout=target.timeout
        )
    except Exception as exc:  # noqa: BLE001
        return err(_EVENTS, f"umami events failed: {exc}")
    rows = data if isinstance(data, list) else []
    return ok(
        _EVENTS,
        {
            "website_id": target.umami_website_id,
            "window": window,
            "count": len(rows),
            "events": rows,
        },
    )


def active(target: TargetConfig) -> dict:
    """Currently-active visitors on the website (Umami's ~5-minute live window)."""
    headers, error = _prepare(target, _ACTIVE)
    if error:
        return error
    try:
        data = get_json(_api(target, "/active"), headers=headers, timeout=target.timeout)
    except Exception as exc:  # noqa: BLE001
        return err(_ACTIVE, f"umami active failed: {exc}")
    return ok(_ACTIVE, {"website_id": target.umami_website_id, "active": data})
