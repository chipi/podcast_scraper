"""Job-state webhook emitter for downstream automation.

The api can emit a fire-and-forget POST to a configured URL whenever a
pipeline job transitions to a terminal state (succeeded / failed /
cancelled). Default behaviour: nothing happens unless
``PODCAST_JOB_WEBHOOK_URL`` is set.

This is the **generic** notification surface — it does not call Slack or
Sentry directly. The operator points it at whichever sink they want:

* A Slack incoming-webhook URL (the most common case in pre-prod).
* A Home Assistant webhook trigger (so a smart bulb can flip red on a
  pipeline failure).
* An iOS Shortcuts handler (via a relay), Discord, Telegram, etc.

Sinks subscribe to the same payload shape; the api stays decoupled from
any specific destination's auth / rate-limit / retry semantics.

See: RFC-081 §Layer 4, issue #682.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Mapping

_LOGGER = logging.getLogger(__name__)

_ENV_URL = "PODCAST_JOB_WEBHOOK_URL"
_ENV_TIMEOUT = "PODCAST_JOB_WEBHOOK_TIMEOUT_SEC"
_DEFAULT_TIMEOUT_SEC = 5.0


def _webhook_url() -> str | None:
    raw = os.environ.get(_ENV_URL, "").strip()
    return raw or None


def _timeout_sec() -> float:
    raw = os.environ.get(_ENV_TIMEOUT, "").strip()
    if not raw:
        return _DEFAULT_TIMEOUT_SEC
    try:
        return max(0.5, float(raw))
    except ValueError:
        return _DEFAULT_TIMEOUT_SEC


async def emit_job_state_change(record: Mapping[str, Any]) -> None:
    """Fire-and-forget POST of *record* to the configured webhook URL.

    Wraps the network call in a timeout + broad except so a misbehaving
    webhook (slow, 5xx, DNS-failure, anything) cannot break the api's
    job-state finalisation. Logs a warning on failure; never raises.

    *record* is the in-memory job dict at terminal state, plus any
    enrichment fields the caller wants (e.g., a ``log_url``). The api
    does not redact this — assume the configured webhook is trusted by
    the operator. (Slack incoming-webhooks are URL-secret; Home Assistant
    webhooks are local-network only by default.)
    """
    url = _webhook_url()
    if not url:
        # No-op path. The default for everyone who hasn't set the env
        # var, including local dev / CI / unconfigured pre-prod.
        return

    payload = {"event": "job_state_changed", "job": dict(record)}

    try:
        # ``httpx`` is already a dep via openai/anthropic SDKs; using
        # it here avoids a new HTTP-client surface. The async context
        # manager + explicit timeout are belt-and-suspenders against
        # an unresponsive webhook hanging the finalize task.
        import httpx

        async with httpx.AsyncClient(timeout=_timeout_sec()) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code >= 400:
                _LOGGER.warning(
                    "job webhook returned %s for url=%s job=%s",
                    resp.status_code,
                    url,
                    record.get("job_id"),
                )
    except ImportError:
        # Defensive: httpx should always be installed via provider extras
        # but if a deployment somehow lacks it, log loudly (not crash).
        _LOGGER.warning(
            "PODCAST_JOB_WEBHOOK_URL is set but httpx is not installed; "
            "skipping webhook for job=%s",
            record.get("job_id"),
        )
    except asyncio.CancelledError:
        # Caller is being cancelled — propagate without swallowing.
        raise
    except Exception as exc:  # pragma: no cover - broad-except by design
        # Log + swallow. Do not let a webhook outage break the api's
        # job-state finalisation path.
        _LOGGER.warning(
            "job webhook POST failed url=%s job=%s err=%s",
            url,
            record.get("job_id"),
            exc,
        )
