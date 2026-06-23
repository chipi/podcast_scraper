"""Recent Langfuse traces for the deploy — the AI-quality lens in the control plane (#1052).

Complements the cost/error sources: where ``loki.cost_today`` answers "how much did we
spend", Langfuse answers "what did each LLM call *do*" (model / token usage / cost per
generation, grouped per run). The probe only **reads** the Langfuse public API (HTTP Basic
auth with the same public/secret key pair the pipeline traces with), so the control plane
stays light — no ``langfuse`` SDK here, just ``httpx``.
"""

from __future__ import annotations

import base64
import hashlib

from .._http import get_json
from ..config import TargetConfig
from ..result import err, ok

_SOURCE = "langfuse.traces"
_CLOUD = "https://cloud.langfuse.com"


def _auth_header(target: TargetConfig) -> str:
    token = base64.b64encode(
        f"{target.langfuse_public_key}:{target.langfuse_secret_key}".encode("utf-8")
    ).decode("ascii")
    return f"Basic {token}"


def trace_id_for_run(run_id: str) -> str:
    """The Langfuse trace id our pipeline seeds for a run (#1053).

    Mirrors the SDK's ``create_trace_id(seed=run_id)`` — ``sha256(seed)[:16].hex()`` — so
    the control plane can address a run's trace with *no* SDK, just the run_id.
    """
    return hashlib.sha256(run_id.encode("utf-8")).digest()[:16].hex()


def trace_by_run(target: TargetConfig, run_id: str) -> dict:
    """The Langfuse trace for ``run_id`` (model/cost/tokens per LLM call) — for correlation."""
    if not target.langfuse_public_key or not target.langfuse_secret_key:
        return err(
            "langfuse.trace",
            "langfuse keys not set (LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY)",
            configured=False,
        )
    base = (target.langfuse_base_url or _CLOUD).rstrip("/")
    trace_id = trace_id_for_run(run_id)
    url = f"{base}/api/public/traces/{trace_id}"
    headers = {"Authorization": _auth_header(target)}
    try:
        payload = get_json(url, headers=headers, timeout=target.timeout)
    except Exception:  # noqa: BLE001 — 404 = no LLM calls traced for this run (valid)
        return ok(
            "langfuse.trace",
            {"run_id": run_id, "trace_id": trace_id, "found": False},
        )
    observations = [
        {
            "name": o.get("name"),
            "model": o.get("model"),
            "cost": o.get("calculatedTotalCost") or o.get("totalCost"),
            "usage": o.get("usageDetails") or o.get("usage"),
            # surface the correlation context so an agent can attribute each call
            # to a stage AND an episode (the join keys live in the span metadata).
            "stage": (o.get("metadata") or {}).get("stage"),
            "episode_id": (o.get("metadata") or {}).get("episode_id"),
            "provider": (o.get("metadata") or {}).get("provider"),
        }
        for o in (payload.get("observations") or [])
        if isinstance(o, dict)
    ]
    data = {
        "run_id": run_id,
        "trace_id": trace_id,
        "found": True,
        "name": payload.get("name"),
        "totalCost": payload.get("totalCost"),
        "observation_count": len(observations),
        "observations": observations,
    }
    return ok("langfuse.trace", data)


def recent_traces(target: TargetConfig, limit: int = 10) -> dict:
    """Most recent Langfuse traces (count + id/name/timestamp) for the deploy."""
    if not target.langfuse_public_key or not target.langfuse_secret_key:
        return err(
            _SOURCE,
            "langfuse keys not set (LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY)",
            configured=False,
        )
    base = (target.langfuse_base_url or _CLOUD).rstrip("/")
    headers = {"Authorization": _auth_header(target)}
    url = f"{base}/api/public/traces"
    params = {"limit": max(limit, 1)}
    try:
        payload = get_json(url, headers=headers, params=params, timeout=target.timeout)
    except Exception as exc:  # noqa: BLE001
        return err(_SOURCE, str(exc))
    items = payload.get("data") if isinstance(payload, dict) else None
    traces = [
        {
            "id": t.get("id"),
            "name": t.get("name"),
            "timestamp": t.get("timestamp"),
            "latency": t.get("latency"),
            "totalCost": t.get("totalCost"),
        }
        for t in (items or [])
        if isinstance(t, dict)
    ]
    data = {
        "base_url": base,
        "count": len(traces),
        "traces": traces,
    }
    return ok(_SOURCE, data)
