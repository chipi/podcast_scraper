"""Recent Langfuse traces for the deploy — the AI-quality lens in the control plane (#1052).

Complements the cost/error sources: where ``loki.cost_today`` answers "how much did we
spend", Langfuse answers "what did each LLM call *do*" (model / token usage / cost per
generation, grouped per run). The probe only **reads** the Langfuse public API (HTTP Basic
auth with the same public/secret key pair the pipeline traces with), so the control plane
stays light — no ``langfuse`` SDK here, just ``httpx``.
"""

from __future__ import annotations

import base64

from .._http import get_json
from ..config import TargetConfig
from ..result import err, ok

_SOURCE = "langfuse.traces"
_CLOUD = "https://cloud.langfuse.com"


def recent_traces(target: TargetConfig, limit: int = 10) -> dict:
    """Most recent Langfuse traces (count + id/name/timestamp) for the deploy."""
    if not target.langfuse_public_key or not target.langfuse_secret_key:
        return err(
            _SOURCE,
            "langfuse keys not set (LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY)",
            configured=False,
        )
    base = (target.langfuse_base_url or _CLOUD).rstrip("/")
    token = base64.b64encode(
        f"{target.langfuse_public_key}:{target.langfuse_secret_key}".encode("utf-8")
    ).decode("ascii")
    headers = {"Authorization": f"Basic {token}"}
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
