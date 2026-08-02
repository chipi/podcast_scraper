"""GET /api/ops/llm-gateway — prod LiteLLM gateway per-key spend (#53 / #1357, ADR-142).

Feeds the Ops view's Prod-LLM card. Reads ``litellm_key_spend_usd`` /
``litellm_key_max_budget_usd`` / ``litellm_key_budget_burn_ratio{box="prod"}`` from homelab
VictoriaMetrics via ``podcast_obs`` (``victoriametrics_url`` from ``PODCAST_OBS_*`` env; the
api on the box reaches homelab over the tailnet). The gateway sidecar pushes these every
30 min.

Tailnet-only operator surface (spend is cost data) — mounted with ``ops`` in
``_OPERATOR_READ_ROUTES``, NOT the public operator serve. Degrades honestly:
``configured=False`` when the VM URL isn't wired, ``reachable=False`` on a query error, and
an empty ``keys`` list when no series exist yet — never hangs or 500s the dashboard.
"""

from __future__ import annotations

from dataclasses import replace

from fastapi import APIRouter

router = APIRouter(tags=["ops"])

# The metrics push carries box="prod"; this endpoint only surfaces that box.
_BOX = "prod"
# A configured-but-unreachable VM must not hang the Ops dashboard — keep the probe tight.
_TIMEOUT = 4.0
# The sidecar pushes every 30 min, but a VictoriaMetrics *instant* query only looks back
# ~5 min for a fresh sample — so a plain query returns empty between pushes (live-verified
# 2026-08-02). last_over_time[..] takes the most recent sample within the window instead;
# spend is monotonic/slow, so a 1h lookback shows the current value robustly.
_LOOKBACK = "1h"
# response field -> VictoriaMetrics series name (spend-to-vm.sh emits these three).
_METRICS = {
    "spend_usd": "litellm_key_spend_usd",
    "max_budget_usd": "litellm_key_max_budget_usd",
    "burn_ratio": "litellm_key_budget_burn_ratio",
}


# SYNC ``def`` (like ops_summary): podcast_obs makes blocking httpx calls, so Starlette runs
# this in a threadpool and it must not block the event loop.
@router.get("/ops/llm-gateway")
def ops_llm_gateway() -> dict:
    """Per-key spend / budget / burn for the prod LiteLLM gateway.

    Shape: ``{configured, reachable, keys: [{key_alias, spend_usd, max_budget_usd,
    burn_ratio}]}`` sorted by spend desc. ``configured=False`` when VictoriaMetrics is not
    wired for this target.
    """
    from podcast_obs.config import ObservabilityConfig
    from podcast_obs.sources import victoria

    target = replace(ObservabilityConfig.load().target(), timeout=_TIMEOUT)
    if not target.victoriametrics_url:
        return {"configured": False, "reachable": False, "keys": []}

    keys: dict[str, dict] = {}
    reachable = True
    for field, metric in _METRICS.items():
        query = f'last_over_time({metric}{{box="{_BOX}"}}[{_LOOKBACK}])'
        res = victoria.metrics_instant(target, query)
        if not res.get("ok"):
            reachable = False
            continue
        for series in res.get("data", {}).get("series", []):
            alias = (series.get("metric") or {}).get("key_alias")
            value = series.get("value")  # [timestamp, "stringified-number"]
            if not alias or not value:
                continue
            try:
                keys.setdefault(alias, {"key_alias": alias})[field] = float(value[1])
            except (TypeError, ValueError, IndexError):
                continue

    return {
        "configured": True,
        "reachable": reachable,
        "keys": sorted(keys.values(), key=lambda k: k.get("spend_usd", 0.0), reverse=True),
    }
