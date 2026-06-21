"""GET /api/ops/summary — prod observability control-plane glance (#803).

Returns the same cross-source summary the ``podcast_obs`` CLI/MCP produce, so the viewer's Ops
view shows a human exactly what an agent sees. Reads ``PODCAST_OBS_*`` from the server env;
defaults the observed target to *this* server so health/version/runs work with zero config,
and the external sources (deploys / cost / logs / errors / alerts) light up when their read-scoped
tokens are present. ``podcast_obs`` is light-dep and pulls no MCP SDK, so importing it is cheap.
"""

from __future__ import annotations

from dataclasses import replace

from fastapi import APIRouter

router = APIRouter(tags=["ops"])

# The api container serves /api on 8000 (GH-745); observe ourselves when no target is configured.
_LOCAL_API_BASE = "http://127.0.0.1:8000"
# Keep the endpoint responsive: a configured-but-unreachable source shouldn't hang the dashboard.
# summary() fans out to several backends sequentially, so keep the per-probe timeout tight.
_WEB_TIMEOUT = 4.0


# Deliberately a SYNC ``def`` (not ``async``): ``summary()`` makes blocking ``httpx`` calls, so
# Starlette runs this handler in a threadpool — it must not block the event loop.
@router.get("/ops/summary")
def ops_summary() -> dict:
    """Cross-source prod-state summary (live / unconfigured / failed + per-source envelopes)."""
    from podcast_obs.aggregate import summary as obs_summary
    from podcast_obs.config import ObservabilityConfig

    target = ObservabilityConfig.load().target()
    overrides: dict = {"timeout": _WEB_TIMEOUT}
    if not target.api_base:
        overrides["api_base"] = _LOCAL_API_BASE
    target = replace(target, **overrides)
    data: dict = obs_summary(target)["data"]
    return data
