"""Operator surface for resilience (ADR-113 gap closure).

* ``GET /api/resilience`` — a normal read (like /api/health): which LLM/RSS breakers are open, their
  cooldowns, and the configured fuse budgets. Queryable by the o11y MCP tool and dashboards.
* ``POST /api/ops/resilience/reset`` — force-close breakers early (the "plug it back in" control).
  Lives under ``/api/ops`` so the OperatorWriteGuard middleware admin-gates it like every other
  operator write. Fuses are not resettable (per-run hard stops); this only touches breakers.

Both are thin wrappers over ``utils.resilience_status`` — the single source of truth.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query

from podcast_scraper.utils.resilience_status import reset_resilience, resilience_snapshot

router = APIRouter(tags=["resilience"])


@router.get("/resilience")
async def resilience_status() -> dict:
    """What resilience is doing right now: open breakers, cooldowns, fuse budgets."""
    return resilience_snapshot()


@router.post("/ops/resilience/reset")
async def resilience_reset(
    scope: Optional[str] = Query(
        default="all",
        description="Which breakers to force-close: 'llm', 'rss', or 'all' (default).",
    ),
) -> dict:
    """Force-close breakers early instead of waiting out the cooldown (admin-gated by the
    OperatorWriteGuard middleware). Does not touch fuses."""
    return reset_resilience(scope or "all")
