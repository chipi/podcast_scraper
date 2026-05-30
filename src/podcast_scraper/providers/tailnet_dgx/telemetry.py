"""Sentry breadcrumbs for DGX fallback events (ADR-096)."""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def emit_dgx_fallback_breadcrumb(
    *,
    stage: str,
    model: str,
    failure_reason: str,
    episode_id: Optional[str] = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """Record that a DGX-primary call fell back to cloud."""
    data: dict[str, Any] = {
        "stage": stage,
        "model": model,
        "failure_reason": failure_reason,
        "dgx_fallback_active": True,
    }
    if episode_id is not None:
        data["episode_id"] = episode_id
    if extra:
        data.update(extra)
    try:
        import sentry_sdk

        sentry_sdk.add_breadcrumb(
            category="dgx.fallback",
            message=f"DGX fallback active for {stage}",
            level="warning",
            data=data,
        )
    except Exception:
        logger.warning(
            "DGX fallback for stage=%s model=%s reason=%s episode=%s",
            stage,
            model,
            failure_reason,
            episode_id,
        )
