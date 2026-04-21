"""GET /api/health — always available."""

from __future__ import annotations

from fastapi import APIRouter, Request

from podcast_scraper.server.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Server health check."""
    st = request.app.state
    return HealthResponse().model_copy(
        update={
            "feeds_api": bool(getattr(st, "feeds_api_enabled", False)),
            "operator_config_api": bool(getattr(st, "operator_config_api_enabled", False)),
            "jobs_api": bool(getattr(st, "jobs_api_enabled", False)),
        }
    )
