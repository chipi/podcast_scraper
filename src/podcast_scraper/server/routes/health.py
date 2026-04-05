"""GET /api/health — always available."""

from __future__ import annotations

from fastapi import APIRouter

from podcast_scraper.server.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Server health check."""
    return HealthResponse()
