"""The graph API's routes (ADR-183): ``GET /api/graph/catalogue``.

Built: the catalogue, the engine's constants (``db.graph.catalogue``). Not yet: the presets'
routes and ``POST /api/graph/query``.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Response

from podcast_scraper.db import graph

router = APIRouter(tags=["graph"])


@router.get("/graph/catalogue")
def graph_catalogue(response: Response) -> dict[str, Any]:
    """What this build's graph engine serves (decision 11). The same for every revision, so it is
    cached as long as the process runs."""
    response.headers["Cache-Control"] = "public, max-age=300"
    return graph.catalogue()
