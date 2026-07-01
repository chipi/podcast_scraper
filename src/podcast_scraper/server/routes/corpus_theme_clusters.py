"""GET /api/corpus/theme-clusters — ``topic_theme_clusters.json`` overlay.

Theme clusters group topics *discussed together* (co-occurrence lift), as
opposed to ``/api/corpus/topic-clusters`` which serves the *semantic*
(embedding-similarity) clusters. The two are complementary and themed apart in
the consumer. Produced by the ``topic_theme_clusters`` enricher under
``enrichments/`` (not ``search/`` — different producer).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from podcast_scraper.server.pathutil import resolve_corpus_path_param
from podcast_scraper.utils.path_validation import safe_resolve_directory

logger = logging.getLogger(__name__)

router = APIRouter(tags=["corpus"])

_THEME_CLUSTERS_REL = "enrichments/topic_theme_clusters.json"


def _resolve_corpus_root(path: str | None, fallback: Path | None) -> Path | None:
    if path is not None and str(path).strip():
        return resolve_corpus_path_param(path, fallback)
    return fallback


@router.get("/corpus/theme-clusters")
async def corpus_theme_clusters(
    request: Request,
    path: str | None = Query(
        default=None,
        description=(
            "Corpus output dir (contains enrichments/). Omit to use server default output_dir."
        ),
    ),
) -> JSONResponse:
    """Return ``<corpus>/enrichments/topic_theme_clusters.json`` when present."""
    fallback = getattr(request.app.state, "output_dir", None)
    root = _resolve_corpus_root(path, fallback)
    if root is None:
        raise HTTPException(
            status_code=400,
            detail="path query parameter is required when the server has no default output_dir.",
        )

    root_dir = safe_resolve_directory(root)
    if root_dir is None:
        raise HTTPException(status_code=400, detail="Invalid corpus path.")

    root_s = os.path.normpath(str(root_dir))
    safe_prefix = root_s + os.sep
    parts = [p for p in _THEME_CLUSTERS_REL.replace("\\", "/").split("/") if p and p != "."]
    if any(p == ".." for p in parts):
        raise HTTPException(status_code=400, detail="Invalid corpus path.")
    joined = os.path.normpath(os.path.join(root_s, *parts))
    not_found = JSONResponse(
        status_code=404,
        content={
            "detail": "topic_theme_clusters.json not found under corpus enrichments/",
            "available": False,
        },
    )
    if joined != root_s and not joined.startswith(safe_prefix):
        return not_found
    # codeql[py/path-injection] -- joined under root_s (Type 1; CODEQL_DISMISSALS.md).
    if not os.path.isfile(joined):
        return not_found

    try:
        # codeql[py/path-injection] -- joined sanitized above.
        with open(joined, encoding="utf-8") as fh:
            payload = json.loads(fh.read())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("corpus_theme_clusters: failed to read %s: %s", joined, exc)
        raise HTTPException(
            status_code=500,
            detail="topic_theme_clusters.json is unreadable or invalid JSON.",
        ) from exc

    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=500,
            detail="topic_theme_clusters.json must be a JSON object.",
        )
    _clusters = payload.get("clusters")
    _n = len(_clusters) if isinstance(_clusters, list) else None
    logger.debug(
        "corpus_theme_clusters: serving schema_version=%s cluster_entries=%s",
        payload.get("schema_version"),
        _n,
    )
    return JSONResponse(content=payload)
