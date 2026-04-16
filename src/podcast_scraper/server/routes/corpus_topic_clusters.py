"""GET /api/corpus/topic-clusters — RFC-075 ``topic_clusters.json`` overlay for the viewer."""

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

_TOPIC_CLUSTERS_REL = "search/topic_clusters.json"


def _resolve_corpus_root(path: str | None, fallback: Path | None) -> Path | None:
    if path is not None and str(path).strip():
        return resolve_corpus_path_param(path, fallback)
    return fallback


@router.get("/corpus/topic-clusters")
async def corpus_topic_clusters(
    request: Request,
    path: str | None = Query(
        default=None,
        description="Corpus output dir (contains search/). Omit to use server default output_dir.",
    ),
) -> JSONResponse:
    """Return ``<corpus>/search/topic_clusters.json`` when present (RFC-075)."""
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
    parts = [p for p in _TOPIC_CLUSTERS_REL.replace("\\", "/").split("/") if p and p != "."]
    if any(p == ".." for p in parts):
        raise HTTPException(status_code=400, detail="Invalid corpus path.")
    joined = os.path.normpath(os.path.join(root_s, *parts))
    if joined != root_s and not joined.startswith(safe_prefix):
        return JSONResponse(
            status_code=404,
            content={
                "detail": "topic_clusters.json not found under corpus search/",
                "available": False,
            },
        )
    if not os.path.isfile(joined):
        return JSONResponse(
            status_code=404,
            content={
                "detail": "topic_clusters.json not found under corpus search/",
                "available": False,
            },
        )

    try:
        with open(joined, encoding="utf-8") as fh:
            payload = json.loads(fh.read())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("corpus_topic_clusters: failed to read %s: %s", joined, exc)
        raise HTTPException(
            status_code=500,
            detail="topic_clusters.json is unreadable or invalid JSON.",
        ) from exc

    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=500,
            detail="topic_clusters.json must be a JSON object.",
        )
    _clusters = payload.get("clusters")
    _n = len(_clusters) if isinstance(_clusters, list) else None
    logger.debug(
        "corpus_topic_clusters: serving schema_version=%s cluster_entries=%s",
        payload.get("schema_version"),
        _n,
    )
    return JSONResponse(content=payload)
