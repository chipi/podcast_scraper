"""GET/PUT /api/feeds — structured corpus feed list (``feeds.spec.yaml``).

Root document: ``{ feeds: [...] }`` (RFC-077 / #626). Each entry is a URL string or an object
with ``url`` plus optional per-feed overrides validated server-side.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Union

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import ValidationError

from podcast_scraper.rss.feeds_spec import (
    dump_feeds_spec_yaml,
    FEEDS_SPEC_DEFAULT_BASENAME,
    FeedsSpecDocument,
    load_feeds_spec_file,
)
from podcast_scraper.server.atomic_write import atomic_write_text
from podcast_scraper.server.pathutil import resolve_corpus_path_param
from podcast_scraper.server.schemas import FeedsListResponse, FeedsPutBody

logger = logging.getLogger(__name__)

router = APIRouter(tags=["feeds"])

FEEDS_SPEC_BASENAME = FEEDS_SPEC_DEFAULT_BASENAME


def _feeds_path(root: Path) -> Path:
    return root / FEEDS_SPEC_BASENAME


def _dedupe_feeds_put(items: List[Union[str, dict[str, Any]]]) -> List[Union[str, dict[str, Any]]]:
    seen: set[str] = set()
    out: List[Union[str, dict[str, Any]]] = []
    for item in items:
        if isinstance(item, str):
            u = item.strip()
            if not u or u in seen:
                continue
            seen.add(u)
            out.append(u)
        elif isinstance(item, dict):
            u = str(item.get("url") or item.get("rss") or "").strip()
            if not u or u in seen:
                continue
            seen.add(u)
            out.append(dict(item))
        else:
            raise HTTPException(status_code=400, detail="Each feed must be a string or object")
    return out


def _document_feeds_to_api_list(doc: FeedsSpecDocument) -> List[Union[str, dict[str, Any]]]:
    """API list: bare string when only ``url`` is set."""
    out: List[Union[str, dict[str, Any]]] = []
    for e in doc.feeds:
        d = e.model_dump(mode="json", exclude_none=True)
        if set(d.keys()) == {"url"}:
            out.append(str(d["url"]))
        else:
            out.append(d)
    return out


@router.get("/feeds", response_model=FeedsListResponse)
async def get_feeds(
    request: Request,
    path: str = Query(..., description="Corpus root directory (resolved under server anchor)."),
) -> FeedsListResponse:
    """Return structured ``feeds.spec.yaml`` for the resolved corpus root."""
    anchor = getattr(request.app.state, "output_dir", None)
    root = resolve_corpus_path_param(path, anchor)
    fp = _feeds_path(root)
    feeds: List[Union[str, dict[str, Any]]] = []
    if fp.is_file():
        try:
            doc = load_feeds_spec_file(fp)
        except (OSError, ValueError, ValidationError) as exc:
            logger.warning("Invalid feeds spec at %s: %s", fp, exc)
            raise HTTPException(
                status_code=500,
                detail=f"Invalid feeds spec file (fix on disk): {exc}",
            ) from exc
        feeds = _document_feeds_to_api_list(doc)
    return FeedsListResponse(
        path=str(root.resolve()),
        file_relpath=FEEDS_SPEC_BASENAME,
        feeds=feeds,
    )


@router.put("/feeds", response_model=FeedsListResponse)
async def put_feeds(
    request: Request,
    body: FeedsPutBody,
    path: str = Query(..., description="Corpus root directory (resolved under server anchor)."),
) -> FeedsListResponse:
    """Persist structured feeds to ``feeds.spec.yaml`` under the corpus root."""
    anchor = getattr(request.app.state, "output_dir", None)
    root = resolve_corpus_path_param(path, anchor)
    if len(body.feeds) > 5000:
        raise HTTPException(status_code=400, detail="Too many feed entries (max 5000).")
    cleaned = _dedupe_feeds_put(list(body.feeds))
    try:
        doc = FeedsSpecDocument.model_validate({"feeds": cleaned})
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    fp = _feeds_path(root)
    text = dump_feeds_spec_yaml(doc)
    atomic_write_text(fp, text)
    return FeedsListResponse(
        path=str(root.resolve()),
        file_relpath=FEEDS_SPEC_BASENAME,
        feeds=_document_feeds_to_api_list(doc),
    )
