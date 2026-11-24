"""News items over the corpus (News 1.0, RFC-134 §10; UXS-019).

Every item comes from :func:`podcast_scraper.server.news_access.cached_served_items`, so an item
whose ``serveable`` is ``none`` never appears and no response carries the article body (ADR-179).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request

from podcast_scraper.news_items import item_kg
from podcast_scraper.server.news_access import cached_served_items
from podcast_scraper.server.pathutil import resolve_corpus_path_param
from podcast_scraper.server.schemas import (
    NewsItem,
    NewsItemDetailResponse,
    NewsItemsResponse,
    NewsKgNode,
    NewsSummaryResponse,
)

router = APIRouter(tags=["corpus"])


def _root(request: Request, path: Optional[str]) -> Path:
    anchor = getattr(request.app.state, "output_dir", None)
    if path is not None and str(path).strip():
        return resolve_corpus_path_param(path, anchor)
    if anchor is not None:
        return Path(anchor).expanduser().resolve()
    raise HTTPException(status_code=400, detail="path is required without a default corpus.")


@router.get("/corpus/news/items", response_model=NewsItemsResponse)
def corpus_news_items(
    request: Request,
    path: Optional[str] = Query(default=None, description="Corpus root."),
    publisher_id: Optional[str] = Query(
        default=None, description="Only this publisher's items (a registry org: id)."
    ),
    since: Optional[str] = Query(
        default=None, description="Only items published on or after this day (YYYY-MM-DD)."
    ),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> NewsItemsResponse:
    """The news items a surface may show, newest first."""
    root = _root(request, path)
    items = cached_served_items(root)
    if since and since.strip():
        day = since.strip()[:10]
        items = [i for i in items if (i.get("published_at") or "")[:10] >= day]
    if publisher_id:
        items = [i for i in items if (i.get("publisher") or {}).get("id") == publisher_id]
    page = items[offset : offset + limit]
    return NewsItemsResponse(path=str(root), items=[NewsItem(**i) for i in page], total=len(items))


@router.get("/corpus/news/item", response_model=NewsItemDetailResponse)
def corpus_news_item(
    request: Request,
    item_id: str = Query(..., description="The item's id (item:news_article:...)."),
    path: Optional[str] = Query(default=None, description="Corpus root."),
) -> NewsItemDetailResponse:
    """One news item, its KG file and the topics, people and organisations it names."""
    root = _root(request, path)
    item = next((i for i in cached_served_items(root) if i["item_id"] == item_id), None)
    if item is None:
        raise HTTPException(status_code=404, detail="No shown news item has this id.")
    kg = item_kg(root, item["metadata_relative_path"])
    return NewsItemDetailResponse(
        path=str(root),
        item=NewsItem(**item),
        kg_relative_path=kg["kg_relative_path"],
        kg_nodes=[NewsKgNode(**n) for n in kg["kg_nodes"]],
    )


@router.get("/corpus/news/summary", response_model=NewsSummaryResponse)
def corpus_news_summary(
    request: Request,
    path: Optional[str] = Query(default=None, description="Corpus root."),
) -> NewsSummaryResponse:
    """Counts over every shown news item: items, publishers, teasers (and videos), paywalled."""
    root = _root(request, path)
    items = cached_served_items(root)
    return NewsSummaryResponse(
        path=str(root),
        items=len(items),
        publishers=len({(i.get("publisher") or {}).get("id") for i in items} - {None}),
        teasers=sum(1 for i in items if i.get("text_status") in ("teaser", "video")),
        paywalled=sum(1 for i in items if i.get("paywalled") is True),
    )
