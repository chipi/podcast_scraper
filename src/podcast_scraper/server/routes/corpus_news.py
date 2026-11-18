"""News items over the corpus (News 1.0, RFC-134 §10; UXS-019).

Every item goes through :func:`podcast_scraper.news_items.served_item`, so an item whose
``serveable`` is ``none`` never appears and no response carries the article body (ADR-179).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request

from podcast_scraper import perf_cache
from podcast_scraper.news_items import item_kg, served_items
from podcast_scraper.publishers import load_registry
from podcast_scraper.server.pathutil import resolve_corpus_path_param
from podcast_scraper.server.schemas import (
    NewsItem,
    NewsItemDetailResponse,
    NewsItemsResponse,
    NewsKgNode,
    NewsSummaryResponse,
)

router = APIRouter(tags=["corpus"])

_NS = "news_served_items"


def _root(request: Request, path: Optional[str]) -> Path:
    anchor = getattr(request.app.state, "output_dir", None)
    if path is not None and str(path).strip():
        return resolve_corpus_path_param(path, anchor)
    if anchor is not None:
        return Path(anchor).expanduser().resolve()
    raise HTTPException(status_code=400, detail="path is required without a default corpus.")


def cached_served_items(root: Path) -> List[Dict[str, Any]]:
    """:func:`served_items` for ``root``, cached until the corpus changes (an ingest moves the
    corpus token)."""
    items: List[Dict[str, Any]] = perf_cache.get_or_compute(
        _NS,
        str(root),
        perf_cache.corpus_mtime(root),
        lambda: served_items(root, load_registry()),
    )
    return items


def _node_index(root: Path) -> Dict[str, List[str]]:
    """KG node id -> the shown news items naming it (topics, people, organisations, objects)."""
    index: Dict[str, List[str]] = {}
    for item in cached_served_items(root):
        for node in item_kg(root, item["metadata_relative_path"])["kg_nodes"]:
            index.setdefault(str(node["id"]), []).append(item["item_id"])
    return index


def news_items_naming(root: Path, node_id: str) -> List[Dict[str, Any]]:
    """The shown news items whose KG names ``node_id``, newest first (cached per corpus)."""
    index: Dict[str, List[str]] = perf_cache.get_or_compute(
        _NS + "_nodes", str(root), perf_cache.corpus_mtime(root), lambda: _node_index(root)
    )
    wanted = set(index.get(node_id, ()))
    return [i for i in cached_served_items(root) if i["item_id"] in wanted]


#: Where each grouping keeps its members' ``news_item_ids`` (Common's piece A): themes (``tc:``)
#: in the search directory's topic clusters, storylines (``thc:``) in the theme-clusters
#: enrichment, under ``data``.
_THEMES_FILE = "search/topic_clusters.json"
_STORYLINES_FILE = "enrichments/topic_theme_clusters.json"


def _cluster_item_ids(root: Path, cluster_id: str) -> List[str]:
    from podcast_scraper.server.corpus_access import cached_json_artifact

    relpath = _STORYLINES_FILE if cluster_id.startswith("thc:") else _THEMES_FILE
    doc = cached_json_artifact(root, relpath) or {}
    data = doc.get("data")
    holder: Dict[str, Any] = data if isinstance(data, dict) else doc
    ids: Dict[str, None] = {}
    for cluster in holder.get("clusters") or []:
        if not isinstance(cluster, dict) or cluster.get("graph_compound_parent_id") != cluster_id:
            continue
        for member in cluster.get("members") or []:
            if isinstance(member, dict):
                for item_id in member.get("news_item_ids") or []:
                    if isinstance(item_id, str) and item_id:
                        ids[item_id] = None
    return list(ids)


def news_items_in_cluster(root: Path, cluster_id: str) -> List[Dict[str, Any]]:
    """The shown news items of a theme (``tc:``) or storyline (``thc:``): the union of its
    members' ``news_item_ids``, newest first. A ``serveable: none`` item is never among them."""
    wanted = set(_cluster_item_ids(root, cluster_id))
    return [i for i in cached_served_items(root) if i["item_id"] in wanted]


def corpus_has_news(root: Path) -> bool:
    """Whether the corpus holds a news item some surface may show."""
    return bool(cached_served_items(root))


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
