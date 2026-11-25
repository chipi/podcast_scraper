"""The shown news items of a corpus, for every reader of them (News 1.0, RFC-134 §10; UXS-019).

Every item goes through :func:`podcast_scraper.news_items.served_items`, so an item whose
``serveable`` is ``none`` never appears and nothing here carries the article body (ADR-179).

This module imports no fastapi: the Player's board summarizer runs the cluster cards' news path
(:func:`podcast_scraper.server.relational_view._with_news`) in public's pipeline-llm image, which
has none. The routes in :mod:`podcast_scraper.server.routes.corpus_news` import from here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from podcast_scraper import perf_cache
from podcast_scraper.news_items import item_kg, served_items
from podcast_scraper.publishers import load_registry, registry_path

_NS = "news_served_items"


def _news_token(root: Path) -> float:
    """What the news caches are valid for: the corpus token (an ingest moves it), the newest
    ``metadata/`` folder of any news run (the writer replaces a metadata file by renaming a new one
    into place, which moves its folder's time, so one item's rights changing shows here), and the
    publisher registry file. Stats one folder per run, not one file per item."""
    stamps = [perf_cache.corpus_mtime(root)]
    for meta in (root / "sources" / "news_article").glob("*/*/metadata"):
        try:
            stamps.append(meta.stat().st_mtime)
        except OSError:
            continue
    try:
        stamps.append(registry_path().stat().st_mtime)
    except OSError:
        pass
    return max(stamps)


def cached_served_items(root: Path) -> List[Dict[str, Any]]:
    """:func:`served_items` for ``root``, cached until a news item or the registry changes
    (:func:`_news_token`)."""
    items: List[Dict[str, Any]] = perf_cache.get_or_compute(
        _NS,
        str(root),
        _news_token(root),
        lambda: served_items(root, load_registry()),
    )
    return items


def _node_index(root: Path) -> Dict[str, Dict[str, Optional[str]]]:
    """KG node id -> {item_id: the node's role in it} for the shown news items naming it (topics,
    people, organisations, objects); the role is ``author`` for a byline author, else ``None``."""
    index: Dict[str, Dict[str, Optional[str]]] = {}
    for item in cached_served_items(root):
        for node in item_kg(root, item["metadata_relative_path"])["kg_nodes"]:
            roles = index.setdefault(str(node["id"]), {})
            role = node.get("role")
            roles[item["item_id"]] = roles.get(item["item_id"]) or (role if role else None)
    return index


def news_items_naming(root: Path, node_id: str) -> List[Dict[str, Any]]:
    """The shown news items whose KG names ``node_id``, newest first (cached per corpus), each
    with ``node_role``: the node's role in that item (``author`` when it wrote it, else ``None``).
    """
    index: Dict[str, Dict[str, Optional[str]]] = perf_cache.get_or_compute(
        _NS + "_nodes", str(root), _news_token(root), lambda: _node_index(root)
    )
    roles = index.get(node_id, {})
    return [
        {**i, "node_role": roles[i["item_id"]]}
        for i in cached_served_items(root)
        if i["item_id"] in roles
    ]


#: Where the themes keep their members' ``news_item_ids`` (Common's piece A), under ``data``: the
#: one per-topic source of news, which storylines read too (``_storyline_item_ids``).
_THEMES_FILE = "search/topic_clusters.json"


def _cluster_item_ids(root: Path, cluster_id: str) -> List[str]:
    """A theme's (``tc:``) members' ``news_item_ids``."""
    from podcast_scraper.server.corpus_access import cached_json_artifact

    doc = cached_json_artifact(root, _THEMES_FILE) or {}
    data = doc.get("data")
    holder: Dict[str, Any] = data if isinstance(data, dict) else doc
    ids: Dict[str, None] = {}
    for cluster in holder.get("clusters") or []:
        if not isinstance(cluster, dict) or cluster.get("graph_compound_parent_id") != cluster_id:
            continue
        # Podcast topics (members) and news-only topics that joined by meaning (news_members,
        # themes schema 3) both carry news_item_ids.
        for member in [*(cluster.get("members") or []), *(cluster.get("news_members") or [])]:
            if isinstance(member, dict):
                for item_id in member.get("news_item_ids") or []:
                    if isinstance(item_id, str) and item_id:
                        ids[item_id] = None
    return list(ids)


def _item_ids_by_topic(root: Path) -> Dict[str, List[str]]:
    """``topic_id -> news_item_ids`` from the themes file, the one per-topic source of news."""
    from podcast_scraper.server.corpus_access import cached_json_artifact

    doc = cached_json_artifact(root, _THEMES_FILE) or {}
    data = doc.get("data")
    holder: Dict[str, Any] = data if isinstance(data, dict) else doc
    by_topic: Dict[str, Dict[str, None]] = {}
    for cluster in holder.get("clusters") or []:
        if not isinstance(cluster, dict):
            continue
        for member in [*(cluster.get("members") or []), *(cluster.get("news_members") or [])]:
            if not isinstance(member, dict) or not isinstance(member.get("topic_id"), str):
                continue
            ids = by_topic.setdefault(member["topic_id"], {})
            for item_id in member.get("news_item_ids") or []:
                if isinstance(item_id, str) and item_id:
                    ids[item_id] = None
    return {topic: list(ids) for topic, ids in by_topic.items()}


#: The storylines' membership (Common's theme-clusters enrichment, under ``data``).
_STORYLINES_FILE = "enrichments/topic_theme_clusters.json"


def _storyline_topics(root: Path, storyline_id: str) -> List[str]:
    """The member topics of *storyline_id*, read as Common's ``storyline_map_by_topic`` reads them
    (each member's ``topic_id``; a topic in two storylines belongs to the last), so the set is the
    one ``relational_view._build_cluster_card`` walks for the episodes, without needing Common
    installed."""
    from podcast_scraper.server.corpus_access import cached_json_artifact

    doc = cached_json_artifact(root, _STORYLINES_FILE) or {}
    data = doc.get("data")
    holder: Dict[str, Any] = data if isinstance(data, dict) else doc
    owner: Dict[str, str] = {}
    for cluster in holder.get("clusters") or []:
        gpid = cluster.get("graph_compound_parent_id") if isinstance(cluster, dict) else None
        if not isinstance(gpid, str) or not gpid.strip():
            continue
        for member in cluster.get("members") or []:
            tid = member.get("topic_id") if isinstance(member, dict) else None
            if isinstance(tid, str) and tid.strip():
                owner[tid.strip()] = gpid.strip()
    return [topic for topic, sid in owner.items() if sid == storyline_id]


def _storyline_item_ids(root: Path, storyline_id: str) -> List[str]:
    """A storyline's articles by the rule its episodes and top voices follow (next-cycle item 25,
    the operator 2026-11-24: one rule for everything a storyline shows): the union over its member
    topics of each topic's articles from the themes file. The storyline file's own
    ``news_item_ids`` covered 45 articles on prod against 253 for the same topics there."""
    by_topic = _item_ids_by_topic(root)
    ids: Dict[str, None] = {}
    for topic_id in _storyline_topics(root, storyline_id):
        for item_id in by_topic.get(topic_id, []):
            ids[item_id] = None
    return list(ids)


def news_items_in_cluster(root: Path, cluster_id: str) -> List[Dict[str, Any]]:
    """The shown news items of a theme (``tc:``) or storyline (``thc:``), newest first: a theme's
    members' ``news_item_ids``; a storyline's member topics' articles (``_storyline_item_ids``). A
    ``serveable: none`` item is never among them."""
    if cluster_id.startswith("thc:"):
        wanted = set(_storyline_item_ids(root, cluster_id))
    else:
        wanted = set(_cluster_item_ids(root, cluster_id))
    return [i for i in cached_served_items(root) if i["item_id"] in wanted]


def corpus_has_news(root: Path) -> bool:
    """Whether the corpus holds a news item some surface may show."""
    return bool(cached_served_items(root))
