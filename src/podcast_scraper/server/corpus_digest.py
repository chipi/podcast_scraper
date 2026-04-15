"""Corpus digest selection (RFC-068): time windows, feed diversity, topic config."""

from __future__ import annotations

import calendar
import re
from collections import defaultdict, deque
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from podcast_scraper.graph_id_utils import slugify_label, topic_node_id_from_slug
from podcast_scraper.search.corpus_scope import normalize_feed_id
from podcast_scraper.server.corpus_catalog import (
    CatalogEpisodeRow,
    episode_list_summary_preview,
    resolve_feed_description,
    resolve_feed_display_title,
    resolve_feed_rss_url,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_DIGEST_TOPICS: list[dict[str, str]] = [
    {
        "id": "science",
        "label": "Science & research",
        "query": "scientific research discovery experiment",
    },
    {
        "id": "technology",
        "label": "Technology",
        "query": "software artificial intelligence technology engineering",
    },
]

DIGEST_MAX_TOPICS_PER_REQUEST = 5
DIGEST_TOPIC_SEARCH_TIMEOUT_SEC = 0.8
DIGEST_TOPIC_SEARCH_TOP_K = 24


def digest_topics_config_path() -> Path:
    """Filesystem path to optional ``config/digest_topics.yaml`` (RFC-068)."""
    return _REPO_ROOT / "config" / "digest_topics.yaml"


def load_digest_topics() -> list[dict[str, str]]:
    """Return topic dicts with keys id, label, query (RFC-068)."""
    path = digest_topics_config_path()
    if not path.is_file():
        return list(DEFAULT_DIGEST_TOPICS)
    try:
        import yaml

        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return list(DEFAULT_DIGEST_TOPICS)
    if not isinstance(raw, dict):
        return list(DEFAULT_DIGEST_TOPICS)
    items = raw.get("topics")
    if not isinstance(items, list):
        return list(DEFAULT_DIGEST_TOPICS)
    out: list[dict[str, str]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        tid = it.get("id")
        label = it.get("label")
        query = it.get("query")
        if (
            isinstance(tid, str)
            and tid.strip()
            and isinstance(label, str)
            and label.strip()
            and isinstance(query, str)
            and query.strip()
        ):
            out.append({"id": tid.strip(), "label": label.strip(), "query": query.strip()})
    return out or list(DEFAULT_DIGEST_TOPICS)


def utc_bounds_for_window(
    window: str,
    *,
    since: Optional[str],
    now_utc: datetime,
) -> tuple[datetime, datetime]:
    """Return inclusive-ish window [start, end] in UTC (end = now)."""
    end = now_utc.astimezone(timezone.utc)
    if window == "all":
        start = datetime(1970, 1, 1, tzinfo=timezone.utc)
    elif window == "24h":
        start = end - timedelta(hours=24)
    elif window == "7d":
        start = end - timedelta(days=7)
    elif window == "1mo":
        # Previous calendar month in UTC (first day 00:00 through last day end).
        d = end.date()
        first_this_month = date(d.year, d.month, 1)
        last_day_prev = first_this_month - timedelta(days=1)
        first_prev = date(last_day_prev.year, last_day_prev.month, 1)
        start = datetime(
            first_prev.year,
            first_prev.month,
            first_prev.day,
            tzinfo=timezone.utc,
        )
        y, m = last_day_prev.year, last_day_prev.month
        last_dom = calendar.monthrange(y, m)[1]
        end = datetime(y, m, last_dom, 23, 59, 59, 999999, tzinfo=timezone.utc)
    elif window == "since":
        if not since or not re.match(r"^\d{4}-\d{2}-\d{2}$", since.strip()):
            raise ValueError("since_required")
        d = date.fromisoformat(since.strip()[:10])
        start = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
        if start > end:
            start = end
    else:
        raise ValueError("bad_window")
    return start, end


def episode_in_utc_window(row: CatalogEpisodeRow, start: datetime, end: datetime) -> bool:
    """True when publish_date (UTC midnight) lies within [start, end]."""
    if not row.publish_date:
        return False
    try:
        d = date.fromisoformat(row.publish_date[:10])
    except ValueError:
        return False
    ep_dt = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    return start <= ep_dt <= end


def filter_rows_in_window(
    rows: list[CatalogEpisodeRow],
    start: datetime,
    end: datetime,
) -> list[CatalogEpisodeRow]:
    """Preserve catalog order (newest-first within global sort)."""
    return [r for r in rows if episode_in_utc_window(r, start, end)]


def diversify_digest_rows(
    rows_newest_first: list[CatalogEpisodeRow],
    *,
    max_rows: int,
    per_feed_cap: int,
) -> list[CatalogEpisodeRow]:
    """Round-robin by feed_id on a newest-first stream (RFC-068)."""
    if max_rows <= 0 or per_feed_cap <= 0:
        return []

    buckets: dict[str, list[CatalogEpisodeRow]] = defaultdict(list)
    feed_order: list[str] = []
    seen: set[str] = set()
    for r in rows_newest_first:
        buckets[r.feed_id].append(r)
        if r.feed_id not in seen:
            seen.add(r.feed_id)
            feed_order.append(r.feed_id)

    feed_queues: dict[str, deque[CatalogEpisodeRow]] = {
        fid: deque(buckets[fid]) for fid in feed_order
    }
    counts: dict[str, int] = {}
    out: list[CatalogEpisodeRow] = []
    while len(out) < max_rows:
        progressed = False
        for fid in feed_order:
            if len(out) >= max_rows:
                break
            if counts.get(fid, 0) >= per_feed_cap:
                continue
            dq = feed_queues.get(fid)
            if not dq:
                continue
            out.append(dq.popleft())
            counts[fid] = counts.get(fid, 0) + 1
            progressed = True
        if not progressed:
            break
    return out


def digest_row_dict(
    row: CatalogEpisodeRow,
    *,
    feed_titles_by_feed_id: Optional[Mapping[str, str]] = None,
    feed_rss_urls_by_feed_id: Optional[Mapping[str, str]] = None,
    feed_descriptions_by_feed_id: Optional[Mapping[str, str]] = None,
) -> dict[str, Any]:
    """JSON-serializable digest row for API responses (glance and full digest)."""
    bullets = list(row.summary_bullets[:4])
    bullet_graph_topic_ids = [
        topic_node_id_from_slug(slugify_label(str(b) if b is not None else "")) for b in bullets
    ]
    titles = feed_titles_by_feed_id or {}
    rss_by = feed_rss_urls_by_feed_id or {}
    desc_by = feed_descriptions_by_feed_id or {}
    display_title = resolve_feed_display_title(row, titles)
    return {
        "metadata_relative_path": row.metadata_relative_path,
        "feed_id": row.feed_id,
        "feed_display_title": display_title,
        "feed_rss_url": resolve_feed_rss_url(row, rss_by),
        "feed_description": resolve_feed_description(row, desc_by),
        "episode_id": row.episode_id,
        "episode_title": row.episode_title,
        "publish_date": row.publish_date,
        "summary_title": row.summary_title,
        "summary_bullets_preview": bullets,
        "summary_bullet_graph_topic_ids": bullet_graph_topic_ids,
        "summary_preview": episode_list_summary_preview(row),
        "has_gi": row.has_gi,
        "has_kg": row.has_kg,
        "gi_relative_path": row.gi_relative_path,
        "kg_relative_path": row.kg_relative_path,
        "feed_image_url": row.feed_image_url,
        "episode_image_url": row.episode_image_url,
        "duration_seconds": row.duration_seconds,
        "episode_number": row.episode_number,
        "feed_image_local_relpath": row.feed_image_local_relpath,
        "episode_image_local_relpath": row.episode_image_local_relpath,
    }


def since_str_for_search(start: datetime) -> str:
    """YYYY-MM-DD for GET /api/search ``since`` (UTC start-of-day)."""
    return start.astimezone(timezone.utc).date().isoformat()


def lookup_scope_index(
    rows: list[CatalogEpisodeRow],
) -> dict[tuple[str, str], CatalogEpisodeRow]:
    """Map (normalized feed_id, episode_id) -> catalog row."""
    out: dict[tuple[str, str], CatalogEpisodeRow] = {}
    for row in rows:
        if not row.episode_id:
            continue
        fn = normalize_feed_id(row.feed_id) or ""
        key = (fn, str(row.episode_id).strip())
        out[key] = row
    return out


def meta_episode_key(meta: dict[str, Any]) -> Optional[tuple[str, str]]:
    """Map search hit metadata to ``(normalized_feed_id, episode_id)`` for catalog joins."""
    ep = meta.get("episode_id")
    if not isinstance(ep, str) or not ep.strip():
        return None
    fn = normalize_feed_id(meta.get("feed_id"))
    return (fn or "", ep.strip())
