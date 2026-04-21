"""GET /api/corpus/coverage — GI/KG artifact presence by month and feed (dashboard)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from fastapi import APIRouter, Query, Request

from podcast_scraper.server.corpus_catalog import build_catalog_rows
from podcast_scraper.server.pathutil import resolved_corpus_root_str
from podcast_scraper.server.routes.corpus_library import _resolve_corpus_root
from podcast_scraper.server.schemas import (
    CorpusCoverageResponse,
    CoverageByMonthItem,
    CoverageFeedItem,
)

router = APIRouter(tags=["corpus"])


@dataclass
class _MonthAgg:
    total: int = 0
    with_gi: int = 0
    with_kg: int = 0
    with_both: int = 0


@dataclass
class _FeedAgg:
    total: int = 0
    with_gi: int = 0
    with_kg: int = 0
    display_title: str = ""


@router.get("/corpus/coverage", response_model=CorpusCoverageResponse)
async def corpus_coverage(
    request: Request,
    path: str | None = Query(
        default=None,
        description="Corpus root. Omit to use server default output_dir.",
    ),
) -> CorpusCoverageResponse:
    """Scan catalog once; check sibling ``*.gi.json`` / ``*.kg.json`` existence per episode."""
    anchor = getattr(request.app.state, "output_dir", None)
    root = _resolve_corpus_root(path, anchor)
    root_safe = resolved_corpus_root_str(root, anchor)
    rows = build_catalog_rows(root)

    total_episodes = len(rows)
    with_gi = sum(1 for r in rows if r.has_gi)
    with_kg = sum(1 for r in rows if r.has_kg)
    with_both = sum(1 for r in rows if r.has_gi and r.has_kg)
    with_neither = sum(1 for r in rows if not r.has_gi and not r.has_kg)

    by_month_map: Dict[str, _MonthAgg] = {}
    for r in rows:
        pd = r.publish_date
        if not pd or len(pd) < 7:
            continue
        month = pd[:7]
        agg = by_month_map.setdefault(month, _MonthAgg())
        agg.total += 1
        if r.has_gi:
            agg.with_gi += 1
        if r.has_kg:
            agg.with_kg += 1
        if r.has_gi and r.has_kg:
            agg.with_both += 1

    by_month = [
        CoverageByMonthItem(
            month=m,
            total=a.total,
            with_gi=a.with_gi,
            with_kg=a.with_kg,
            with_both=a.with_both,
        )
        for m, a in sorted(by_month_map.items(), key=lambda kv: kv[0])
    ]

    by_feed_map: Dict[str, _FeedAgg] = {}
    for r in rows:
        fid = (r.feed_id or "").strip() or "(unknown)"
        feed_agg = by_feed_map.setdefault(fid, _FeedAgg())
        feed_agg.total += 1
        if r.has_gi:
            feed_agg.with_gi += 1
        if r.has_kg:
            feed_agg.with_kg += 1
        ft = r.feed_title
        if isinstance(ft, str) and ft.strip() and not feed_agg.display_title:
            feed_agg.display_title = ft.strip()

    by_feed = [
        CoverageFeedItem(
            feed_id=fid,
            display_title=feed_agg.display_title or fid,
            total=feed_agg.total,
            with_gi=feed_agg.with_gi,
            with_kg=feed_agg.with_kg,
        )
        for fid, feed_agg in by_feed_map.items()
    ]
    by_feed.sort(key=lambda x: ((x.with_gi / x.total) if x.total else 0.0, x.feed_id))

    return CorpusCoverageResponse(
        path=str(root_safe),
        total_episodes=total_episodes,
        with_gi=with_gi,
        with_kg=with_kg,
        with_both=with_both,
        with_neither=with_neither,
        by_month=by_month,
        by_feed=by_feed,
    )
