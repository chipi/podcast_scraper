"""GET /api/app/search — library-wide grounded search for the consumer platform (#1068).

Extractive grounded retrieval over the existing hybrid index (RFC-090); **no request-time
LLM** (D6). Spans the whole shared corpus by default; ``scope=mine`` (P3 Consolidation, #1120)
restricts the result set to the signed-in user's heard∪captured episodes — grounded recall over
the user's own experience, with honest zero-coverage. Episode-scoped search lives in
``app_episodes``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from podcast_scraper.search.capability import structured_corpus_search
from podcast_scraper.search.corpus_similar import episode_scope_key
from podcast_scraper.search.query_log import append_query_event
from podcast_scraper.search.theme_clusters import STORYLINE_DOC_TYPE
from podcast_scraper.server.app_artwork import artwork_url
from podcast_scraper.server.app_catalog_cache import cached_catalog
from podcast_scraper.server.app_search_view import build_search_response
from podcast_scraper.server.app_slugs import slug_for_row
from podcast_scraper.server.app_user_corpus import user_episode_set
from podcast_scraper.server.app_user_store import User
from podcast_scraper.server.corpus_catalog import (
    index_rows_by_feed_episode,
)
from podcast_scraper.server.query_enricher_helper import apply_query_enrichers
from podcast_scraper.server.routes.app_auth import get_current_user
from podcast_scraper.server.schemas import CorpusSearchApiResponse

router = APIRouter(tags=["app"])


def _corpus_root(request: Request) -> Path:
    anchor = getattr(request.app.state, "output_dir", None)
    if anchor is None:
        raise HTTPException(status_code=503, detail="No corpus configured for the platform API.")
    return Path(anchor)


def _attach_consumer_slugs(root: Path, resp: CorpusSearchApiResponse) -> None:
    """Add ``episode_slug`` / titles to each hit's metadata so the client can jump to the
    episode + moment (corpus search hits carry feed/episode ids, not the consumer slug)."""
    if not resp.results:
        return
    by_scope = index_rows_by_feed_episode(cached_catalog(root))
    for hit in resp.results:
        row = by_scope.get(episode_scope_key(hit.metadata) or ("", ""))
        if row is not None:
            local_art = row.episode_image_local_relpath or row.feed_image_local_relpath
            hit.metadata = {
                **hit.metadata,
                "episode_slug": slug_for_row(row),
                "episode_title": row.episode_title,
                "podcast_title": row.feed_title or "",
                # Small show/episode artwork so search results read like library cards.
                "episode_artwork": (
                    artwork_url(local_art, "thumb") or row.episode_image_url or row.feed_image_url
                ),
            }


def _storyline_slugs(root: Path, meta: dict) -> set[str]:
    """Consumer slugs for the episodes a storyline draws on.

    ``scope=mine`` is expressed in slugs; the theme-cluster artifact records episode IDS. The
    catalogue bridges them. Keyed by episode id alone because a cluster member records no feed.
    """
    raw = meta.get("storyline_episode_ids")
    if not isinstance(raw, list) or not raw:
        return set()
    wanted = {str(x) for x in raw if isinstance(x, str) and x.strip()}
    if not wanted:
        return set()
    return {
        slug_for_row(row)
        for row in cached_catalog(root)
        if row.episode_id and row.episode_id in wanted
    }


def _in_listening_scope(root: Path, hit: object, mine: set[str]) -> bool:
    """Whether a hit belongs to the user's heard∪captured set.

    A passage is in scope when ITS episode is. A STORYLINE has no episode, so filtering it by
    ``episode_slug`` dropped every one of them from recall mode — silently, as a consequence of
    filtering an episode-less row by episode rather than a decision anyone made. That hid the
    result type best suited to recall: a storyline is the shape of a recurring thread across things
    you have actually heard (operator 2026-09-17).

    ANY overlap counts — one episode of the cluster's thirty-six makes the storyline yours
    (operator's call). The alternative, a threshold, makes a storyline near-unreachable in the scope
    where it is most useful.
    """
    meta = getattr(hit, "metadata", None)
    if not isinstance(meta, dict):
        return False
    if meta.get("doc_type") == STORYLINE_DOC_TYPE:
        return bool(_storyline_slugs(root, meta) & mine)
    return meta.get("episode_slug") in mine


def _data_dir(request: Request) -> Path | None:
    raw = getattr(request.app.state, "app_data_dir", None)
    return Path(raw) if raw is not None else None


@router.get("/search", response_model=CorpusSearchApiResponse)
async def app_search(
    request: Request,
    q: str = Query(min_length=1, description="Natural-language query."),
    top_k: int = Query(default=10, ge=1, le=100),
    grounded_only: bool = Query(default=False),
    scope: Literal["all", "mine"] = Query(
        default="all",
        description="'all' = the shared corpus; 'mine' = the signed-in user's heard∪captured set.",
    ),
    enrich_results: bool = Query(
        default=False,
        description=(
            "RFC-088 Phase 4: run the QueryEnricher chain (currently "
            "``query_topic_relatedness``) over results so each hit carries "
            "``metadata.query_enrichments.related_topics``. Chain failures are "
            "swallowed — response degrades to the plain top-k page."
        ),
    ),
    user: User = Depends(get_current_user),
) -> CorpusSearchApiResponse:
    """Grounded library-wide search (extractive grounded passages; no request-time LLM).

    ``scope=mine`` (P3) filters hits to the user's heard∪captured episodes (RFC-101 §1). It requires
    a signed-in user (401 otherwise) and returns honest zero-coverage — an empty result set when the
    user's corpus has nothing on ``q`` — never widening back to the global corpus.
    """
    root = _corpus_root(request)
    mine: set[str] | None = None
    if scope == "mine":
        if user is None:
            raise HTTPException(status_code=401, detail="Sign in to search your corpus.")
        data_dir = _data_dir(request)
        mine = user_episode_set(root, data_dir, user.user_id) if data_dir is not None else set()
        if not mine:
            # The user has heard/captured nothing yet — honest empty, no global fallback.
            return build_search_response(q, {"query": q, "results": []})
    # Over-fetch when scoping so post-filtering to the user's set still fills the page.
    fetch_k = min(top_k * 5, 100) if mine is not None else top_k
    # Offload the blocking, CPU-bound search (query embedding + LanceDB) off the event loop so it
    # doesn't freeze concurrent requests — the first call also lazy-loads MiniLM (~seconds). Safe
    # under concurrency: index_pool serves a warm, shared LanceDB backend (the pool is what removed
    # the #1205 concurrent-cold-init SIGSEGV).
    outcome = await asyncio.to_thread(
        structured_corpus_search, root, q, top_k=fetch_k, grounded_only=grounded_only
    )
    if not outcome.get("error"):
        append_query_event(root, str(outcome.get("query_type") or ""))
    resp = build_search_response(q, outcome)
    _attach_consumer_slugs(root, resp)
    if mine is not None:
        resp.results = [r for r in resp.results if _in_listening_scope(root, r, mine)][:top_k]
    # The cluster's episode ids were carried only to answer the scope question. Strip them: a
    # storyline can span dozens of episodes and the client has no use for the raw ids.
    for r in resp.results:
        if isinstance(r.metadata, dict) and "storyline_episode_ids" in r.metadata:
            r.metadata = {k: v for k, v in r.metadata.items() if k != "storyline_episode_ids"}
    if enrich_results and resp.results:
        await apply_query_enrichers(request, root, q, resp.results)
    return resp
