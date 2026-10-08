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
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from podcast_scraper.search.capability import structured_corpus_search
from podcast_scraper.search.corpus_similar import episode_scope_key
from podcast_scraper.search.query_log import append_query_event
from podcast_scraper.search.router import classify_query, tier_for_doc_type
from podcast_scraper.search.storylines import (
    STORYLINE_DOC_TYPE,
    storyline_episode_ids,
    storyline_index_rows,
    storyline_lifecycle,
)
from podcast_scraper.server.app_episode_search import time_hits_by_episode
from podcast_scraper.server.app_moments import episode_has_moments
from podcast_scraper.server.app_search_view import build_search_response
from podcast_scraper.server.app_user_corpus import world_episode_set
from podcast_scraper.server.app_user_store import User
from podcast_scraper.server.artwork import artwork_url
from podcast_scraper.server.catalog_cache import cached_catalog
from podcast_scraper.server.corpus_catalog import (
    index_rows_by_feed_episode,
)
from podcast_scraper.server.kg_index import normalize_label
from podcast_scraper.server.query_enricher_helper import apply_query_enrichers
from podcast_scraper.server.routes.app_auth import get_current_user
from podcast_scraper.server.schemas import CorpusSearchApiResponse, SearchCategoryModel
from podcast_scraper.server.slugs import slug_for_row

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
                "episode_has_gi": bool(row.has_gi),
                # Whether the result can offer the episode's Moments reel (same rule as the reel).
                "episode_has_moments": episode_has_moments(root, row),
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


def _world_search(
    root: Path, q: str, mine: set[str], *, top_k: int, grounded_only: bool
) -> dict[str, Any]:
    """Search the listener's world with the world IN the query (operator 2026-10-10).

    It used to rank the whole corpus, keep a few hundred, and keep the listener's episodes after —
    on a large corpus their own episodes' matches never made the cut and "Mine" came back thin or
    empty. Now two scoped queries: the world's episodes (their ids as a prefilter), and storylines
    (a storyline has no single episode, so it cannot match an episode filter; it belongs to the
    world when any episode it draws on does — ``_in_listening_scope``, applied by the caller).
    Merged by score.
    """
    ids = _world_episode_ids(root, mine)
    episodes = structured_corpus_search(
        root, q, top_k=top_k, grounded_only=grounded_only, episode_ids=ids
    )
    if episodes.get("error"):
        return episodes
    storylines = structured_corpus_search(root, q, top_k=top_k, doc_types=[STORYLINE_DOC_TYPE])
    story_rows = [
        r
        for r in storylines.get("results") or []
        if (r.get("metadata") or {}).get("doc_type") == STORYLINE_DOC_TYPE
    ]
    merged = [*(episodes.get("results") or []), *story_rows]
    merged.sort(key=lambda r: float(r.get("score") or 0.0), reverse=True)
    return {**episodes, "results": merged}


#: The player's search in category mode (operator 2026-10-13): each kind of result is its own
#: capped search, so no kind crowds out the others. A topic row exists per episode per topic, and
#: a common word ("risk") filled all twelve slots with topic rows, which the player lists by NAME
#: above the results and drops from them, so the search showed no episode at all. Topic and person
#: rows are not asked for here: the subjects at the top come from the name search.
SEARCH_CATEGORIES: dict[str, tuple[str, ...]] = {
    # Not quotes: each insight's quote is the same moment again (on the fixture word for word), so
    # the section listed every insight twice.
    "insights": ("insight",),
    "transcript": ("transcript",),
    "episodes": ("episode_title", "episode_description", "summary_short", "summary"),
    "mentions": ("kg_entity",),
    STORYLINE_DOC_TYPE + "s": (STORYLINE_DOC_TYPE,),
}


def _storylines_named(root: Path, q: str) -> list[dict[str, Any]]:
    """Storylines whose current name or a name they had before (RFC-128) IS the query, as search
    rows, so they lead the storylines category.

    A storyline is named after one of its member topics, so its name is always a topic's name too:
    the name resolver opens the topic, and the storyline is left to the embedding search, which
    ranked "model distillation" below three other storylines and missed an old name outright
    (prod, 2026-10-15: 29 of 33 names first, 2 lower, 2 missing). A name typed exactly is not a
    similarity question.
    """
    norm = normalize_label(q)
    if not norm:
        return []
    lifecycle = storyline_lifecycle(root)
    named: list[tuple[str, str, dict[str, Any]]] = []
    for doc_id, text, meta in storyline_index_rows(root):
        sid = str(meta.get("source_id") or "")
        names = [
            str(meta.get("storyline_label") or ""),
            *(lifecycle.get(sid) or {}).get("aliases", []),
        ]
        if any(normalize_label(n) == norm for n in names):
            named.append((doc_id, text, meta))
    if not named:
        return []
    episodes = storyline_episode_ids(root)
    return [
        {
            "doc_id": doc_id,
            "score": 1.0,
            "metadata": {
                **meta,
                "storyline_episode_ids": sorted(episodes.get(str(meta.get("source_id")), ())),
            },
            "text": text,
            "source_tier": tier_for_doc_type(STORYLINE_DOC_TYPE),
        }
        for doc_id, text, meta in named
    ]


def _category_page(
    root: Path,
    q: str,
    category: str,
    *,
    offset: int,
    limit: int,
    grounded_only: bool,
    episode_ids: list[str] | None,
) -> tuple[list[dict[str, Any]], bool, dict[str, Any] | None]:
    """One category's results ``[offset, offset + limit)``, whether more follow, and the error
    outcome when the search failed. A storyline has no episode, so it is never episode-scoped
    (the caller applies ``_in_listening_scope``); a person is listed by name, never here."""
    doc_types = SEARCH_CATEGORIES[category]
    storyline = doc_types == (STORYLINE_DOC_TYPE,)
    want = offset + limit + 1
    # Mentions drop person rows after the search, so ask for more to fill the page.
    fetch = min(100, want * 3 if category == "mentions" else want)
    outcome = structured_corpus_search(
        root,
        q,
        top_k=fetch,
        grounded_only=grounded_only,
        doc_types=list(doc_types),
        episode_ids=None if storyline else episode_ids,
    )
    if outcome.get("error"):
        return [], False, outcome
    rows = [
        r
        for r in outcome.get("results") or []
        if (r.get("metadata") or {}).get("doc_type") in doc_types
        and not str((r.get("metadata") or {}).get("source_id") or "").startswith("person:")
    ]
    if storyline:
        pinned = _storylines_named(root, q)
        if pinned:
            sids = {p["metadata"]["source_id"] for p in pinned}
            rows = pinned + [
                r for r in rows if (r.get("metadata") or {}).get("source_id") not in sids
            ]
    for r in rows:
        r["metadata"] = {**(r.get("metadata") or {}), "search_category": category}
    return rows[offset : offset + limit], len(rows) > offset + limit, None


def _categories_search(
    root: Path,
    q: str,
    *,
    categories: list[str],
    offset: int,
    limit: int,
    grounded_only: bool,
    episode_ids: list[str] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Each category's page, concatenated in category order, and what each holds."""
    results: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    query_type = ""
    for category in categories:
        rows, more, error = _category_page(
            root,
            q,
            category,
            offset=offset,
            limit=limit,
            grounded_only=grounded_only,
            episode_ids=episode_ids,
        )
        if error is not None:
            return error, []
        results.extend(rows)
        summary.append({"id": category, "returned": len(rows), "has_more": more})
    if categories:
        query_type = classify_query(q)
    return {"query": q, "results": results, "query_type": query_type}, summary


def _world_episode_ids(root: Path, mine: set[str]) -> list[str]:
    return sorted(
        {
            row.episode_id
            for row in cached_catalog(root)
            if row.episode_id and slug_for_row(row) in mine
        }
    )


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
        description=(
            "'all' = the shared corpus; 'mine' = the signed-in user's world: episodes heard or "
            "captured from, saved, or of a show they follow."
        ),
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
    per_category: int | None = Query(
        default=None,
        ge=1,
        le=50,
        description=(
            "Category mode (the player): each kind of result (``SEARCH_CATEGORIES``) is its own "
            "search capped at this many, and ``categories`` says which have more. Unset: the "
            "plain top-k page."
        ),
    ),
    category: str | None = Query(
        default=None,
        description='Category mode: only this category ("Show more"), from ``offset``.',
    ),
    offset: int = Query(default=0, ge=0, le=90),
    user: User = Depends(get_current_user),
) -> CorpusSearchApiResponse:
    """Grounded library-wide search (extractive grounded passages; no request-time LLM).

    ``scope=mine`` filters hits to the episodes of the user's world — heard or captured from,
    saved, or of a show they follow (:func:`world_episode_set`, 2026-10-09; was heard∪captured).
    It requires a signed-in user (401 otherwise) and returns honest zero-coverage — an empty result
    set when the user's world has nothing on ``q`` — never widening back to the global corpus.
    """
    root = _corpus_root(request)
    mine: set[str] | None = None
    if scope == "mine":
        if user is None:
            raise HTTPException(status_code=401, detail="Sign in to search your corpus.")
        data_dir = _data_dir(request)
        mine = world_episode_set(root, data_dir, user.user_id) if data_dir is not None else set()
        if not mine:
            # Nothing in the user's world yet — honest empty, no global fallback.
            return build_search_response(q, {"query": q, "results": []})
    if category is not None and category not in SEARCH_CATEGORIES:
        raise HTTPException(status_code=422, detail=f"Unknown search category: {category}")
    # Offload the blocking, CPU-bound search (query embedding + LanceDB) off the event loop so it
    # doesn't freeze concurrent requests — the first call also lazy-loads MiniLM (~seconds). Safe
    # under concurrency: index_pool serves a warm, shared LanceDB backend (the pool is what removed
    # the #1205 concurrent-cold-init SIGSEGV).
    summary: list[dict[str, Any]] | None = None
    if per_category is not None or category is not None:
        outcome, summary = await asyncio.to_thread(
            _categories_search,
            root,
            q,
            categories=[category] if category else list(SEARCH_CATEGORIES),
            offset=offset if category else 0,
            limit=per_category or 3,
            grounded_only=grounded_only,
            episode_ids=_world_episode_ids(root, mine) if mine is not None else None,
        )
    elif mine is not None:
        outcome = await asyncio.to_thread(
            _world_search, root, q, mine, top_k=top_k, grounded_only=grounded_only
        )
    else:
        outcome = await asyncio.to_thread(
            structured_corpus_search, root, q, top_k=top_k, grounded_only=grounded_only
        )
    if not outcome.get("error"):
        append_query_event(root, str(outcome.get("query_type") or ""))
    resp = build_search_response(q, outcome)
    _attach_consumer_slugs(root, resp)
    # A real time on every transcript result (operator 2026-10-10): the index stores none, so
    # "Play from" jumped to 0:00. Found by the result's words in its episode's timed transcript.
    await asyncio.to_thread(time_hits_by_episode, root, resp.results, cached_catalog(root))
    if mine is not None:
        kept = [r for r in resp.results if _in_listening_scope(root, r, mine)]
        resp.results = kept if summary is not None else kept[:top_k]
    if summary is not None and not resp.error:
        counts: dict[str, int] = {}
        for r in resp.results:
            cat = str((r.metadata or {}).get("search_category") or "")
            counts[cat] = counts.get(cat, 0) + 1
        resp.categories = [
            SearchCategoryModel(id=c["id"], returned=counts.get(c["id"], 0), has_more=c["has_more"])
            for c in summary
        ]
    # The cluster's episode ids were carried only to answer the scope question. Strip them: a
    # storyline can span dozens of episodes and the client has no use for the raw ids.
    for r in resp.results:
        if isinstance(r.metadata, dict) and "storyline_episode_ids" in r.metadata:
            r.metadata = {k: v for k, v in r.metadata.items() if k != "storyline_episode_ids"}
    if enrich_results and resp.results:
        await apply_query_enrichers(request, root, q, resp.results)
    return resp
