"""Consumer knowledge-card routes (``/api/app/persons/*``, ``/api/app/topics/*``).

Dedicated, read-only person/topic cards for the end-user app (PRD-043 FR2/FR3; RFC-102;
#1095/#1096). KG-grounded projections of the single shared corpus — NOT a proxy of the
operator relational API (the consumer/operator boundary stays clean). Mounted under the
``/api/app`` namespace alongside the other consumer routes.
"""

from __future__ import annotations

import asyncio
import time
from collections import Counter
from pathlib import Path
from typing import Literal, TypeVar

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse

from podcast_scraper.server.app_corpus_access import corpus_root_or_503
from podcast_scraper.server.app_relational_view import (
    build_cluster_perspectives,
    build_org_card,
    build_person_card,
    build_storyline_card,
    build_theme_card,
    build_topic_card,
    build_topic_perspectives,
    resolve_entity,
)
from podcast_scraper.server.app_user_corpus import user_episode_set
from podcast_scraper.server.app_user_store import User
from podcast_scraper.server.routes.app_auth import get_current_user
from podcast_scraper.server.schemas import (
    AppClusterCard,
    AppEntitySearchResponse,
    AppMonthCount,
    AppOrgCard,
    AppPersonCard,
    AppTopicCard,
    AppTopicConversationArcResponse,
    AppTopicPerspectivesResponse,
    AppTopShow,
    CilTopicConversationArcWeek,
)

router = APIRouter(tags=["app"])

_Card = TypeVar("_Card", AppPersonCard, AppTopicCard)


def _user_set(request: Request, user: User | None) -> set[str]:
    """The signed-in user's heard∪captured slugs; 401 when ``scope=mine`` but signed out,
    503 when the user store isn't configured (so ``scope=mine`` doesn't masquerade as a 404
    empty result on a misconfigured server)."""
    if user is None:
        raise HTTPException(status_code=401, detail="Sign in to scope to your corpus.")
    root = corpus_root_or_503(request)
    data_dir = getattr(request.app.state, "app_data_dir", None)
    if data_dir is None:
        raise HTTPException(status_code=503, detail="User store is not configured.")
    return user_episode_set(root, Path(data_dir), user.user_id)


# Episodes PAGED on the server (operator 2026-10-08: "what's the point of paging if you're not
# actually paging?"). The cards returned every episode — a storyline 99 (1.29 MB on prod), a person
# 70 (416 KB) — and the client sliced to five. `episodes_limit` + `episodes_offset` page the list;
# `episode_count` stays the TOTAL, so "Show more (N)" is still right. Always paged: a request
# without `episodes_limit` gets the first 20. No pre-1.0.3 client is left (2026-10-10,
# docs/wip/TODO-remove-pre-1.0.3-compat.md).
_DEFAULT_EPISODES = 20
_EpisodesOffset = Query(default=0, ge=0, description="Skip this many episodes (paging).")
_EpisodesLimit = Query(
    default=_DEFAULT_EPISODES, ge=1, le=100, description="Return at most this many episodes."
)


_Paged = TypeVar("_Paged", AppTopicCard, AppPersonCard, AppOrgCard, AppClusterCard)


def _page_episodes(card: _Paged, offset: int, limit: int) -> _Paged:
    return card.model_copy(
        update={
            "episodes": card.episodes[offset : offset + limit],
            "episodes_total": len(card.episodes),
        }
    )


def _with_topic_aggregates(card: AppTopicCard) -> AppTopicCard:
    """Month counts and top shows over ALL the topic's episodes — the client drew both from the
    full list it no longer gets once the list is paged."""
    months: Counter[str] = Counter()
    shows: dict[str, AppTopShow] = {}
    for e in card.episodes:
        if e.publish_date and len(e.publish_date) >= 7:
            months[e.publish_date[:7]] += 1
        if e.feed_id:
            cur = shows.get(e.feed_id)
            if cur is None:
                shows[e.feed_id] = AppTopShow(
                    feed_id=e.feed_id,
                    title=e.podcast_title,
                    count=1,
                    artwork_url=e.feed_artwork_url,
                    image_url=e.feed_image_url,
                )
            else:
                cur.count += 1
                cur.artwork_url = cur.artwork_url or e.feed_artwork_url
    return card.model_copy(
        update={
            "episode_months": [AppMonthCount(month=m, count=c) for m, c in sorted(months.items())],
            "top_shows": sorted(shows.values(), key=lambda s: -s.count)[:5],
        }
    )


# Perspectives PAGED (2026-10-08): a storyline's perspectives were 160 KB and 1.1 s on prod — every
# speaker with every take — for a section that shows three speakers with two takes each.
# `insights_per_speaker` caps each speaker's takes (insight_count stays their total);
# `speakers_offset`/`speakers_limit` page the speakers (perspective_count stays the total). With
# none of them the full response is returned, as before.
_SpeakersOffset = Query(default=0, ge=0, description="Skip this many speakers (paging).")
_SpeakersLimit = Query(default=None, ge=1, le=100, description="At most this many speakers.")
_PerSpeaker = Query(
    default=None, ge=1, le=100, description="At most this many insights per speaker."
)


def _page_perspectives(
    resp: AppTopicPerspectivesResponse, offset: int, limit: int | None, per_speaker: int | None
) -> AppTopicPerspectivesResponse:
    if offset == 0 and limit is None and per_speaker is None:
        return resp
    end = None if limit is None else offset + limit
    speakers = resp.perspectives[offset:end]
    if per_speaker is not None:
        speakers = [p.model_copy(update={"insights": p.insights[:per_speaker]}) for p in speakers]
    return resp.model_copy(update={"perspectives": speakers})


def _scope_to_corpus(card: _Card, mine: set[str]) -> _Card:
    """Filter a card's appears-in episodes to the user's set (the "you heard X in …" lens),
    recomputing ``episode_count`` so the card reads honestly per RFC-101 §4. For a person card the
    per-show breakdown is re-scoped too — drop shows with no heard episode, recount the rest."""
    card.episodes = [e for e in card.episodes if e.slug in mine]
    card.episode_count = len(card.episodes)
    if isinstance(card, AppPersonCard):
        heard_by_feed = Counter(e.feed_id for e in card.episodes)
        card.shows = [
            s.model_copy(update={"episode_count": heard_by_feed[s.feed_id]})
            for s in card.shows
            if heard_by_feed.get(s.feed_id)
        ]
    return card


@router.get("/persons/{person_id}", response_model=AppPersonCard)
async def person_card(
    request: Request,
    person_id: str,
    scope: Literal["all", "mine"] = Query(default="all"),
    episodes_offset: int = _EpisodesOffset,
    episodes_limit: int = _EpisodesLimit,
    exclude_host_shows: bool = Query(
        default=False,
        description="Leave out episodes of the shows this person HOSTS (their back-catalogue), "
        "before paging — the card lists their appearances elsewhere.",
    ),
    user: User = Depends(get_current_user),
) -> AppPersonCard:
    """Person profile card: appears-in episodes + related people/topics (KG co-occurrence).

    404 when the person appears in no episode's KG (rather than an empty card), so the
    client can distinguish "unknown person" from "person with a thin footprint".

    ``scope=mine`` (P3 #1122) is the "your corpus" lens — the guest *across the episodes you have
    heard* ("you also heard them in …"); auth-gated (401 signed out).
    """
    root = corpus_root_or_503(request)
    # Offload the KG-grounded card build off the event loop — it iterates episode KGs and, even with
    # the catalog cached, must not block concurrent requests while it runs.
    card = await asyncio.to_thread(build_person_card, root, person_id.strip())
    if card is None:
        raise HTTPException(status_code=404, detail="Unknown person id.")
    if scope == "mine":
        card = _scope_to_corpus(card, _user_set(request, user))
    if exclude_host_shows:
        hosted = {sh.feed_id for sh in card.shows if (sh.role or "").lower() == "host"}
        if hosted:
            card = card.model_copy(
                update={"episodes": [e for e in card.episodes if e.feed_id not in hosted]}
            )
    return _page_episodes(card, episodes_offset, episodes_limit)


@router.get("/organizations/{org_id}", response_model=AppOrgCard)
async def org_card(
    request: Request,
    org_id: str,
    episodes_offset: int = _EpisodesOffset,
    episodes_limit: int = _EpisodesLimit,
    _user: User = Depends(get_current_user),
) -> AppOrgCard:
    """Organization card (#2031): mentioned-in episodes + co-occurring people/orgs/topics.

    KG-grounded via MENTIONS_ORG. Leaner than the person card — orgs have no web bio/photo. 404
    when the org appears in no episode's KG, so the client can tell "unknown org" from "thin
    footprint".
    """
    root = corpus_root_or_503(request)
    # Off the event loop — the build iterates episode KGs (same rationale as the person card).
    card = await asyncio.to_thread(build_org_card, root, org_id.strip())
    if card is None:
        raise HTTPException(status_code=404, detail="Unknown org id.")
    return _page_episodes(card, episodes_offset, episodes_limit)


@router.get("/organizations/{org_id}/logo")
def org_logo(request: Request, org_id: str) -> FileResponse:
    """Serve the org's self-hosted logo (org_web enricher, #2035).

    Deliberately UNAUTHENTICATED, same reason as ``person_photo`` above and ``serve_avatar``
    (#2109): an ``<img src=…>`` cannot send an ``Authorization`` header. Opened together with the
    person photo rather than left behind — the two are rendered by the same card surfaces, and
    fixing one while the other keeps 401'ing is how a defect class survives its own fix.

    The logo lives under ``enrichments/org_logos/`` (downloaded + license-validated at enrichment
    time); the stem is sanitized and the filename is a fixed glob, so the path cannot traverse out.
    404 when no logo is hosted (the common case — logos are often non-free)."""
    from podcast_scraper.enrichment.enrichers.org_web import org_logo_path

    root = corpus_root_or_503(request)
    found = org_logo_path(root, org_id.strip())
    if found is None:
        raise HTTPException(status_code=404, detail="No logo.")
    path, media = found
    # codeql[py/path-injection] -- org_id is sanitized to [a-z0-9._-] by _safe_name and the
    # filename is a fixed glob; nosniff so the browser can't reinterpret the allow-listed bytes.
    return FileResponse(
        path=str(path), media_type=media, headers={"X-Content-Type-Options": "nosniff"}
    )


@router.get("/persons/{person_id}/photo")
def person_photo(request: Request, person_id: str) -> FileResponse:
    """Serve the person's self-hosted photo (wave-G, person_web enricher).

    Deliberately UNAUTHENTICATED, for exactly the reason ``serve_avatar`` is (#2109, operator
    2026-09-16): an ``<img src=…>`` cannot send an ``Authorization`` header, and the native shell
    carries its session in precisely that header rather than a cookie. Session-gated, every photo
    401'd on device and ``ProfileAvatar`` silently fell back to initials — so 652 hosted photos
    existed on disk and not one of them ever appeared.

    What is exposed is already public: these are Wikipedia/Wikimedia images, fetched from public
    URLs and stored with their CC credit. Show and episode ARTWORK is already served
    unauthenticated by this same API, and the avatar — a USER's own upload — was opened on the
    same reasoning; a corpus person's public encyclopedia portrait is strictly less sensitive.

    The photo lives under ``enrichments/person_images/``; the stem is sanitized and matched
    against a directory enumeration, so the path cannot traverse out. 404 when none is hosted."""
    from podcast_scraper.enrichment.enrichers.person_web import person_image_path

    root = corpus_root_or_503(request)
    found = person_image_path(root, person_id.strip())
    if found is None:
        raise HTTPException(status_code=404, detail="No photo.")
    path, media = found
    # codeql[py/path-injection] -- person_id is sanitized to [a-z0-9._-] by _safe_name and the
    # filename is a fixed glob; nosniff so the browser can't reinterpret the allow-listed bytes.
    return FileResponse(
        path=str(path), media_type=media, headers={"X-Content-Type-Options": "nosniff"}
    )


@router.get("/topics/{topic_id}/perspectives", response_model=AppTopicPerspectivesResponse)
async def topic_perspectives_route(
    request: Request,
    topic_id: str,
    scope: Literal["all", "mine"] = Query(default="all"),
    speakers_offset: int = _SpeakersOffset,
    speakers_limit: int | None = _SpeakersLimit,
    insights_per_speaker: int | None = _PerSpeaker,
    user: User = Depends(get_current_user),
) -> AppTopicPerspectivesResponse:
    """Multi-perspective synthesis — each speaker's take on the topic (#1146).

    ``scope=mine`` (#1149) restricts to the episodes the user has heard∪captured; auth-gated
    (401 signed out). 404 when the topic has no speaker-attributable insight in the (scoped) GI.
    """
    root = corpus_root_or_503(request)
    mine = _user_set(request, user) if scope == "mine" else None
    resp = await asyncio.to_thread(
        build_topic_perspectives, root, topic_id.strip(), mine_slugs=mine
    )
    if resp is None:
        raise HTTPException(status_code=404, detail="No perspectives for this topic.")
    return _page_perspectives(resp, speakers_offset, speakers_limit, insights_per_speaker)


@router.get(
    "/topics/{topic_id}/conversation-arc",
    response_model=AppTopicConversationArcResponse,
)
async def topic_conversation_arc_route(
    request: Request,
    topic_id: str,
    _user: User = Depends(get_current_user),
) -> AppTopicConversationArcResponse:
    """Consumer conversation arc — weekly volume × sentiment for a topic (ADR-108).

    The aggregate-first overview so a big topic (1000s of insights) shows as a compact time-shape.
    KG-grounded read over the shared corpus; empty ``weeks`` when the topic has no dated insights.
    """
    from podcast_scraper.server import cil_queries

    root = str(corpus_root_or_503(request))
    tid = cil_queries.canonical_cil_entity_id(topic_id.strip())
    # Blocking os.walk + per-episode JSON reads — keep it off the event loop (audit #3).
    raw = await asyncio.to_thread(_conversation_arc, root, tid)
    return AppTopicConversationArcResponse(
        topic_id=tid,
        weeks=[CilTopicConversationArcWeek(**w) for w in raw],
    )


@router.get("/entities/search", response_model=AppEntitySearchResponse)
async def entity_search(
    request: Request,
    q: str = Query(min_length=1, description="Query to resolve to a person/topic entity."),
    _user: User = Depends(get_current_user),
) -> AppEntitySearchResponse:
    """Resolve a query to a person/topic card (PRD-043 FR3 / 3.4) — exact/near-exact name only.

    The consumer search view calls this alongside `/search` and renders the matched entity as a
    card above the passage results. Returns ``entity: null`` (200) when nothing matches.
    """
    root = corpus_root_or_503(request)
    entity = await asyncio.to_thread(resolve_entity, root, q)
    return AppEntitySearchResponse(query=q, entity=entity)


# The conversation arc is an uncached corpus-wide scan (``topic_timeline`` walks every episode
# bundle). The topic card needs its WEEK COUNT up front — so the card can decide whether the arc
# section exists before drawing it, rather than drawing a placeholder that then vanishes for most
# topics (#2202) — and the arc route needs the weeks themselves a moment later. Memoized per
# (corpus, topic) for a short window so the two requests share ONE scan. Brief staleness is fine:
# the arc only moves when the corpus is re-ingested.
_CONVERSATION_ARC_TTL_SECONDS = 30.0
_conversation_arc_cache: dict[tuple[str, str], tuple[float, list[dict]]] = {}


def _conversation_arc(root: str, topic_id: str) -> list[dict]:
    from podcast_scraper.server import cil_queries

    key = (root, topic_id)
    now = time.monotonic()
    cached = _conversation_arc_cache.get(key)
    if cached is not None and now - cached[0] < _CONVERSATION_ARC_TTL_SECONDS:
        return cached[1]
    weeks = cil_queries.topic_conversation_arc(root, root, topic_id)
    _conversation_arc_cache[key] = (now, weeks)
    return weeks


@router.get("/storylines/{storyline_id}", response_model=AppClusterCard)
async def storyline_card(
    request: Request,
    storyline_id: str,
    episodes_offset: int = _EpisodesOffset,
    episodes_limit: int = _EpisodesLimit,
    user: User = Depends(get_current_user),
) -> AppClusterCard:
    """Storyline card: the member topics, their MERGED episodes, and the people across them.

    Accepts EITHER the storyline's own `thc:` id or one of its member topics' ids. The second form
    keeps ``/storyline/:id`` working — that page routes by ANCHOR TOPIC, because it was built before
    any storyline endpoint existed and derived everything from the anchor's topic card.

    That derivation is what this replaces. The topic card's ``episodes`` are the ANCHOR's episodes,
    so the storyline page said "Discussed in 30 episodes" when the storyline actually spans 40 —
    it was showing one member's corpus and calling it the storyline's.

    404 when the id names neither a storyline nor a topic inside one.
    """
    root = corpus_root_or_503(request)
    card = await asyncio.to_thread(build_storyline_card, root, storyline_id.strip())
    if card is None:
        raise HTTPException(status_code=404, detail="Unknown storyline id.")
    return _page_episodes(card, episodes_offset, episodes_limit)


@router.get("/themes/{theme_id}", response_model=AppClusterCard)
async def theme_card(
    request: Request,
    theme_id: str,
    episodes_offset: int = _EpisodesOffset,
    episodes_limit: int = _EpisodesLimit,
    user: User = Depends(get_current_user),
) -> AppClusterCard:
    """Theme card: the member topics, their MERGED episodes, and the people across them.

    A theme is a grouping of topics that mean the same thing, not an entity — it is never a node on
    an episode. It therefore needs its own endpoint: ``/topics/{id}`` builds a card by matching a
    topic node by id, so a `tc:` id matched nothing and the page rendered empty.

    No ``scope`` parameter, deliberately. The topic card's ``scope=mine`` narrows to the user's
    heard set; a theme page answers "what is this grouping, across the corpus", and a
    personally-filtered union would quietly answer a different question. Add it when a surface
    actually asks.

    404 when the theme id is unknown or none of its members appear in any episode's KG.
    """
    root = corpus_root_or_503(request)
    card = await asyncio.to_thread(build_theme_card, root, theme_id.strip())
    if card is None:
        raise HTTPException(status_code=404, detail="Unknown theme id.")
    return _page_episodes(card, episodes_offset, episodes_limit)


@router.get("/storylines/{storyline_id}/perspectives", response_model=AppTopicPerspectivesResponse)
async def storyline_perspectives_route(
    request: Request,
    storyline_id: str,
    speakers_offset: int = _SpeakersOffset,
    speakers_limit: int | None = _SpeakersLimit,
    insights_per_speaker: int | None = _PerSpeaker,
    user: User = Depends(get_current_user),
) -> AppTopicPerspectivesResponse:
    """What is SAID across a storyline — its members' insights, grouped by speaker.

    Scoped to the UNION of the storyline's member topics, which is what the storyline is. Accepts
    the `thc:` id or an anchor topic id, like the card route above and for the same reason.

    404 when no member has a speaker-attributable insight. That is an ABSENCE, not a fault, and it
    is the normal outcome for a grouping whose members are abstract labels nobody says aloud — the
    client renders nothing rather than an error.
    """
    root = corpus_root_or_503(request)
    resp = await asyncio.to_thread(
        build_cluster_perspectives, root, storyline_id.strip(), "storyline"
    )
    if resp is None:
        raise HTTPException(status_code=404, detail="No perspectives for this storyline.")
    return _page_perspectives(resp, speakers_offset, speakers_limit, insights_per_speaker)


@router.get("/themes/{theme_id}/perspectives", response_model=AppTopicPerspectivesResponse)
async def theme_perspectives_route(
    request: Request,
    theme_id: str,
    speakers_offset: int = _SpeakersOffset,
    speakers_limit: int | None = _SpeakersLimit,
    insights_per_speaker: int | None = _PerSpeaker,
    user: User = Depends(get_current_user),
) -> AppTopicPerspectivesResponse:
    """What is SAID across a theme — its members' insights, grouped by speaker.

    No ``scope`` parameter, for the same reason the theme card has none: the page asks what the
    grouping is across the corpus, and a personally-filtered union answers a different question.

    404 when no member has a speaker-attributable insight — an absence, not a fault.
    """
    root = corpus_root_or_503(request)
    resp = await asyncio.to_thread(build_cluster_perspectives, root, theme_id.strip(), "theme")
    if resp is None:
        raise HTTPException(status_code=404, detail="No perspectives for this theme.")
    return _page_perspectives(resp, speakers_offset, speakers_limit, insights_per_speaker)


@router.get("/topics/{topic_id}", response_model=AppTopicCard)
async def topic_card(
    request: Request,
    topic_id: str,
    scope: Literal["all", "mine"] = Query(default="all"),
    episodes_offset: int = _EpisodesOffset,
    episodes_limit: int = _EpisodesLimit,
    user: User = Depends(get_current_user),
) -> AppTopicCard:
    """Topic card: episodes-about + cluster siblings + related people (KG-grounded).

    404 when the topic appears in no episode's KG. ``scope=mine`` (P3 #1122) restricts the
    episodes-about to the user's heard∪captured set; auth-gated (401 signed out).
    """
    from podcast_scraper.server import cil_queries

    root = corpus_root_or_503(request)
    arc_id = cil_queries.canonical_cil_entity_id(topic_id.strip())
    # Built side by side: the arc scan runs while the card is assembled, not after it.
    built: tuple[AppTopicCard | None, list[dict]] = await asyncio.gather(
        asyncio.to_thread(build_topic_card, root, topic_id.strip()),
        asyncio.to_thread(_conversation_arc, str(root), arc_id),
    )
    card, arc = built
    if card is None:
        raise HTTPException(status_code=404, detail="Unknown topic id.")
    if scope == "mine":
        # The arc is corpus-wide with no per-user cut; under "My corpus" the card shows none.
        scoped = _with_topic_aggregates(_scope_to_corpus(card, _user_set(request, user)))
        return _page_episodes(scoped, episodes_offset, episodes_limit)
    card = _with_topic_aggregates(card.model_copy(update={"conversation_arc_weeks": len(arc)}))
    return _page_episodes(card, episodes_offset, episodes_limit)
