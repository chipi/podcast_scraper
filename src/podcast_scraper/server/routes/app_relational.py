"""Consumer knowledge-card routes (``/api/app/persons/*``, ``/api/app/topics/*``).

Dedicated, read-only person/topic cards for the end-user app (PRD-043 FR2/FR3; RFC-102;
#1095/#1096). KG-grounded projections of the single shared corpus — NOT a proxy of the
operator relational API (the consumer/operator boundary stays clean). Mounted under the
``/api/app`` namespace alongside the other consumer routes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypeVar

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from podcast_scraper.server.app_corpus_access import corpus_root_or_503
from podcast_scraper.server.app_relational_view import (
    build_person_card,
    build_topic_card,
    build_topic_perspectives,
    resolve_entity,
)
from podcast_scraper.server.app_user_corpus import user_episode_set
from podcast_scraper.server.app_user_store import User
from podcast_scraper.server.routes.app_auth import get_optional_user
from podcast_scraper.server.schemas import (
    AppEntitySearchResponse,
    AppPersonCard,
    AppTopicCard,
    AppTopicPerspectivesResponse,
)

router = APIRouter(tags=["app"])

_Card = TypeVar("_Card", AppPersonCard, AppTopicCard)


def _user_set(request: Request, user: User | None) -> set[str]:
    """The signed-in user's heard∪captured slugs; 401 when ``scope=mine`` but signed out."""
    if user is None:
        raise HTTPException(status_code=401, detail="Sign in to scope to your corpus.")
    root = corpus_root_or_503(request)
    data_dir = getattr(request.app.state, "app_data_dir", None)
    return user_episode_set(root, Path(data_dir), user.user_id) if data_dir is not None else set()


def _scope_to_corpus(card: _Card, mine: set[str]) -> _Card:
    """Filter a card's appears-in episodes to the user's set (the "you heard X in …" lens),
    recomputing ``episode_count`` so the card reads honestly per RFC-101 §4."""
    card.episodes = [e for e in card.episodes if e.slug in mine]
    card.episode_count = len(card.episodes)
    return card


@router.get("/persons/{person_id}", response_model=AppPersonCard)
async def person_card(
    request: Request,
    person_id: str,
    scope: Literal["all", "mine"] = Query(default="all"),
    user: User | None = Depends(get_optional_user),
) -> AppPersonCard:
    """Person profile card: appears-in episodes + related people/topics (KG co-occurrence).

    404 when the person appears in no episode's KG (rather than an empty card), so the
    client can distinguish "unknown person" from "person with a thin footprint".

    ``scope=mine`` (P3 #1122) is the "your corpus" lens — the guest *across the episodes you have
    heard* ("you also heard them in …"); auth-gated (401 signed out).
    """
    root = corpus_root_or_503(request)
    card = build_person_card(root, person_id.strip())
    if card is None:
        raise HTTPException(status_code=404, detail="Unknown person id.")
    if scope == "mine":
        card = _scope_to_corpus(card, _user_set(request, user))
    return card


@router.get("/topics/{topic_id}/perspectives", response_model=AppTopicPerspectivesResponse)
async def topic_perspectives_route(
    request: Request,
    topic_id: str,
    scope: Literal["all", "mine"] = Query(default="all"),
    user: User | None = Depends(get_optional_user),
) -> AppTopicPerspectivesResponse:
    """Multi-perspective synthesis — each speaker's take on the topic (#1146).

    ``scope=mine`` (#1149) restricts to the episodes the user has heard∪captured; auth-gated
    (401 signed out). 404 when the topic has no speaker-attributable insight in the (scoped) GI.
    """
    root = corpus_root_or_503(request)
    mine = _user_set(request, user) if scope == "mine" else None
    resp = build_topic_perspectives(root, topic_id.strip(), mine_slugs=mine)
    if resp is None:
        raise HTTPException(status_code=404, detail="No perspectives for this topic.")
    return resp


@router.get("/entities/search", response_model=AppEntitySearchResponse)
async def entity_search(
    request: Request,
    q: str = Query(min_length=1, description="Query to resolve to a person/topic entity."),
) -> AppEntitySearchResponse:
    """Resolve a query to a person/topic card (PRD-043 FR3 / 3.4) — exact/near-exact name only.

    The consumer search view calls this alongside `/search` and renders the matched entity as a
    card above the passage results. Returns ``entity: null`` (200) when nothing matches.
    """
    root = corpus_root_or_503(request)
    return AppEntitySearchResponse(query=q, entity=resolve_entity(root, q))


@router.get("/topics/{topic_id}", response_model=AppTopicCard)
async def topic_card(
    request: Request,
    topic_id: str,
    scope: Literal["all", "mine"] = Query(default="all"),
    user: User | None = Depends(get_optional_user),
) -> AppTopicCard:
    """Topic card: episodes-about + cluster siblings + related people (KG-grounded).

    404 when the topic appears in no episode's KG. ``scope=mine`` (P3 #1122) restricts the
    episodes-about to the user's heard∪captured set; auth-gated (401 signed out).
    """
    root = corpus_root_or_503(request)
    card = build_topic_card(root, topic_id.strip())
    if card is None:
        raise HTTPException(status_code=404, detail="Unknown topic id.")
    if scope == "mine":
        card = _scope_to_corpus(card, _user_set(request, user))
    return card
