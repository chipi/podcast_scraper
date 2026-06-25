"""Consumer knowledge-card routes (``/api/app/persons/*``, ``/api/app/topics/*``).

Dedicated, read-only person/topic cards for the end-user app (PRD-043 FR2/FR3; RFC-102;
#1095/#1096). KG-grounded projections of the single shared corpus — NOT a proxy of the
operator relational API (the consumer/operator boundary stays clean). Mounted under the
``/api/app`` namespace alongside the other consumer routes.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from podcast_scraper.server.app_corpus_access import corpus_root_or_503
from podcast_scraper.server.app_relational_view import (
    build_person_card,
    build_topic_card,
    resolve_entity,
)
from podcast_scraper.server.schemas import (
    AppEntitySearchResponse,
    AppPersonCard,
    AppTopicCard,
)

router = APIRouter(tags=["app"])


@router.get("/persons/{person_id}", response_model=AppPersonCard)
async def person_card(request: Request, person_id: str) -> AppPersonCard:
    """Person profile card: appears-in episodes + related people/topics (KG co-occurrence).

    404 when the person appears in no episode's KG (rather than an empty card), so the
    client can distinguish "unknown person" from "person with a thin footprint".
    """
    root = corpus_root_or_503(request)
    card = build_person_card(root, person_id.strip())
    if card is None:
        raise HTTPException(status_code=404, detail="Unknown person id.")
    return card


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
async def topic_card(request: Request, topic_id: str) -> AppTopicCard:
    """Topic card: episodes-about + cluster siblings + related people (KG-grounded).

    404 when the topic appears in no episode's KG.
    """
    root = corpus_root_or_503(request)
    card = build_topic_card(root, topic_id.strip())
    if card is None:
        raise HTTPException(status_code=404, detail="Unknown topic id.")
    return card
