"""Consumer knowledge-card routes (``/api/app/persons/*``, ``/api/app/topics/*``).

Dedicated, read-only person/topic cards for the end-user app (PRD-043 FR2/FR3; RFC-102;
#1095/#1096). KG-grounded projections of the single shared corpus — NOT a proxy of the
operator relational API (the consumer/operator boundary stays clean). Mounted under the
``/api/app`` namespace alongside the other consumer routes.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from podcast_scraper.server.app_corpus_access import corpus_root_or_503
from podcast_scraper.server.app_relational_view import build_person_card, build_topic_card
from podcast_scraper.server.schemas import AppPersonCard, AppTopicCard

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
