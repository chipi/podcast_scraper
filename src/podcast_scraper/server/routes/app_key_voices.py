"""Per-user key-voices route ``/api/app/key-voices`` (wave-G).

Auth-gated — the signed-in user's most-present people across their own heard∪captured corpus.
KG-grounded projection over the shared corpus (no external data); the per-topic flavor lives on the
topic card.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, Depends, Query, Request

from podcast_scraper.server import app_key_voices
from podcast_scraper.server.app_corpus_access import corpus_root_or_503
from podcast_scraper.server.app_user_store import User
from podcast_scraper.server.routes.app_auth import get_current_user
from podcast_scraper.server.schemas import KeyVoicesResponse

router = APIRouter(tags=["app"])


@router.get("/key-voices", response_model=KeyVoicesResponse)
async def key_voices(
    request: Request,
    limit: int = Query(default=8, ge=1, le=24),
    user: User = Depends(get_current_user),
) -> KeyVoicesResponse:
    """The signed-in user's key voices — people ranked by presence in their own corpus."""
    root = corpus_root_or_503(request)
    data_dir = Path(request.app.state.app_data_dir)
    # KG-grounded scan iterates per-episode artifacts — keep it off the event loop.
    voices = await asyncio.to_thread(
        app_key_voices.key_voices_for_user, root, data_dir, user.user_id, limit=limit
    )
    return KeyVoicesResponse.model_validate({"voices": voices})
