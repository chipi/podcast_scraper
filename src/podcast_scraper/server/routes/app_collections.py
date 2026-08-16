"""Collections / boards routes — the curation layer (#1417, PRD-046 FR4 / RFC-111 §1).

Auth-gated, per-user. A collection is a named set of highlight ids spanning episodes; the detail
view hydrates those ids against the user's capture store so the client renders the highlight cards
(dropping ids whose highlight was since deleted).
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request

from podcast_scraper.server import app_collections_store, app_user_state
from podcast_scraper.server.app_user_store import User
from podcast_scraper.server.routes.app_auth import get_current_user
from podcast_scraper.server.schemas import (
    Collection,
    CollectionCreate,
    CollectionDetail,
    CollectionItemBody,
    CollectionsResponse,
    Highlight,
)

router = APIRouter(tags=["app"])


def _data_dir(request: Request) -> Path:
    return Path(request.app.state.app_data_dir)


@router.get("/collections", response_model=CollectionsResponse)
async def list_collections(
    request: Request, user: User = Depends(get_current_user)
) -> CollectionsResponse:
    """The user's collections, newest-first, each with its item count."""
    rows = app_collections_store.list_collections(_data_dir(request), user.user_id)
    return CollectionsResponse(items=[Collection(**c) for c in rows])


@router.post("/collections", response_model=Collection, status_code=201)
async def create_collection(
    request: Request, body: CollectionCreate, user: User = Depends(get_current_user)
) -> Collection:
    """Create a named collection."""
    created = app_collections_store.create_collection(_data_dir(request), user.user_id, body.name)
    return Collection(**created)


@router.delete("/collections/{collection_id}", response_model=CollectionsResponse)
async def delete_collection(
    request: Request, collection_id: str, user: User = Depends(get_current_user)
) -> CollectionsResponse:
    """Delete a collection (its membership goes; the highlights themselves stay)."""
    app_collections_store.delete_collection(_data_dir(request), user.user_id, collection_id)
    rows = app_collections_store.list_collections(_data_dir(request), user.user_id)
    return CollectionsResponse(items=[Collection(**c) for c in rows])


@router.get("/collections/{collection_id}", response_model=CollectionDetail)
async def collection_detail(
    request: Request, collection_id: str, user: User = Depends(get_current_user)
) -> CollectionDetail:
    """A collection + its highlights, hydrated from the capture store (missing ids dropped)."""
    data_dir = _data_dir(request)
    rows = app_collections_store.list_collections(data_dir, user.user_id)
    meta = next((c for c in rows if c["id"] == collection_id), None)
    if meta is None:
        raise HTTPException(status_code=404, detail="collection not found")
    ids = app_collections_store.get_items(data_dir, user.user_id, collection_id)
    by_id = {h["id"]: h for h in app_user_state.get_highlights(data_dir, user.user_id)}
    highlights = [Highlight(**by_id[i]) for i in ids if i in by_id]
    return CollectionDetail(collection=Collection(**meta), highlights=highlights)


@router.post("/collections/{collection_id}/items", response_model=Collection)
async def add_item(
    request: Request,
    collection_id: str,
    body: CollectionItemBody,
    user: User = Depends(get_current_user),
) -> Collection:
    """Add a highlight to a collection (idempotent). 404 when the collection is unknown."""
    data_dir = _data_dir(request)
    try:
        app_collections_store.add_item(data_dir, user.user_id, collection_id, body.highlight_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="collection not found") from exc
    rows = app_collections_store.list_collections(data_dir, user.user_id)
    return Collection(**next(c for c in rows if c["id"] == collection_id))


@router.delete("/collections/{collection_id}/items/{highlight_id}", response_model=Collection)
async def remove_item(
    request: Request,
    collection_id: str,
    highlight_id: str,
    user: User = Depends(get_current_user),
) -> Collection:
    """Remove a highlight from a collection."""
    data_dir = _data_dir(request)
    app_collections_store.remove_item(data_dir, user.user_id, collection_id, highlight_id)
    rows = app_collections_store.list_collections(data_dir, user.user_id)
    meta = next((c for c in rows if c["id"] == collection_id), None)
    if meta is None:
        raise HTTPException(status_code=404, detail="collection not found")
    return Collection(**meta)
