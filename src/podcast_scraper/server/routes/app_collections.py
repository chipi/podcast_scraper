"""Collections / boards routes — the curation layer (#1417 / RFC-111 §1; RFC-119 typed items).

Auth-gated, per-user. A collection is a named MIXED bucket of typed items (highlight / episode /
show / search / topic / person / link — RFC-119). The detail view resolves each item best-effort:
highlights are hydrated from the capture store here (the client can't fetch one by id); every other
kind returns its ``{kind, ref, deep_link}`` and the client hydrates display (episode/show/topic/…)
through its existing endpoints. A dangling highlight (deleted since) is dropped, matching the count.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Request

from podcast_scraper.server import app_collections_store, app_user_state
from podcast_scraper.server.app_user_store import User
from podcast_scraper.server.routes.app_auth import get_current_user
from podcast_scraper.server.schemas import (
    Collection,
    CollectionCreate,
    CollectionDetail,
    CollectionItem,
    CollectionItemBody,
    CollectionsResponse,
)

router = APIRouter(tags=["app"])


def _data_dir(request: Request) -> Path:
    return Path(request.app.state.app_data_dir)


def _live_highlight_ids(data_dir: Path, user_id: str) -> set[str]:
    """The ids of highlights that still exist — what an honest item count must be measured against.

    Deleting a highlight touches only ``highlights.json``; every collection that held it keeps the
    id. Non-highlight kinds resolve from shared stores and are always counted as present.
    """
    return {str(h["id"]) for h in app_user_state.get_highlights(data_dir, user_id) if h.get("id")}


def _rows(data_dir: Path, user_id: str) -> list[dict]:
    return app_collections_store.list_collections(
        data_dir, user_id, live_item_ids=_live_highlight_ids(data_dir, user_id)
    )


def _deslug(ns_id: str) -> str:
    """``topic:risk-management`` → ``risk management`` — a readable fallback label."""
    return ns_id.split(":", 1)[-1].replace("-", " ").replace("_", " ")


def _resolve_item(item: dict, highlights_by_id: dict[str, dict]) -> CollectionItem | None:
    """A stored typed item → a resolved CollectionItem, or None to drop (dangling highlight)."""
    kind = str(item.get("kind"))
    ref = str(item.get("ref"))
    scope = item.get("scope")
    if kind == "highlight":
        h = highlights_by_id.get(ref)
        if h is None:
            return None  # deleted since — drop, so it matches the count
        slug = h.get("episode_slug")
        return CollectionItem(
            kind=kind,
            ref=ref,
            title=(h.get("quote_text") or "Highlight"),
            deep_link=f"/player/{slug}" if slug else None,
        )
    if kind == "episode":
        return CollectionItem(kind=kind, ref=ref, deep_link=f"/episode/{ref}")
    if kind == "show":
        return CollectionItem(kind=kind, ref=ref, deep_link=f"/podcast/{ref}")
    if kind == "topic":
        return CollectionItem(kind=kind, ref=ref, title=_deslug(ref), deep_link=f"/topic/{ref}")
    if kind == "person":
        return CollectionItem(kind=kind, ref=ref, title=_deslug(ref), deep_link=f"/person/{ref}")
    if kind == "search":
        q = quote(ref, safe="")
        link = f"/search?q={q}" + (f"&scope={quote(str(scope), safe='')}" if scope else "")
        return CollectionItem(kind=kind, ref=ref, title=ref, deep_link=link, scope=scope)
    if kind == "link":
        return CollectionItem(kind=kind, ref=ref, title=(item.get("title") or ref), deep_link=ref)
    return None


@router.get("/collections", response_model=CollectionsResponse)
async def list_collections(
    request: Request, user: User = Depends(get_current_user)
) -> CollectionsResponse:
    """The user's collections, newest-first, each with its item count."""
    data_dir = _data_dir(request)
    return CollectionsResponse(items=[Collection(**c) for c in _rows(data_dir, user.user_id)])


@router.post("/collections", response_model=Collection, status_code=201)
async def create_collection(
    request: Request, body: CollectionCreate, user: User = Depends(get_current_user)
) -> Collection:
    """Create a named collection. 422 when the per-user collection cap is reached (#51)."""
    try:
        created = app_collections_store.create_collection(
            _data_dir(request), user.user_id, body.name
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return Collection(**created)


@router.delete("/collections/{collection_id}", response_model=CollectionsResponse)
async def delete_collection(
    request: Request, collection_id: str, user: User = Depends(get_current_user)
) -> CollectionsResponse:
    """Delete a collection (its membership goes; the referenced things stay)."""
    app_collections_store.delete_collection(_data_dir(request), user.user_id, collection_id)
    data_dir = _data_dir(request)
    return CollectionsResponse(items=[Collection(**c) for c in _rows(data_dir, user.user_id)])


@router.get("/collections/{collection_id}", response_model=CollectionDetail)
async def collection_detail(
    request: Request, collection_id: str, user: User = Depends(get_current_user)
) -> CollectionDetail:
    """A collection + its resolved typed items (dangling highlights dropped)."""
    data_dir = _data_dir(request)
    by_id = {h["id"]: h for h in app_user_state.get_highlights(data_dir, user.user_id)}
    rows = app_collections_store.list_collections(data_dir, user.user_id, live_item_ids=set(by_id))
    meta = next((c for c in rows if c["id"] == collection_id), None)
    if meta is None:
        raise HTTPException(status_code=404, detail="collection not found")
    stored = app_collections_store.get_items(data_dir, user.user_id, collection_id)
    items = [it for it in (_resolve_item(m, by_id) for m in stored) if it is not None]
    return CollectionDetail(collection=Collection(**meta), items=items)


@router.post("/collections/{collection_id}/items", response_model=Collection)
async def add_item(
    request: Request,
    collection_id: str,
    body: CollectionItemBody,
    user: User = Depends(get_current_user),
) -> Collection:
    """Add a typed item to a collection (idempotent by kind+ref).

    404 when the collection is unknown, or when a ``highlight`` item's id doesn't exist (the
    we resolve here — an unknown highlight would be uncountable + unrenderable). Other kinds are
    accepted as-is; a bad ref simply won't resolve in the detail view.
    """
    data_dir = _data_dir(request)
    live = _live_highlight_ids(data_dir, user.user_id)
    rows = app_collections_store.list_collections(data_dir, user.user_id, live_item_ids=live)
    if not any(c["id"] == collection_id for c in rows):
        raise HTTPException(status_code=404, detail="collection not found")
    if body.kind == "highlight" and body.ref not in live:
        raise HTTPException(status_code=404, detail="highlight not found")
    item = {"kind": body.kind, "ref": body.ref}
    if body.scope is not None:
        item["scope"] = body.scope
    if body.title is not None:
        item["title"] = body.title
    try:
        app_collections_store.add_item(data_dir, user.user_id, collection_id, item)
    except KeyError as exc:  # lost a race with a concurrent delete
        raise HTTPException(status_code=404, detail="collection not found") from exc
    except ValueError as exc:  # invalid item / the per-collection cap (#51)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    rows = app_collections_store.list_collections(data_dir, user.user_id, live_item_ids=live)
    return Collection(**next(c for c in rows if c["id"] == collection_id))


@router.delete("/collections/{collection_id}/items", response_model=Collection)
async def remove_item(
    request: Request,
    collection_id: str,
    kind: str,
    ref: str,
    user: User = Depends(get_current_user),
) -> Collection:
    """Remove the item identified by ``?kind=&ref=`` from a collection."""
    data_dir = _data_dir(request)
    app_collections_store.remove_item(data_dir, user.user_id, collection_id, kind, ref)
    rows = _rows(data_dir, user.user_id)
    meta = next((c for c in rows if c["id"] == collection_id), None)
    if meta is None:
        raise HTTPException(status_code=404, detail="collection not found")
    return Collection(**meta)
