"""Operator overrides for feed and episode fields (#2283), next to the feed list they extend.

Mounted with ``/api/feeds`` on the operator CONTROL plane — the tailnet-only api that owns the
corpus and already writes ``feeds.spec.yaml`` (operator, 2026-10-05: "from tailscale and only to
that one place where other overrides basically exist today"). The public planes mount the corpus
read-only and never mount this router.

Authorization is the operator API's: ``/api/feeds/overrides`` is under the ``/api/feeds`` base, so
:class:`~podcast_scraper.server.app_operator_guard.OperatorWriteGuard` requires an admin session or
the operator key for every method, reads included, and audits the write with WHO. This module adds
the value before and after to that record.

The store and its validation live in :mod:`podcast_scraper.overrides`, which the pipeline reads;
this module is only the authorized, audited way to change it.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import ValidationError

from podcast_scraper import overrides
from podcast_scraper.server.app_audit import append_audit
from podcast_scraper.server.pathutil import resolve_corpus_path_param

logger = logging.getLogger(__name__)

router = APIRouter(tags=["feeds"])

_PATH_DOC = "Corpus root directory (resolved under server anchor), as for /api/feeds."


def _root(request: Request, path: str) -> Path:
    """The corpus root, which must be the server's own anchor; *path* only has to name it.

    THE FILE LOCATION NEVER COMES FROM THE REQUEST. Overrides live at the corpus root the pipeline
    reads (``app.state.output_dir``, ``/app/output`` on the control plane), so the request's
    ``path`` is validated against the anchor (``resolve_corpus_path_param`` refuses anything
    outside it) and compared as a string; the path returned is built from the server's own
    setting. CodeQL's ``py/path-injection`` flagged every overrides read/write while the location
    was derived from the request, including through a normpath/startswith sanitizer.
    """
    anchor = getattr(request.app.state, "output_dir", None)
    if anchor is None:
        raise HTTPException(status_code=400, detail="No corpus anchor configured.")
    anchor_s = os.path.normpath(str(Path(anchor).expanduser().resolve()))
    requested = os.path.normpath(str(resolve_corpus_path_param(path, anchor)))
    if requested != anchor_s:
        raise HTTPException(
            status_code=400,
            detail="Overrides live at the corpus root; path must name it.",
        )
    return Path(anchor_s)


def _audit(request: Request, **record: object) -> None:
    who = getattr(request.state, "operator_who", None) or {"via": "none"}
    append_audit(getattr(request.app.state, "audit_path", None), {**record, **who})


def _load(root: Path) -> overrides.OverridesDocument:
    try:
        return overrides.load_overrides(root)
    except (ValueError, json.JSONDecodeError) as exc:
        # A broken file must be visible, not silently replaced: fix it on disk.
        raise HTTPException(status_code=500, detail=f"overrides.json is invalid: {exc}") from exc


def _validated(model: Any, body: Dict[str, Any]) -> Any:
    try:
        return model.model_validate(body)
    except ValidationError as exc:
        # No `ctx`: it carries the raw ValueError, which is not JSON — the refusal itself 500'd.
        detail = exc.errors(include_url=False, include_context=False)
        raise HTTPException(status_code=422, detail=detail) from exc


@router.get("/feeds/overrides")
def get_overrides(
    request: Request, path: str = Query(..., description=_PATH_DOC)
) -> Dict[str, Any]:
    """Every override in the corpus."""
    return _load(_root(request, path)).model_dump(mode="json", exclude_none=True)


@router.put("/feeds/overrides/feed")
def put_feed_override(
    request: Request,
    body: Dict[str, Any],
    path: str = Query(..., description=_PATH_DOC),
    url: str = Query(..., min_length=1, description="The feed's RSS URL."),
) -> Dict[str, Any]:
    """Replace a feed's field overrides with *body* (fields left out are not overridden)."""
    fields = _validated(overrides.FeedFields, body)
    root = _root(request, path)
    _load(root)  # refuse to write over a broken file
    try:
        before, after = overrides.set_feed_fields(root, url, fields)
    except ValueError as exc:  # a blank url passes min_length=1 ("   ")
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    _audit(
        request,
        action="override_feed_set",
        feed=overrides.feed_key(url),
        before=before,
        after=after,
    )
    return {"feed": overrides.feed_key(url), "fields": after}


@router.delete("/feeds/overrides/feed")
def delete_feed_override(
    request: Request,
    path: str = Query(..., description=_PATH_DOC),
    url: str = Query(..., min_length=1),
) -> Dict[str, Any]:
    """Remove a feed's overrides, including every episode override under it."""
    root = _root(request, path)
    _load(root)
    removed = overrides.delete_feed(root, url)
    if removed is None:
        raise HTTPException(status_code=404, detail="No overrides for that feed.")
    _audit(
        request,
        action="override_feed_deleted",
        feed=overrides.feed_key(url),
        before=removed,
        after=None,
    )
    return {"removed": removed}


@router.put("/feeds/overrides/episode")
def put_episode_override(
    request: Request,
    body: Dict[str, Any],
    path: str = Query(..., description=_PATH_DOC),
    url: str = Query(..., min_length=1, description="The feed's RSS URL."),
    guid: str = Query(..., min_length=1, description="The episode's GUID in that feed."),
) -> Dict[str, Any]:
    """Replace one episode's field overrides with *body*."""
    fields = _validated(overrides.EpisodeFields, body)
    root = _root(request, path)
    _load(root)
    try:
        before, after = overrides.set_episode_fields(root, url, guid, fields)
    except ValueError as exc:  # a blank url or guid passes min_length=1
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    _audit(
        request,
        action="override_episode_set",
        feed=overrides.feed_key(url),
        guid=guid.strip(),
        before=before,
        after=after,
    )
    return {"feed": overrides.feed_key(url), "guid": guid.strip(), "fields": after}


@router.delete("/feeds/overrides/episode")
def delete_episode_override(
    request: Request,
    path: str = Query(..., description=_PATH_DOC),
    url: str = Query(..., min_length=1),
    guid: str = Query(..., min_length=1),
) -> Dict[str, Optional[Dict[str, Any]]]:
    """Remove one episode's overrides."""
    root = _root(request, path)
    _load(root)
    removed = overrides.delete_episode(root, url, guid)
    if removed is None:
        raise HTTPException(status_code=404, detail="No overrides for that episode.")
    _audit(
        request,
        action="override_episode_deleted",
        feed=overrides.feed_key(url),
        guid=guid.strip(),
        before=removed,
        after=None,
    )
    return {"removed": removed}
