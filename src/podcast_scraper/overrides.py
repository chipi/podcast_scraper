"""Operator overrides for feed- and episode-level fields (#2283).

One file per corpus, ``<corpus>/overrides.json``, beside ``feeds.spec.yaml`` — the same place and
the same delivery path, so whatever reads the spec reads the overrides. Written only through the
operator endpoint (``/api/feeds/overrides``), which audits every change; read by the pipeline at
the seams where each field is decided, so every downstream reader sees the overridden value.

WHAT CAN BE OVERRIDDEN is an explicit, typed list — not arbitrary keys — because a field the
pipeline does not apply would be an override that silently does nothing:

* feed: ``title``, ``description``, ``authors``, ``image_url``, ``language``, ``hosts``;
* episode (keyed by feed URL + GUID): ``title``, ``description``, ``published_date``,
  ``image_url``, ``language``, ``hosts``, ``guests``, ``speaker_renames``.

``hosts`` / ``guests`` REPLACE what the pipeline derives (they are not unioned with it), and
``speaker_renames`` maps a published speaker name to the name to publish instead.

``language`` must be a recognised ISO code (it is normalised on write); an override is how a feed
that declares no ``<language>`` gets one — without it, such a feed is refused before download.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator

from podcast_scraper.languages import normalize_language_tag

logger = logging.getLogger(__name__)

OVERRIDES_BASENAME = "overrides.json"
SCHEMA_VERSION = 1

_write_lock = threading.Lock()


def _clean_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _clean_names(values: Optional[List[str]]) -> Optional[List[str]]:
    if values is None:
        return None
    out: List[str] = []
    for raw in values:
        name = " ".join(str(raw or "").split())
        if name and name.lower() not in {n.lower() for n in out}:
            out.append(name)
    return out


def _check_image_url(value: Optional[str]) -> Optional[str]:
    text = _clean_text(value)
    if text is not None and not text.lower().startswith(("https://", "http://")):
        raise ValueError("image_url must be an http(s) URL")
    return text


def _check_language(value: Optional[str]) -> Optional[str]:
    text = _clean_text(value)
    if text is None:
        return None
    code = normalize_language_tag(text)
    if code is None:
        raise ValueError(f"language {text!r} is not an ISO language code")
    return code


class FeedFields(BaseModel):
    """Overridable feed (show) fields. ``None`` = not overridden."""

    model_config = ConfigDict(extra="forbid")

    title: Optional[str] = None
    description: Optional[str] = None
    authors: Optional[List[str]] = None
    image_url: Optional[str] = None
    language: Optional[str] = None
    hosts: Optional[List[str]] = None

    @field_validator("title", "description")
    @classmethod
    def _text(cls, value: Optional[str]) -> Optional[str]:
        return _clean_text(value)

    @field_validator("authors", "hosts")
    @classmethod
    def _names(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        return _clean_names(value)

    @field_validator("image_url")
    @classmethod
    def _image(cls, value: Optional[str]) -> Optional[str]:
        return _check_image_url(value)

    @field_validator("language")
    @classmethod
    def _language(cls, value: Optional[str]) -> Optional[str]:
        return _check_language(value)


class EpisodeFields(BaseModel):
    """Overridable episode fields. ``None`` = not overridden."""

    model_config = ConfigDict(extra="forbid")

    title: Optional[str] = None
    description: Optional[str] = None
    published_date: Optional[date] = None
    image_url: Optional[str] = None
    language: Optional[str] = None
    hosts: Optional[List[str]] = None
    guests: Optional[List[str]] = None
    speaker_renames: Optional[Dict[str, str]] = None

    @field_validator("title", "description")
    @classmethod
    def _text(cls, value: Optional[str]) -> Optional[str]:
        return _clean_text(value)

    @field_validator("hosts", "guests")
    @classmethod
    def _names(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        return _clean_names(value)

    @field_validator("image_url")
    @classmethod
    def _image(cls, value: Optional[str]) -> Optional[str]:
        return _check_image_url(value)

    @field_validator("language")
    @classmethod
    def _language(cls, value: Optional[str]) -> Optional[str]:
        return _check_language(value)

    @field_validator("speaker_renames")
    @classmethod
    def _renames(cls, value: Optional[Dict[str, str]]) -> Optional[Dict[str, str]]:
        if value is None:
            return None
        out: Dict[str, str] = {}
        for old, new in value.items():
            old_c, new_c = " ".join(str(old).split()), " ".join(str(new).split())
            if not old_c or not new_c:
                raise ValueError("speaker_renames needs a non-empty name on both sides")
            out[old_c] = new_c
        return out


class FeedOverride(BaseModel):
    """One feed's overrides, and its episodes' (keyed by GUID)."""

    model_config = ConfigDict(extra="forbid")

    fields: FeedFields = Field(default_factory=FeedFields)
    episodes: Dict[str, EpisodeFields] = Field(default_factory=dict)


class OverridesDocument(BaseModel):
    """The whole ``overrides.json``: feed URL -> :class:`FeedOverride`."""

    model_config = ConfigDict(extra="forbid")

    version: int = SCHEMA_VERSION
    feeds: Dict[str, FeedOverride] = Field(default_factory=dict)


def feed_key(url: str) -> str:
    """The key a feed is stored under: its URL, trimmed. One spelling per feed, like the spec."""
    return (url or "").strip()


def overrides_path(corpus_root: Path) -> Path:
    """Where a corpus keeps its overrides: ``<corpus>/overrides.json``."""
    return Path(corpus_root) / OVERRIDES_BASENAME


def load_overrides(corpus_root: Path) -> OverridesDocument:
    """The corpus's overrides; an empty document when there is no file.

    A file that exists but does not parse RAISES: overrides change what is published, and a run
    that silently ignored a broken file would publish exactly what the operator overrode.
    """
    path = overrides_path(corpus_root)
    if not path.is_file():
        return OverridesDocument()
    raw = json.loads(path.read_text(encoding="utf-8"))
    return OverridesDocument.model_validate(raw)


def save_overrides(corpus_root: Path, doc: OverridesDocument) -> None:
    """Atomic write (temp file + rename), so a reader never sees a half-written file."""
    path = overrides_path(corpus_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = doc.model_dump(mode="json", exclude_none=True)
    fd, tmp = tempfile.mkstemp(prefix=".overrides.", suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _dump(model: Optional[BaseModel]) -> Optional[Dict[str, Any]]:
    return None if model is None else model.model_dump(mode="json", exclude_none=True)


def feed_fields_for(doc: OverridesDocument, url: str) -> Optional[FeedFields]:
    """The feed-level overrides for *url*, or ``None`` when it has none."""
    entry = doc.feeds.get(feed_key(url))
    return entry.fields if entry is not None else None


def episode_fields_for(doc: OverridesDocument, url: str, guid: str) -> Optional[EpisodeFields]:
    """One episode's overrides (feed *url* + *guid*), or ``None`` when it has none."""
    entry = doc.feeds.get(feed_key(url))
    if entry is None:
        return None
    return entry.episodes.get((guid or "").strip())


def set_feed_fields(
    corpus_root: Path, url: str, fields: FeedFields
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Replace a feed's field overrides. Returns ``(before, after)`` for the audit."""
    key = feed_key(url)
    if not key:
        raise ValueError("feed url is required")
    with _write_lock:
        doc = load_overrides(corpus_root)
        entry = doc.feeds.setdefault(key, FeedOverride())
        before = _dump(entry.fields) or None
        entry.fields = fields
        save_overrides(corpus_root, doc)
    return before, _dump(fields) or {}


def set_episode_fields(
    corpus_root: Path, url: str, guid: str, fields: EpisodeFields
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Replace one episode's field overrides. Returns ``(before, after)`` for the audit."""
    key, ep = feed_key(url), (guid or "").strip()
    if not key or not ep:
        raise ValueError("feed url and episode guid are required")
    with _write_lock:
        doc = load_overrides(corpus_root)
        entry = doc.feeds.setdefault(key, FeedOverride())
        before = _dump(entry.episodes.get(ep))
        entry.episodes[ep] = fields
        save_overrides(corpus_root, doc)
    return before, _dump(fields) or {}


def delete_feed(corpus_root: Path, url: str) -> Optional[Dict[str, Any]]:
    """Remove a feed's overrides AND its episodes'. Returns what was removed (or ``None``)."""
    with _write_lock:
        doc = load_overrides(corpus_root)
        entry = doc.feeds.pop(feed_key(url), None)
        if entry is None:
            return None
        save_overrides(corpus_root, doc)
    return entry.model_dump(mode="json", exclude_none=True)


def delete_episode(corpus_root: Path, url: str, guid: str) -> Optional[Dict[str, Any]]:
    """Remove one episode's overrides. Returns what was removed (or ``None``)."""
    with _write_lock:
        doc = load_overrides(corpus_root)
        entry = doc.feeds.get(feed_key(url))
        removed = entry.episodes.pop((guid or "").strip(), None) if entry else None
        if removed is None:
            return None
        if entry is not None and not entry.episodes and not _dump(entry.fields):
            doc.feeds.pop(feed_key(url), None)
        save_overrides(corpus_root, doc)
    return _dump(removed)
