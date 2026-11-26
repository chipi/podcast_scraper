"""The pipeline's writes to the corpus database: an item's transcript documents and segments.

CRFC-005 §3.5, tracker row ``w-public``. After a stage writes an episode's transcript files the
pipeline syncs them, one transaction per item on the ``pipeline_rw`` engine
(:func:`~podcast_scraper.db.engines.pipeline_rw_engine`). The files stay the source until the
cut-over; no reader reads these rows before it.

The rules, settled with the database session (tracker ``w-public``, ``note_rules_2``):

- The files and their variants are the importer's (``db.importers.corpus_documents``
  ``transcript_files``), read the same way (``read_body``), so a document hashes the same
  whichever side wrote it.
- Only an item whose ``content_items`` row exists is written; any other is counted. The item
  upsert is the next slice.
- A body that equals the newest version's writes nothing. A different body is a new version that
  supersedes it, keeping the old version's segments; ``change_kind`` is ``created`` on version 1
  and the caller's cause on the later ones.
- A failure is logged and counted before the cut-over; with ``PODCAST_CORPUS_WRITE_STRICT`` set
  it fails the item.

A podcast item only, in this slice: an article body (``workflow/text_in.py``) syncs in the next.
"""

from __future__ import annotations

import collections
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import yaml
from sqlalchemy import and_, func, insert, select
from sqlalchemy.engine import Connection

from podcast_scraper.db.corpus_keys import is_news, item_natural_id, KeyNotDerivable
from podcast_scraper.db.importers.corpus_documents import (
    DocumentRefused,
    joined_body,
    read_body,
    read_segments,
    TRANSCRIPT,
    transcript_files,
    TranscriptFile,
)
from podcast_scraper.db.schema import content_items, documents, segments
from podcast_scraper.db.settings import pipeline_is_configured
from podcast_scraper.document_cut import cut_body, DocumentCutError
from podcast_scraper.languages import episode_language

logger = logging.getLogger(__name__)

ENV_STRICT = "PODCAST_CORPUS_WRITE_STRICT"

CREATED = "created"
RETRANSCRIBED = "retranscribed"
REPUNCTUATED = "repunctuated"
TRANSLATED = "translated"
CORRECTED = "corrected"
CHANGE_KINDS = (RETRANSCRIBED, REPUNCTUATED, TRANSLATED, CORRECTED)

#: How each sync ended, by outcome, since the process started; the run's metrics read it.
OUTCOMES: collections.Counter[str] = collections.Counter()

_SEGMENTS_BATCH = 1000


class SyncFailed(RuntimeError):
    """A sync failed while ``PODCAST_CORPUS_WRITE_STRICT`` is set."""


@dataclass(frozen=True)
class SyncedDocument:
    """One variant after a sync: its newest version, and whether this sync wrote it."""

    variant: str
    document_id: int
    version: int
    segments: int
    written: bool


def _strict() -> bool:
    return os.environ.get(ENV_STRICT, "").strip().lower() in ("1", "true", "yes")


def item_id_for(conn: Connection, natural_id: str) -> Optional[int]:
    """The ``content_items`` row id of ``natural_id``, or ``None`` when there is no row."""
    found: Optional[int] = conn.execute(
        select(content_items.c.item_id).where(content_items.c.natural_id == natural_id)
    ).scalar_one_or_none()
    return found


def _newest(conn: Connection, item_id: int, variant: str) -> Optional[tuple[int, int, str]]:
    row = conn.execute(
        select(documents.c.document_id, documents.c.version, documents.c.body_sha256)
        .where(
            and_(
                documents.c.item_id == item_id,
                documents.c.kind == TRANSCRIPT,
                documents.c.variant == variant,
            )
        )
        .order_by(documents.c.version.desc())
        .limit(1)
    ).first()
    return (row[0], row[1], row[2]) if row else None


def _segment_count(conn: Connection, item_id: int, document_id: int) -> int:
    count: int = conn.execute(
        select(func.count()).where(
            and_(segments.c.item_id == item_id, segments.c.document_id == document_id)
        )
    ).scalar_one()
    return count


def sync_documents(
    conn: Connection, item_id: int, files: Sequence[TranscriptFile], *, change_kind: str
) -> list[SyncedDocument]:
    """Write each file whose body differs from its variant's newest version, in the caller's
    transaction. ``change_kind`` names why a later version exists; version 1 is ``created``."""
    if change_kind not in CHANGE_KINDS:
        raise ValueError(f"change_kind {change_kind!r} is not one of {CHANGE_KINDS}")
    synced: list[SyncedDocument] = []
    for file in files:
        body = read_body(file.path)
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        newest = _newest(conn, item_id, file.variant)
        if newest is not None and newest[2] == digest:
            document_id, version, _ = newest
            count = _segment_count(conn, item_id, document_id)
            synced.append(SyncedDocument(file.variant, document_id, version, count, False))
            continue
        try:
            cuts = cut_body(body, read_segments(file.segments_path))
        except (DocumentCutError, ValueError) as exc:
            raise DocumentRefused(f"{file.path.name}: {exc}") from exc
        version = 1 if newest is None else newest[1] + 1
        document_id = conn.execute(
            insert(documents)
            .values(
                item_id=item_id,
                kind=TRANSCRIPT,
                variant=file.variant,
                version=version,
                supersedes_id=None if newest is None else newest[0],
                change_kind=CREATED if newest is None else change_kind,
                body_sha256=digest,
            )
            .returning(documents.c.document_id)
        ).scalar_one()
        rows = [
            {
                "item_id": item_id,
                "document_id": document_id,
                "seq": cut.seq,
                "start_ms": cut.start_ms,
                "end_ms": cut.end_ms,
                "text": cut.text,
            }
            for cut in cuts
        ]
        for start in range(0, len(rows), _SEGMENTS_BATCH):
            conn.execute(insert(segments), rows[start : start + _SEGMENTS_BATCH])
        if joined_body(conn, item_id, document_id) != body:
            raise DocumentRefused(f"{file.path.name}: the written segments do not join back")
        synced.append(SyncedDocument(file.variant, document_id, version, len(rows), True))
    return synced


def _load(metadata_path: Path) -> Mapping[str, Any]:
    text = metadata_path.read_text(encoding="utf-8")
    doc = (
        yaml.safe_load(text)
        if metadata_path.suffix.lower() in (".yaml", ".yml")
        else json.loads(text)
    )
    if not isinstance(doc, Mapping):
        raise ValueError(f"{metadata_path.name} is not a metadata document")
    return doc


def _base(transcript_path: Path) -> str:
    """The episode's base name: the transcript's name without its extension (``.txt``, or a
    downloaded ``.vtt`` / ``.srt``, whose ``.txt`` siblings are what is written)."""
    return transcript_path.name.rsplit(".", 1)[0]


def _outcome(synced: Sequence[SyncedDocument]) -> str:
    written = [d for d in synced if d.written]
    if any(d.version > 1 for d in written):
        return "superseded"
    return "written" if written else "unchanged"


def sync_episode(
    metadata_path: Path, transcript_path: Path, *, change_kind: str = RETRANSCRIBED
) -> str:
    """Sync one podcast episode's transcript documents; returns and counts the outcome.

    ``written`` (version 1 of every variant written), ``superseded`` (a later version written: its
    ``change_kind`` is the caller's, so the metadata stage's ``retranscribed`` is assumed until the
    translation, repunctuate and retranscript_only stages sync themselves), ``unchanged``,
    ``no_files`` (no transcript body beside ``transcript_path``), ``no_item_row``, ``refused`` (the
    files cannot be named one way, or a body does not cut), ``failed`` (anything else),
    ``news_item`` (not this slice's) or ``not_configured`` (no pipeline database URL). Never raises
    unless the strict setting is on. An item the pipeline marked not served is synced too: which
    items are served is the read side's (``content_items.served``), as in the importer.
    """
    if not pipeline_is_configured():
        return _count("not_configured")
    try:
        doc = _load(metadata_path)
        if is_news(doc):
            return _count("news_item")
        natural_id = item_natural_id(doc)
        directory = transcript_path.parent
        files = (
            transcript_files(directory, _base(transcript_path), episode_language(dict(doc)))
            if directory.is_dir()
            else []
        )
        if not files:
            return _count("no_files")
        from podcast_scraper.db.engines import pipeline_rw_engine

        with pipeline_rw_engine().begin() as conn:
            item_id = item_id_for(conn, natural_id)
            if item_id is None:
                return _count("no_item_row")
            synced = sync_documents(conn, item_id, files, change_kind=change_kind)
        return _count(_outcome(synced))
    except (DocumentRefused, KeyNotDerivable) as exc:
        return _fail("refused", metadata_path, exc)
    except Exception as exc:  # noqa: BLE001 - a corpus write never stops the pipeline (yet)
        return _fail("failed", metadata_path, exc)


def _count(outcome: str) -> str:
    OUTCOMES[outcome] += 1
    return outcome


def _fail(outcome: str, metadata_path: Path, exc: BaseException) -> str:
    _count(outcome)
    logger.warning("corpus write %s for %s: %s", outcome, metadata_path.name, exc)
    if _strict():
        raise SyncFailed(f"corpus write {outcome} for {metadata_path.name}: {exc}") from exc
    return outcome
