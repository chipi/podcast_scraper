"""The pipeline's writes to the corpus database: an item's source and item rows, its transcript
documents and their segments.

CRFC-005 §3.5, tracker row ``w-public``. After a stage writes an episode's files the pipeline syncs
them, one transaction per item on the ``pipeline_rw`` engine
(:func:`~podcast_scraper.db.engines.pipeline_rw_engine`). The files stay the source until the
cut-over; no reader reads these rows before it.

Each sync takes its own job leases (:mod:`podcast_scraper.db.leases`): ``corpus`` shared, then its
source's exclusive, released when the sync ends; the transaction starts with their heartbeat. The
lease lives for one episode, so its TTL never needs a heartbeat between episodes. While a row
migration or ``db migrate`` holds ``corpus`` exclusive, every episode synced in that window is
``lease_refused`` and has no rows until something syncs it again: before the cut-over the
importer's run writes every item from the files, and after it the strict setting fails the item,
which the run's retry path picks up. The importer takes no lease: it runs into an empty database
with the pipeline stopped (the cut-over runbook), so the two never write one source at once.

The rules, settled with the database session (tracker ``w-public``, ``note_rules_2``):

- The files and their variants are the importer's (``db.importers.corpus_documents``
  ``transcript_files``), read the same way (``read_body``), so a document hashes the same
  whichever side wrote it.
- The source row is upserted on its natural id. A served item that is its source's newest
  (:func:`~podcast_scraper.db.corpus_rows.newest_key`, the order ``source_row_for`` uses)
  overwrites the source's columns with each one it states; an older served item only fills the
  columns still null; an item not served changes nothing but inserts a missing source. A known
  gap: when the newest served item later stops being served, the source keeps its columns, where
  ``source_row_for`` over every item would take the next newest's; seeing that needs every served
  item's metadata, so the cut-over import reconciles it, and after it a row migration does.
- The item row is upserted on its natural id, every column ``item_row`` decides overwritten.
  ``has_transcript`` is this writer's: whether the item has a transcript document after the sync.
  The other ``has_*`` flags, the publisher and the artwork are their writers'. An item whose
  stored source differs, or whose slug or ``(source_id, external_item_id)`` is another item's, is
  refused: moving or merging items is a migration.
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
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import yaml
from sqlalchemy import and_, func, insert, or_, select, update
from sqlalchemy.engine import Connection

from podcast_scraper.db import leases
from podcast_scraper.db.corpus_keys import (
    is_news,
    item_natural_id,
    KeyNotDerivable,
    source_natural_id,
)
from podcast_scraper.db.corpus_rows import (
    item_row,
    newest_key,
    NewestKey,
    SERVED,
    served_for,
    source_row,
)
from podcast_scraper.db.importers.corpus_documents import (
    DocumentRefused,
    joined_body,
    read_body,
    read_segments,
    TRANSCRIPT,
    transcript_files,
    TranscriptFile,
)
from podcast_scraper.db.schema import content_items, documents, segments, sources
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


_SOURCE_COLUMNS = ("title", "language", "category", "licence")


class SyncFailed(RuntimeError):
    """A sync failed while ``PODCAST_CORPUS_WRITE_STRICT`` is set."""


class RowRefused(RuntimeError):
    """The item's rows contradict the stored ones in a way only a migration may change."""


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


def _newest_other(conn: Connection, source_id: int, natural_id: str) -> Optional[NewestKey]:
    """The newest served item of the source other than ``natural_id``, as :func:`newest_key`."""
    row = conn.execute(
        select(content_items.c.published_at, content_items.c.natural_id)
        .where(
            and_(
                content_items.c.source_id == source_id,
                content_items.c.served > 0,
                content_items.c.natural_id != natural_id,
            )
        )
        .order_by(
            content_items.c.published_at.desc().nulls_last(), content_items.c.natural_id.desc()
        )
        .limit(1)
    ).first()
    return newest_key(row[0], row[1]) if row else None


def upsert_source(conn: Connection, doc: Mapping[str, Any]) -> int:
    """Insert or update the source row of a podcast item ``doc``; returns its ``source_id``."""
    wanted = source_row(doc)
    stored = (
        conn.execute(select(sources).where(sources.c.natural_id == wanted["natural_id"]))
        .mappings()
        .first()
    )
    if stored is None:
        source_id: int = conn.execute(
            insert(sources).values(**wanted).returning(sources.c.source_id)
        ).scalar_one()
        return source_id
    source_id = stored["source_id"]
    if stored["content_type"] != wanted["content_type"] or (
        stored["raw_feed_id"] is not None
        and wanted["raw_feed_id"] is not None
        and stored["raw_feed_id"] != wanted["raw_feed_id"]
    ):
        raise RowRefused(f"source {wanted['natural_id']}: its stored type or feed id differs")
    if served_for(doc) != SERVED:
        return source_id
    item = item_row(doc)
    other = _newest_other(conn, source_id, item["natural_id"])
    newest = other is None or newest_key(item["published_at"], item["natural_id"]) > other
    changes = {
        column: wanted[column]
        for column in ("raw_feed_id",) + _SOURCE_COLUMNS
        if wanted[column] is not None
        and stored[column] != wanted[column]
        and (stored[column] is None or (newest and column != "raw_feed_id"))
    }
    if changes:
        conn.execute(
            update(sources)
            .where(sources.c.source_id == source_id)
            .values(**changes, updated_at=func.now())
        )
    return source_id


def upsert_item(conn: Connection, doc: Mapping[str, Any], source_id: int) -> int:
    """Insert or update the item row of a podcast item ``doc`` under ``source_id``; returns its
    ``item_id``."""
    wanted = {**item_row(doc), "source_id": source_id}
    clash = conn.execute(
        select(content_items.c.natural_id)
        .where(
            and_(
                content_items.c.natural_id != wanted["natural_id"],
                or_(
                    content_items.c.slug == wanted["slug"],
                    and_(
                        content_items.c.source_id == source_id,
                        content_items.c.external_item_id == wanted["external_item_id"],
                    ),
                ),
            )
        )
        .limit(1)
    ).scalar_one_or_none()
    if clash is not None:
        raise RowRefused(f"item {wanted['natural_id']}: its slug or source id is {clash}'s")
    stored = (
        conn.execute(
            select(content_items).where(content_items.c.natural_id == wanted["natural_id"])
        )
        .mappings()
        .first()
    )
    if stored is None:
        item_id: int = conn.execute(
            insert(content_items).values(**wanted).returning(content_items.c.item_id)
        ).scalar_one()
        return item_id
    if stored["source_id"] != source_id:
        raise RowRefused(f"item {wanted['natural_id']}: it is stored under another source")
    changes = {column: value for column, value in wanted.items() if stored[column] != value}
    if changes:
        conn.execute(
            update(content_items)
            .where(content_items.c.item_id == stored["item_id"])
            .values(**changes, updated_at=func.now())
        )
    item_id = stored["item_id"]
    return item_id


def mark_has_transcript(conn: Connection, item_id: int) -> bool:
    """Set the item's ``has_transcript`` to whether it has a transcript document; returns it."""
    has = bool(
        conn.execute(
            select(func.count()).where(
                and_(documents.c.item_id == item_id, documents.c.kind == TRANSCRIPT)
            )
        ).scalar_one()
    )
    conn.execute(
        update(content_items)
        .where(and_(content_items.c.item_id == item_id, content_items.c.has_transcript != has))
        .values(has_transcript=has, updated_at=func.now())
    )
    return has


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
    metadata_path: Path, transcript_path: Optional[Path], *, change_kind: str = RETRANSCRIBED
) -> str:
    """Sync one podcast episode's source and item rows and its transcript documents; returns and
    counts the outcome.

    The outcome names the documents: ``written`` (version 1 of every variant written),
    ``superseded`` (a later version written: its ``change_kind`` is the caller's, so the metadata
    stage's ``retranscribed`` is assumed until the translation, repunctuate and retranscript_only
    stages sync themselves), ``unchanged`` or ``no_files`` (no ``transcript_path``, or no
    transcript body beside it); the rows are upserted in each of these. Otherwise nothing is
    written: ``lease_refused`` (another job holds a conflicting lease), ``lease_lost``,
    ``refused`` (the keys cannot be derived, a stored row contradicts the item, or a body does not
    cut), ``failed`` (anything else), ``news_item`` (not this slice's) or ``not_configured`` (no
    pipeline database URL). Never raises unless the strict setting is on. An item the pipeline
    marked not served is synced too, with ``served`` 0: which items are served is the read side's.
    """
    if not pipeline_is_configured():
        return _count("not_configured")
    engine = None
    holder = ""
    try:
        doc = _load(metadata_path)
        if is_news(doc):
            return _count("news_item")
        natural_id = item_natural_id(doc)
        source = source_natural_id(doc)
        files = (
            transcript_files(
                transcript_path.parent, _base(transcript_path), episode_language(dict(doc))
            )
            if transcript_path is not None and transcript_path.parent.is_dir()
            else []
        )
        from podcast_scraper.db.engines import pipeline_rw_engine

        engine = pipeline_rw_engine()
        holder = f"pipeline:{uuid.uuid4().hex[:12]}:{natural_id}"
        held = leases.acquire(
            engine,
            [
                leases.LeaseRequest(leases.CORPUS, leases.SHARED),
                leases.LeaseRequest(leases.source_resource(source), leases.EXCLUSIVE),
            ],
            holder,
        )
        with engine.begin() as conn:
            leases.heartbeat(conn, holder, held)
            item_id = upsert_item(conn, doc, upsert_source(conn, doc))
            synced = sync_documents(conn, item_id, files, change_kind=change_kind)
            mark_has_transcript(conn, item_id)
        return _count(_outcome(synced) if files else "no_files")
    except leases.LeaseRefused as exc:
        return _fail("lease_refused", metadata_path, exc)
    except leases.LeaseLost as exc:
        return _fail("lease_lost", metadata_path, exc)
    except (DocumentRefused, KeyNotDerivable, RowRefused) as exc:
        return _fail("refused", metadata_path, exc)
    except Exception as exc:  # noqa: BLE001 - a corpus write never stops the pipeline (yet)
        return _fail("failed", metadata_path, exc)
    finally:
        if engine is not None and holder:
            _release(engine, holder)


def _release(engine: Any, holder: str) -> None:
    try:
        leases.release(engine, holder)
    except Exception as exc:  # noqa: BLE001 - an unreleased lease expires with its TTL
        logger.warning("corpus write could not release %s: %s", holder, exc)


def _count(outcome: str) -> str:
    OUTCOMES[outcome] += 1
    return outcome


def _fail(outcome: str, metadata_path: Path, exc: BaseException) -> str:
    _count(outcome)
    logger.warning("corpus write %s for %s: %s", outcome, metadata_path.name, exc)
    if _strict():
        raise SyncFailed(f"corpus write {outcome} for {metadata_path.name}: {exc}") from exc
    return outcome
