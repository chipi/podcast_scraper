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

An episode refused or losing its lease is not left without rows: it is remembered, retried once
when the run ends (:func:`flush_deferred`), written to the run directory's ``DEFERRED_FILE`` if
still refused, and retried from there when a later run starts (:func:`retry_deferred`) until it
syncs. A sync that already landed writes nothing, so a repeated retry is harmless.

The rules, settled with the database session (tracker ``w-public``, ``note_rules_2``):

- The files and their variants are the importer's (``db.importers.corpus_documents``
  ``transcript_files``), read the same way (``read_body``), so a document hashes the same
  whichever side wrote it.
- The source row is found on its natural id, inserted from the item when missing. Each item row
  stores the source columns its own metadata states (``source_attrs``, corpus_0008); after the
  item is written, the source's columns are merged again from every served item's
  ``source_attrs``, newest first (:func:`~podcast_scraper.db.corpus_rows.newest_key`), by the
  merge ``source_row_for`` uses. So the row is the same whatever order items arrive in, and an
  item that stops being served stops deciding it. A source with no served item keeps its columns.
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
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import yaml
from sqlalchemy import and_, delete, exists, func, insert, or_, select, update
from sqlalchemy.engine import Connection

from podcast_scraper.builders.bridge_artifact_paths import bridge_json_path_adjacent_to_metadata
from podcast_scraper.db import audio_fingerprints, leases
from podcast_scraper.db.corpus_cards import card_row, FLAGS
from podcast_scraper.db.corpus_keys import (
    external_item_id,
    is_news,
    item_natural_id,
    KeyNotDerivable,
    raw_feed_id,
    source_natural_id,
)
from podcast_scraper.db.corpus_rows import (
    _instant,
    item_row,
    merge_source_attrs,
    newest_key,
    source_attrs,
    source_row,
    unreadable_times,
)
from podcast_scraper.db.importers.corpus_documents import (
    ARTICLE,
    DocumentRefused,
    joined_body,
    read_body,
    read_segments,
    TRANSCRIPT,
    transcript_files,
    TranscriptFile,
)
from podcast_scraper.db.schema import content_items, documents, item_cards, segments, sources
from podcast_scraper.db.settings import pipeline_is_configured
from podcast_scraper.document_cut import cut_body, DocumentCutError, paragraph_cuts
from podcast_scraper.languages import episode_language
from podcast_scraper.publishers import load_registry, Registry
from podcast_scraper.server.corpus_catalog import _gi_kg_relpaths_from_metadata
from podcast_scraper.utils.corpus_walk import corpus_rglob

logger = logging.getLogger(__name__)

ENV_STRICT = "PODCAST_CORPUS_WRITE_STRICT"

CREATED = "created"
RETRANSCRIBED = "retranscribed"
REPUNCTUATED = "repunctuated"
TRANSLATED = "translated"
CORRECTED = "corrected"
#: A news article's later version (CRFC-006 R6); a news ``corrected`` is only for a version a source's
#: notice pattern or the operator marks so, and nothing marks one yet.
UPDATED = "updated"
CHANGE_KINDS = (RETRANSCRIBED, REPUNCTUATED, TRANSLATED, CORRECTED, UPDATED)

#: The change a later version of an episode's transcript records, by the run's ``pipeline_stage``.
#: Every reprocess stage reaches the metadata stage, whose hook syncs the episode, so the stage
#: names its change there: ``translate_only`` re-translates, ``repunctuate_only`` re-transcribes
#: windows to restore punctuation, and the stages that keep the transcription and change the text
#: around it (speaker names: ``relabel_only``, ``rediarize_only``; the cleaned render:
#: ``rederive_only``) correct it. A full run, ``retranscript_only`` and any other stage
#: re-transcribe.
STAGE_CHANGE_KINDS = {
    "translate_only": TRANSLATED,
    "repunctuate_only": REPUNCTUATED,
    "relabel_only": CORRECTED,
    "rediarize_only": CORRECTED,
    "rederive_only": CORRECTED,
}


def change_kind_for(pipeline_stage: Optional[str]) -> str:
    """The ``change_kind`` a sync in a run of ``pipeline_stage`` records: its
    :data:`STAGE_CHANGE_KINDS` entry, else ``retranscribed``."""
    return STAGE_CHANGE_KINDS.get(pipeline_stage or "", RETRANSCRIBED)


#: How each sync ended, by outcome, since the process started; the run's metrics read it.
OUTCOMES: collections.Counter[str] = collections.Counter()
#: Items written with a time column null because the metadata states it unreadably, by column
#: (``corpus_rows.unreadable_times``), since the process started.
UNREADABLE: collections.Counter[str] = collections.Counter()

_SEGMENTS_BATCH = 1000

#: Items written with no ``type_attrs.metadata_relative_path``: the metadata file is in no corpus
#: layout ``corpus_scope.classify_metadata_relpath`` knows, since the process started.
UNPLACED: collections.Counter[str] = collections.Counter()

#: The run directory's record of episodes whose sync is to be tried again, one JSON object a line.
DEFERRED_FILE = "corpus_sync_deferred.jsonl"
#: This process's episodes whose sync is to be tried again (a lease refused or lost, or the write
#: ``failed``): metadata path to (transcript, change_kind). ``refused`` is final: the same files
#: would be refused again.
_DEFERRED: dict[str, tuple[Optional[str], str]] = {}


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


#: ``documents.change_kind`` of a withdrawal (``db.news_recheck.withdraw``, CRFC-006 B.1).
WITHDRAW = "withdraw"


def is_withdrawn(conn: Connection, natural_id: str) -> bool:
    """Whether the item's newest version of any document series is a withdrawal: the re-check
    withdrew it on the rows, and its files, which still state it served, must not undo that.

    A withdrawal is final: no code path writes a version over a withdraw row (the row sync stops
    here, and a withdrawn item is never re-checked again). Undoing a wrong one is a row migration
    written by hand that adds the next version over the withdraw row (CRFC-006 B.5: a wrong
    version is undone by a new version, never a delete); then this is false and the files sync
    again."""
    newer = documents.alias("newer")
    return bool(
        conn.execute(
            select(
                exists()
                .select_from(
                    documents.join(content_items, content_items.c.item_id == documents.c.item_id)
                )
                .where(
                    content_items.c.natural_id == natural_id,
                    documents.c.change_kind == WITHDRAW,
                    ~exists().where(newer.c.supersedes_id == documents.c.document_id),
                )
            )
        ).scalar()
    )


def item_id_for(conn: Connection, natural_id: str) -> Optional[int]:
    """The ``content_items`` row id of ``natural_id``, or ``None`` when there is no row."""
    found: Optional[int] = conn.execute(
        select(content_items.c.item_id).where(content_items.c.natural_id == natural_id)
    ).scalar_one_or_none()
    return found


def _newest(
    conn: Connection, item_id: int, variant: str, kind: str = TRANSCRIPT
) -> Optional[tuple[int, int, str]]:
    row = conn.execute(
        select(documents.c.document_id, documents.c.version, documents.c.body_sha256)
        .where(
            and_(
                documents.c.item_id == item_id,
                documents.c.kind == kind,
                documents.c.variant == variant,
            )
        )
        .order_by(documents.c.version.desc())
        .limit(1)
    ).first()
    return (row[0], row[1], row[2]) if row else None


def ensure_source(
    conn: Connection, doc: Mapping[str, Any], *, registry: Optional[Registry] = None
) -> int:
    """The ``source_id`` of item ``doc``'s source, inserting the row from ``doc`` when it is
    missing; :func:`merge_source` decides its columns once the item's row is written. A news
    item's source needs ``registry`` for its title."""
    wanted = source_row(doc, registry=registry)
    stored = (
        conn.execute(
            select(sources.c.source_id, sources.c.content_type, sources.c.raw_feed_id).where(
                sources.c.natural_id == wanted["natural_id"]
            )
        )
        .mappings()
        .first()
    )
    if stored is None:
        if wanted["title"] is None:
            raise RowRefused(f"source {wanted['natural_id']}: the metadata names no source title")
        source_id: int = conn.execute(
            insert(sources).values(**wanted).returning(sources.c.source_id)
        ).scalar_one()
        return source_id
    if stored["content_type"] != wanted["content_type"] or (
        stored["raw_feed_id"] is not None
        and wanted["raw_feed_id"] is not None
        and stored["raw_feed_id"] != wanted["raw_feed_id"]
    ):
        raise RowRefused(f"source {wanted['natural_id']}: its stored type or feed id differs")
    source_id = stored["source_id"]
    return source_id


def merge_source(conn: Connection, source_id: int) -> bool:
    """Set the source's columns to :func:`~podcast_scraper.db.corpus_rows.merge_source_attrs`
    over its served items' ``source_attrs``, newest first (``newest_key``); returns whether a
    column changed. A source with no served item keeps its columns."""
    rows = conn.execute(
        select(
            content_items.c.published_at, content_items.c.natural_id, content_items.c.source_attrs
        ).where(and_(content_items.c.source_id == source_id, content_items.c.served > 0))
    ).all()
    if not rows:
        return False
    ordered = sorted(rows, key=lambda row: newest_key(row[0], row[1]), reverse=True)
    wanted = merge_source_attrs(row[2] for row in ordered)
    stored = conn.execute(select(sources).where(sources.c.source_id == source_id)).mappings().one()
    changes = {
        column: value
        for column, value in wanted.items()
        if stored[column] != value and (value is not None or column != "title")
    }
    if not changes:
        return False
    conn.execute(
        update(sources)
        .where(sources.c.source_id == source_id)
        .values(**changes, updated_at=func.now())
    )
    return True


def corpus_relpath(metadata_path: Path) -> Optional[str]:
    """The metadata file's path from the served corpus root: the longest end of ``metadata_path``
    that ``corpus_scope.classify_metadata_relpath`` recognises as a corpus layout (the importer
    passes the same path, relative to the corpus root); ``None`` when there is none."""
    from podcast_scraper.search.corpus_scope import classify_metadata_relpath

    parts = metadata_path.parts[1:] if metadata_path.is_absolute() else metadata_path.parts
    for start in range(len(parts)):
        rel = "/".join(parts[start:])
        if classify_metadata_relpath(rel) is not None:
            return rel
    return None


def upsert_item(
    conn: Connection,
    doc: Mapping[str, Any],
    source_id: int,
    *,
    metadata_relpath: str = "",
    has_bridge: Optional[bool] = None,
    registry: Optional[Registry] = None,
) -> int:
    """Insert or update the item row of ``doc`` under ``source_id``; returns its ``item_id``.
    ``metadata_relpath`` and ``has_bridge`` go to :func:`item_row`; a news item's
    ``source_attrs`` need ``registry``."""
    wanted = {
        **item_row(doc, metadata_relpath=metadata_relpath, has_bridge=has_bridge),
        "source_id": source_id,
        "source_attrs": source_attrs(doc, registry=registry),
    }
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


def bump_revision(conn: Connection, item_id: int, source_id: int, wrote: bool) -> bool:
    """Bump ``corpus_revision`` when this transaction wrote content; returns whether it did. Last
    in the transaction (DECISIONS rule 7): the graph's slices and the catalog's token read it.
    Content is ``wrote`` (a document version, a card) or the item's or source's row, whose
    ``updated_at`` every writer sets to ``now()``, this transaction's start. That start is also why
    the catalog needs the revision: a sync that began before another committed, and commits after
    it, leaves the newest ``updated_at`` where it was (#2614). An unchanged sync bumps nothing."""
    from podcast_scraper.upgrade.rows import bump_corpus_revision

    if not wrote:
        wrote = bool(
            conn.execute(
                select(
                    or_(
                        exists().where(
                            and_(
                                content_items.c.item_id == item_id,
                                content_items.c.updated_at == func.now(),
                            )
                        ),
                        exists().where(
                            and_(
                                sources.c.source_id == source_id, sources.c.updated_at == func.now()
                            )
                        ),
                    )
                )
            ).scalar_one()
        )
    return wrote and bump_corpus_revision(conn)


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


def write_card(
    conn: Connection, corpus_root: Path, metadata_relpath: str, item_id: int, item: dict[str, Any]
) -> bool:
    """The episode's ``item_cards`` row and its item's ``has_*`` flags (CRFC-005 §3.2: the item's
    writers write its card in the item's transaction), from ``corpus_cards.card_row``, the
    importer's; returns whether it changed the card (written, changed or removed). One the catalog
    does not show (unusable) has none, so a card written before is removed and its flags are
    cleared, as the importer leaves an item it writes no card for. A card that cannot be read
    (``ValueError``, a corrupt GI file) refuses the item, as the importer refuses it, rather than
    failing it on every run. The flags are the item row's: :func:`bump_revision` sees them there."""
    try:
        card = card_row(corpus_root, metadata_relpath, item)
    except ValueError as exc:
        raise RowRefused(f"item {item.get('natural_id')}: its card cannot be read: {exc}") from exc
    if card is None:
        removed = conn.execute(delete(item_cards).where(item_cards.c.item_id == item_id)).rowcount
        conn.execute(
            update(content_items)
            .where(
                and_(
                    content_items.c.item_id == item_id,
                    or_(*(content_items.c[flag].is_(True) for flag in FLAGS)),
                )
            )
            .values(**{flag: False for flag in FLAGS}, updated_at=func.now())
        )
        return bool(removed)
    changed = True
    stored = (
        conn.execute(select(item_cards).where(item_cards.c.item_id == item_id)).mappings().first()
    )
    if stored is None:
        conn.execute(insert(item_cards).values(item_id=item_id, **card))
    else:
        changes = {column: value for column, value in card.items() if stored[column] != value}
        changed = bool(changes)
        if changes:
            conn.execute(
                update(item_cards)
                .where(item_cards.c.item_id == item_id)
                .values(**changes, updated_at=func.now())
            )
    flags = {flag: card[flag] for flag in FLAGS}
    conn.execute(
        update(content_items)
        .where(
            and_(
                content_items.c.item_id == item_id,
                or_(*(content_items.c[flag] != value for flag, value in flags.items())),
            )
        )
        .values(**flags, updated_at=func.now())
    )
    return changed


_REGISTRY: Optional[Registry] = None


def _registry() -> Registry:
    """The publisher registry a news source's title comes from, read once per process."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = load_registry()
    return _REGISTRY


def _corpus_root(metadata_path: Path, metadata_relpath: str) -> Path:
    return metadata_path.parents[len(Path(metadata_relpath).parts) - 1]


def artifact_paths(metadata_path: Path) -> tuple[Optional[Path], Optional[Path]]:
    """The item's GI and KG files beside its metadata (the catalog's naming rule), each ``None``
    when it is not there."""
    gi, kg = _gi_kg_relpaths_from_metadata(str(metadata_path))
    return (
        Path(gi) if os.path.isfile(gi) else None,
        Path(kg) if os.path.isfile(kg) else None,
    )


def _segment_count(conn: Connection, item_id: int, document_id: int) -> int:
    count: int = conn.execute(
        select(func.count()).where(
            and_(segments.c.item_id == item_id, segments.c.document_id == document_id)
        )
    ).scalar_one()
    return count


def sync_documents(
    conn: Connection,
    item_id: int,
    files: Sequence[TranscriptFile],
    *,
    change_kind: str,
    kind: str = TRANSCRIPT,
    fetched_at: Optional[datetime] = None,
    body_sha256: Optional[str] = None,
    extraction: Optional[str] = None,
    extractor_version: Optional[str] = None,
) -> list[SyncedDocument]:
    """Write each file whose body differs from its variant's newest version of ``kind``, in the
    caller's transaction. ``change_kind`` names why a later version exists; version 1 is
    ``created``. An ``article`` is cut into paragraphs (``paragraph_cuts``) and carries
    ``fetched_at``, ``extraction`` and ``extractor_version`` (G9), and its ``raw`` body must hash
    to the metadata's ``body_sha256`` when one is given, as the importer writes it."""
    if change_kind not in CHANGE_KINDS:
        raise ValueError(f"change_kind {change_kind!r} is not one of {CHANGE_KINDS}")
    synced: list[SyncedDocument] = []
    for file in files:
        body = read_body(file.path)
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        if body_sha256 and file.variant == "raw" and digest != body_sha256:
            raise DocumentRefused(
                f"{file.path.name}: its sha256 is not the metadata's body_sha256 {body_sha256}"
            )
        newest = _newest(conn, item_id, file.variant, kind)
        if newest is not None and newest[2] == digest:
            document_id, version, _ = newest
            count = _segment_count(conn, item_id, document_id)
            synced.append(SyncedDocument(file.variant, document_id, version, count, False))
            continue
        try:
            cuts = (
                paragraph_cuts(body)
                if kind == ARTICLE
                else cut_body(body, read_segments(file.segments_path))
            )
        except (DocumentCutError, ValueError) as exc:
            raise DocumentRefused(f"{file.path.name}: {exc}") from exc
        version = 1 if newest is None else newest[1] + 1
        document_id = conn.execute(
            insert(documents)
            .values(
                item_id=item_id,
                kind=kind,
                variant=file.variant,
                version=version,
                supersedes_id=None if newest is None else newest[0],
                change_kind=CREATED if newest is None else change_kind,
                body_sha256=digest,
                fetched_at=fetched_at,
                extraction=extraction,
                extractor_version=extractor_version,
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


#: The outcomes in which the item's rows are written or already right: the artifact hooks run.
ROWS_KEPT = ("written", "superseded", "unchanged", "no_files")


def sync_episode(
    metadata_path: Path, transcript_path: Optional[Path], *, change_kind: str = RETRANSCRIBED
) -> str:
    """Sync one item's rows and documents (:func:`sync_rows`), then, when its rows are kept
    (:data:`ROWS_KEPT`), run every ``artifact_rows`` stage hook on its metadata, GI and KG files
    (:func:`artifact_paths`), after the rows' transaction: those writers own their transaction,
    their counts. A hook that fails is counted in ``OUTCOMES["hook_failed"]`` and deferred like a
    failed row write (retried at the run's end and the next run, its rows then ``unchanged``); a
    hook that cannot succeed returns ``"refused"`` instead. It never stops the others or the
    pipeline, unless ``PODCAST_CORPUS_WRITE_STRICT`` is set (:class:`SyncFailed`)."""
    outcome = sync_rows(metadata_path, transcript_path, change_kind=change_kind)
    if outcome in ROWS_KEPT:
        from podcast_scraper import stage_hooks

        hooked = stage_hooks.artifact_rows(metadata_path, *artifact_paths(metadata_path))
        failed = hooked.count("failed")
        if failed:
            OUTCOMES["hook_failed"] += failed
            _defer(metadata_path, transcript_path, change_kind)
            if _strict():
                raise SyncFailed(f"{failed} artifact_rows hooks failed for {metadata_path.name}")
    return outcome


def sync_rows(
    metadata_path: Path, transcript_path: Optional[Path], *, change_kind: str = RETRANSCRIBED
) -> str:
    """Sync one item's source and item rows and its documents (a podcast episode's transcripts
    and card, a news item's article); returns and counts the outcome.

    The outcome names the documents: ``written`` (version 1 of every variant written),
    ``superseded`` (a later version written, recording ``change_kind``: the metadata hook passes
    :func:`change_kind_for` its run's stage; a news item's later article is ``updated``,
    CRFC-006 R6), ``unchanged`` or ``no_files`` (no ``transcript_path``, or no body beside it);
    the rows are upserted in each of these. Otherwise nothing is written: ``lease_refused``
    (another job holds a conflicting lease), ``lease_lost``, ``refused`` (the keys cannot be
    derived, a stored row contradicts the item, or a body does not cut or hash), ``failed``
    (anything else), ``not_configured`` (no pipeline database URL) or ``withdrawn`` (a news item
    the re-check withdrew, :func:`is_withdrawn`: its rows are left as they are). A news item's
    ``transcript_path`` is its body file, and it has no card yet (the importer writes none
    either). Never raises unless the strict setting is on. An item the pipeline marked not served
    is synced too, with ``served`` 0: which items are served is the read side's.
    """
    if not pipeline_is_configured():
        return _count("not_configured")
    engine = None
    holder = ""
    try:
        doc = _load(metadata_path)
        news = is_news(doc)
        registry = _registry() if news else None
        natural_id = item_natural_id(doc)
        source = source_natural_id(doc)
        relpath = corpus_relpath(metadata_path)
        has_bridge = os.path.isfile(bridge_json_path_adjacent_to_metadata(str(metadata_path)))
        language = doc.get("language") if news else episode_language(dict(doc))
        files = (
            transcript_files(transcript_path.parent, _base(transcript_path), language)
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
            if news and is_withdrawn(conn, natural_id):
                # A withdrawal is the rows' own (CRFC-006 B.1): a sync of the item's files, which
                # still state it served, would serve it again and write its body over the
                # withdraw row.
                _DEFERRED.pop(str(metadata_path), None)
                return _count("withdrawn")
            source_id = ensure_source(conn, doc, registry=registry)
            item_id = upsert_item(
                conn,
                doc,
                source_id,
                metadata_relpath=relpath or "",
                has_bridge=has_bridge,
                registry=registry,
            )
            # The gate's fingerprints held for this feed (db.audio_fingerprints): after the item,
            # so a fingerprint links to it; one an earlier sync wrote for this episode links now.
            feed = None if news else raw_feed_id(doc)
            fingerprints = audio_fingerprints.write_held(conn, feed, source_id)
            if not news:
                audio_fingerprints.link_item(conn, source_id, external_item_id(doc), item_id)
            merge_source(conn, source_id)
            carded = False
            if news:
                synced = sync_documents(
                    conn,
                    item_id,
                    files,
                    change_kind=UPDATED,
                    kind=ARTICLE,
                    fetched_at=_instant(doc.get("fetched_at")),
                    body_sha256=_str_or_none(doc.get("body_sha256")),
                    extraction=_str_or_none(_own(doc).get("extraction")),
                    extractor_version=_str_or_none(_own(doc).get("extractor_version")),
                )
            else:
                synced = sync_documents(conn, item_id, files, change_kind=change_kind)
                mark_has_transcript(conn, item_id)
                if relpath is not None:
                    item = item_row(doc, metadata_relpath=relpath, has_bridge=has_bridge)
                    root = _corpus_root(metadata_path, relpath)
                    carded = write_card(conn, root, relpath, item_id, item)
            bump_revision(conn, item_id, source_id, carded or any(d.written for d in synced))
        audio_fingerprints.forget_held(feed, fingerprints)
        _DEFERRED.pop(str(metadata_path), None)
        if relpath is None:
            UNPLACED["metadata_relative_path"] += 1
            logger.warning(
                "corpus write: %s is in no corpus layout; written with no metadata_relative_path",
                metadata_path,
            )
        for column in unreadable_times(doc):
            UNREADABLE[column] += 1
            logger.warning(
                "corpus write: %s states a %s that is not ISO 8601; written null",
                metadata_path.name,
                column,
            )
        return _count(_outcome(synced) if files else "no_files")
    except (leases.LeaseRefused, leases.LeaseLost) as exc:
        _defer(metadata_path, transcript_path, change_kind)
        outcome = "lease_refused" if isinstance(exc, leases.LeaseRefused) else "lease_lost"
        return _fail(outcome, metadata_path, exc)
    except (DocumentRefused, KeyNotDerivable, RowRefused) as exc:
        return _fail("refused", metadata_path, exc)
    except Exception as exc:  # noqa: BLE001 - a corpus write never stops the pipeline (yet)
        _defer(metadata_path, transcript_path, change_kind)
        return _fail("failed", metadata_path, exc)
    finally:
        if engine is not None and holder:
            _release(engine, holder)


def _defer(metadata_path: Path, transcript_path: Optional[Path], change_kind: str) -> None:
    _DEFERRED[str(metadata_path)] = (
        None if transcript_path is None else str(transcript_path),
        change_kind,
    )


def _own(doc: Mapping[str, Any]) -> Mapping[str, Any]:
    own = doc.get("news_article")
    return own if isinstance(own, Mapping) else {}


def _str_or_none(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value else None


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


def _retry(entries: dict[str, tuple[Optional[str], str]]) -> dict[str, tuple[Optional[str], str]]:
    """Sync each entry once more, never raising; returns the ones to try again still (what
    :func:`sync_episode` put back in ``_DEFERRED``: a lease refused or lost, or ``failed``). Each
    outcome, a strict failure included, is the sync's own and is counted and logged there."""
    left: dict[str, tuple[Optional[str], str]] = {}
    for metadata, (transcript, change_kind) in entries.items():
        _DEFERRED.pop(metadata, None)
        if not Path(metadata).is_file():
            continue
        try:
            sync_episode(
                Path(metadata),
                None if transcript is None else Path(transcript),
                change_kind=change_kind,
            )
        except SyncFailed:
            pass
        if _DEFERRED.pop(metadata, None) is not None:
            left[metadata] = (transcript, change_kind)
    return left


def _write_deferred(path: Path, entries: dict[str, tuple[Optional[str], str]]) -> None:
    if not entries:
        path.unlink(missing_ok=True)
        return
    lines = [
        json.dumps({"metadata": m, "transcript": t, "change_kind": c}, sort_keys=True)
        for m, (t, c) in sorted(entries.items())
    ]
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def flush_deferred(run_dir: str, *, retry: bool = True) -> int:
    """At the end of a run: retry this process's deferred episodes once (not when ``retry`` is
    false: a run that raised keeps them as they are) and write those left to ``run_dir``'s
    :data:`DEFERRED_FILE`; returns how many are left."""
    if not _DEFERRED or not pipeline_is_configured():
        return 0
    left = _retry(dict(_DEFERRED)) if retry else dict(_DEFERRED)
    _DEFERRED.clear()
    _write_deferred(Path(run_dir) / DEFERRED_FILE, left)
    if left:
        logger.warning("corpus write: %d episodes to try again, kept in %s", len(left), run_dir)
    return len(left)


def retry_deferred(output_root: str) -> int:
    """At the start of a run: retry every :data:`DEFERRED_FILE` under ``output_root`` and rewrite
    each with what is still to try again (removed when empty); returns how many are left."""
    if not pipeline_is_configured():
        return 0
    left_total = 0
    for path in sorted(corpus_rglob(Path(output_root), DEFERRED_FILE, include_sources=True)):
        entries: dict[str, tuple[Optional[str], str]] = {}
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    entries[rec["metadata"]] = (rec.get("transcript"), rec["change_kind"])
        except (OSError, ValueError, KeyError) as exc:
            logger.warning("corpus write: unreadable %s: %s", path, exc)
            continue
        left = _retry(entries)
        _write_deferred(path, left)
        left_total += len(left)
    return left_total
