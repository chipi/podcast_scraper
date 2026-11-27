"""The audio dedup gate (#1656) on ``audio_fingerprints`` rows (tracker w-public, corpus_0012).

The same four operations ``utils.audio_fingerprint`` runs on a feed's
``.podcast_scraper/audio-fingerprints.json``, keyed by the feed's source row instead of its
directory. Every one is a single statement on the primary key ``(source_id, digest)`` or a range of
it:

- :func:`duplicate_of`: the episode already transcribed from these bytes, unless it is this
  identity (a retry);
- :func:`record`: a transcript exists for these bytes; a re-record keeps the identities already
  skipped as its copies, as the file version does;
- :func:`record_duplicate`: an identity was skipped as a copy; nothing when the digest has no row
  (a same-run twin's original may still fail) or the identity is already listed;
- :func:`known_duplicate_identities`: every identity a source's gate skipped, which
  ``episode_selection=unprocessed`` must not offer again.

:func:`source_for_feed` finds the source by ``raw_feed_id``, the ``sha256:`` of the normalised feed
URL (``metadata_generation.generate_feed_id``), so the gate needs no metadata file to find it. A
row's ``item_id`` is its source's item whose external id is the identity, as the importer links
it.
"""

from __future__ import annotations

import collections
import logging
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Connection

from podcast_scraper.db.schema import audio_fingerprints, content_items, sources

logger = logging.getLogger(__name__)

_FP = audio_fingerprints.c


def source_for_feed(conn: Connection, raw_feed_id: str) -> Optional[int]:
    """The source row of this feed id, or None when no item of the feed has been written."""
    found: Optional[int] = conn.execute(
        select(sources.c.source_id).where(sources.c.raw_feed_id == raw_feed_id)
    ).scalar_one_or_none()
    return found


def duplicate_of(
    conn: Connection, source_id: int, digest: str, identity: str
) -> Optional[dict[str, Any]]:
    """The episode recorded under ``digest`` when it is not ``identity``: its ``identity``,
    ``episode_title``, ``transcript_path`` and ``recorded_at``; else None."""
    row = (
        conn.execute(
            select(_FP.identity, _FP.episode_title, _FP.transcript_path, _FP.recorded_at).where(
                _FP.source_id == source_id, _FP.digest == digest, _FP.identity != identity
            )
        )
        .mappings()
        .first()
    )
    return dict(row) if row is not None else None


def record(
    conn: Connection,
    source_id: int,
    digest: str,
    *,
    identity: str,
    episode_title: Optional[str],
    transcript_path: Optional[str],
    recorded_at: datetime,
) -> None:
    """Record that a transcript exists for ``digest``; a second record of the digest replaces
    everything but ``duplicates``."""
    item = (
        select(content_items.c.item_id)
        .where(
            content_items.c.source_id == source_id,
            content_items.c.external_item_id == identity,
        )
        .scalar_subquery()
    )
    values = {
        "identity": identity,
        "item_id": item,
        "episode_title": episode_title,
        "transcript_path": transcript_path,
        "recorded_at": recorded_at,
    }
    statement = insert(audio_fingerprints).values(source_id=source_id, digest=digest, **values)
    conn.execute(
        statement.on_conflict_do_update(
            index_elements=[_FP.source_id, _FP.digest],
            set_={name: statement.excluded[name] for name in values},
        )
    )


def record_duplicate(conn: Connection, source_id: int, digest: str, identity: str) -> bool:
    """Add ``identity`` to the copies skipped as ``digest``'s episode; whether a row changed."""
    result = conn.execute(
        audio_fingerprints.update()
        .where(
            _FP.source_id == source_id,
            _FP.digest == digest,
            ~_FP.duplicates.contains([identity]),
        )
        .values(duplicates=func.array_append(_FP.duplicates, identity))
    )
    return bool(result.rowcount)


def known_duplicate_identities(conn: Connection, source_id: int) -> set[str]:
    """Every identity this source's gate skipped as a copy of an episode already transcribed."""
    copies = func.unnest(_FP.duplicates).label("identity")
    rows = conn.execute(select(copies).where(_FP.source_id == source_id))
    return {str(row.identity) for row in rows}


# --- the gate's side, once the corpus is served from rows ------------------------------------


@dataclass(frozen=True)
class Held:
    """A fingerprint recorded before its episode's rows are written."""

    identity: str
    episode_title: Optional[str]
    transcript_path: Optional[str]
    recorded_at: datetime


#: raw feed id -> digest -> the fingerprint, until the episode's ``sync_rows`` writes it. The gate
#: records right after the transcript is saved, before the metadata stage, so a feed's first
#: episode has no source row yet, and the gate has no feed title to create one with; the sync
#: that creates the source writes the held fingerprints in the same transaction, under its
#: leases. A process that dies between the two loses the fingerprint: that republish costs one
#: more transcription, as with a gate that never saw it.
_HELD: dict[str, dict[str, Held]] = {}
_HELD_LOCK = threading.Lock()

#: How each of the gate's row operations ended, since the process started.
OUTCOMES: collections.Counter[str] = collections.Counter()


def hold(raw_feed_id: str, digest: str, fingerprint: Held) -> None:
    """Hold a fingerprint until its feed's next ``sync_rows`` writes it."""
    with _HELD_LOCK:
        _HELD.setdefault(raw_feed_id, {})[digest] = fingerprint


def held(raw_feed_id: str, digest: str) -> Optional[Held]:
    """The fingerprint held for this feed and digest, if any."""
    with _HELD_LOCK:
        return _HELD.get(raw_feed_id, {}).get(digest)


def write_held(conn: Connection, raw_feed_id: Optional[str], source_id: int) -> list[str]:
    """Write this feed's held fingerprints in the caller's transaction; the digests written, for
    :func:`forget_held` once it commits."""
    if raw_feed_id is None:
        return []
    with _HELD_LOCK:
        pending = dict(_HELD.get(raw_feed_id, {}))
    for digest, f in sorted(pending.items()):
        record(
            conn,
            source_id,
            digest,
            identity=f.identity,
            episode_title=f.episode_title,
            transcript_path=f.transcript_path,
            recorded_at=f.recorded_at,
        )
    return sorted(pending)


def link_item(conn: Connection, source_id: int, external_item_id: str, item_id: int) -> int:
    """Link this item's fingerprints that have no item yet; the rows linked.

    :func:`write_held` writes every fingerprint held for the feed, so one written in another
    episode's sync, before its own item row existed, has no ``item_id`` until its own sync.
    """
    result = conn.execute(
        audio_fingerprints.update()
        .where(
            _FP.source_id == source_id,
            _FP.identity == external_item_id,
            _FP.item_id.is_(None),
        )
        .values(item_id=item_id)
    )
    return int(result.rowcount)


def forget_held(raw_feed_id: Optional[str], digests: list[str]) -> None:
    """Drop the held fingerprints :func:`write_held` wrote, once their transaction committed."""
    if raw_feed_id is None or not digests:
        return
    with _HELD_LOCK:
        feed = _HELD.get(raw_feed_id, {})
        for digest in digests:
            feed.pop(digest, None)
        if not feed:
            _HELD.pop(raw_feed_id, None)


def _engine() -> Any:
    from podcast_scraper.db.engines import pipeline_rw_engine

    return pipeline_rw_engine()


def _source(conn: Connection, raw_feed_id: str) -> Optional[tuple[int, str]]:
    row = conn.execute(
        select(sources.c.source_id, sources.c.natural_id).where(
            sources.c.raw_feed_id == raw_feed_id
        )
    ).first()
    return (row.source_id, row.natural_id) if row is not None else None


def gate_duplicate_of(raw_feed_id: str, digest: str, identity: str) -> Optional[dict[str, Any]]:
    """The gate's check on rows: a fingerprint held for this feed under another identity (its
    transcript exists, its row is not written yet), else the row's episode."""
    waiting = held(raw_feed_id, digest)
    if waiting is not None:
        if waiting.identity == identity:
            return None
        return {
            "identity": waiting.identity,
            "episode_title": waiting.episode_title,
            "in_flight": True,
        }
    from podcast_scraper.db.engines import read_connection

    with read_connection(_engine()) as conn:
        found = _source(conn, raw_feed_id)
        return None if found is None else duplicate_of(conn, found[0], digest, identity)


def gate_known_duplicates(raw_feed_id: str) -> set[str]:
    """The gate's skipped copies of this feed on rows; none for a feed with no source row."""
    from podcast_scraper.db.engines import read_connection

    with read_connection(_engine()) as conn:
        found = _source(conn, raw_feed_id)
        return set() if found is None else known_duplicate_identities(conn, found[0])


def gate_record_duplicate(raw_feed_id: str, digest: str, identity: str) -> str:
    """Add a skipped copy under the ingest's leases (``corpus`` shared, the source exclusive);
    returns and counts ``recorded``, ``listed`` (already there), ``no_row`` (the digest's
    fingerprint is not a row: held, or none) or ``lease_refused``. A copy not recorded is offered
    again next run and skipped again by its digest, so nothing is deferred."""
    from podcast_scraper.db import leases
    from podcast_scraper.db.engines import read_connection

    engine = _engine()
    with read_connection(engine) as conn:
        found = _source(conn, raw_feed_id)
    if found is None:
        OUTCOMES["no_row"] += 1
        return "no_row"
    source_id, source_natural_id = found
    holder = f"pipeline:{uuid.uuid4().hex[:12]}:fingerprint"
    try:
        taken = leases.acquire(
            engine,
            [
                leases.LeaseRequest(leases.CORPUS, leases.SHARED),
                leases.LeaseRequest(leases.source_resource(source_natural_id), leases.EXCLUSIVE),
            ],
            holder,
        )
    except leases.LeaseRefused:
        OUTCOMES["lease_refused"] += 1
        return "lease_refused"
    try:
        with engine.begin() as conn:
            leases.heartbeat(conn, holder, taken)
            if record_duplicate(conn, source_id, digest, identity):
                outcome = "recorded"
            elif conn.execute(
                select(_FP.digest).where(_FP.source_id == source_id, _FP.digest == digest)
            ).first():
                outcome = "listed"
            else:
                outcome = "no_row"
    finally:
        try:
            leases.release(engine, holder)
        except Exception as exc:  # noqa: BLE001 - an unreleased lease expires with its TTL
            logger.warning("audio fingerprint: could not release %s: %s", holder, exc)
    OUTCOMES[outcome] += 1
    return outcome
