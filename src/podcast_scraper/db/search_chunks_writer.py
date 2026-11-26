"""The index build's transcript chunks as ``search_chunks`` rows (CRFC-005 §6, tracker r-search).

Search stays on LanceDB in E06, but a hit is read back from rows: a transcript chunk's row says
which document it was cut from and where, as a range over that document's segments
(``document_cut.cut_range``), so the lift from a hit to its insight can compare it with
``evidence``'s ranges.

Each chunk the index build cuts (``search.indexer``) is written with:

- ``natural_id``: its LanceDB id (``chunk:<scope>:<n>``, a source-layer chunk ``…:src``);
- ``document_id``: the served transcript document of the variant its body file names
  (``kg.as_v3.document_variant``), and only when that document's ``body_sha256`` is the hash of
  the text that was chunked: a body the pipeline has not synced yet is skipped (``stale``), never
  cut against another text;
- the range: ``cut_range`` over the document's segments, in ``seq`` order;
- ``speakers``: the distinct ``segments.speaker_id`` of the range (empty until the identity
  writers set them).

An item's chunks are replaced together, in one transaction under the job leases the pipeline's
writer takes, and only when they differ from the stored ones, so an unchanged episode costs reads
only. It runs for every episode the build walks, the ones its fingerprint skips included, so an
episode indexed before this existed gets its rows. Failures are logged and counted, never raised
(the index build must not stop), unless ``PODCAST_CORPUS_WRITE_STRICT`` is set.

Not here: chunks of an item the build no longer walks keep their rows until its documents go (the
foreign key cascades); news paragraphs (the ``news`` table) are not ``search_chunks``.
"""

from __future__ import annotations

import collections
import hashlib
import logging
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from sqlalchemy import and_, delete, insert, select
from sqlalchemy.engine import Connection

from podcast_scraper.db import leases
from podcast_scraper.db.corpus_keys import item_natural_id, KeyNotDerivable, source_natural_id
from podcast_scraper.db.schema import content_items, documents, search_chunks, segments
from podcast_scraper.db.settings import pipeline_is_configured
from podcast_scraper.document_cut import Cut, cut_range
from podcast_scraper.kg.as_v3 import document_variant

logger = logging.getLogger(__name__)

ENV_STRICT = "PODCAST_CORPUS_WRITE_STRICT"
TRANSCRIPT = "transcript"

#: How each episode's chunk sync ended, since the process started.
OUTCOMES: collections.Counter[str] = collections.Counter()


@dataclass(frozen=True)
class ChunkSpan:
    """One chunk as the index build cut it: its LanceDB id and its span in the body."""

    natural_id: str
    char_start: int
    char_end: int


@dataclass(frozen=True)
class ChunkedBody:
    """A body the index build chunked: its file, the exact text it chunked, and its chunks."""

    path: Path
    text: str
    spans: Sequence[ChunkSpan]


class ChunksFailed(RuntimeError):
    """A chunk sync failed while ``PODCAST_CORPUS_WRITE_STRICT`` is set."""


def _served_document(
    conn: Connection, item_id: int, variant: str
) -> Optional[tuple[int, Optional[str]]]:
    row = conn.execute(
        select(documents.c.document_id, documents.c.body_sha256)
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
    return (row[0], row[1]) if row else None


def _cuts(conn: Connection, item_id: int, document_id: int) -> tuple[list[Cut], dict[int, Any]]:
    rows = conn.execute(
        select(segments.c.seq, segments.c.text, segments.c.speaker_id)
        .where(and_(segments.c.item_id == item_id, segments.c.document_id == document_id))
        .order_by(segments.c.seq)
    ).all()
    cuts = [Cut(seq=r[0], text=r[1], start_ms=None, end_ms=None) for r in rows]
    return cuts, {r[0]: r[2] for r in rows}


def chunk_rows(
    conn: Connection, item_id: int, bodies: Sequence[ChunkedBody], language: Optional[str]
) -> tuple[list[dict[str, Any]], int]:
    """The ``search_chunks`` rows of an item's chunked ``bodies``; returns them and how many
    bodies were skipped as stale (no served document of their variant with that text)."""
    rows: list[dict[str, Any]] = []
    stale = 0
    for body in bodies:
        if not body.spans:
            continue
        served = _served_document(conn, item_id, document_variant(body.path.name, language))
        digest = hashlib.sha256(body.text.encode("utf-8")).hexdigest()
        if served is None or served[1] != digest:
            stale += 1
            continue
        document_id = served[0]
        cuts, speaker_by_seq = _cuts(conn, item_id, document_id)
        for span in body.spans:
            found = cut_range(cuts, span.char_start, span.char_end)
            speakers = sorted(
                {
                    speaker_by_seq[seq]
                    for seq in range(found.start_seq, found.end_seq + 1)
                    if speaker_by_seq.get(seq) is not None
                }
            )
            rows.append(
                {
                    "natural_id": span.natural_id,
                    "item_id": item_id,
                    "document_id": document_id,
                    "start_seq": found.start_seq,
                    "start_offset": found.start_offset,
                    "end_seq": found.end_seq,
                    "end_offset": found.end_offset,
                    "speakers": speakers,
                }
            )
    return rows, stale


_COMPARED = (
    "natural_id",
    "document_id",
    "start_seq",
    "start_offset",
    "end_seq",
    "end_offset",
    "speakers",
)


def _key(row: Any) -> tuple[Any, ...]:
    return tuple(tuple(row[c]) if c == "speakers" else row[c] for c in _COMPARED)


def replace_item_chunks(conn: Connection, item_id: int, rows: Sequence[Mapping[str, Any]]) -> bool:
    """Make the item's ``search_chunks`` exactly ``rows``; returns whether anything changed."""
    stored = conn.execute(
        select(*(search_chunks.c[c] for c in _COMPARED)).where(search_chunks.c.item_id == item_id)
    ).mappings()
    if sorted(_key(r) for r in stored) == sorted(_key(r) for r in rows):
        return False
    conn.execute(delete(search_chunks).where(search_chunks.c.item_id == item_id))
    if rows:
        conn.execute(insert(search_chunks), [dict(r) for r in rows])
    return True


def _strict() -> bool:
    return os.environ.get(ENV_STRICT, "").strip().lower() in ("1", "true", "yes")


def _count(outcome: str) -> str:
    OUTCOMES[outcome] += 1
    return outcome


def sync_episode_chunks(
    doc: Mapping[str, Any], bodies: Sequence[ChunkedBody], language: Optional[str]
) -> str:
    """Write one episode's chunks (``doc`` is its metadata); returns and counts the outcome:
    ``written``, ``unchanged``, ``stale`` (a body with no matching served document; its chunks are
    left out, the others written), ``no_item_row`` (the pipeline has not synced the episode),
    ``lease_refused``, ``refused`` (a span outside its document), ``failed`` or
    ``not_configured``."""
    if not pipeline_is_configured():
        return _count("not_configured")
    engine = None
    holder = ""
    try:
        natural_id = item_natural_id(doc)
        source = source_natural_id(doc)
        from podcast_scraper.db.engines import pipeline_rw_engine

        engine = pipeline_rw_engine()
        holder = f"index:{uuid.uuid4().hex[:12]}:{natural_id}"
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
            item_id = conn.execute(
                select(content_items.c.item_id).where(content_items.c.natural_id == natural_id)
            ).scalar_one_or_none()
            if item_id is None:
                return _count("no_item_row")
            rows, stale = chunk_rows(conn, item_id, bodies, language)
            changed = replace_item_chunks(conn, item_id, rows)
        if stale:
            logger.warning(
                "search chunks: %s has %d chunked bodies with no matching served document",
                natural_id,
                stale,
            )
            return _count("stale")
        return _count("written" if changed else "unchanged")
    except (leases.LeaseRefused, leases.LeaseLost) as exc:
        return _fail("lease_refused", doc, exc)
    except (KeyNotDerivable, ValueError) as exc:
        return _fail("refused", doc, exc)
    except Exception as exc:  # noqa: BLE001 - the index build must not stop on a row write
        return _fail("failed", doc, exc)
    finally:
        if engine is not None and holder:
            try:
                leases.release(engine, holder)
            except Exception as exc:  # noqa: BLE001 - an unreleased lease expires with its TTL
                logger.warning("search chunks could not release %s: %s", holder, exc)


def _fail(outcome: str, doc: Mapping[str, Any], exc: BaseException) -> str:
    _count(outcome)
    episode = doc.get("episode")
    name = episode.get("episode_id") if isinstance(episode, Mapping) else None
    logger.warning("search chunks %s for %s: %s", outcome, name, exc)
    if _strict():
        raise ChunksFailed(f"search chunks {outcome} for {name}: {exc}") from exc
    return outcome
