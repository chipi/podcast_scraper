"""A corpus root's sources, items and documents as rows (CRFC-005 §3.1, §3.5; tracker s-importer).

:func:`import_corpus` reads the corpus the way the catalog does and writes it in the caller's
transaction:

- Membership is the catalog's: ``search.corpus_scope.discover_metadata_files`` (every feed and
  source run, the newest run per episode), news sources included. An episode the pipeline could
  not complete is written too, with ``served`` 0 (``corpus_rows.served_for``): which items are
  served is the read side's.
- A source's columns are ``corpus_rows.source_row_for`` over its served items in
  ``metadata_relpath`` order (the catalog's rule); a source with no served item takes them from all
  its items. Each column comes from the newest item that states it.
- Each item is ``corpus_rows.item_row`` under its source.
- Each item's transcript files (a podcast) or article files (news, ``kind`` article, with the
  metadata's ``fetched_at`` and ``body_sha256``) are written by
  ``importers.corpus_documents.import_documents``.

What cannot be written is refused, never guessed: an item whose keys cannot be derived, whose
documents cannot be named one way or do not cut, or whose slug or ``(source, external id)`` another
item holds. A refused item is listed in the report with its reason and nothing of it is written;
the rest of the corpus is.

It runs into an empty database with the pipeline stopped (the cut-over runbook) and takes no job
lease. An item already present is left as it is when its row is equal and refused otherwise.
"""

from __future__ import annotations

import collections
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from sqlalchemy import insert, select
from sqlalchemy.engine import Connection

from podcast_scraper.db.corpus_keys import is_news, KeyNotDerivable, source_natural_id
from podcast_scraper.db.corpus_rows import (
    _instant,
    item_row,
    SERVED,
    served_for,
    source_row_for,
)
from podcast_scraper.db.importers.corpus_documents import (
    ARTICLE,
    DocumentRefused,
    import_documents,
    TRANSCRIPT,
    transcript_files,
)
from podcast_scraper.db.schema import content_items, sources
from podcast_scraper.languages import episode_language
from podcast_scraper.publishers import load_registry, Registry
from podcast_scraper.search.corpus_scope import discover_metadata_files

_SUFFIX = ".metadata.json"


@dataclass
class ImportReport:
    """What one :func:`import_corpus` wrote and refused."""

    sources: int = 0
    items: int = 0
    documents: int = 0
    segments: int = 0
    by_kind: collections.Counter[str] = field(default_factory=collections.Counter)
    refused: list[tuple[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class _Member:
    relpath: str
    path: Path
    doc: Mapping[str, Any]


def _members(root: Path) -> tuple[list[_Member], list[tuple[str, str]]]:
    members: list[_Member] = []
    refused: list[tuple[str, str]] = []
    for path in discover_metadata_files(root, include_sources=True):
        relpath = path.relative_to(root).as_posix()
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            refused.append((relpath, f"unreadable metadata: {exc}"))
            continue
        if not isinstance(doc, dict):
            refused.append((relpath, "metadata is not an object"))
            continue
        members.append(_Member(relpath, path, doc))
    members.sort(key=lambda m: m.relpath)
    return members, refused


def _write_documents(conn: Connection, item_id: int, member: _Member) -> tuple[int, int, str]:
    base = member.path.name[: -len(_SUFFIX)]
    directory = member.path.parent.parent / "transcripts"
    if not directory.is_dir():
        return 0, 0, ""
    if is_news(member.doc):
        files = transcript_files(directory, base, member.doc.get("language"))
        written = import_documents(
            conn,
            item_id,
            files,
            kind=ARTICLE,
            fetched_at=_instant(member.doc.get("fetched_at")),
            body_sha256=member.doc.get("body_sha256"),
        )
        kind = ARTICLE
    else:
        files = transcript_files(directory, base, episode_language(dict(member.doc)))
        written = import_documents(conn, item_id, files)
        kind = TRANSCRIPT
    return len(written), sum(d.segments for d in written), kind


def _upsert_item(conn: Connection, row: dict[str, Any]) -> int:
    stored = (
        conn.execute(select(content_items).where(content_items.c.natural_id == row["natural_id"]))
        .mappings()
        .first()
    )
    if stored is not None:
        differs = sorted(column for column, value in row.items() if stored[column] != value)
        if differs:
            raise DocumentRefused(f"already present with other {', '.join(differs)}")
        item_id: int = stored["item_id"]
        return item_id
    clash = conn.execute(
        select(content_items.c.natural_id).where(
            (content_items.c.slug == row["slug"])
            | (
                (content_items.c.source_id == row["source_id"])
                & (content_items.c.external_item_id == row["external_item_id"])
            )
        )
    ).scalar_one_or_none()
    if clash is not None:
        raise DocumentRefused(f"its slug or (source, external id) is {clash}'s")
    item_id = conn.execute(
        insert(content_items).values(**row).returning(content_items.c.item_id)
    ).scalar_one()
    return item_id


def _source_id(conn: Connection, row: dict[str, Any]) -> tuple[int, bool]:
    found = conn.execute(
        select(sources.c.source_id).where(sources.c.natural_id == row["natural_id"])
    ).scalar_one_or_none()
    if found is not None:
        return found, False
    source_id: int = conn.execute(
        insert(sources).values(**row).returning(sources.c.source_id)
    ).scalar_one()
    return source_id, True


def import_corpus(
    conn: Connection, corpus_root: Path, *, registry: Optional[Registry] = None
) -> ImportReport:
    """Write the corpus at ``corpus_root`` as sources, items and documents, in the caller's
    transaction; returns what was written and refused."""
    root = Path(corpus_root)
    report = ImportReport()
    members, report.refused = _members(root)
    registry = registry if registry is not None else load_registry()

    by_source: dict[str, list[_Member]] = collections.defaultdict(list)
    for member in members:
        try:
            by_source[source_natural_id(member.doc)].append(member)
        except KeyNotDerivable as exc:
            report.refused.append((member.relpath, f"no source key: {exc}"))

    for source_key in sorted(by_source):
        group = by_source[source_key]
        served = [m.doc for m in group if served_for(m.doc) == SERVED]
        try:
            row = source_row_for(served or [m.doc for m in group], registry=registry)
        except ValueError as exc:
            report.refused.extend((m.relpath, f"source {source_key}: {exc}") for m in group)
            continue
        source_id, inserted = _source_id(conn, row)
        report.sources += int(inserted)
        for member in group:
            savepoint = conn.begin_nested()
            try:
                item = {**item_row(member.doc, metadata_relpath=member.relpath)}
                item["source_id"] = source_id
                item_id = _upsert_item(conn, item)
                documents, segments, kind = _write_documents(conn, item_id, member)
            except (DocumentRefused, KeyNotDerivable, ValueError) as exc:
                savepoint.rollback()
                report.refused.append((member.relpath, str(exc)))
                continue
            savepoint.commit()
            report.items += 1
            report.documents += documents
            report.segments += segments
            if kind:
                report.by_kind[kind] += documents
    return report
