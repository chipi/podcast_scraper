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
- Each episode the catalog shows gets its card (``corpus_cards.card_row``, the functions today's
  card is built by) and its item's ``has_*`` flags from the same values. News cards come later.
- Each item's KG file next to its metadata gives its entities, ``item_entities`` and
  ``person_roles`` (``importers.corpus_kg.import_kg``).

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
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from sqlalchemy import insert, select, update
from sqlalchemy.engine import Connection

from podcast_scraper.builders.bridge_artifact_paths import bridge_json_path_adjacent_to_metadata
from podcast_scraper.db.corpus_cards import card_row, FLAGS
from podcast_scraper.db.corpus_keys import is_news, KeyNotDerivable, source_natural_id
from podcast_scraper.db.corpus_rows import (
    _instant,
    item_row,
    SERVED,
    served_for,
    source_attrs,
    source_row_for,
)
from podcast_scraper.db.corpus_writer import mark_has_transcript
from podcast_scraper.db.importers.corpus_documents import (
    ARTICLE,
    DocumentRefused,
    import_documents,
    TRANSCRIPT,
    transcript_files,
)
from podcast_scraper.db.importers.corpus_kg import import_kg, KgWritten, read_kg
from podcast_scraper.db.schema import content_items, item_cards, sources
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
    cards: int = 0
    entities: int = 0
    entity_edges: int = 0
    person_roles: int = 0
    by_kind: collections.Counter[str] = field(default_factory=collections.Counter)
    refused: list[tuple[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class _Member:
    relpath: str
    path: Path
    doc: Mapping[str, Any]


def _write_card(
    conn: Connection, root: Path, member: _Member, item_id: int, item: dict[str, Any]
) -> int:
    """The episode's card and its item's ``has_*`` flags; 1 when a card was written."""
    if is_news(member.doc):
        return 0
    card = card_row(root, member.relpath, item)
    if card is None:
        return 0
    if conn.execute(select(item_cards.c.item_id).where(item_cards.c.item_id == item_id)).first():
        return 0
    conn.execute(insert(item_cards).values(item_id=item_id, **card))
    conn.execute(
        update(content_items)
        .where(content_items.c.item_id == item_id)
        .values(**{flag: card[flag] for flag in FLAGS})
    )
    return 1


def _write_kg(
    conn: Connection, member: _Member, item_id: int, item: dict[str, Any]
) -> Optional[KgWritten]:
    """The item's KG file next to its metadata as rows, or ``None`` when it has none."""
    path = member.path.with_name(member.path.name[: -len(_SUFFIX)] + ".kg.json")
    if not path.is_file():
        return None
    language = None if is_news(member.doc) else episode_language(dict(member.doc))
    return import_kg(
        conn,
        item_id,
        read_kg(path, language),
        published_at=item["published_at"],
        content_type=item["content_type"],
    )


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


def _transcripts_of(member: _Member) -> tuple[Path, str]:
    """The episode's transcripts directory and base name, the files ``corpus_writer`` is handed.

    An episode that names its transcript (``content.transcript_file_path``) is found by that path:
    the metadata's own name can truncate the title while the transcripts carry it whole, and a
    later run can reuse an earlier run's transcript, so the path may lead into a sibling run of
    the same feed. A relative path is read from the run directory; an absolute one is the path in
    the container that wrote it (``/app/output/feeds/<feed>/...``) and is read from what follows
    its ``feeds/<this feed>/``. A path leading out of the feed is refused. An episode that names
    none is found by the metadata's name in its run's ``transcripts/``."""
    run = member.path.parent.parent
    feed = run.parent
    content = member.doc.get("content")
    named = content.get("transcript_file_path") if isinstance(content, Mapping) else None
    if isinstance(named, str) and named.strip() and not is_news(member.doc):
        stated = Path(named.strip())
        if stated.is_absolute():
            parts = stated.parts
            starts = [
                i + 2
                for i in range(len(parts) - 2)
                if parts[i] == "feeds" and parts[i + 1] == feed.name
            ]
            if not starts:
                raise DocumentRefused(f"its transcript path {named!r} leads out of its feed")
            stated = feed / Path(*parts[starts[-1] :])
        path = Path(os.path.normpath(run / stated))
        if not path.is_relative_to(feed):
            raise DocumentRefused(f"its transcript path {named!r} leads out of its feed")
        return path.parent, path.name.rsplit(".", 1)[0]
    return run / "transcripts", member.path.name[: -len(_SUFFIX)]


def _own_str(doc: Mapping[str, Any], name: str) -> Optional[str]:
    own = doc.get("news_article")
    value = own.get(name) if isinstance(own, Mapping) else None
    return value if isinstance(value, str) and value else None


def _write_documents(conn: Connection, item_id: int, member: _Member) -> tuple[int, int, str]:
    directory, base = _transcripts_of(member)
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
            extraction=_own_str(member.doc, "extraction"),
            extractor_version=_own_str(member.doc, "extractor_version"),
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
                item = {
                    **item_row(
                        member.doc,
                        metadata_relpath=member.relpath,
                        has_bridge=os.path.isfile(
                            bridge_json_path_adjacent_to_metadata(str(member.path))
                        ),
                    )
                }
                item["source_id"] = source_id
                # The source columns this item states: the pipeline's writer merges a source again
                # from its items' source_attrs, so an imported item must carry its own.
                item["source_attrs"] = source_attrs(member.doc, registry=registry)
                item_id = _upsert_item(conn, item)
                documents, segments, kind = _write_documents(conn, item_id, member)
                mark_has_transcript(conn, item_id)
                report.cards += _write_card(conn, root, member, item_id, item)
                written_kg = _write_kg(conn, member, item_id, item)
            except (DocumentRefused, KeyNotDerivable, ValueError) as exc:
                savepoint.rollback()
                report.refused.append((member.relpath, str(exc)))
                continue
            savepoint.commit()
            report.items += 1
            report.documents += documents
            report.segments += segments
            if written_kg is not None:
                report.entities += written_kg.entities
                report.entity_edges += written_kg.edges
                report.person_roles += written_kg.roles
            if kind:
                report.by_kind[kind] += documents
    return report
