"""A podcast item's transcript files as ``documents`` and ``segments`` rows (CRFC-005 §3.5).

The variant of each file is ``kg.as_v3.document_variant``'s, the rule the pipeline writes by. A
file's body becomes its document's segments through ``document_cut.cut_body`` (with the file's own
``.segments.json`` when it has one, else one cut per line), so the segments joined in ``seq``
order are the file byte for byte; :func:`import_documents` asserts that per document after it
writes.

What is refused, never guessed (the item is listed and nothing of it is written):

- ``<base>.cleaned.adfree.txt``: a stacked variant (none on snapshot-prod-20261124);
- ``<base>.<lang>.txt`` without ``<base>.translation.json``: a source text nothing says was
  translated;
- an ``en`` document (a non-English item's ``<base>.txt``) with no ``<base>.<lang>.txt`` beside
  it: the rule assumes a translation swap that did not happen, so the file is the source text;
- an ``en`` document with no ``.segments.json`` while its ``<base>.<lang>.txt`` has one: a swap
  that stopped between its renames;
- a body that does not cut (a ``.segments.json`` that is not JSON, or cuts that do not join back);
- two files with one variant;
- an existing document of the same (item, kind, variant) version 1 with another body: a new
  version is the writer's, not the importer's.

``<base>.anon.txt`` is skipped: it is a derived copy, not a document.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from sqlalchemy import and_, func, insert, select
from sqlalchemy.engine import Connection

from podcast_scraper.db.schema import documents, segments
from podcast_scraper.document_cut import cut_body, DocumentCutError
from podcast_scraper.kg.as_v3 import document_variant

TRANSCRIPT = "transcript"

_SEGMENTS_BATCH = 1000


class DocumentRefused(ValueError):
    """The item's transcript files cannot be imported as they are; nothing of it is written."""


@dataclass(frozen=True)
class TranscriptFile:
    """One transcript file of an item, its variant, and its `.segments.json` when it has one."""

    path: Path
    variant: str
    segments_path: Optional[Path]


@dataclass(frozen=True)
class ImportedDocument:
    """A written (or already present) document and its segment count."""

    document_id: int
    variant: str
    segments: int
    inserted: bool


def _lang_files(transcripts_dir: Path, base: str) -> list[tuple[str, Path]]:
    pattern = re.compile(re.escape(base) + r"\.([a-z]{2,3})\.txt")
    found = []
    for path in sorted(transcripts_dir.iterdir()):
        match = pattern.fullmatch(path.name)
        if match:
            found.append((match.group(1), path))
    return found


def _segments_path(body: Path) -> Optional[Path]:
    path = body.with_name(body.name[: -len(".txt")] + ".segments.json")
    return path if path.is_file() else None


def transcript_files(
    transcripts_dir: Path, base: str, language: Optional[str] = None
) -> list[TranscriptFile]:
    """The item's transcript files, each with its variant; raises :class:`DocumentRefused`."""
    if (transcripts_dir / f"{base}.cleaned.adfree.txt").exists():
        raise DocumentRefused(f"{base}: a stacked .cleaned.adfree.txt")
    translated = (transcripts_dir / f"{base}.translation.json").is_file()
    lang_files = _lang_files(transcripts_dir, base)
    if lang_files and not translated:
        raise DocumentRefused(f"{base}: {lang_files[0][1].name} with no {base}.translation.json")
    paths = [
        transcripts_dir / f"{base}{suffix}" for suffix in (".txt", ".cleaned.txt", ".adfree.txt")
    ]
    candidates = [p for p in paths if p.is_file()] + [p for _, p in lang_files]
    files: list[TranscriptFile] = []
    seen: dict[str, Path] = {}
    for path in candidates:
        variant = document_variant(path.name, language)
        if variant == "en" and not lang_files:
            raise DocumentRefused(
                f"{base}: {path.name} reads as the English render of a {language} item, but no "
                f"{base}.<lang>.txt holds its source"
            )
        if variant == "en" and not _segments_path(path):
            # The swap is finished only when the render's segments moved in beside it
            # (translation/artifacts.py _swap_in_translation); a crash between its renames
            # leaves the source's segments tagged and the render's missing.
            tagged = [p for _, p in lang_files if _segments_path(p)]
            if tagged:
                raise DocumentRefused(
                    f"{base}: {tagged[0].name} has its segments but {path.name} has none: an "
                    "unfinished translation swap"
                )
        if variant in seen:
            raise DocumentRefused(
                f"{base}: {seen[variant].name} and {path.name} are both variant {variant}"
            )
        seen[variant] = path
        files.append(TranscriptFile(path, variant, _segments_path(path)))
    return files


def read_body(path: Path) -> str:
    """The file's text exactly: no newline translation, so ``\\r\\n`` survives."""
    return path.read_bytes().decode("utf-8")


def _read_segments(path: Optional[Path]) -> Optional[Sequence[Any]]:
    if path is None:
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    found = raw.get("segments") if isinstance(raw, dict) else raw
    return found if isinstance(found, list) else None


def _existing(conn: Connection, item_id: int, variant: str) -> Optional[tuple[int, Optional[str]]]:
    row = conn.execute(
        select(documents.c.document_id, documents.c.body_sha256).where(
            and_(
                documents.c.item_id == item_id,
                documents.c.kind == TRANSCRIPT,
                documents.c.variant == variant,
                documents.c.version == 1,
            )
        )
    ).first()
    return (row[0], row[1]) if row else None


def joined_body(conn: Connection, item_id: int, document_id: int) -> str:
    """The document's segments joined in ``seq`` order."""
    texts = conn.execute(
        select(segments.c.text)
        .where(and_(segments.c.item_id == item_id, segments.c.document_id == document_id))
        .order_by(segments.c.seq)
    ).scalars()
    return "".join(texts)


def import_documents(
    conn: Connection, item_id: int, files: Sequence[TranscriptFile]
) -> list[ImportedDocument]:
    """Write each file as version 1 of its (item, transcript, variant) and its segments, then check
    every document joins back to its file. Idempotent: a document already there with the same
    body is left as it is. Runs in the caller's transaction."""
    imported: list[ImportedDocument] = []
    for file in files:
        body = read_body(file.path)
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
        existing = _existing(conn, item_id, file.variant)
        if existing is not None:
            document_id, stored = existing
            if stored != digest:
                raise DocumentRefused(
                    f"{file.path.name}: item {item_id} already has a {file.variant} transcript "
                    "with another body"
                )
            count = conn.execute(
                select(func.count()).where(
                    and_(segments.c.item_id == item_id, segments.c.document_id == document_id)
                )
            ).scalar_one()
            imported.append(ImportedDocument(document_id, file.variant, count, False))
            continue
        try:
            cuts = cut_body(body, _read_segments(file.segments_path))
        except (DocumentCutError, ValueError) as exc:
            raise DocumentRefused(f"{file.path.name}: {exc}") from exc
        document_id = conn.execute(
            insert(documents)
            .values(item_id=item_id, kind=TRANSCRIPT, variant=file.variant, body_sha256=digest)
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
        imported.append(ImportedDocument(document_id, file.variant, len(rows), True))
    return imported
