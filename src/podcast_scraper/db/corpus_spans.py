"""A character span of a document as a segment range (CRFC-005 §3.4 ``evidence``).

A document's body is its segments joined in ``seq`` order (``importers.corpus_documents``), so a
character offset into the body is a position in one segment. :func:`segment_range` turns a span
``[start, end)`` into ``(start_seq, start_offset, end_seq, end_offset)``: the start in the segment
that holds its first character, the end in the segment that holds its last one (so a span never
ends at offset 0 of the next segment). Offsets count characters, as Python's ``len`` and
Postgres's ``length`` do. The GI importer and the pipeline's evidence writer both call it.
"""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import and_, func, select
from sqlalchemy.engine import Connection

from podcast_scraper.db.schema import segments


class SpanOutside(ValueError):
    """The span is empty, negative, or runs past the document's end."""


def segment_range(
    cuts: Sequence[tuple[int, int]], start: int, end: int
) -> tuple[int, int, int, int]:
    """``(start_seq, start_offset, end_seq, end_offset)`` of ``[start, end)`` over ``cuts``.

    ``cuts`` is the document's ``(seq, length)`` pairs in ``seq`` order.
    """
    total = sum(length for _seq, length in cuts)
    if start < 0 or end <= start or end > total:
        raise SpanOutside(f"span [{start}, {end}) is outside a document of {total} characters")
    first = last = None
    at = 0
    for seq, length in cuts:
        if first is None and at <= start < at + length:
            first = (seq, start - at)
        if at < end <= at + length:
            last = (seq, end - at)
            break
        at += length
    if first is None or last is None:  # only zero-length cuts left; the bounds check rules it out
        raise SpanOutside(f"span [{start}, {end}) is in no segment")
    return first[0], first[1], last[0], last[1]


def segment_lengths(conn: Connection, item_id: int, document_id: int) -> list[tuple[int, int]]:
    """The stored document's ``(seq, length)`` pairs in ``seq`` order."""
    rows = conn.execute(
        select(segments.c.seq, func.length(segments.c.text))
        .where(and_(segments.c.item_id == item_id, segments.c.document_id == document_id))
        .order_by(segments.c.seq)
    ).all()
    return [(int(seq), int(length)) for seq, length in rows]
