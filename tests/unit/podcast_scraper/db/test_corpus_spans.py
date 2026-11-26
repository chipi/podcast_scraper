"""A character span of a document as a segment range (CRFC-005 §3.4)."""

from __future__ import annotations

import pytest

from podcast_scraper.db.corpus_spans import segment_range, SpanOutside

# "Hello " + "big " + "" + "world" with seq numbers that do not start at 0 or run without gaps.
CUTS = [(3, 6), (4, 4), (7, 0), (9, 5)]
BODY = "Hello big world"


def _text(start_seq: int, start_off: int, end_seq: int, end_off: int) -> str:
    texts = {3: "Hello ", 4: "big ", 7: "", 9: "world"}
    order = [s for s, _ in CUTS]
    out = []
    for seq in order[order.index(start_seq) : order.index(end_seq) + 1]:
        t = texts[seq]
        lo = start_off if seq == start_seq else 0
        hi = end_off if seq == end_seq else len(t)
        out.append(t[lo:hi])
    return "".join(out)


@pytest.mark.parametrize(
    "start,end,want",
    [
        (0, 5, (3, 0, 3, 5)),
        (6, 10, (4, 0, 4, 4)),  # a whole segment ends in it, not at offset 0 of the next
        (6, 11, (4, 0, 9, 1)),  # the empty segment is never a start or an end
        (4, 13, (3, 4, 9, 3)),
        (0, 15, (3, 0, 9, 5)),
        (14, 15, (9, 4, 9, 5)),
    ],
)
def test_a_span_maps_to_the_segments_holding_its_first_and_last_character(
    start: int, end: int, want: tuple[int, int, int, int]
) -> None:
    got = segment_range(CUTS, start, end)
    assert got == want
    assert _text(*got) == BODY[start:end]


@pytest.mark.parametrize("start,end", [(-1, 3), (5, 5), (6, 5), (0, 16), (15, 16)])
def test_a_span_outside_the_document_is_refused(start: int, end: int) -> None:
    with pytest.raises(SpanOutside):
        segment_range(CUTS, start, end)


def test_every_span_of_a_document_round_trips() -> None:
    for start in range(len(BODY)):
        for end in range(start + 1, len(BODY) + 1):
            assert _text(*segment_range(CUTS, start, end)) == BODY[start:end]
