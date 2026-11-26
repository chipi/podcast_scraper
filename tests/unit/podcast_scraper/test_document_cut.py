"""A body cut into segment rows: every character in exactly one cut (CRFC-005 §3.5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.document_cut import (
    Cut,
    cut_body,
    cut_range,
    CutRange,
    DocumentCutError,
    paragraph_cuts,
    seconds_to_ms,
)

_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures/app-validation-corpus/public/feeds/p01/run_20260101-000000/transcripts"
)


def _join(cuts: list[Cut]) -> str:
    return "".join(c.text for c in cuts)


def test_a_fixture_transcript_joins_back_and_keeps_every_segments_times() -> None:
    body = (_FIXTURE / "p01_e01.txt").read_text(encoding="utf-8")
    segments = json.loads((_FIXTURE / "p01_e01.segments.json").read_text(encoding="utf-8"))
    cuts = cut_body(body, segments)
    assert _join(cuts) == body
    assert [c.seq for c in cuts] == list(range(len(cuts)))
    assert len(cuts) == len(segments)
    assert (cuts[1].start_ms, cuts[1].end_ms) == (12225, 19256)


def test_a_speaker_label_between_turns_goes_to_the_turn_after_it() -> None:
    body = "Intro.\nMAYA: Hello there.\nLIAM: Hi."
    segments = [
        {"text": "Hello there.", "start": 0.0, "end": 1.5},
        {"text": "Hi.", "start": 1.5, "end": 2.0},
    ]
    cuts = cut_body(body, segments)
    assert [c.text for c in cuts] == ["Intro.\nMAYA: Hello there.", "\nLIAM: Hi."]
    assert (cuts[0].start_ms, cuts[1].start_ms) == (0, 1500)


def test_explicit_char_spans_are_used_when_every_segment_has_a_pair() -> None:
    body = "[AD] Hello. World. [AD]"
    segments = [
        {"text": "Hello.", "char_start": 5, "char_end": 11, "start": 1.0, "end": 2.0},
        {"text": "World.", "char_start": 12, "char_end": 18, "start": 2.0, "end": 3.0},
    ]
    cuts = cut_body(body, segments)
    assert [c.text for c in cuts] == ["[AD] Hello.", " World. [AD]"]


def test_unusable_char_spans_fall_back_to_finding_the_text() -> None:
    body = "One. Two."
    segments = [{"text": "One.", "char_start": 0}, {"text": "Two.", "char_start": 5}]
    assert [c.text for c in cut_body(body, segments)] == ["One.", " Two."]
    backwards = [
        {"text": "One.", "char_start": 5, "char_end": 9},
        {"text": "Two.", "char_start": 0, "char_end": 4},
    ]
    assert [c.text for c in cut_body(body, backwards)] == ["One.", " Two."]


def test_a_segment_not_found_in_the_body_joins_the_cut_after_it() -> None:
    body = "Alpha. Beta. Gamma."
    segments = [{"text": "Alpha."}, {"text": "missing"}, {"text": "Gamma."}]
    assert [c.text for c in cut_body(body, segments)] == ["Alpha.", " Beta. Gamma."]


def test_zero_length_cuts_are_dropped_and_seq_renumbered() -> None:
    body = "Same. Same."
    segments = [
        {"text": "Same.", "char_start": 0, "char_end": 5},
        {"text": "", "char_start": 5, "char_end": 5},
        {"text": "Same.", "char_start": 6, "char_end": 11},
    ]
    cuts = cut_body(body, segments)
    assert [(c.seq, c.text) for c in cuts] == [(0, "Same."), (1, " Same.")]


def test_no_segments_or_none_found_cut_one_line_each_without_times() -> None:
    body = "line one\r\nline two\nlast"
    for segments in (None, [], [{"text": "nowhere"}]):
        cuts = cut_body(body, segments)
        assert [c.text for c in cuts] == ["line one\r\n", "line two\n", "last"]
        assert all(c.start_ms is None and c.end_ms is None for c in cuts)


def test_an_empty_body_has_no_cuts() -> None:
    assert cut_body("", [{"text": "x"}]) == []


def test_unicode_survives_the_cut() -> None:
    body = "Café — crème.\nNaïve façade."
    segments = [{"text": "Café — crème."}, {"text": "Naïve façade."}]
    assert _join(cut_body(body, segments)) == body


@pytest.mark.parametrize(
    "seconds,ms",
    [(0.0, 0), (12.225, 12225), (1.0005, 1001), (1.0004, 1000), ("2.5", 2500), (None, None)],
)
def test_seconds_round_half_up_to_milliseconds(seconds: object, ms: int | None) -> None:
    assert seconds_to_ms(seconds) == ms


def test_an_end_before_its_start_nulls_both_times() -> None:
    cuts = cut_body("A. B.", [{"text": "A.", "start": 2.0, "end": 1.0}, {"text": "B."}])
    assert (cuts[0].start_ms, cuts[0].end_ms) == (None, None)


def test_article_paragraphs_keep_their_blank_lines() -> None:
    body = "First para\nstill first.\n\nSecond.\n\n\nThird."
    cuts = paragraph_cuts(body)
    assert [c.text for c in cuts] == ["First para\nstill first.\n\n", "Second.\n\n\n", "Third."]
    assert _join(cuts) == body


def test_a_cut_that_does_not_join_back_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    from podcast_scraper import document_cut

    monkeypatch.setattr(document_cut, "_line_cuts", lambda body: [Cut(0, "other", None, None)])
    with pytest.raises(DocumentCutError):
        cut_body("body", None)


def _three() -> list[Cut]:
    return [Cut(0, "Hello", None, None), Cut(1, " there", None, None), Cut(2, " you.", None, None)]


@pytest.mark.parametrize(
    "span,expected",
    [
        ((0, 5), CutRange(0, 0, 0, 5)),
        ((2, 8), CutRange(0, 2, 1, 3)),
        ((5, 11), CutRange(1, 0, 1, 6)),
        ((5, 12), CutRange(1, 0, 2, 1)),
        ((0, 16), CutRange(0, 0, 2, 5)),
        ((5, 5), CutRange(1, 0, 1, 0)),
        ((16, 16), CutRange(2, 5, 2, 5)),
    ],
)
def test_a_span_maps_to_its_cut_range(span: tuple[int, int], expected: CutRange) -> None:
    assert cut_range(_three(), *span) == expected


@pytest.mark.parametrize("span", [(-1, 3), (3, 2), (0, 17)])
def test_a_span_outside_the_body_or_reversed_is_refused(span: tuple[int, int]) -> None:
    with pytest.raises(ValueError):
        cut_range(_three(), *span)


def test_every_fixture_chunk_span_round_trips_through_its_range() -> None:
    from podcast_scraper.search.chunker import chunk_transcript

    body = (_FIXTURE / "p01_e01.txt").read_text(encoding="utf-8")
    segments = json.loads((_FIXTURE / "p01_e01.segments.json").read_text(encoding="utf-8"))
    cuts = cut_body(body, segments)
    for chunk in chunk_transcript(body, target_tokens=120, overlap_tokens=20):
        r = cut_range(cuts, chunk.char_start, chunk.char_end)
        text = "".join(
            c.text[
                r.start_offset if c.seq == r.start_seq else 0 : (
                    r.end_offset if c.seq == r.end_seq else None
                )
            ]
            for c in cuts[r.start_seq : r.end_seq + 1]
        )
        assert text == body[chunk.char_start : chunk.char_end]
