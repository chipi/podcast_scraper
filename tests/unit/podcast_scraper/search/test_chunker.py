"""Unit tests for transcript chunking helpers (search.chunker)."""

from __future__ import annotations

from typing import Any, cast

import pytest

from podcast_scraper.search.chunker import (
    _merge_time_for_span,
    _paragraph_spans,
    _sentences_with_spans,
    _token_count,
    chunk_transcript,
)

pytestmark = pytest.mark.unit


def test_token_count() -> None:
    assert _token_count("") == 0
    assert _token_count("a b c") == 3


def test_sentences_with_spans_basic() -> None:
    text = "First. Second! Third?"
    spans = _sentences_with_spans(text)
    assert len(spans) == 3
    joined = "".join(text[s[1] : s[2]] for s in spans)
    assert "First" in joined and "Second" in joined


def test_sentences_with_spans_empty_and_whitespace_only() -> None:
    assert _sentences_with_spans("") == []
    assert _sentences_with_spans("   \n\t  ") == []


def test_sentences_with_spans_no_punctuation_falls_back_to_strip() -> None:
    text = "  only words here  "
    spans = _sentences_with_spans(text)
    assert len(spans) == 1
    assert spans[0][0] == "only words here"


def test_paragraph_spans_double_newline() -> None:
    text = "Para one.\n\nPara two.\n\n"
    ps = _paragraph_spans(text)
    assert len(ps) == 2
    assert ps[0][0].startswith("Para one")


def test_merge_time_for_span_empty_and_overlap() -> None:
    assert _merge_time_for_span(0, 10, []) == (None, None)
    segs = [
        {"char_start": 0, "char_end": 5, "start_ms": 0, "end_ms": 100},
        {"char_start": 3, "char_end": 20, "start_ms": 50, "end_ms": 200},
    ]
    assert _merge_time_for_span(2, 8, segs) == (0, 200)
    assert _merge_time_for_span(100, 200, segs) == (None, None)


def test_merge_time_for_span_alias_and_bad_values() -> None:
    segs = cast(
        list[dict[str, Any]],
        [
            {
                "char_start": "0",
                "char_end": "100",
                "timestamp_start_ms": "10",
                "timestamp_end_ms": "20",
            },
            {"char_start": "x", "char_end": 1, "start_ms": 1, "end_ms": 2},
        ],
    )
    assert _merge_time_for_span(0, 50, segs) == (10, 20)


def test_chunk_transcript_validation() -> None:
    with pytest.raises(ValueError, match="target_tokens"):
        chunk_transcript("a", target_tokens=0)
    with pytest.raises(ValueError, match="overlap_tokens"):
        chunk_transcript("a", overlap_tokens=-1)
    with pytest.raises(ValueError, match="overlap_tokens"):
        chunk_transcript("a", target_tokens=10, overlap_tokens=10)


def test_chunk_transcript_empty() -> None:
    assert chunk_transcript("") == []
    assert chunk_transcript("   \n") == []


def test_chunk_transcript_with_timestamps() -> None:
    text = "One. Two. Three."
    ts = [
        {"char_start": 0, "char_end": 5, "start_ms": 0, "end_ms": 1000},
        {"char_start": 6, "char_end": 11, "start_ms": 1000, "end_ms": 2000},
    ]
    chunks = chunk_transcript(text, target_tokens=2, overlap_tokens=0, timestamps=ts)
    assert chunks
    assert chunks[0].timestamp_start_ms is not None


def test_chunk_transcript_multiline_long_sentence_fallback() -> None:
    line = "word " * 80
    text = "\n".join([line, line, line])
    chunks = chunk_transcript(text, target_tokens=40, overlap_tokens=5)
    assert len(chunks) >= 2


def test_chunk_transcript_overlap_advances_window() -> None:
    parts = [f"Seg{i} " + "word " * 10 for i in range(30)]
    text = ". ".join(parts)
    chunks = chunk_transcript(text, target_tokens=25, overlap_tokens=8)
    assert len(chunks) >= 2
    assert all(c.char_end > c.char_start for c in chunks)
