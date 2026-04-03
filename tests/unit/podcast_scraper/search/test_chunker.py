"""Unit tests for transcript chunker (#484 Step 2)."""

from __future__ import annotations

import pytest

from podcast_scraper.search.chunker import chunk_transcript


@pytest.mark.unit
def test_chunk_transcript_empty() -> None:
    assert chunk_transcript("") == []
    assert chunk_transcript("   ") == []


@pytest.mark.unit
def test_chunk_transcript_invalid_overlap_raises() -> None:
    with pytest.raises(ValueError, match="overlap_tokens"):
        chunk_transcript("Hello.", target_tokens=10, overlap_tokens=10)


@pytest.mark.unit
def test_chunk_transcript_single_sentence() -> None:
    chunks = chunk_transcript("Only one sentence here.", target_tokens=50, overlap_tokens=5)
    assert len(chunks) == 1
    assert chunks[0].text == "Only one sentence here."
    assert chunks[0].chunk_index == 0
    assert chunks[0].char_start < chunks[0].char_end


@pytest.mark.unit
def test_chunk_transcript_char_spans_within_transcript() -> None:
    text = "First part. Second part! Third part?"
    chunks = chunk_transcript(text, target_tokens=5, overlap_tokens=1)
    assert len(chunks) >= 1
    for ch in chunks:
        assert 0 <= ch.char_start < ch.char_end <= len(text)
        joined = ch.text.replace(" ", "")
        raw = text[ch.char_start : ch.char_end].replace(" ", "").replace("\n", "")
        assert joined == raw


@pytest.mark.unit
def test_chunk_transcript_overlap_reuses_tail_sentences() -> None:
    sents = " ".join(f"Sentence number {i} ends here." for i in range(12))
    chunks = chunk_transcript(sents, target_tokens=15, overlap_tokens=5)
    assert len(chunks) >= 2
    toks0 = set(chunks[0].text.lower().split())
    toks1 = set(chunks[1].text.lower().split())
    assert len(toks0 & toks1) > 0


@pytest.mark.unit
def test_chunk_transcript_timestamps_interpolation() -> None:
    text = "Alpha. Beta."
    segments = [
        {"char_start": 0, "char_end": 6, "start_ms": 0, "end_ms": 1000},
        {"char_start": 7, "char_end": 12, "start_ms": 1000, "end_ms": 2000},
    ]
    chunks = chunk_transcript(
        text,
        target_tokens=100,
        overlap_tokens=5,
        timestamps=segments,
    )
    assert len(chunks) == 1
    assert chunks[0].timestamp_start_ms == 0
    assert chunks[0].timestamp_end_ms == 2000


@pytest.mark.unit
def test_paragraph_fallback_splits_long_text() -> None:
    body = "\n\n".join(f"Paragraph {i} without punctuation marks" for i in range(5))
    chunks = chunk_transcript(body, target_tokens=8, overlap_tokens=2)
    assert len(chunks) >= 2
