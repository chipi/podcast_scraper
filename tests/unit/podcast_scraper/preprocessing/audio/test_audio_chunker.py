"""Unit tests for API audio chunking."""

from __future__ import annotations

import pytest

from podcast_scraper.preprocessing.audio.chunker import (
    _dedupe_overlap_text,
    AudioChunk,
    AudioChunker,
)

pytestmark = pytest.mark.unit


class TestDedupeOverlapText:
    def test_removes_shared_suffix_prefix(self) -> None:
        prev = "one two three four five"
        nxt = "four five six seven"
        assert _dedupe_overlap_text(prev, nxt, overlap_seconds=5.0) == "six seven"

    def test_no_overlap_returns_next(self) -> None:
        assert (
            _dedupe_overlap_text("alpha beta", "gamma delta", overlap_seconds=5.0) == "gamma delta"
        )


class TestMergeTranscriptResults:
    def test_offsets_and_text_merge(self) -> None:
        chunker = AudioChunker(max_bytes=1024, overlap_seconds=5.0)
        chunks = [
            AudioChunk(path="a.mp3", start_seconds=0.0, index=0),
            AudioChunk(path="b.mp3", start_seconds=95.0, index=1),
        ]
        results = [
            (
                {
                    "text": "hello world",
                    "segments": [{"start": 0.0, "end": 1.0, "text": "hello world"}],
                },
                1.0,
            ),
            (
                {
                    "text": "world again",
                    "segments": [{"start": 0.0, "end": 1.0, "text": "world again"}],
                },
                1.5,
            ),
        ]
        merged, elapsed = chunker.merge_transcript_results(results, chunks)
        assert elapsed == 2.5
        assert "hello" in merged["text"]
        assert len(merged["segments"]) == 2
        assert merged["segments"][1]["start"] == pytest.approx(95.0)


class TestNeedsChunking:
    def test_needs_chunking_when_over_limit(self, tmp_path) -> None:
        audio = tmp_path / "big.mp3"
        audio.write_bytes(b"x" * 2000)
        chunker = AudioChunker(max_bytes=1000)
        assert chunker.needs_chunking(str(audio))

    def test_no_chunking_when_under_limit(self, tmp_path) -> None:
        audio = tmp_path / "small.mp3"
        audio.write_bytes(b"x" * 100)
        chunker = AudioChunker(max_bytes=1000)
        assert not chunker.needs_chunking(str(audio))
