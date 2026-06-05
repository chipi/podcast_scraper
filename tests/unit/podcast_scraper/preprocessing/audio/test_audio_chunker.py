"""Unit tests for API audio chunking."""

from __future__ import annotations

from subprocess import CompletedProcess
from unittest.mock import patch

import pytest

from podcast_scraper.preprocessing.audio.chunker import (
    _dedupe_overlap_text,
    AudioChunk,
    AudioChunker,
)
from podcast_scraper.utils.audio_payload_limits import (
    OPENAI_GPT4O_TRANSCRIBE_MAX_DURATION_SECONDS,
    transcription_max_chunk_duration_seconds,
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


class TestDurationChunking:
    @patch("podcast_scraper.preprocessing.audio.chunker._probe_duration_seconds")
    def test_needs_chunking_when_duration_over_limit(self, mock_probe, tmp_path) -> None:
        mock_probe.return_value = 1500.0
        audio = tmp_path / "short.mp3"
        audio.write_bytes(b"x" * 100)
        chunker = AudioChunker(max_bytes=1_000_000, max_duration_seconds=1400.0)
        assert chunker.needs_chunking(str(audio))

    def test_transcription_duration_limit_openai_gpt4o(self) -> None:
        from podcast_scraper import config

        cfg = config.Config(
            rss="https://example.com/feed.xml",
            transcription_provider="openai",
            openai_api_key="sk-test",
            openai_transcription_model="gpt-4o-transcribe",
        )
        assert transcription_max_chunk_duration_seconds(cfg) == (
            OPENAI_GPT4O_TRANSCRIBE_MAX_DURATION_SECONDS
        )


class TestSplitAudio:
    @patch("podcast_scraper.preprocessing.audio.chunker._run_text_subprocess")
    @patch("podcast_scraper.preprocessing.audio.chunker._probe_duration_seconds")
    @patch("podcast_scraper.preprocessing.audio.chunker.shutil.which")
    def test_split_invokes_ffmpeg_stream_copy(
        self, mock_which, mock_probe, mock_run, tmp_path
    ) -> None:
        mock_which.return_value = "/usr/bin/ffmpeg"
        mock_probe.return_value = 3000.0
        audio = tmp_path / "long.mp3"
        audio.write_bytes(b"x" * (26 * 1024 * 1024))

        def _fake_ffmpeg(cmd, **kwargs):
            out_path = cmd[-1]
            with open(out_path, "wb") as handle:
                handle.write(b"chunk-bytes")
            return CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

        mock_run.side_effect = _fake_ffmpeg
        chunker = AudioChunker(max_bytes=1024 * 1024, max_duration_seconds=1400.0)
        chunks = chunker.split(str(audio), work_dir=str(tmp_path / "chunks"))
        assert len(chunks) >= 2
        assert "-c" in mock_run.call_args[0][0]
        assert "copy" in mock_run.call_args[0][0]
