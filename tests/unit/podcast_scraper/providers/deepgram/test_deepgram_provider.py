"""Unit tests for Deepgram transcription provider."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from podcast_scraper import config
from podcast_scraper.providers.deepgram.deepgram_provider import (
    _words_to_segments,
    DeepgramTranscriptionProvider,
    parse_deepgram_transcript,
)

pytestmark = pytest.mark.unit


def _provider_with_mock_client(mock_client, **cfg_overrides):
    """Build an initialized provider whose SDK client is the given mock."""
    cfg = config.Config(
        rss="https://example.com/feed.xml",
        transcription_provider="deepgram",
        deepgram_api_key="dg-test-key",
        deepgram_model="nova-3",
        **cfg_overrides,
    )
    provider = DeepgramTranscriptionProvider(cfg)
    provider._client = mock_client
    provider._initialized = True
    return provider


class TestParseDeepgramTranscript:
    def test_utterances_map_to_segments(self) -> None:
        payload = {
            "results": {
                "channels": [{"alternatives": [{"transcript": "hello world"}]}],
                "utterances": [
                    {
                        "start": 0.0,
                        "end": 1.2,
                        "transcript": "hello world",
                        "speaker": 0,
                    }
                ],
            }
        }
        parsed = parse_deepgram_transcript(payload)
        assert parsed["text"] == "hello world"
        assert len(parsed["segments"]) == 1
        assert parsed["segments"][0]["speaker"] == 0

    def test_empty_dict_response_yields_empty_result(self) -> None:
        """A well-formed-but-empty payload degrades to empty, not a crash (D6)."""
        assert parse_deepgram_transcript({}) == {"text": "", "segments": []}

    def test_malformed_missing_results_yields_empty(self) -> None:
        """Missing ``results``/``channels`` keys → empty result (D6)."""
        assert parse_deepgram_transcript({"metadata": {"x": 1}}) == {"text": "", "segments": []}

    def test_transcript_synthesized_from_segments_when_top_level_blank(self) -> None:
        """When the alternative transcript is blank, text is rebuilt from segments (D6)."""
        payload = {
            "results": {
                "channels": [{"alternatives": [{"transcript": ""}]}],
                "utterances": [
                    {"start": 0.0, "end": 1.0, "transcript": "one", "speaker": 0},
                    {"start": 1.0, "end": 2.0, "transcript": "two", "speaker": 1},
                ],
            }
        }
        parsed = parse_deepgram_transcript(payload)
        assert parsed["text"] == "one two"
        assert [s["speaker"] for s in parsed["segments"]] == [0, 1]

    def test_words_fallback_when_no_utterances(self) -> None:
        payload = {
            "results": {
                "channels": [
                    {
                        "alternatives": [
                            {
                                "transcript": "hello there",
                                "words": [
                                    {
                                        "word": "hello",
                                        "start": 0.0,
                                        "end": 0.4,
                                        "speaker": 0,
                                    },
                                    {
                                        "word": "there",
                                        "start": 0.5,
                                        "end": 0.9,
                                        "speaker": 0,
                                    },
                                ],
                            }
                        ]
                    }
                ],
                "utterances": [],
            }
        }
        parsed = parse_deepgram_transcript(payload)
        assert parsed["text"] == "hello there"
        assert len(parsed["segments"]) == 1


class TestDeepgramTranscriptionProvider:
    def test_initialize_requires_api_key(self) -> None:
        cfg = config.Config.model_construct(
            rss="https://example.com/feed.xml",
            transcription_provider="deepgram",
            deepgram_api_key=None,
        )
        provider = DeepgramTranscriptionProvider(cfg)
        with pytest.raises(ValueError, match="Deepgram API key"):
            provider.initialize()

    @patch("podcast_scraper.providers.deepgram.deepgram_provider._create_deepgram_client")
    def test_transcribe_with_segments_maps_response(self, mock_create_client) -> None:
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client
        mock_client.listen.v1.media.transcribe_file.return_value = {
            "results": {
                "channels": [{"alternatives": [{"transcript": "episode text"}]}],
                "utterances": [
                    {
                        "start": 0.5,
                        "end": 2.0,
                        "transcript": "episode text",
                        "speaker": 1,
                    }
                ],
            }
        }

        cfg = config.Config(
            rss="https://example.com/feed.xml",
            transcription_provider="deepgram",
            deepgram_api_key="dg-test-key",
            deepgram_model="nova-3",
        )
        provider = DeepgramTranscriptionProvider(cfg)
        provider.initialize()

        with (
            patch("builtins.open", create=True) as mock_open,
            patch(
                "podcast_scraper.providers.deepgram.deepgram_provider.os.path.exists",
                return_value=True,
            ),
        ):
            mock_open.return_value.__enter__.return_value.read.return_value = b"audio-bytes"
            result, elapsed = provider.transcribe_with_segments("/tmp/ep.mp3", language="en")

        assert result["text"] == "episode text"
        assert result["segments"][0]["text"] == "episode text"
        assert elapsed >= 0
        mock_client.listen.v1.media.transcribe_file.assert_called_once()
        call_kwargs = mock_client.listen.v1.media.transcribe_file.call_args.kwargs
        assert call_kwargs["model"] == "nova-3"
        assert call_kwargs["diarize"] is True
        mock_create_client.assert_called_once_with("dg-test-key")

    @patch(
        "podcast_scraper.providers.deepgram.deepgram_provider._create_deepgram_client",
        side_effect=RuntimeError("deepgram-sdk is required"),
    )
    def test_initialize_requires_deepgram_sdk(self, _mock_create) -> None:
        cfg = config.Config(
            rss="https://example.com/feed.xml",
            transcription_provider="deepgram",
            deepgram_api_key="dg-test-key",
        )
        provider = DeepgramTranscriptionProvider(cfg)
        with pytest.raises(RuntimeError, match="deepgram-sdk is required"):
            provider.initialize()


class TestDeepgramRobustness:
    def test_unparseable_response_returns_empty(self) -> None:
        """An unrecognized response shape degrades to empty, not a crash (D4)."""
        parsed = parse_deepgram_transcript(object())
        assert parsed == {"text": "", "segments": []}

    def test_deepgram_retryable_exceptions_non_empty(self) -> None:
        """The retryable set is wired so the API call gets backoff like siblings (D2)."""
        from podcast_scraper.utils.provider_metrics import _safe_deepgram_retryable

        retryable = _safe_deepgram_retryable()
        assert isinstance(retryable, tuple) and retryable
        assert all(isinstance(e, type) and issubclass(e, Exception) for e in retryable)


class TestDeepgramScreenplay:
    def _provider(self) -> DeepgramTranscriptionProvider:
        cfg = config.Config(
            rss="https://example.com/feed.xml",
            transcription_provider="deepgram",
            deepgram_api_key="dg-test-key",
        )
        return DeepgramTranscriptionProvider(cfg)

    def test_format_screenplay_maps_native_speakers_to_names(self) -> None:
        """Deepgram's integer speakers become a named screenplay (D1)."""
        segments = [
            {"start": 0.0, "end": 2.0, "text": "hello", "speaker": 0},
            {"start": 2.0, "end": 4.0, "text": "hi there", "speaker": 1},
            {"start": 4.0, "end": 6.0, "text": "back to me", "speaker": 0},
        ]
        out = self._provider().format_screenplay_from_segments(segments, None, ["Alice", "Bob"])
        assert out is not None
        assert "Alice: hello" in out
        assert "Bob: hi there" in out
        assert "Alice: back to me" in out

    def test_format_screenplay_falls_back_to_speaker_n(self) -> None:
        segments = [{"start": 0.0, "end": 1.0, "text": "solo", "speaker": 0}]
        out = self._provider().format_screenplay_from_segments(segments, None, None)
        assert out is not None
        assert "Speaker 1: solo" in out

    def test_format_screenplay_none_when_no_segments(self) -> None:
        assert self._provider().format_screenplay_from_segments([], None, ["Alice"]) is None


class TestDeepgramConfigCoercion:
    def test_deepgram_keeps_screenplay_but_drops_pyannote_diarize(self) -> None:
        """Native-diarization provider keeps screenplay; the pyannote pass is coerced off (D1)."""
        config.reset_diarize_coerce_log_for_tests()
        config.reset_screenplay_transcription_api_coerce_log_for_tests()
        cfg = config.Config(
            rss="https://example.com/feed.xml",
            transcription_provider="deepgram",
            deepgram_api_key="dg-test-key",
            screenplay=True,
            diarize=True,
        )
        assert cfg.screenplay is True
        assert cfg.diarize is False


class TestDeepgramConfigValidation:
    def test_missing_key_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Deepgram API key"):
            config.Config(
                rss="https://example.com/feed.xml",
                transcription_provider="deepgram",
            )

    def test_with_key_accepts(self) -> None:
        cfg = config.Config(
            rss="https://example.com/feed.xml",
            transcription_provider="deepgram",
            deepgram_api_key="dg-test",
        )
        assert cfg.deepgram_model == "nova-3"


class TestWordsToSegments:
    """Multi-speaker grouping in the words->segments fallback (D6)."""

    def test_groups_consecutive_words_by_speaker(self) -> None:
        words = [
            {"word": "hello", "punctuated_word": "Hello,", "start": 0.0, "end": 0.4, "speaker": 0},
            {"word": "there", "punctuated_word": "there.", "start": 0.4, "end": 0.8, "speaker": 0},
            {"word": "hi", "punctuated_word": "Hi!", "start": 1.0, "end": 1.3, "speaker": 1},
            {"word": "yes", "punctuated_word": "Yes,", "start": 2.0, "end": 2.3, "speaker": 0},
        ]
        segs = _words_to_segments(words)
        # Speaker runs 0,1,0 -> three segments preserving order + boundaries.
        assert [s["speaker"] for s in segs] == [0, 1, 0]
        assert segs[0]["text"] == "Hello, there."
        assert segs[0]["start"] == 0.0 and segs[0]["end"] == 0.8
        assert segs[1]["text"] == "Hi!"
        assert segs[2]["text"] == "Yes,"

    def test_empty_words_yields_no_segments(self) -> None:
        assert _words_to_segments([]) == []


class TestDeepgramCapabilities:
    def test_capabilities_are_transcription_only_with_gi_timing(self) -> None:
        cfg = config.Config(
            rss="https://example.com/feed.xml",
            transcription_provider="deepgram",
            deepgram_api_key="dg-test",
        )
        caps = DeepgramTranscriptionProvider(cfg).get_capabilities()
        assert caps.supports_transcription is True
        assert caps.supports_audio_input is True
        assert caps.supports_gi_segment_timing is True
        assert caps.supports_summarization is False
        assert caps.provider_name == "deepgram"


class TestDeepgramTranscribeGuards:
    def test_transcribe_before_initialize_raises(self) -> None:
        cfg = config.Config(
            rss="https://example.com/feed.xml",
            transcription_provider="deepgram",
            deepgram_api_key="dg-test",
        )
        provider = DeepgramTranscriptionProvider(cfg)
        with pytest.raises(RuntimeError, match="not initialized"):
            provider.transcribe_with_segments("/tmp/whatever.mp3")

    def test_transcribe_missing_file_raises_filenotfound(self) -> None:
        provider = _provider_with_mock_client(MagicMock())
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            provider.transcribe_with_segments("/no/such/file.mp3")

    def test_api_error_propagates(self) -> None:
        """A non-retryable SDK error surfaces (logged + re-raised), not swallowed (D6)."""
        mock_client = MagicMock()
        mock_client.listen.v1.media.transcribe_file.side_effect = ValueError("boom")
        provider = _provider_with_mock_client(mock_client)
        with (
            patch("builtins.open", create=True) as mock_open,
            patch(
                "podcast_scraper.providers.deepgram.deepgram_provider.os.path.exists",
                return_value=True,
            ),
        ):
            mock_open.return_value.__enter__.return_value.read.return_value = b"audio"
            with pytest.raises(ValueError, match="boom"):
                provider.transcribe_with_segments("/tmp/ep.mp3")


class TestDeepgramCost:
    """D5: per-minute transcription cost is recorded from the feed duration."""

    def _run(self, episode_duration_seconds):
        mock_client = MagicMock()
        mock_client.listen.v1.media.transcribe_file.return_value = {
            "results": {"channels": [{"alternatives": [{"transcript": "hi"}]}], "utterances": []}
        }
        provider = _provider_with_mock_client(mock_client)
        call_metrics = MagicMock()
        call_metrics.estimated_cost = None
        pipeline_metrics = MagicMock()
        with (
            patch("builtins.open", create=True) as mock_open,
            patch(
                "podcast_scraper.providers.deepgram.deepgram_provider.os.path.exists",
                return_value=True,
            ),
        ):
            mock_open.return_value.__enter__.return_value.read.return_value = b"audio"
            provider.transcribe_with_segments(
                "/tmp/ep.mp3",
                pipeline_metrics=pipeline_metrics,
                episode_duration_seconds=episode_duration_seconds,
                call_metrics=call_metrics,
            )
        return call_metrics, pipeline_metrics

    def test_cost_recorded_from_episode_duration(self) -> None:
        """10 min at nova-3 ($0.0043/min) -> the cost helper is invoked with 10.0 minutes."""
        call_metrics, pipeline_metrics = self._run(episode_duration_seconds=600)
        # pipeline_metrics records the audio-minute transcription call (10.0 min).
        pipeline_metrics.record_llm_transcription_call.assert_called_once()
        args, kwargs = pipeline_metrics.record_llm_transcription_call.call_args
        assert args[0] == pytest.approx(10.0)

    def test_no_cost_call_when_duration_unknown_and_no_file(self) -> None:
        """With no duration and an unstattable path, audio_minutes stays 0 -> no cost call."""
        mock_client = MagicMock()
        mock_client.listen.v1.media.transcribe_file.return_value = {
            "results": {"channels": [{"alternatives": [{"transcript": "hi"}]}], "utterances": []}
        }
        provider = _provider_with_mock_client(mock_client)
        pipeline_metrics = MagicMock()
        with (
            patch("builtins.open", create=True) as mock_open,
            patch(
                "podcast_scraper.providers.deepgram.deepgram_provider.os.path.exists",
                return_value=True,
            ),
            patch(
                "podcast_scraper.providers.deepgram.deepgram_provider.os.path.getsize",
                side_effect=OSError("gone"),
            ),
        ):
            mock_open.return_value.__enter__.return_value.read.return_value = b"audio"
            provider.transcribe_with_segments(
                "/tmp/ep.mp3", pipeline_metrics=pipeline_metrics, episode_duration_seconds=None
            )
        pipeline_metrics.record_llm_transcription_call.assert_not_called()


class TestDeepgramPayloadCap:
    def test_deepgram_uses_large_single_request_cap(self) -> None:
        """D3: Deepgram's byte cap is its real 2 GB ceiling, not the 25 MiB default —
        so whole episodes go in one request and diarization speakers stay consistent."""
        from podcast_scraper.utils.audio_payload_limits import transcription_max_bytes

        cfg = config.Config(
            rss="https://example.com/feed.xml",
            transcription_provider="deepgram",
            deepgram_api_key="dg-test",
        )
        assert transcription_max_bytes(cfg) == 2 * 1024 * 1024 * 1024
