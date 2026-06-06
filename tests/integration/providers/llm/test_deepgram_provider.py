"""Integration tests for the Deepgram transcription provider.

Drive ``initialize -> transcribe_with_segments -> parse -> screenplay -> cost``
against a **realistic, full-shape** Deepgram Nova-3 response (multi-speaker:
``results.channels[].alternatives[].{transcript,words}`` + ``utterances[]`` with
``speaker``/``start``/``end``). The SDK *method* is mocked (no network), but the
payload mirrors a real diarized response — so this catches our-side parse /
screenplay / cost regressions the toy unit dicts can't.

The complementary mock-server round-trip (real SDK against the e2e HTTP server)
lives in ``tests/integration/infrastructure/test_deepgram_mock.py``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from podcast_scraper import config
from podcast_scraper.providers.deepgram.deepgram_provider import DeepgramTranscriptionProvider

pytestmark = [pytest.mark.integration, pytest.mark.llm]


# A realistic two-speaker Deepgram Nova-3 response: the host (speaker 0) and a
# guest (speaker 1), with utterances (preferred by the parser) AND a words-level
# alternative (the fallback path), so both extraction routes have real data.
_REAL_SHAPE_RESPONSE = {
    "metadata": {"model_info": {"name": "nova-3"}, "duration": 7.5, "channels": 1},
    "results": {
        "channels": [
            {
                "alternatives": [
                    {
                        "transcript": (
                            "Welcome to the show. Thanks for having me. "
                            "Let's get started. Absolutely, let's dive in."
                        ),
                        "words": [
                            {
                                "word": "welcome",
                                "punctuated_word": "Welcome",
                                "start": 0.0,
                                "end": 0.4,
                                "speaker": 0,
                            },
                            {
                                "word": "thanks",
                                "punctuated_word": "Thanks",
                                "start": 2.0,
                                "end": 2.4,
                                "speaker": 1,
                            },
                            {
                                "word": "lets",
                                "punctuated_word": "Let's",
                                "start": 4.0,
                                "end": 4.3,
                                "speaker": 0,
                            },
                            {
                                "word": "absolutely",
                                "punctuated_word": "Absolutely",
                                "start": 6.0,
                                "end": 6.5,
                                "speaker": 1,
                            },
                        ],
                    }
                ]
            }
        ],
        "utterances": [
            {"start": 0.0, "end": 1.8, "transcript": "Welcome to the show.", "speaker": 0},
            {"start": 2.0, "end": 3.6, "transcript": "Thanks for having me.", "speaker": 1},
            {"start": 4.0, "end": 5.6, "transcript": "Let's get started.", "speaker": 0},
            {"start": 6.0, "end": 7.5, "transcript": "Absolutely, let's dive in.", "speaker": 1},
        ],
    },
}


def _provider(model: str = "nova-3") -> DeepgramTranscriptionProvider:
    cfg = config.Config(
        rss="https://example.com/feed.xml",
        transcription_provider="deepgram",
        deepgram_api_key="dg-test-key",
        deepgram_model=model,
    )
    return DeepgramTranscriptionProvider(cfg)


def _transcribe(provider, response, **kwargs):
    """Run transcribe_with_segments with the SDK call returning ``response``."""
    mock_client = MagicMock()
    mock_client.listen.v1.media.transcribe_file.return_value = response
    with patch(
        "podcast_scraper.providers.deepgram.deepgram_provider._create_deepgram_client",
        return_value=mock_client,
    ):
        provider.initialize()
    with (
        patch("builtins.open", create=True) as mock_open,
        patch(
            "podcast_scraper.providers.deepgram.deepgram_provider.os.path.exists",
            return_value=True,
        ),
    ):
        mock_open.return_value.__enter__.return_value.read.return_value = b"audio-bytes"
        return provider.transcribe_with_segments("/tmp/ep.mp3", **kwargs)


class TestDeepgramRealShapeParse:
    def test_multispeaker_utterances_become_segments(self) -> None:
        provider = _provider()
        result, elapsed = _transcribe(provider, _REAL_SHAPE_RESPONSE, language="en")

        assert elapsed >= 0
        segments = result["segments"]
        # Utterances are preferred over the words fallback -> 4 turns, alternating speakers.
        assert [s["speaker"] for s in segments] == [0, 1, 0, 1]
        assert segments[0]["text"] == "Welcome to the show."
        assert segments[1]["start"] == 2.0 and segments[1]["end"] == 3.6
        assert "welcome to the show" in result["text"].lower()
        assert "dive in" in result["text"].lower()

    def test_native_diarization_renders_named_screenplay(self) -> None:
        """The diarized segments map onto detected names in a screenplay (D1)."""
        provider = _provider()
        result, _ = _transcribe(provider, _REAL_SHAPE_RESPONSE)

        screenplay = provider.format_screenplay_from_segments(
            result["segments"], None, ["Maya", "Liam"]
        )
        assert screenplay is not None
        assert "Maya: Welcome to the show." in screenplay
        assert "Liam: Thanks for having me." in screenplay
        # Speaker 0 reappears as Maya (stable first-appearance mapping).
        assert "Maya: Let's get started." in screenplay

    def test_words_fallback_when_utterances_absent(self) -> None:
        """With no utterances, the words-level alternative drives multi-speaker segments."""
        response = {
            "metadata": {"duration": 7.5},
            "results": {
                "channels": _REAL_SHAPE_RESPONSE["results"]["channels"],
                "utterances": [],
            },
        }
        provider = _provider()
        result, _ = _transcribe(provider, response)
        # 4 single-word turns, each a distinct speaker run.
        assert [s["speaker"] for s in result["segments"]] == [0, 1, 0, 1]
        assert result["segments"][0]["text"] == "Welcome"

    def test_cost_recorded_from_feed_duration(self) -> None:
        """D5: a 5-minute episode records the per-minute transcription call."""
        provider = _provider()
        pipeline_metrics = MagicMock()
        call_metrics = MagicMock()
        call_metrics.estimated_cost = None
        _transcribe(
            provider,
            _REAL_SHAPE_RESPONSE,
            pipeline_metrics=pipeline_metrics,
            episode_duration_seconds=300,
            call_metrics=call_metrics,
        )
        pipeline_metrics.record_llm_transcription_call.assert_called_once()
        minutes = pipeline_metrics.record_llm_transcription_call.call_args.args[0]
        assert minutes == pytest.approx(5.0)

    def test_model_dump_response_object_is_parsed(self) -> None:
        """A typed SDK response (Pydantic-like, exposes model_dump) parses like a dict —
        mirrors the real v7 ListenV1Response our _response_to_dict normalizes."""

        class _TypedResponse:
            def model_dump(self):
                return _REAL_SHAPE_RESPONSE

        provider = _provider()
        result, _ = _transcribe(provider, _TypedResponse())
        assert [s["speaker"] for s in result["segments"]] == [0, 1, 0, 1]
