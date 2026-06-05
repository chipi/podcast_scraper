"""Unit tests for Deepgram transcription provider."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from podcast_scraper import config
from podcast_scraper.providers.deepgram.deepgram_provider import (
    DeepgramTranscriptionProvider,
    parse_deepgram_transcript,
)

pytestmark = pytest.mark.unit


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
