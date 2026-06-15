"""Unit tests for ``DeepgramDiarizationProvider`` (#913 follow-up).

The provider POSTs audio to Deepgram's Listen API and parses speaker
turns. These tests cover the response-parsing logic (handles dict and
object response shapes), the speaker grouping (consecutive words by
the same speaker become one segment), and the factory wiring.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from podcast_scraper.providers.ml.diarization.base import DiarizationResult
from podcast_scraper.providers.ml.diarization.deepgram_provider import (
    DeepgramDiarizationProvider,
)


def _make_dict_response(words):
    return {
        "results": {
            "channels": [
                {"alternatives": [{"words": words}]},
            ],
        },
    }


def _make_object_response(words):
    channel = SimpleNamespace(alternatives=[SimpleNamespace(words=words)])
    results = SimpleNamespace(channels=[channel])
    return SimpleNamespace(results=results)


def _word(start, end, speaker):
    return {"start": start, "end": end, "speaker": speaker}


class TestExtractSpeakerTurns:
    def test_empty_response_returns_empty(self):
        result = DeepgramDiarizationProvider._extract_speaker_turns({})
        assert result == []

    def test_missing_results_returns_empty(self):
        result = DeepgramDiarizationProvider._extract_speaker_turns({"foo": "bar"})
        assert result == []

    def test_no_channels_returns_empty(self):
        result = DeepgramDiarizationProvider._extract_speaker_turns({"results": {"channels": []}})
        assert result == []

    def test_no_words_returns_empty(self):
        result = DeepgramDiarizationProvider._extract_speaker_turns(_make_dict_response([]))
        assert result == []

    def test_single_speaker_one_segment(self):
        words = [_word(0.0, 0.5, 0), _word(0.5, 1.0, 0), _word(1.0, 1.5, 0)]
        segs = DeepgramDiarizationProvider._extract_speaker_turns(_make_dict_response(words))
        assert len(segs) == 1
        assert segs[0].start == 0.0
        assert segs[0].end == 1.5
        assert segs[0].speaker == "SPEAKER_00"

    def test_two_speakers_two_segments(self):
        words = [
            _word(0.0, 0.5, 0),
            _word(0.5, 1.0, 0),
            _word(1.0, 1.5, 1),
            _word(1.5, 2.0, 1),
        ]
        segs = DeepgramDiarizationProvider._extract_speaker_turns(_make_dict_response(words))
        assert len(segs) == 2
        assert segs[0].speaker == "SPEAKER_00"
        assert segs[1].speaker == "SPEAKER_01"
        assert segs[0].end == 1.0
        assert segs[1].start == 1.0

    def test_alternating_speakers_collapse_correctly(self):
        words = [
            _word(0.0, 0.5, 0),
            _word(0.5, 1.0, 1),
            _word(1.0, 1.5, 0),
            _word(1.5, 2.0, 0),
        ]
        segs = DeepgramDiarizationProvider._extract_speaker_turns(_make_dict_response(words))
        assert len(segs) == 3
        assert [s.speaker for s in segs] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]
        # Final speaker_00 segment covers the last 0.5s.
        assert segs[-1].start == 1.0
        assert segs[-1].end == 2.0

    def test_object_response_shape_works(self):
        # Deepgram SDK can return either dict or object responses.
        words = [
            _word(0.0, 0.5, 0),
            _word(0.5, 1.0, 1),
        ]
        segs = DeepgramDiarizationProvider._extract_speaker_turns(_make_object_response(words))
        assert len(segs) == 2

    def test_words_with_missing_fields_are_skipped(self):
        # A word with no speaker or no timestamps shouldn't crash the parser.
        words = [
            _word(0.0, 0.5, 0),
            {"start": 0.5, "end": 1.0},  # no speaker
            _word(1.0, 1.5, 0),
        ]
        segs = DeepgramDiarizationProvider._extract_speaker_turns(_make_dict_response(words))
        # The two real words with speaker=0 collapse into one segment.
        assert len(segs) == 1
        assert segs[0].speaker == "SPEAKER_00"

    def test_malformed_response_returns_empty_without_crashing(self):
        """Defensive: a response that doesn't match either shape just
        returns [] (and logs WARNING) instead of crashing the pipeline."""
        result = DeepgramDiarizationProvider._extract_speaker_turns("not a dict or object")
        assert result == []


class TestProviderConstruction:
    def test_missing_api_key_raises(self):
        with pytest.raises(ValueError, match="Deepgram API key required"):
            DeepgramDiarizationProvider(api_key="")

    def test_default_model_is_nova3_general(self):
        p = DeepgramDiarizationProvider(api_key="test")
        assert p.model == "nova-3-general"

    def test_diarize_missing_file_raises(self):
        p = DeepgramDiarizationProvider(api_key="test")
        p._client = MagicMock()  # bypass initialize
        with pytest.raises(FileNotFoundError):
            p.diarize("/no/such/file.mp3")


class TestDiarizeAgainstMockedClient:
    @pytest.fixture
    def tmp_audio(self, tmp_path):
        audio = tmp_path / "test.mp3"
        audio.write_bytes(b"fake-mp3-bytes")
        return str(audio)

    def test_diarize_returns_diarization_result(self, tmp_audio):
        p = DeepgramDiarizationProvider(api_key="test")
        fake_client = MagicMock()
        fake_client.listen.v1.media.transcribe_file.return_value = _make_dict_response(
            [
                _word(0.0, 1.0, 0),
                _word(1.0, 2.0, 1),
            ]
        )
        p._client = fake_client

        result = p.diarize(tmp_audio)
        assert isinstance(result, DiarizationResult)
        assert result.num_speakers == 2
        assert len(result.segments) == 2
        assert result.model_name == "deepgram/nova-3-general"
        # Verify the API was called with diarize=True
        call_kwargs = fake_client.listen.v1.media.transcribe_file.call_args.kwargs
        assert call_kwargs["diarize"] is True
        assert call_kwargs["model"] == "nova-3-general"


class TestFactoryWiring:
    def test_factory_dispatches_to_deepgram(self, monkeypatch):
        from podcast_scraper import config as cfg_module
        from podcast_scraper.providers.ml.diarization.factory import (
            create_diarization_provider,
        )

        cfg = cfg_module.Config.model_validate(
            {
                "rss_url": "https://example.com/feed.xml",
                "diarization_provider": "deepgram",
                "deepgram_api_key": "test-key",
            }
        )

        with patch.object(DeepgramDiarizationProvider, "initialize", autospec=True) as mocked_init:
            provider = create_diarization_provider(cfg)
            mocked_init.assert_called_once()
        assert isinstance(provider, DeepgramDiarizationProvider)
        assert provider.api_key == "test-key"

    def test_factory_raises_without_api_key(self):
        from podcast_scraper import config as cfg_module
        from podcast_scraper.providers.ml.diarization.factory import (
            create_diarization_provider,
        )

        cfg = cfg_module.Config.model_validate(
            {
                "rss_url": "https://example.com/feed.xml",
                "diarization_provider": "deepgram",
            }
        )
        # Make sure no env var is providing a key.
        import os

        old = os.environ.pop("DEEPGRAM_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="DEEPGRAM_API_KEY required"):
                create_diarization_provider(cfg)
        finally:
            if old is not None:
                os.environ["DEEPGRAM_API_KEY"] = old
