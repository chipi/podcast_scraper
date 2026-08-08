"""Unit tests for ``DeepgramDiarizationProvider`` (#913, 2026-06-15).

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

    def test_diarize_without_cfg_skips_cost_recording(self, tmp_audio):
        """No cfg passed (legacy construction) -> cost_usd stays None, no crash (BUG 1)."""
        p = DeepgramDiarizationProvider(api_key="test")
        fake_client = MagicMock()
        fake_client.listen.v1.media.transcribe_file.return_value = _make_dict_response(
            [_word(0.0, 60.0, 0)]
        )
        p._client = fake_client

        result = p.diarize(tmp_audio)
        assert result.cost_usd is None


class TestDeepgramDiarizationCost:
    """BUG 1: the diarization API call is a full billed audio pass that must emit its own
    ``llm_cost`` event — before this fix, only the transcription stage ever recorded cost, so a
    profile pairing ``diarization_provider: deepgram`` with a non-Deepgram transcriber silently
    dropped ALL diarization spend from telemetry (0 events, despite every call being billed)."""

    @pytest.fixture
    def tmp_audio(self, tmp_path):
        audio = tmp_path / "test.mp3"
        audio.write_bytes(b"fake-mp3-bytes")
        return str(audio)

    def _cfg(self):
        from podcast_scraper import config as cfg_module

        return cfg_module.Config.model_validate(
            {
                "rss_url": "https://example.com/feed.xml",
                "diarization_provider": "deepgram",
                "deepgram_api_key": "test-key",
            }
        )

    def test_diarize_emits_exactly_one_deepgram_diarization_cost_event(self, tmp_audio, caplog):
        import json
        import logging

        p = DeepgramDiarizationProvider(api_key="test", cfg=self._cfg())
        fake_client = MagicMock()
        # 10 minutes (600s) of audio across two speakers.
        fake_client.listen.v1.media.transcribe_file.return_value = _make_dict_response(
            [
                _word(0.0, 300.0, 0),
                _word(300.0, 600.0, 1),
            ]
        )
        p._client = fake_client

        with caplog.at_level(logging.INFO, logger="podcast_scraper.workflow.cost_monitoring"):
            result = p.diarize(tmp_audio)

        cost_events = [
            json.loads(r.message)
            for r in caplog.records
            if r.name == "podcast_scraper.workflow.cost_monitoring"
        ]
        assert len(cost_events) == 1
        event = cost_events[0]
        assert event["provider"] == "deepgram"
        assert event["stage"] == "diarization"
        assert event["estimated_cost_usd"] > 0.0
        # 10 min * $0.0043/min (nova-3 rate, deepgram.transcription.nova-3 pricing row).
        assert event["estimated_cost_usd"] == pytest.approx(0.043, rel=1e-3)
        assert result.cost_usd == pytest.approx(0.043, rel=1e-3)

    def test_diarize_no_cost_event_for_zero_duration(self, tmp_audio, caplog):
        import logging

        p = DeepgramDiarizationProvider(api_key="test", cfg=self._cfg())
        fake_client = MagicMock()
        fake_client.listen.v1.media.transcribe_file.return_value = _make_dict_response([])
        p._client = fake_client

        with caplog.at_level(logging.INFO, logger="podcast_scraper.workflow.cost_monitoring"):
            result = p.diarize(tmp_audio)

        cost_events = [
            r for r in caplog.records if r.name == "podcast_scraper.workflow.cost_monitoring"
        ]
        assert cost_events == []
        assert result.cost_usd is None


class TestInitializeUsesGenerousTimeout:
    """BUG 4: the diarization upload is the same size/cost as the transcription upload — it must
    get the same generous SDK timeout override, not just the transcription provider."""

    def test_no_base_url_uses_generous_write_timeout(self) -> None:
        # Inject a fake ``deepgram`` module (the SDK is an optional extra, absent in the unit-CI
        # env) so ``initialize()``'s ``from deepgram import DeepgramClient`` resolves hermetically.
        import sys

        from podcast_scraper import config_constants

        fake_deepgram = MagicMock()
        with patch.dict(sys.modules, {"deepgram": fake_deepgram}):
            p = DeepgramDiarizationProvider(api_key="dg-key")
            p.initialize()

        fake_deepgram.DeepgramClient.assert_called_once_with(
            api_key="dg-key", timeout=config_constants.DEEPGRAM_SDK_TIMEOUT_SECONDS
        )

    def test_base_url_override_uses_generous_write_timeout(self) -> None:
        import sys

        from podcast_scraper import config_constants

        fake_deepgram = MagicMock()
        fake_env_module = MagicMock()
        fake_env_module.DeepgramClientEnvironment.return_value = "FAKE_ENV"
        with patch.dict(
            sys.modules,
            {"deepgram": fake_deepgram, "deepgram.environment": fake_env_module},
        ):
            p = DeepgramDiarizationProvider(api_key="dg-key", api_base="http://self-hosted:8080")
            p.initialize()

        fake_deepgram.DeepgramClient.assert_called_once_with(
            api_key="dg-key",
            environment="FAKE_ENV",
            timeout=config_constants.DEEPGRAM_SDK_TIMEOUT_SECONDS,
        )


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
        assert provider.cfg is cfg  # BUG 1: cfg threaded through so cost can be recorded

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
