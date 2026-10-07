"""The artifacts record what language went to ASR and what the service said back (#2187).

WHY THIS EXISTS. V.6b's first real ASR run on non-English audio (Spanish, 2026-10-07) could not
answer two of its own questions — did the per-episode language reach the ASR service, and does
the service's reported language agree — because nothing recorded either: `asr.json` and the
manifest's `asr` stage carried only the model. The DGX provider merged the two into one
`language` field (`requested or detected`), which makes a disagreement invisible by construction.
These tests pin the split: the provider returns both halves, `asr.json` carries a `language`
block, and the manifest flags `asr_language_mismatch` when the primary subtags differ.
"""

from __future__ import annotations

import json
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from podcast_scraper.config import Config
from podcast_scraper.providers.tailnet_dgx import whisper_provider as wp
from podcast_scraper.providers.tailnet_dgx.whisper_provider import (
    TailnetDgxWhisperTranscriptionProvider,
)
from podcast_scraper.workflow import episode_processor, processing_manifest as pm

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_breaker():
    wp._whisper_breaker.reset()
    yield
    wp._whisper_breaker.reset()


def _cfg(**kw):
    base = dict(
        rss_url="https://example.com/feed.xml",
        run_id="run-xyz",
        transcription_provider="tailnet_dgx_whisper",
        dgx_whisper_model="turbo",
        diarization_provider="pyannote",
    )
    base.update(kw)
    return SimpleNamespace(**base)


class TestTheProviderReturnsBothHalves:
    def _provider(self) -> TailnetDgxWhisperTranscriptionProvider:
        cfg = Config.model_validate(
            {
                "rss_url": "https://example.com/feed.xml",
                "transcription_provider": "tailnet_dgx_whisper",
                "transcription_fallback_provider": "openai",
                "dgx_tailnet_host": "dgx-llm-1.tail-test.ts.net",
                "openai_api_key": "sk-test",
            }
        )
        provider = TailnetDgxWhisperTranscriptionProvider(cfg)
        provider._initialized = True
        return provider

    @pytest.mark.parametrize(("asked", "server_says"), [("es", "es"), ("es", "pt"), (None, "de")])
    def test_requested_and_reported_are_kept_apart(self, tmp_path, asked, server_says) -> None:
        audio = tmp_path / "ep.mp3"
        audio.write_bytes(b"\x00\x01")
        provider = self._provider()

        def fake_call(*_a, **_k):
            provider._last_detected_language = server_says  # what the server's payload said
            return "texto", [{"start": 0.0, "end": 1.0, "text": "texto"}], 0.5

        with (
            patch(
                "podcast_scraper.providers.tailnet_dgx.whisper_provider.check_faster_whisper_health",
                return_value=True,
            ),
            patch.object(TailnetDgxWhisperTranscriptionProvider, "_transcribe_dgx", fake_call),
        ):
            result, _ = provider.transcribe_with_segments(str(audio), language=asked)

        assert result["language_requested"] == asked
        assert result["language_reported"] == server_says
        assert result["language"] == (asked or server_says)  # the merged field is unchanged


class TestTheRecord:
    def test_a_match(self) -> None:
        rec = episode_processor._asr_language_record(
            {"language_requested": "es", "language_reported": "es"}, _cfg()
        )
        assert rec == {"requested": "es", "reported": "es", "mismatch": False}

    def test_a_mismatch_compares_primary_subtags(self) -> None:
        rec = episode_processor._asr_language_record(
            {"language_requested": "pt-BR", "language_reported": "es"}, _cfg()
        )
        assert rec["mismatch"] is True
        same = episode_processor._asr_language_record(
            {"language_requested": "pt-BR", "language_reported": "pt"}, _cfg()
        )
        assert same["mismatch"] is False

    def test_reported_is_never_filled_from_requested(self) -> None:
        rec = episode_processor._asr_language_record({"language_requested": "es"}, _cfg())
        assert rec == {"requested": "es", "reported": None, "mismatch": None}

    def test_requested_falls_back_to_the_episodes_resolved_language(self) -> None:
        cfg = _cfg(feed_declared_language="de-DE")
        rec = episode_processor._asr_language_record({}, cfg)
        assert rec["requested"] == "de"
        assert rec["reported"] is None


class TestTheArtifacts:
    def _setup(self):
        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, "transcripts"))
        open(os.path.join(d, "transcripts", "0006 - X.txt"), "w").close()
        return d, "transcripts/0006 - X.txt"

    def test_asr_json_carries_the_language_block(self) -> None:
        d, rel = self._setup()
        result = {
            "speech_audio_ratio": 0.93,
            "language_requested": "es",
            "language_reported": "pt",
        }
        episode_processor._save_asr_provenance_file(result, _cfg(), rel, d)
        rec = json.load(open(os.path.join(d, "transcripts", "0006 - X.asr.json")))
        assert rec["language"] == {"requested": "es", "reported": "pt", "mismatch": True}

    def test_the_manifest_flags_a_mismatch(self) -> None:
        from podcast_scraper.models.entities import TranscriptionJob

        d, rel = self._setup()
        job = TranscriptionJob(idx=6, ep_title="X", ep_title_safe="X", temp_media="", episode=None)
        result = {"speech_audio_ratio": 0.93, "language_requested": "es", "language_reported": "pt"}
        episode_processor._write_processing_manifest(result, _cfg(), job, rel, d)
        data = json.load(open(pm.manifest_path(d, rel)))
        asr = data["stages"]["asr"]
        assert asr["metrics"]["language_requested"] == "es"
        assert asr["metrics"]["language_reported"] == "pt"
        assert "asr_language_mismatch" in data["quality_flags"]

    def test_the_manifest_does_not_flag_a_match(self) -> None:
        from podcast_scraper.models.entities import TranscriptionJob

        d, rel = self._setup()
        job = TranscriptionJob(idx=6, ep_title="X", ep_title_safe="X", temp_media="", episode=None)
        result = {"speech_audio_ratio": 0.93, "language_requested": "es", "language_reported": "es"}
        episode_processor._write_processing_manifest(result, _cfg(), job, rel, d)
        data = json.load(open(pm.manifest_path(d, rel)))
        assert "asr_language_mismatch" not in data.get("quality_flags", [])
