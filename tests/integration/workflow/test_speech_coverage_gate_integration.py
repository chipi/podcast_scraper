"""Integration: ADR-131 speech-coverage gate over the REAL diarization denominator.

Couples ``apply_diarization_to_result`` (which computes ``diarization_speech_seconds`` from a real
diarization) with ``_maybe_speech_coverage_failover`` (the gate decision). Only the diarization
*provider* and the failover *re-transcription* are mocked; the speech-denominator maths and the gate
threshold run for real — the layer the unit tests stub out.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from podcast_scraper import config as podcast_config
from podcast_scraper.models.entities import TranscriptionJob
from podcast_scraper.providers.ml.diarization.base import DiarizationResult, DiarizationSegment
from podcast_scraper.providers.ml.diarization.pipeline import apply_diarization_to_result
from podcast_scraper.workflow import episode_processor

pytestmark = pytest.mark.integration

_PIPELINE = "podcast_scraper.providers.ml.diarization.pipeline"


def _cfg():
    return podcast_config.Config(
        rss="https://example.com/feed.xml",
        transcription_provider="whisper",
        diarize=True,
        hf_token="hf-test",
        dgx_whisper_model="turbo",
        transcription_speech_coverage_min=0.85,
        transcription_coverage_failover_model="large-v3",
    )


def _job():
    return TranscriptionJob(idx=1, ep_title="t", ep_title_safe="t", temp_media="", episode=None)


def _diarized(diar_segments, transcript_segments, cfg=None):
    """Run the REAL apply_diarization_to_result to get a result carrying diarization_speech_seconds."""
    provider = MagicMock()
    provider.diarize.return_value = DiarizationResult(
        segments=diar_segments, num_speakers=2, model_name="test"
    )
    base = {
        "text": " ".join(s["text"] for s in transcript_segments),
        "segments": transcript_segments,
    }
    with patch(f"{_PIPELINE}.create_diarization_provider", return_value=provider):
        return apply_diarization_to_result(base, "/tmp/a.wav", cfg or _cfg(), ["Guest"])


def test_music_heavy_episode_does_not_failover() -> None:
    # The "Move Over Humans" case: the diarizer found 400s of SPEECH; turbo transcribed ~390s of it.
    # The audio may be much longer (music/ads), but that is not in the diarization denominator, so
    # speech coverage ~0.97 >= 0.85 and NO failover fires — even though RAW coverage would be low.
    diar = [
        DiarizationSegment(0.0, 60.0, "SPEAKER_00"),
        DiarizationSegment(60.0, 400.0, "SPEAKER_01"),
    ]
    host = "Welcome back. I'm Katie Martin. " + ("Markets talk. " * 40)
    guest = "Thanks. " + ("A long answer about robots and China. " * 40)
    trans = [
        {"start": 0.0, "end": 60.0, "text": host},
        {"start": 60.0, "end": 390.0, "text": guest},
    ]
    result = _diarized(diar, trans)
    assert result["diarization_speech_seconds"] == 400.0

    out = episode_processor._maybe_speech_coverage_failover(
        result, "/tmp/a.wav", _cfg(), _job(), "/out", None, 1200.0
    )
    assert "speech_coverage_failover" not in out  # ~0.975 speech coverage -> no re-transcription


def test_long_episode_speech_drop_failovers() -> None:
    # The Ezra Klein cliff: the diarizer heard a speaker talking for 1000s, but turbo only
    # transcribed 400s of it -> 0.40 speech coverage -> re-transcribe on the failover model.
    diar = [
        DiarizationSegment(0.0, 60.0, "SPEAKER_00"),
        DiarizationSegment(60.0, 1000.0, "SPEAKER_01"),
    ]
    host = "Welcome. I'm the host. " + ("Intro. " * 20)
    guest = "Hi. " + ("Answer. " * 20)
    trans = [
        {"start": 0.0, "end": 60.0, "text": host},
        {"start": 60.0, "end": 400.0, "text": guest},
    ]
    result = _diarized(diar, trans)
    assert result["diarization_speech_seconds"] == 1000.0

    fo_enriched = {
        "segments": [{"start": 0.0, "end": 980.0, "text": "recovered"}],
        "text": "recovered",
        "diarization_speech_seconds": 1000.0,
        "model_used": None,
    }
    with (
        patch(f"{_PIPELINE}.apply_diarization_to_result", return_value=fo_enriched),
        patch.object(
            episode_processor,
            "_transcribe_with_segments_maybe_chunked",
            return_value=(dict(fo_enriched), 1.0),
        ),
        patch(
            "podcast_scraper.transcription.factory.create_transcription_provider",
            return_value=MagicMock(),
        ),
    ):
        out = episode_processor._maybe_speech_coverage_failover(
            result, "/tmp/a.wav", _cfg(), _job(), "/out", None, 1200.0
        )
    assert out["speech_coverage_failover"]["primary_speech_coverage"] == 0.4
    assert "large-v3" in out["model_used"]


def test_speech_gate_failover_routes_to_moss_provider() -> None:
    """The speech gate's failover must honor ``transcription_coverage_failover_provider='moss'`` —
    re-transcribe on the MOSS service (transcription_provider='moss' + moss_model), NOT send the MOSS
    model id to the speaches whisper endpoint. Parity with the ADR-123 raw-gate factory routing so
    switching profiles from the raw gate to the speech gate does not silently break MOSS failover.
    """
    cfg = podcast_config.Config(
        rss="https://example.com/feed.xml",
        transcription_provider="whisper",
        diarize=True,
        hf_token="hf-test",
        dgx_whisper_model="turbo",
        transcription_speech_coverage_min=0.85,
        transcription_coverage_failover_model="OpenMOSS-Team/MOSS-Transcribe-Diarize",
        transcription_coverage_failover_provider="moss",
    )
    # 400s transcribed of 1000s speech -> 0.40 < 0.85 -> failover fires.
    diar = [DiarizationSegment(0.0, 1000.0, "SPEAKER_01")]
    trans = [{"start": 0.0, "end": 400.0, "text": "short " * 20}]
    result = _diarized(diar, trans, cfg)
    assert result["diarization_speech_seconds"] == 1000.0

    fo_enriched = {
        "segments": [{"start": 0.0, "end": 980.0, "text": "recovered"}],
        "text": "recovered",
        "diarization_speech_seconds": 1000.0,
        "model_used": None,
    }
    captured: dict = {}

    def _capture(fo_cfg):
        captured["provider"] = fo_cfg.transcription_provider
        captured["moss_model"] = fo_cfg.moss_model
        captured["dgx_whisper_model"] = fo_cfg.dgx_whisper_model
        return MagicMock()

    with (
        patch(f"{_PIPELINE}.apply_diarization_to_result", return_value=fo_enriched),
        patch.object(
            episode_processor,
            "_transcribe_with_segments_maybe_chunked",
            return_value=(dict(fo_enriched), 1.0),
        ),
        patch(
            "podcast_scraper.transcription.factory.create_transcription_provider",
            side_effect=_capture,
        ),
    ):
        episode_processor._maybe_speech_coverage_failover(
            result, "/tmp/a.wav", cfg, _job(), "/out", None, 1200.0
        )
    # Routed to the MOSS provider with the MOSS model — never to the whisper endpoint.
    assert captured["provider"] == "moss"
    assert captured["moss_model"] == "OpenMOSS-Team/MOSS-Transcribe-Diarize"
    # dgx_whisper_model must NOT have been overwritten with the MOSS id (that would hit speaches).
    assert captured["dgx_whisper_model"] == "turbo"


def test_asr_provenance_records_speech_audio_ratio_without_gate(tmp_path) -> None:
    """speech_audio_ratio is a METRIC, not a gate: it is written to .asr.json for EVERY episode,
    including when no coverage gate ran (no speech_coverage, no failover). Guards the retirement of
    the raw ADR-123 gate into an observability signal."""
    import json

    cfg = _cfg()
    (tmp_path / "t.txt").write_text("x", encoding="utf-8")
    result = {"speech_audio_ratio": 0.713, "model_used": "turbo"}  # no gate fields at all
    episode_processor._save_asr_provenance_file(result, cfg, "t.txt", str(tmp_path))
    prov = json.loads((tmp_path / "t.asr.json").read_text(encoding="utf-8"))
    assert prov["speech_audio_ratio"] == 0.713
    assert prov["speech_coverage"] is None  # gate did not run
    assert prov["failed_over"] is False
