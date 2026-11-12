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
    from podcast_scraper.providers.ml.diarization.base import DiarizationResult
    from podcast_scraper.providers.ml.diarization.pipeline import apply_diarization_to_result

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
