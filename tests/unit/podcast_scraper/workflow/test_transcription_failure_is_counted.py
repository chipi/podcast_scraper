"""A transcription that fails is recorded as a failed episode, not left "ok" from queueing.

Camera seduta 717 (2026-10-12): Whisper returned 417 words for 66 minutes of speech, the reprocess
policy surfaced its fuse-open error, nothing was written, and the feed reported
``episodes=3 ok=3 failed=0``. Behaviour through ``transcribe_media_to_text``.
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from podcast_scraper import config as config_mod
from podcast_scraper.providers.guardrails import GuardrailViolation
from podcast_scraper.providers.resilience import ResilienceFuseOpenError


def _run(monkeypatch, exc: Exception) -> Dict[str, Any]:
    from podcast_scraper.workflow import episode_processor as ep

    seen: Dict[str, Any] = {"recorded": [], "saved": False}
    monkeypatch.setattr(ep, "_bind_episode_correlation", lambda job, cfg: None)
    monkeypatch.setattr(ep, "_maybe_dispatch_reprocess_stage", lambda *a, **k: None)
    monkeypatch.setattr(ep, "_check_and_reuse_existing_transcript", lambda *a, **k: None)
    monkeypatch.setattr(ep, "_check_transcript_cache", lambda *a, **k: None)
    monkeypatch.setattr(ep, "_preprocess_audio_if_needed", lambda job, cfg, m, pm: m)
    monkeypatch.setattr(ep, "_cleanup_temp_media", lambda *a, **k: None)
    monkeypatch.setattr(ep, "_append_transcription_incident", lambda *a, **k: None)
    monkeypatch.setattr(ep, "_capture_stage_exception", lambda *a, **k: None)

    def _raise(*a: Any, **k: Any) -> Any:
        raise exc

    monkeypatch.setattr(ep, "_transcribe_with_segments_maybe_chunked", _raise)

    def _saved(*a: Any, **k: Any) -> str:
        seen["saved"] = True
        return "transcripts/x.txt"

    monkeypatch.setattr(ep, "_save_transcript_file", _saved)
    monkeypatch.setattr(
        ep,
        "_record_unresolved_transcript",
        lambda job, cfg, pm, stage, *, error_type="", detail=None: seen["recorded"].append(
            {"stage": stage, "error_type": error_type, "detail": detail}
        ),
    )
    cfg = config_mod.Config(rss="https://example.com/f.xml").model_copy(
        update={"feed_declared_language": "it"}
    )
    job = MagicMock()
    job.idx = 2
    job.temp_media = None
    seen["return"] = ep.transcribe_media_to_text(
        job, cfg, None, None, "/tmp/out", MagicMock(), None
    )
    return seen


@pytest.mark.parametrize(
    "exc",
    [
        ResilienceFuseOpenError("dgx-whisper: endpoint did not recover within 900s"),
        GuardrailViolation("whisper", "length_floor_violated", "word_count=417 floor=4965"),
    ],
    ids=["fuse_open", "guardrail"],
)
def test_a_failed_transcription_is_recorded_as_failed(monkeypatch, exc: Exception) -> None:
    seen = _run(monkeypatch, exc)
    assert seen["return"][0] is False and seen["return"][1] is None
    assert seen["saved"] is False
    assert [r["error_type"] for r in seen["recorded"]] == ["TranscriptionFailed"]
    assert seen["recorded"][0]["stage"] == "transcription"
