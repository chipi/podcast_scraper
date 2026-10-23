"""Transcript naming's shadow answer reaches the speaker-diagnostics file on disk.

The stage's output goes through the episode processor's own writer
(``_save_speaker_diagnostics_file``) and is read back: the ladder's fields are untouched and the
``transcript_naming`` key sits beside them, for any later reader comparing the two namers. The
model is a canned answer; no provider is built. All fixtures are synthetic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Literal

import pytest

from podcast_scraper.config import Config
from podcast_scraper.providers.ml.diarization import pipeline as P, transcript_naming as tn
from podcast_scraper.providers.ml.diarization.base import DiarizationResult, DiarizationSegment
from podcast_scraper.workflow.episode_processor import _save_speaker_diagnostics_file

pytestmark = [pytest.mark.integration, pytest.mark.critical_path]

WORDS = " ".join(["indeed"] * 45)
ANSWER = (
    '{"voices": {"SPEAKER_00": {"name": "Tobias Wren", "role": "host", "confidence": "high",'
    ' "evidence": "[00:00] I\'m Tobias Wren"}, "SPEAKER_01": {"name": "Maria Lindqvist",'
    ' "role": "guest", "confidence": "medium", "evidence": ""}}}'
)


def _diagnostics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: Literal["off", "shadow", "publish"]
) -> Dict[str, Any]:
    monkeypatch.setattr(P, "_resolve_voices_via_llm", lambda *a, **k: ({}, {}))
    monkeypatch.setattr(
        P,
        "_ask_transcript_naming",
        lambda cfg, aligned, title, desc: tn.ask(
            aligned,
            lambda prompt: ANSWER,
            feed_title="The Wren Hour",
            feed_description=None,
            episode_title=title,
            episode_description=desc,
        ),
    )
    P._copresence_cache.clear()
    out = P.apply_diarization_to_result(
        {
            "text": "x",
            "segments": [
                {"start": 0, "end": 30, "text": "Welcome. I'm Tobias Wren. " + WORDS},
                {"start": 30, "end": 90, "text": "Thanks for having me. " + WORDS},
            ],
        },
        "",
        Config(output_dir=str(tmp_path), speaker_resolution_llm=False, transcript_naming=mode),
        ["Maria Lindqvist"],
        precomputed_diarization=DiarizationResult(
            segments=[
                DiarizationSegment(0, 30, "SPEAKER_00"),
                DiarizationSegment(30, 90, "SPEAKER_01"),
            ],
            num_speakers=2,
        ),
        feed_hosts=["Tobias Wren"],
        episode_title="Ep 1",
    )
    rel = "transcripts/0001 - ep.txt"
    (tmp_path / "transcripts").mkdir(parents=True, exist_ok=True)
    _save_speaker_diagnostics_file(out["speaker_diagnostics"], rel, str(tmp_path))
    path = tmp_path / "transcripts" / "0001 - ep.speakers.diagnostics.json"
    loaded: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def test_the_shadow_key_is_written_beside_the_ladders_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    off = _diagnostics(monkeypatch, tmp_path / "off", "off")
    shadow = _diagnostics(monkeypatch, tmp_path / "shadow", "shadow")
    assert tn.SIDECAR_KEY not in off
    assert {k: v for k, v in shadow.items() if k != tn.SIDECAR_KEY} == off
    side = shadow[tn.SIDECAR_KEY]
    assert side["version"] == tn.SIDECAR_VERSION and side["mode"] == "shadow" and side["ok"]
    assert set(side["voices"]) == {v["voice"] for v in off["voices"]}
    for voice, entry in side["voices"].items():
        assert set(entry) >= {"ladder", "model", "model_result", "outcome"}, voice
    assert side["voices"]["SPEAKER_01"]["model"]["confidence"] == "medium"
    assert side["report"]["prompt_version"] == tn.PROMPT_VERSION
    assert sum(side["outcomes"].values()) == len(side["voices"])
