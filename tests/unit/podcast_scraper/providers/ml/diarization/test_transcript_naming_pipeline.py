"""Transcript naming through ``apply_diarization_to_result``: off, shadow and publish.

Shadow must change nothing the pipeline publishes except the new sidecar key; publish names the
roster from the model's answer. The model is a canned answer (no provider is built). All fixtures
are synthetic (never-commit-real-episodes).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import pytest

from podcast_scraper.providers.ml.diarization import pipeline as P, transcript_naming as tn
from podcast_scraper.providers.ml.diarization.base import DiarizationResult, DiarizationSegment

pytestmark = pytest.mark.unit

Mode = Literal["off", "shadow", "publish"]

WORDS = " ".join(["indeed"] * 45)
# The ladder names SPEAKER_00 the host from his self-introduction and SPEAKER_01 the guest; the
# model swaps them, so publish visibly differs from the ladder.
SWAPPED = (
    '{"voices": {"SPEAKER_00": {"name": "Maria Lindqvist", "role": "guest", "confidence": "low",'
    ' "evidence": ""}, "SPEAKER_01": {"name": "Tobias Wren", "role": "host",'
    ' "confidence": "low", "evidence": ""}}}'
)


def _run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: Mode, answer: Optional[str] = SWAPPED
) -> Dict[str, Any]:
    from podcast_scraper.config import Config

    monkeypatch.setattr(P, "_resolve_voices_via_llm", lambda *a, **k: ({}, {}))
    calls = []

    def fake_ask(cfg: Any, aligned: Any, title: Any, description: Any) -> Any:
        calls.append(mode)
        if answer is None:
            return None
        return tn.ask(
            aligned,
            lambda prompt: answer,
            feed_title="The Wren Hour",
            feed_description=None,
            episode_title=title,
            episode_description=description,
        )

    monkeypatch.setattr(P, "_ask_transcript_naming", fake_ask)
    P._copresence_cache.clear()
    diar = DiarizationResult(
        segments=[
            DiarizationSegment(0, 30, "SPEAKER_00"),
            DiarizationSegment(30, 90, "SPEAKER_01"),
        ],
        num_speakers=2,
    )
    result = {
        "text": "Welcome. I'm Tobias Wren. Thanks for having me.",
        "segments": [
            {"start": 0, "end": 30, "text": "Welcome to the show. I'm Tobias Wren. " + WORDS},
            {"start": 30, "end": 90, "text": "Thanks for having me. " + WORDS},
        ],
    }
    cfg = Config(output_dir=str(tmp_path), speaker_resolution_llm=False, transcript_naming=mode)
    out: Dict[str, Any] = P.apply_diarization_to_result(
        copy.deepcopy(result),
        "",
        cfg,
        ["Maria Lindqvist"],
        precomputed_diarization=diar,
        feed_hosts=["Tobias Wren"],
        episode_title="Ep 1",
    )
    out["_asked"] = list(calls)
    return out


def _published(out: Dict[str, Any]) -> str:
    """Everything the stage publishes, with the new sidecar key taken out."""
    view = {k: v for k, v in out.items() if k != "_asked"}
    view["speaker_diagnostics"] = {
        k: v for k, v in view["speaker_diagnostics"].items() if k != tn.SIDECAR_KEY
    }
    return json.dumps(view, sort_keys=True, default=str)


def _names(out: Dict[str, Any]) -> Dict[str, Any]:
    return {v["voice"]: v["resolved_name"] for v in out["speaker_diagnostics"]["voices"]}


def test_off_asks_nothing_and_writes_no_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    out = _run(monkeypatch, tmp_path, "off")
    assert out["_asked"] == []
    assert tn.SIDECAR_KEY not in out["speaker_diagnostics"]


def test_shadow_publishes_exactly_what_off_publishes_and_records_the_model_beside_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    off = _run(monkeypatch, tmp_path, "off")
    shadow = _run(monkeypatch, tmp_path, "shadow")
    assert _published(shadow) == _published(off)
    side = shadow["speaker_diagnostics"][tn.SIDECAR_KEY]
    assert side["mode"] == "shadow" and side["ok"] is True
    assert side["voices"]["SPEAKER_00"]["ladder"]["name"] == "Tobias Wren"
    assert side["voices"]["SPEAKER_00"]["model_result"]["name"] == "Maria Lindqvist"
    assert side["voices"]["SPEAKER_00"]["outcome"] == "changes"
    assert side["outcomes"]["changes"] >= 1


def test_publish_names_the_roster_from_the_answer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    off = _run(monkeypatch, tmp_path, "off")
    published = _run(monkeypatch, tmp_path, "publish")
    assert _names(off)["SPEAKER_00"] == "Tobias Wren"
    assert _names(published)["SPEAKER_00"] == "Maria Lindqvist"
    assert _names(published)["SPEAKER_01"] == "Tobias Wren"
    side = published["speaker_diagnostics"][tn.SIDECAR_KEY]
    assert side["mode"] == "publish"
    # The sidecar compares against what the LADDER would have published, not against itself.
    assert side["voices"]["SPEAKER_00"]["ladder"]["name"] == "Tobias Wren"
    assert side["voices"]["SPEAKER_00"]["outcome"] == "changes"


def test_a_failed_call_in_shadow_or_publish_leaves_the_ladder_and_says_so(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    off = _run(monkeypatch, tmp_path, "off")
    modes: Tuple[Mode, ...] = ("shadow", "publish")
    for mode in modes:
        out = _run(monkeypatch, tmp_path, mode, answer="not json")
        assert _published(out) == _published(off)
        side = out["speaker_diagnostics"][tn.SIDECAR_KEY]
        assert side["ok"] is False and side["report"]["error"] == "unreadable answer"


def test_a_profile_without_an_llm_writes_no_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    off = _run(monkeypatch, tmp_path, "off")
    out = _run(monkeypatch, tmp_path, "shadow", answer=None)
    assert _published(out) == _published(off)
    assert tn.SIDECAR_KEY not in out["speaker_diagnostics"]
