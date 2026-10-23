"""One person split into two diarized voices, merged by fingerprint (``merge_split_voices``).

Synthetic fingerprints only (never-commit-real-episodes).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from podcast_scraper.providers.ml.diarization import pipeline as P, split_voices as sv
from podcast_scraper.providers.ml.diarization.base import DiarizationResult, DiarizationSegment

pytestmark = pytest.mark.unit

HOST = [1.0, 0.0, 0.0]
HOST_AGAIN = [0.95, 0.31, 0.0]  # cos to HOST ~0.95
GUEST = [0.0, 1.0, 0.0]


def _diar(spec: Dict[str, Any], vecs: Optional[Dict[str, List[float]]] = None) -> DiarizationResult:
    segs, t = [], 0.0
    for voice, secs in spec.items():
        segs.append(DiarizationSegment(t, t + secs, voice))
        t += secs
    return DiarizationResult(
        segments=segs, num_speakers=len(spec), model_name="m", speaker_embeddings=vecs or {}
    )


def test_two_voices_of_one_person_become_the_one_that_talks_more() -> None:
    d = _diar(
        {"SPEAKER_00": 300, "SPEAKER_01": 400, "SPEAKER_02": 40},
        {"SPEAKER_00": HOST, "SPEAKER_01": GUEST, "SPEAKER_02": HOST_AGAIN},
    )
    out, report = sv.merge(d, threshold=0.8)
    assert [s.speaker for s in out.segments] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]
    assert out.num_speakers == 2
    assert set(out.speaker_embeddings) == {"SPEAKER_00", "SPEAKER_01"}
    assert report["merged"] == [
        {"voice": "SPEAKER_02", "into": "SPEAKER_00", "similarity": 0.9507, "talk_s": 40.0}
    ]
    # every pair is reported, merged or not, so a threshold can be chosen from the run
    assert len(report["pairs"]) == 3


def test_different_people_are_not_merged() -> None:
    d = _diar({"SPEAKER_00": 300, "SPEAKER_01": 400}, {"SPEAKER_00": HOST, "SPEAKER_01": GUEST})
    out, report = sv.merge(d, threshold=0.8)
    assert out is d and report["merged"] == []


def test_a_chain_of_matches_is_one_person() -> None:
    mid = [0.85, 0.53, 0.0]  # ~0.85 to HOST, ~0.85 to END
    end = [0.44, 0.9, 0.0]  # ~0.44 to HOST
    d = _diar(
        {"SPEAKER_00": 300, "SPEAKER_01": 50, "SPEAKER_02": 60},
        {"SPEAKER_00": HOST, "SPEAKER_01": mid, "SPEAKER_02": end},
    )
    out, report = sv.merge(d, threshold=0.8)
    assert {s.speaker for s in out.segments} == {"SPEAKER_00"}
    assert sorted(m["voice"] for m in report["merged"]) == ["SPEAKER_01", "SPEAKER_02"]


@pytest.mark.parametrize(
    "spec, vecs, why",
    [
        ({"SPEAKER_00": 300, "SPEAKER_01": 40}, {}, "no fingerprints"),
        ({"SPEAKER_00": 300, "SPEAKER_01": 40}, {"SPEAKER_00": HOST}, "one fingerprint"),
        # the second voice is under MIN_TALK_S: its fingerprint is too short to trust
        (
            {"SPEAKER_00": 300, "SPEAKER_01": 5},
            {"SPEAKER_00": HOST, "SPEAKER_01": HOST_AGAIN},
            "too short",
        ),
    ],
)
def test_it_abstains_without_two_trustworthy_fingerprints(spec: Any, vecs: Any, why: str) -> None:
    d = _diar(spec, vecs)
    out, report = sv.merge(d)
    assert out is d, why
    assert report["abstained"] == "fewer than two fingerprinted voices"


def _stage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, on: bool) -> Dict[str, Any]:
    from podcast_scraper.config import Config

    monkeypatch.setattr(P, "_resolve_voices_via_llm", lambda *a, **k: ({}, {}))
    P._copresence_cache.clear()
    words = " ".join(["indeed"] * 60)
    out: Dict[str, Any] = P.apply_diarization_to_result(
        {
            "text": "x",
            "segments": [
                {"start": 0, "end": 300, "text": "Welcome, I'm Tobias Wren. " + words},
                {"start": 300, "end": 700, "text": "Thanks for having me. " + words},
                {"start": 700, "end": 740, "text": "That's all, I'm Tobias Wren. " + words},
            ],
        },
        "",
        Config(output_dir=str(tmp_path), speaker_resolution_llm=False, merge_split_voices=on),
        [],
        precomputed_diarization=_diar(
            {"SPEAKER_00": 300, "SPEAKER_01": 400, "SPEAKER_02": 40},
            {"SPEAKER_00": HOST, "SPEAKER_01": GUEST, "SPEAKER_02": HOST_AGAIN},
        ),
        episode_title="Ep 1",
    )
    return out


def test_the_stage_merges_before_naming_only_when_on(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    off = _stage(monkeypatch, tmp_path, False)
    on = _stage(monkeypatch, tmp_path, True)
    assert "split_voices" not in off["speaker_diagnostics"]
    assert {s["speaker"] for s in off["segments"]} == {"SPEAKER_00", "SPEAKER_01", "SPEAKER_02"}
    assert {s["speaker"] for s in on["segments"]} == {"SPEAKER_00", "SPEAKER_01"}
    assert on["speaker_diagnostics"]["split_voices"]["merged"][0]["voice"] == "SPEAKER_02"
    assert {v["voice"] for v in on["speaker_diagnostics"]["voices"]} == {"SPEAKER_00", "SPEAKER_01"}


def test_off_by_default() -> None:
    from podcast_scraper import config

    assert config.Config.model_fields["merge_split_voices"].default is False
