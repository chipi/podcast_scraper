"""scripts/measure/naming_gold_cases.py: blind case shape, turns, and disjoint selection."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "measure" / "naming_gold_cases.py"
_spec = importlib.util.spec_from_file_location("naming_gold_cases", _SCRIPT)
assert _spec and _spec.loader
gc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gc)


def _episode(corpus: Path, feed: str, eid: str, run: str, segs: list) -> Path:
    root = corpus / "feeds" / feed / run
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    (root / "transcripts").mkdir(exist_ok=True)
    (root / "transcripts" / f"{eid}.segments.json").write_text(json.dumps(segs))
    meta = root / "metadata" / f"{eid}.metadata.json"
    meta.write_text(
        json.dumps(
            {
                "feed": {"title": "Show", "description": "d" * 2000, "authors": ["Pub"]},
                "episode": {"title": "Ep", "description": "e" * 3000, "episode_id": eid},
                "content": {"transcript_file_path": f"transcripts/{eid}.txt"},
            }
        )
    )
    return meta


def _seg(voice: str, start: float, end: float, text: str, label: str = "") -> dict:
    return {
        "speaker": voice,
        "speaker_label": label or voice,
        "start": start,
        "end": end,
        "text": text,
    }


def test_a_case_is_blind_and_carries_each_voices_own_words(tmp_path: Path) -> None:
    segs = [
        _seg("SPEAKER_00", 0.6, 10.0, " Welcome, I'm Ann Host.", "Ann Host"),
        _seg("SPEAKER_00", 10.0, 20.0, " Today, Bo.", "Ann Host"),
        _seg("SPEAKER_01", 20.4, 50.0, " Thanks " + "x" * 300, "Bo Guest"),
    ]
    meta = _episode(tmp_path, "f1", "e1", "run_1", segs)
    case = gc.build_case(meta, tmp_path, "w001", "validation")
    assert case is not None
    assert "Bo Guest" not in json.dumps(case) and "speaker_label" not in json.dumps(case)
    assert [v["voice"] for v in case["voices"]] == ["SPEAKER_01", "SPEAKER_00"]
    host = case["voices"][1]
    assert (host["first_s"], host["last_s"]) == (1, 20)
    assert host["opening"] == " Welcome, I'm Ann Host.  Today, Bo."
    assert [t["start"] for t in case["first_turns"]] == [1, 20]
    assert case["first_turns"][0]["text"] == " Welcome, I'm Ann Host."
    assert len(case["first_turns"][1]["text"]) == gc.TURN_CHARS
    assert len(case["feed"]["description"]) == gc.FEED_DESCRIPTION_CHARS
    assert len(case["episode"]["description"]) == gc.EPISODE_DESCRIPTION_CHARS
    assert case["duration_s"] == 50


def test_one_voice_is_not_a_case(tmp_path: Path) -> None:
    meta = _episode(tmp_path, "f1", "e1", "run_1", [_seg("SPEAKER_00", 0, 9, " Solo.")])
    assert gc.build_case(meta, tmp_path, "w001", "validation") is None


def test_an_episode_in_an_existing_set_is_excluded_even_after_a_reprocess(tmp_path: Path) -> None:
    two = [_seg("SPEAKER_00", 0, 5, " A."), _seg("SPEAKER_01", 5, 9, " B.")]
    old = _episode(tmp_path, "f1", "e1", "run_old", two)
    new = _episode(tmp_path, "f1", "e1", "run_new", two)
    other = _episode(tmp_path, "f1", "e2", "run_new", two)
    held = tmp_path / "dev" / "cases"
    held.mkdir(parents=True)
    (held / "c001.json").write_text(json.dumps({"meta_relpath": str(old.relative_to(tmp_path))}))
    exclude = gc.excluded_episodes([held], tmp_path)
    picked = gc.select([new, other], tmp_path, 5, 1, exclude, "en")
    assert picked == [other]
