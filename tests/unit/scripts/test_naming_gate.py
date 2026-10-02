"""scripts/measure/naming_gate.py — scoring speaker naming against labels, old vs new.

One test per outcome of ``score_voice`` and one end-to-end run over a synthetic labelled episode
(never-commit-real-episodes): identical code scores identically; a variant that drops a labelled
host's name shows up as a REGRESSION.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

pytestmark = pytest.mark.unit

_DIR = Path(__file__).resolve().parents[3] / "scripts" / "measure"
sys.path.insert(0, str(_DIR))
_spec = importlib.util.spec_from_file_location("naming_gate", _DIR / "naming_gate.py")
assert _spec and _spec.loader
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)

import podcast_scraper.providers.ml.diarization.roster as _installed_roster  # noqa: E402

ROSTER_SRC = Path(str(_installed_roster.__file__))


def _lab(role: str, name: Any = None, confidence: str = "high") -> Dict[str, Any]:
    return {"voice": "SPEAKER_00", "role": role, "name": name, "confidence": confidence}


# --- one outcome per test -------------------------------------------------------------------


def test_the_same_person_is_correct() -> None:
    assert gate.score_voice(_lab("host", "Tobias Wren"), "Tobias Wren", "host") == "correct_name"


def test_a_spelling_variant_of_the_same_person_is_correct() -> None:
    assert gate.score_voice(_lab("host", "Tracy Alloway"), "Tracey Alloway", "host") == (
        "correct_name"
    )


def test_a_different_person_is_wrong() -> None:
    assert gate.score_voice(_lab("guest", "Ann Applebaum"), "Jake Sullivan", "guest") == (
        "wrong_name"
    )


def test_a_labelled_name_published_as_nobody_is_missing() -> None:
    assert gate.score_voice(_lab("host", "David Perell"), None, "unknown") == "missing_name"


def test_a_name_the_text_does_not_support_is_spurious() -> None:
    assert gate.score_voice(_lab("guest", None), "Two Carnegie Mellon", "host") == "spurious_name"


def test_an_unnamed_participant_published_unnamed_is_correct() -> None:
    assert gate.score_voice(_lab("guest", None), None, "guest") == "correct_unnamed"


@pytest.mark.parametrize("role", ["ad", "promo", "clip"])
def test_a_non_participant_published_as_a_speaker_is_wrong(role: str) -> None:
    # The Curiosity Shop promo: "I'm Phoebe Judge" — the name is true, the voice is not a guest.
    assert gate.score_voice(_lab(role, "Phoebe Judge"), "Phoebe Judge", "guest") == (
        "non_participant"
    )


@pytest.mark.parametrize("role", ["ad", "promo", "clip"])
def test_a_non_participant_left_unnamed_is_correct(role: str) -> None:
    assert gate.score_voice(_lab(role, "Phoebe Judge"), None, "unknown") == "correct_unnamed"


def test_host_and_guest_swapped_is_a_role_error() -> None:
    assert gate.score_voice(_lab("host", "Katie Martin"), "Katie Martin", "guest") == "role_error"


def test_unknown_and_low_confidence_labels_are_not_scored() -> None:
    assert gate.score_voice(_lab("unknown"), "Anyone", "guest") is None
    assert gate.score_voice(_lab("host", "Tobias Wren", "low"), None, "host") is None


# --- end to end ------------------------------------------------------------------------------


def _labelled(root: Path) -> tuple[Path, Path, Path]:
    run = root / "corpus" / "feeds" / "river" / "run_1"
    (run / "metadata").mkdir(parents=True)
    (run / "transcripts").mkdir()
    (run / "metadata" / "1.metadata.json").write_text(
        json.dumps(
            {
                "feed": {"title": "River Trade Weekly"},
                "episode": {"title": "The ports", "episode_id": "ep-1"},
                "content": {"transcript_file_path": "transcripts/1.txt"},
            }
        )
    )
    turns = [
        ("SPEAKER_00", "Welcome to River Trade Weekly, I'm Tobias Wren.", 30.0),
        ("SPEAKER_01", "The ports moved north because the river silted up.", 400.0),
        ("SPEAKER_00", "And the merchants followed?", 200.0),
        ("SPEAKER_01", "Most of them, within a generation.", 400.0),
    ]
    segs, t = [], 0.0
    for v, text, dur in turns:
        segs.append({"speaker": v, "start": t, "end": t + dur, "text": text})
        t += dur
    (run / "transcripts" / "1.segments.json").write_text(json.dumps(segs))
    (run / "transcripts" / "1.speakers.diagnostics.json").write_text(
        json.dumps({"tried": {"known_hosts": ["Tobias Wren"]}, "voices": []})
    )
    cases, labels = root / "cases", root / "labels"
    cases.mkdir()
    labels.mkdir()
    (cases / "c001.json").write_text(
        json.dumps(
            {
                "case_id": "c001",
                "meta_relpath": "feeds/river/run_1/metadata/1.metadata.json",
                "feed": {"title": "River Trade Weekly"},
                "episode": {"title": "The ports"},
            }
        )
    )
    (labels / "c001.json").write_text(
        json.dumps(
            {
                "case_id": "c001",
                "voices": [
                    {
                        "voice": "SPEAKER_00",
                        "role": "host",
                        "name": "Tobias Wren",
                        "confidence": "high",
                    },
                    {"voice": "SPEAKER_01", "role": "guest", "name": None, "confidence": "high"},
                ],
            }
        )
    )
    return root / "corpus", cases, labels


def test_identical_code_scores_identically(tmp_path: Path) -> None:
    corpus, cases, labels = _labelled(tmp_path)
    variant = gate.R.load_variant({})
    report = gate.run(corpus, gate.load_labelled(cases, labels), variant, variant)
    assert report["old"] == report["new"]
    assert report["hosts_old"] == {"correct_name": 1}
    assert report["regressions"] == [] and report["fixes"] == []


def test_a_variant_that_drops_the_labelled_host_is_a_regression(tmp_path: Path) -> None:
    corpus, cases, labels = _labelled(tmp_path)
    nameless = tmp_path / "roster_nameless.py"
    nameless.write_text(
        ROSTER_SRC.read_text(encoding="utf-8") + "\n_orig = resolve_speaker_roster\n"
        "def resolve_speaker_roster(*a, **k):\n"
        "    r = _orig(*a, **k)\n"
        "    for v, role in list(r.by_voice.items()):\n"
        "        r.by_voice[v] = replace(role, name=v, named=False)\n"
        "    return r\n"
    )
    report = gate.run(
        corpus,
        gate.load_labelled(cases, labels),
        gate.R.load_variant({}),
        gate.R.load_variant({"roster": nameless}),
    )
    assert report["hosts_new"] == {"missing_name": 1}
    assert [(r["voice"], r["old"]["score"], r["new"]["score"]) for r in report["regressions"]] == [
        ("SPEAKER_00", "correct_name", "missing_name")
    ]


def _nameless_roster(tmp_path: Path) -> Path:
    path = tmp_path / "roster_nameless.py"
    path.write_text(
        ROSTER_SRC.read_text(encoding="utf-8") + "\n_orig = resolve_speaker_roster\n"
        "def resolve_speaker_roster(*a, **k):\n"
        "    r = _orig(*a, **k)\n"
        "    for v, role in list(r.by_voice.items()):\n"
        "        r.by_voice[v] = replace(role, name=v, named=False)\n"
        "    return r\n"
    )
    return path


def test_a_wrong_name_that_becomes_no_name_is_better(tmp_path: Path) -> None:
    corpus, cases, labels = _labelled(tmp_path)
    lab = json.loads((labels / "c001.json").read_text())
    lab["voices"][0]["name"] = "Someone Else"  # the code publishes Tobias Wren: a WRONG name
    (labels / "c001.json").write_text(json.dumps(lab))
    report = gate.run(
        corpus,
        gate.load_labelled(cases, labels),
        gate.R.load_variant({}),
        gate.R.load_variant({"roster": _nameless_roster(tmp_path)}),
    )
    assert [(r["old"]["score"], r["new"]["score"]) for r in report["better"]] == [
        ("wrong_name", "missing_name")
    ]
    assert report["worse"] == [] and report["regressions"] == [] and report["fixes"] == []


def test_a_validation_set_with_v_prefixed_files_is_read(tmp_path: Path) -> None:
    _corpus, cases, labels = _labelled(tmp_path)
    for d in (cases, labels):
        (d / "c001.json").rename(d / "v001.json")
    assert [c["case_id"] for c, _l in gate.load_labelled(cases, labels)] == ["c001"]


def test_severity_orders_wrong_above_missing_above_correct() -> None:
    s = gate.SEVERITY
    assert s["wrong_name"] > s["missing_name"] > s["correct_name"] == s["correct_unnamed"]
    assert s["non_participant"] == s["spurious_name"] == s["wrong_name"]
