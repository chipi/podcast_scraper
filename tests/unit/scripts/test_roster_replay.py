"""scripts/measure/roster_replay.py — the old-vs-new roster replay every naming change is decided
with. Synthetic one-episode corpus (never-commit-real-episodes)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "measure" / "roster_replay.py"
_spec = importlib.util.spec_from_file_location("roster_replay", _SCRIPT)
assert _spec and _spec.loader
rr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rr)

import podcast_scraper.providers.ml.diarization.roster as _installed_roster  # noqa: E402

ROSTER_SRC = Path(str(_installed_roster.__file__))


def _corpus(root: Path) -> Path:
    run = root / "feeds" / "river" / "run_1"
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
    return root


def _run(corpus: Path, old: dict, new: dict) -> list:
    return list(rr.compare(rr.corpus_episodes(corpus), rr.load_variant(old), rr.load_variant(new)))


def test_identical_code_reports_no_change(tmp_path: Path) -> None:
    out = _run(_corpus(tmp_path), {}, {})
    assert out == [{"summary": {"episodes": 1}}]


def test_a_changed_variant_is_reported_voice_by_voice(tmp_path: Path) -> None:
    # The new side is the real roster plus a rule that refuses every name.
    changed = tmp_path / "roster_new.py"
    changed.write_text(
        ROSTER_SRC.read_text(encoding="utf-8") + "\n_orig = resolve_speaker_roster\n"
        "def resolve_speaker_roster(*a, **k):\n"
        "    r = _orig(*a, **k)\n"
        "    for v, role in list(r.by_voice.items()):\n"
        "        r.by_voice[v] = replace(role, name=v, named=False)\n"
        "    return r\n"
    )
    out = _run(_corpus(tmp_path), {}, {"roster": changed})
    lost = [r for r in out if r.get("kind") == "lost"]
    assert [(r["voice"], r["old"], r["new"]) for r in lost] == [("SPEAKER_00", "Tobias Wren", None)]
    assert out[-1]["summary"]["lost"] == 1


def test_building_a_variant_leaves_the_installed_modules_in_place(tmp_path: Path) -> None:
    before = {name: sys.modules.get(name) for name in rr.MODULES.values()}
    rr.load_variant({"roster": ROSTER_SRC})
    assert {name: sys.modules.get(name) for name in rr.MODULES.values()} == before
