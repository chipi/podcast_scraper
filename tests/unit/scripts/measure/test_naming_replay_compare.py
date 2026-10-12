"""scripts/measure/naming_replay_compare.py: coverage and changed voices between two replays."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "measure" / "naming_replay_compare.py"
_spec = importlib.util.spec_from_file_location("naming_replay_compare", _SCRIPT)
assert _spec and _spec.loader
nrc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nrc)


def _rec(ep: str, roster: dict) -> dict:
    return {"meta_path": f"/x/{ep}.metadata.json", "feed": "F", "episode": ep, "roster": roster}


def _write(path: Path, recs: list) -> Path:
    lines = [json.dumps(r) for r in recs] + [json.dumps({"summary": {"episodes": len(recs)}})]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_coverage_moves_and_changed_voices(tmp_path: Path) -> None:
    host = {"name": "Ann Host", "role": "host", "named": True}
    raw = {"name": "SPEAKER_01", "role": "unknown", "named": False}
    guest = {"name": "Bo Guest", "role": "guest", "named": True}
    base = _write(
        tmp_path / "a.jsonl", [_rec("e1", {"S0": host, "S1": raw}), _rec("e2", {"S0": raw})]
    )
    new = _write(
        tmp_path / "b.jsonl", [_rec("e1", {"S0": host, "S1": guest}), _rec("e2", {"S0": raw})]
    )
    a, b = nrc.load(base), nrc.load(new)
    assert len(a) == 2  # the summary line is not an episode
    assert nrc.coverage(a) == {
        "episodes": 2,
        "no_named_voice": 1,
        "named_voices": 1,
        "named_host_episodes": 1,
    }
    assert nrc.coverage(b)["named_voices"] == 2
    assert nrc.episode_moves(a, b) == {"gained_first_name": 0, "lost_every_name": 0}
    changed = list(nrc.changed_voices(a, b))
    assert [(k[1], v, nrc._named(y)) for k, v, _, y in changed] == [("e1", "S1", "Bo Guest")]


def test_an_episode_losing_every_name_is_counted(tmp_path: Path) -> None:
    host = {"name": "Ann Host", "role": "host", "named": True}
    raw = {"name": "SPEAKER_00", "role": "host", "named": False}
    a = nrc.load(_write(tmp_path / "a.jsonl", [_rec("e1", {"S0": host})]))
    b = nrc.load(_write(tmp_path / "b.jsonl", [_rec("e1", {"S0": raw})]))
    assert nrc.episode_moves(a, b) == {"gained_first_name": 0, "lost_every_name": 1}
