"""Unit tests for `scripts/eval/score/cil_baseline_v2.py` (#903 audit follow-up)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[5]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

_PATH = ROOT / "scripts" / "eval" / "score" / "cil_baseline_v2.py"
_spec = importlib.util.spec_from_file_location("cil_baseline_v2_under_test", _PATH)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

pytestmark = [pytest.mark.unit]


def _kg(*persons: tuple[str, str]) -> dict:
    return {
        "nodes": [
            {"id": pid, "type": "Entity", "properties": {"label": label}} for pid, label in persons
        ],
        "edges": [],
    }


def _gi() -> dict:
    return {"nodes": [], "edges": []}


def test_two_marcos_stay_distinct() -> None:
    """Different `person:` ids must NOT be merged regardless of display-name overlap."""
    gi = {"p03_e01": _gi(), "p05_e02": _gi()}
    kg = {
        "p03_e01": _kg(("person:marco", "Marco")),
        "p05_e02": _kg(("person:marco-bianchi", "Marco Bianchi")),
    }
    feeds = {"p03_e01": "feed-p03", "p05_e02": "feed-p05"}
    out = _mod.aggregate_cil(gi, kg, feeds)
    person_ids = {p["id"] for p in out["persons"]}
    assert {"person:marco", "person:marco-bianchi"} <= person_ids


def test_person_multi_ep_aggregates_correctly() -> None:
    gi = {"p03_e01": _gi(), "p03_e02": _gi()}
    kg = {
        "p03_e01": _kg(("person:marco", "Marco")),
        "p03_e02": _kg(("person:marco", "Marco")),
    }
    feeds = {"p03_e01": "feed-p03", "p03_e02": "feed-p03"}
    out = _mod.aggregate_cil(gi, kg, feeds)
    marcos = [p for p in out["persons"] if p["id"] == "person:marco"]
    assert len(marcos) == 1
    assert marcos[0]["episode_count"] == 2
    assert marcos[0]["feed_count"] == 1
    assert out["summary"]["person_bridges_multi_ep"] == 1


def test_topic_cross_feed_counts_correctly() -> None:
    gi = {"p02_e01": _gi(), "p05_e01": _gi()}
    kg = {
        "p02_e01": _kg(("topic:risk-management", "risk management")),
        "p05_e01": _kg(("topic:risk-management", "risk management")),
    }
    feeds = {"p02_e01": "feed-p02", "p05_e01": "feed-p05"}
    out = _mod.aggregate_cil(gi, kg, feeds)
    assert out["summary"]["topic_bridges_cross_feed"] == 1
    assert out["ac_pass"]["topic_bridges_cross_feed"] is True


def test_episode_to_feed_warns_on_missing_prefix(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A small share (<=25%) of unresolved paths warns but does not raise."""
    ds = tmp_path / "ds.json"
    eps = [
        {"episode_id": f"a{i}", "transcript_path": f"data/eval/sources/x/feed-p01/a{i}.txt"}
        for i in range(9)
    ]
    eps.append({"episode_id": "c", "transcript_path": "weird/path/c.txt"})
    ds.write_text(json.dumps({"episodes": eps}), encoding="utf-8")
    feeds = _mod._episode_to_feed(ds)
    assert feeds["a0"] == "feed-p01"
    assert feeds["c"] == "unknown"
    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert "c (transcript_path='weird/path/c.txt')" in captured.err


def test_episode_to_feed_raises_when_majority_unresolved(tmp_path: Path) -> None:
    """A dataset where most paths lack the convention should hard-fail."""
    ds = tmp_path / "ds.json"
    ds.write_text(
        json.dumps(
            {
                "episodes": [
                    {"episode_id": str(i), "transcript_path": f"weird/{i}.txt"} for i in range(8)
                ]
                + [
                    {
                        "episode_id": "ok",
                        "transcript_path": "data/eval/sources/x/feed-p01/ok.txt",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="feed-"):
        _mod._episode_to_feed(ds)


def test_load_predictions_wraps_malformed_line(tmp_path: Path) -> None:
    """A KeyError mid-iteration should surface with file:line context."""
    p = tmp_path / "predictions.jsonl"
    p.write_text(
        json.dumps({"episode_id": "ok", "output": {"gil": {}}})
        + "\n"
        + json.dumps({"episode_id": "broken", "output": {}})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="line 2"):
        _mod._load_predictions(p, "gil")
