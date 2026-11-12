"""The corpus scoreboard comes from installed packages; without one, a scored corpus is refused.

The change gate and the outcome hook classify a migration by whether gating numbers moved. Measured
with no scoreboard installed every number is zero before and after, so a corpus that holds files a
scoreboard reads must be refused, never reported as unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from podcast_scraper.extensions import CorpusScoreboard, Extension, use_extensions
from podcast_scraper.quality import change_gate, scoreboard

pytestmark = [pytest.mark.unit]


def _episode(root: Path, name: str, *, gi: bool) -> None:
    meta = root / "feeds" / "f" / "run_1" / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    doc = {"feed": {"feed_id": "f"}, "episode": {"episode_id": name, "language": "en"}}
    (meta / f"{name}.metadata.json").write_text(json.dumps(doc), encoding="utf-8")
    (meta / f"{name}.kg.json").write_text(json.dumps({"nodes": [], "edges": []}), encoding="utf-8")
    if gi:
        (meta / f"{name}.gi.json").write_text(json.dumps({"nodes": []}), encoding="utf-8")


def _board(counts: Dict[str, int], gating=("X_hidden",)) -> Extension:
    def score(meta: Path) -> Optional[Dict[str, Any]]:
        return {"language": "en", "counts": dict(counts)}

    return Extension(
        name="fake",
        corpus_scoreboard=CorpusScoreboard(
            score_episode=score, gating_metrics=gating, keys=("X_hidden", "X_zero")
        ),
    )


def test_without_a_scoreboard_there_are_no_gating_metrics_and_no_rows(tmp_path: Path) -> None:
    _episode(tmp_path, "e1", gi=False)
    with use_extensions([]):
        assert scoreboard.gating_metrics() == []
        assert scoreboard.score_corpus(tmp_path) == {"total": {}, "per_language": {}}


def test_a_corpus_with_scored_files_is_refused_without_a_scoreboard(tmp_path: Path) -> None:
    _episode(tmp_path, "e1", gi=True)
    with use_extensions([]), pytest.raises(scoreboard.ScoreboardMissing, match="e1.gi.json"):
        scoreboard.require_scoreboard(tmp_path)


def test_a_public_only_corpus_needs_no_scoreboard(tmp_path: Path) -> None:
    _episode(tmp_path, "e1", gi=False)
    with use_extensions([]):
        scoreboard.require_scoreboard(tmp_path)


def test_an_installed_scoreboard_is_used_and_its_keys_are_written_with_zeros(
    tmp_path: Path,
) -> None:
    _episode(tmp_path, "e1", gi=True)
    _episode(tmp_path, "e2", gi=True)
    with use_extensions([_board({"X_hidden": 2})]):
        scoreboard.require_scoreboard(tmp_path)
        assert scoreboard.gating_metrics() == ["X_hidden"]
        total = scoreboard.score_corpus(tmp_path)["total"]
    assert total == {"X_hidden": 4, "X_zero": 0}


def test_a_moved_gating_number_makes_the_migration_full() -> None:
    with use_extensions([_board({})]):
        metrics = scoreboard.gating_metrics()
    assert change_gate.gating_moves({"X_hidden": 1}, {"X_hidden": 3, "other": 9}, metrics) == {
        "X_hidden": 2
    }
    assert change_gate.gating_moves({"X_hidden": 1}, {"X_hidden": 1}, metrics) == {}
