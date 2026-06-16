"""Unit tests for scripts/ops/corpus_snapshot/count_distinct_episodes.py (#877)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]

_SPEC = importlib.util.spec_from_file_location(
    "count_distinct_episodes_under_test",
    ROOT / "scripts" / "ops" / "corpus_snapshot" / "count_distinct_episodes.py",
)
assert _SPEC and _SPEC.loader
_mod = importlib.util.module_from_spec(_SPEC)
sys.modules["count_distinct_episodes_under_test"] = _mod
_SPEC.loader.exec_module(_mod)

pytestmark = [pytest.mark.unit]


def _write_episode(corpus: Path, run_seg: str, episode_id: str) -> None:
    doc = {
        "feed": {"feed_id": "f1", "title": "S"},
        "episode": {"episode_id": episode_id, "title": episode_id, "published_date": "2024-01-01"},
    }
    p = corpus / "feeds" / "pod" / run_seg / "metadata" / f"0001 - {episode_id}.metadata.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc), encoding="utf-8")


def test_count_distinguishes_all_runs_from_latest_only(tmp_path: Path) -> None:
    _write_episode(tmp_path, "run_20260416-000000_old", "e1")
    _write_episode(tmp_path, "run_20260417-000000_new", "e2")
    counts = _mod.count_corpus_episodes(str(tmp_path))
    assert counts == {"distinct": 2, "latest_run": 1}


def test_expect_flag_passes_on_match(tmp_path: Path) -> None:
    _write_episode(tmp_path, "run_20260416-000000_old", "e1")
    _write_episode(tmp_path, "run_20260417-000000_new", "e2")
    assert _mod.main([str(tmp_path), "--expect", "2"]) == 0


def test_expect_flag_fails_on_undercapture(tmp_path: Path) -> None:
    _write_episode(tmp_path, "run_20260417-000000_new", "e2")
    # Only one distinct episode present; expecting 2 must fail loudly (exit 2).
    assert _mod.main([str(tmp_path), "--expect", "2"]) == 2


def test_missing_dir_returns_error(tmp_path: Path) -> None:
    assert _mod.main([str(tmp_path / "nope")]) == 1
