"""Unit tests for scripts/ops/dedupe_graph_edges.py — the one-off clean-up of copied edges."""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]

_SPEC = importlib.util.spec_from_file_location(
    "dedupe_graph_edges_under_test", ROOT / "scripts" / "ops" / "dedupe_graph_edges.py"
)
assert _SPEC and _SPEC.loader
_mod = importlib.util.module_from_spec(_SPEC)
sys.modules["dedupe_graph_edges_under_test"] = _mod
_SPEC.loader.exec_module(_mod)

pytestmark = [pytest.mark.unit]

SB = {"type": "SPOKEN_BY", "from": "quote:1", "to": "person:unresolved-twiggy-ep1"}
SUP = {"type": "SUPPORTED_BY", "from": "insight:1", "to": "quote:1"}


def _corpus(tmp_path: Path, edges: list) -> Path:
    meta = tmp_path / "feeds" / "f1" / "run_1" / "metadata"
    meta.mkdir(parents=True)
    (meta / "0001 - ep.metadata.json").write_text(
        json.dumps({"feed": {"feed_id": "f1"}, "episode": {"episode_id": "ep1"}}), encoding="utf-8"
    )
    (meta / "0001 - ep.gi.json").write_text(
        json.dumps({"episode_id": "ep1", "nodes": [], "edges": edges}), encoding="utf-8"
    )
    return meta / "0001 - ep.gi.json"


def test_only_exact_copies_go_and_the_first_stays_in_place() -> None:
    near = {**SB, "properties": {"confidence": 0.5}}
    out, dropped = _mod.dedupe_edges({"edges": [SB, SUP, dict(SB), near, dict(SB)]})
    assert out["edges"] == [SB, SUP, near]
    assert dropped == [SB, SB]


def test_a_file_without_copies_is_returned_untouched() -> None:
    payload = {"edges": [SB, SUP]}
    out, dropped = _mod.dedupe_edges(payload)
    assert out is payload and dropped == []


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    gi = _corpus(tmp_path, [SB, dict(SB)])
    before = gi.read_bytes()
    stats = _mod.run(tmp_path, False, logging.getLogger("t"))
    assert stats["edges_dropped"] == 1 and stats["by_type"] == {"SPOKEN_BY": 1}
    assert gi.read_bytes() == before


def test_apply_cleans_backs_up_and_undo_restores(tmp_path: Path) -> None:
    gi = _corpus(tmp_path, [SB, dict(SB), SUP])
    before = gi.read_bytes()
    assert _mod.main(["--corpus", str(tmp_path), "--apply"]) == 0
    assert json.loads(gi.read_text())["edges"] == [SB, SUP]
    assert (tmp_path / "corpus_edges_stamp.json").is_file(), "the API must reload the edges"
    assert _mod.run(tmp_path, False, logging.getLogger("t"))["edges_dropped"] == 0
    assert _mod.main(["--corpus", str(tmp_path), "--undo"]) == 0
    assert gi.read_bytes() == before
