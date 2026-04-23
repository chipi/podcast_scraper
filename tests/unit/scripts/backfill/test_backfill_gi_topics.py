"""Unit tests for the #653 Part E backfill CLI."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "scripts" / "backfill" / "backfill_gi_topics.py"

_spec = importlib.util.spec_from_file_location("backfill_gi_topics", SCRIPT)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
sys.modules["backfill_gi_topics"] = _mod
_spec.loader.exec_module(_mod)

rewrite_gi_topics = _mod._rewrite_gi_topics
kg_canonical_topic_labels = _mod._kg_canonical_topic_labels

pytestmark = [pytest.mark.unit]


def _gi_with_topics(topic_entries: list[tuple[str, str]]) -> dict:
    """Build a minimal GI artifact with the given [(id, label), ...] topics."""
    nodes = [
        {"id": "episode:ep1", "type": "Episode", "properties": {"id": "ep1"}},
    ]
    edges = []
    for t_id, label in topic_entries:
        nodes.append({"id": t_id, "type": "Topic", "properties": {"label": label}})
        edges.append({"type": "ABOUT", "from": "insight:abc", "to": t_id})
        edges.append({"type": "MENTIONS", "from": t_id, "to": "episode:ep1"})
    nodes.append({"id": "insight:abc", "type": "Insight", "properties": {"text": "x"}})
    return {"nodes": nodes, "edges": edges}


def _kg_with_topics(labels: list[str]) -> dict:
    nodes = [{"id": "episode:ep1", "type": "Episode", "properties": {"id": "ep1"}}]
    for i, lab in enumerate(labels):
        nodes.append({"id": f"topic:kg-{i}", "type": "Topic", "properties": {"label": lab}})
    return {"nodes": nodes, "edges": []}


class TestKgCanonicalTopicLabels:
    def test_collects_topic_labels_in_order(self):
        kg = _kg_with_topics(["AI agents", "regulation", "oil prices"])
        assert kg_canonical_topic_labels(kg) == ["AI agents", "regulation", "oil prices"]

    def test_dedupes_preserving_first(self):
        kg = _kg_with_topics(["AI agents", "AI agents", "regulation"])
        assert kg_canonical_topic_labels(kg) == ["AI agents", "regulation"]

    def test_ignores_non_topic_nodes(self):
        kg = {"nodes": [{"id": "episode:x", "type": "Episode", "properties": {}}]}
        assert kg_canonical_topic_labels(kg) == []


class TestRewriteGiTopics:
    def test_rewrites_labels_and_ids_and_edges(self):
        # Old GI had ugly bullet-slugs; KG has clean canonical topics.
        gi = _gi_with_topics(
            [
                ("topic:the-long-bullet-slug-that-used-to-live-here", "Long bullet summary text"),
            ]
        )
        new_gi, changes = rewrite_gi_topics(gi, ["Prediction markets"])
        assert changes == 1

        topic_nodes = [n for n in new_gi["nodes"] if n["type"] == "Topic"]
        assert len(topic_nodes) == 1
        assert topic_nodes[0]["id"] == "topic:prediction-markets"
        assert topic_nodes[0]["properties"]["label"] == "Prediction markets"

        # Edges pointing at the old id are rewritten to the new id.
        about = [e for e in new_gi["edges"] if e["type"] == "ABOUT"][0]
        assert about["to"] == "topic:prediction-markets"
        mentions = [e for e in new_gi["edges"] if e["type"] == "MENTIONS"][0]
        assert mentions["from"] == "topic:prediction-markets"

    def test_no_change_when_already_canonical(self):
        gi = _gi_with_topics([("topic:prediction-markets", "Prediction markets")])
        new_gi, changes = rewrite_gi_topics(gi, ["Prediction markets"])
        assert changes == 0
        # Artifact shape preserved.
        topic_nodes = [n for n in new_gi["nodes"] if n["type"] == "Topic"]
        assert topic_nodes[0]["id"] == "topic:prediction-markets"

    def test_mismatched_lengths_partial_rewrite(self):
        # GI has 2 topics, KG has only 1 — first topic gets KG canonical,
        # second topic keeps its old label but is re-slugged to canonical form.
        gi = _gi_with_topics(
            [
                ("topic:first-bullet-slug", "First bullet"),
                ("topic:second-bullet-slug", "Second bullet kept"),
            ]
        )
        new_gi, changes = rewrite_gi_topics(gi, ["Prediction markets"])
        topic_nodes = [n for n in new_gi["nodes"] if n["type"] == "Topic"]
        labels = [n["properties"]["label"] for n in topic_nodes]
        assert labels == ["Prediction markets", "Second bullet kept"]
        ids = [n["id"] for n in topic_nodes]
        assert ids[0] == "topic:prediction-markets"
        # Second topic keeps its own label but gets re-slugged to canonical form
        assert ids[1] == "topic:second-bullet-kept"
        assert changes == 2

    def test_empty_gi_returns_unchanged(self):
        new_gi, changes = rewrite_gi_topics({"nodes": [], "edges": []}, ["x"])
        assert changes == 0

    def test_no_topic_nodes_returns_unchanged(self):
        gi = {
            "nodes": [{"id": "episode:1", "type": "Episode", "properties": {}}],
            "edges": [],
        }
        new_gi, changes = rewrite_gi_topics(gi, ["x"])
        assert changes == 0


class TestMainIntegration:
    def _seed_corpus(self, tmp_path: Path, layout: str = "multi") -> Path:
        if layout == "multi":
            run_dir = tmp_path / "feeds" / "feed_a" / "run_001" / "metadata"
        else:
            run_dir = tmp_path / "run_001" / "metadata"
        run_dir.mkdir(parents=True)
        (run_dir / "ep1.gi.json").write_text(
            json.dumps(_gi_with_topics([("topic:old-bullet", "Old bullet label")])),
            encoding="utf-8",
        )
        (run_dir / "ep1.kg.json").write_text(
            json.dumps(_kg_with_topics(["Canonical topic"])), encoding="utf-8"
        )
        return tmp_path

    def test_dry_run_multi_feed_layout_does_not_mutate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = self._seed_corpus(tmp_path, layout="multi")
        monkeypatch.setattr(sys, "argv", ["backfill_gi_topics", str(root)])
        rc = _mod.main()
        assert rc == 0

        gi_path = root / "feeds" / "feed_a" / "run_001" / "metadata" / "ep1.gi.json"
        gi = json.loads(gi_path.read_text())
        topic_nodes = [n for n in gi["nodes"] if n["type"] == "Topic"]
        # Dry-run — original label preserved.
        assert topic_nodes[0]["properties"]["label"] == "Old bullet label"
        assert not (gi_path.with_suffix(gi_path.suffix + ".bak")).exists()

    def test_apply_single_feed_layout_rewrites_and_backs_up(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = self._seed_corpus(tmp_path, layout="single")
        monkeypatch.setattr(sys, "argv", ["backfill_gi_topics", str(root), "--apply"])
        rc = _mod.main()
        assert rc == 0

        gi_path = root / "run_001" / "metadata" / "ep1.gi.json"
        gi = json.loads(gi_path.read_text())
        topic_nodes = [n for n in gi["nodes"] if n["type"] == "Topic"]
        assert topic_nodes[0]["properties"]["label"] == "Canonical topic"
        assert topic_nodes[0]["id"] == "topic:canonical-topic"
        # .bak sibling written.
        assert (gi_path.with_suffix(gi_path.suffix + ".bak")).exists()

    def test_no_artifacts_exits_1(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["backfill_gi_topics", str(tmp_path)])
        rc = _mod.main()
        assert rc == 1
