"""#657 Part B — smoke + graceful-degradation tests for snapshot_quality.py.

Two guarantees:

- Smoke: given a tiny synthetic corpus (5 eps with gi.json + kg.json), the
  script emits a schema-valid JSON with every top-level field populated.
- Graceful degradation: if topic_clusters.json / corpus_manifest.json are
  missing, the snapshot still succeeds and marks the missing pieces with
  ``"status": "not-built"`` rather than crashing the run.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "scripts" / "validate" / "snapshot_quality.py"

_spec = importlib.util.spec_from_file_location("snapshot_quality", SCRIPT)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
sys.modules["snapshot_quality"] = _mod
_spec.loader.exec_module(_mod)


def _make_fixture_corpus(root: Path, n_episodes: int = 5) -> None:
    """Create a minimal gi/kg corpus under ``root/feeds/<slug>/run_X/metadata/``."""
    feed = root / "feeds" / "test_feed" / "run_000000-000000_abcdef01" / "metadata"
    feed.mkdir(parents=True, exist_ok=True)
    for i in range(1, n_episodes + 1):
        stem = f"{i:04d} - Sample Episode {i}"
        gi = {
            "nodes": [
                {"id": f"episode:ep{i}", "type": "Episode", "properties": {}},
                {
                    "id": "topic:ai-agents",
                    "type": "Topic",
                    "properties": {"label": "AI agents"},
                },
                {
                    "id": f"insight:{i}:1",
                    "type": "Insight",
                    "properties": {
                        "text": "AI agents are becoming capable of autonomous reasoning.",
                        "quote_text": "autonomous reasoning",
                    },
                },
            ],
            "edges": [],
        }
        kg = {
            "nodes": [
                {
                    "id": "topic:ai-agents",
                    "type": "Topic",
                    "properties": {"label": "AI agents"},
                },
                {
                    "id": "entity:openai",
                    "type": "Entity",
                    "properties": {"name": "OpenAI", "kind": "org"},
                },
            ],
            "edges": [],
        }
        (feed / f"{stem}.gi.json").write_text(json.dumps(gi), encoding="utf-8")
        (feed / f"{stem}.kg.json").write_text(json.dumps(kg), encoding="utf-8")


class TestSnapshotSmoke:
    def test_snapshot_on_5_ep_fixture_is_schema_valid(self, tmp_path: Path) -> None:
        """Smoke: 5 gi/kg pairs + no optional inputs → schema-valid JSON."""
        _make_fixture_corpus(tmp_path, n_episodes=5)
        out = _mod.build_snapshot(tmp_path)

        # Top-level schema.
        for key in (
            "schema_version",
            "snapshot_date",
            "git_sha",
            "corpus_root",
            "topic_clusters",
            "insight_clusters",
            "bridge_distribution",
            "filter_impact",
            "cost_rollup",
            "gil_quality_metrics",
            "kg_quality_metrics",
        ):
            assert key in out, f"{key} missing from snapshot"

        assert out["schema_version"] == "1.0.0"
        assert len(out["snapshot_date"]) == 10  # ISO date

        # Filter impact always runs even without optional inputs.
        fi = out["filter_impact"]
        assert fi["status"] == "ok"
        assert fi["insights_total"] == 5
        assert fi["topics_total"] == 5
        assert fi["entities_total"] == 5

        # Bridge distribution runs against the gi/kg pairs we created.
        bd = out["bridge_distribution"]
        assert bd["status"] == "ok"
        assert bd["episodes_scanned"] == 5

    def test_snapshot_handles_missing_topic_clusters(self, tmp_path: Path) -> None:
        """Graceful degradation: no ``search/topic_clusters.json`` → field
        marks ``"status": "not-built"`` rather than crashing."""
        _make_fixture_corpus(tmp_path, n_episodes=3)
        # Deliberately don't create search/ dir
        out = _mod.build_snapshot(tmp_path)
        tc = out["topic_clusters"]
        assert tc["status"] in {"not-built", "no-search-dir"}

    def test_snapshot_handles_missing_cost_rollup(self, tmp_path: Path) -> None:
        """Graceful degradation: no ``corpus_manifest.json`` → ``not-built``."""
        _make_fixture_corpus(tmp_path, n_episodes=3)
        out = _mod.build_snapshot(tmp_path)
        cr = out["cost_rollup"]
        assert cr["status"] in {"not-built", "missing-cost-rollup"}

    def test_snapshot_handles_empty_corpus(self, tmp_path: Path) -> None:
        """No gi/kg artifacts → bridge + filter impact return empty-ok or
        no-artifacts status; script doesn't crash."""
        # Empty directory — no feeds/, no run_*/.
        out = _mod.build_snapshot(tmp_path)
        bd = out["bridge_distribution"]
        fi = out["filter_impact"]
        assert bd["status"] in {"ok", "no-gi-artifacts"}
        assert fi["status"] == "ok"
        assert fi["insights_total"] == 0

    def test_snapshot_reads_committed_topic_clusters(self, tmp_path: Path) -> None:
        """Positive case: when topic_clusters.json exists, top-20 is populated."""
        _make_fixture_corpus(tmp_path, n_episodes=3)
        (tmp_path / "search").mkdir()
        (tmp_path / "search" / "topic_clusters.json").write_text(
            json.dumps(
                {
                    "cluster_count": 2,
                    "topic_count": 4,
                    "singletons": 2,
                    "threshold": 0.75,
                    "model": "all-MiniLM-L6-v2",
                    "schema_version": "1.0.0",
                    "clusters": [
                        {
                            "canonical_label": "AI agents",
                            "member_count": 2,
                            "members": [
                                {"label": "AI agents"},
                                {"label": "autonomous agents"},
                            ],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        out = _mod.build_snapshot(tmp_path)
        tc = out["topic_clusters"]
        assert tc["status"] == "ok"
        assert tc["cluster_count"] == 2
        assert tc["top20"][0]["canonical_label"] == "AI agents"
        assert "autonomous agents" in tc["top20"][0]["aliases"]
