"""Integration tests for insight clustering (#599) — no ML models.

Only tests that use JSON I/O and don't require sentence-transformers.
ML-dependent tests (embedding, clustering) are in
tests/e2e/test_insight_clusters_e2e.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.search.insight_clusters import (
    collect_insight_rows_from_corpus,
)

pytestmark = [pytest.mark.integration]


def _write_gi_artifact(
    output_dir: Path,
    episode_id: str,
    insights: list[dict],
) -> None:
    """Write a minimal .gi.json artifact."""
    nodes = []
    edges = []
    for ins in insights:
        nodes.append(
            {
                "id": ins["id"],
                "type": "Insight",
                "properties": {
                    "text": ins["text"],
                    "insight_type": "factual",
                    "grounded": True,
                },
            }
        )
        for i, q in enumerate(ins.get("quotes", [])):
            qid = f"{ins['id']}_q{i}"
            nodes.append(
                {
                    "id": qid,
                    "type": "Quote",
                    "properties": {
                        "text": q,
                        "char_start": i * 100,
                        "char_end": i * 100 + 50,
                    },
                }
            )
            edges.append({"from": ins["id"], "to": qid, "type": "SUPPORTED_BY"})

    gi = {"episode_id": episode_id, "nodes": nodes, "edges": edges}
    ep_dir = output_dir / episode_id
    ep_dir.mkdir(parents=True, exist_ok=True)
    (ep_dir / f"{episode_id}.gi.json").write_text(json.dumps(gi), encoding="utf-8")


def test_collect_insights_from_multi_episode_corpus(tmp_path: Path) -> None:
    """collect_insight_rows_from_corpus finds insights across episodes."""
    _write_gi_artifact(
        tmp_path,
        "ep1",
        [{"id": "ins1", "text": "Index funds beat active managers", "quotes": ["92%"]}],
    )
    _write_gi_artifact(
        tmp_path,
        "ep2",
        [{"id": "ins2", "text": "Passive investing outperforms", "quotes": ["data"]}],
    )
    rows = collect_insight_rows_from_corpus(tmp_path)
    assert len(rows) == 2
    assert {r["episode_id"] for r in rows} == {"ep1", "ep2"}
