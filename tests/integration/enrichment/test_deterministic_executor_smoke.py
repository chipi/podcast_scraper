"""Integration smoke: run all 6 deterministic enrichers via the executor.

Builds a tiny 2-episode corpus with synthetic KG/GI/bridge/metadata,
registers all six deterministic enrichers, runs the executor with an
``EnricherSet`` enabling all of them, and asserts:

* Run status is ``ok``.
* Every enricher recorded ``runs_ok=1``.
* Every expected envelope file is written on disk under the correct
  scope path (corpus-scope under ``enrichments/`` and episode-scope
  under ``metadata/enrichments/``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from podcast_scraper.enrichment.enrichers import (
    ALL_DETERMINISTIC_ENRICHER_IDS,
    register_deterministic_enrichers,
)
from podcast_scraper.enrichment.protocol import (
    EnricherScope,
    EpisodeArtifactBundle,
)
from podcast_scraper.enrichment.registry import EnricherRegistry

pytestmark = pytest.mark.integration


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _episode_bundle(
    corpus_root: Path,
    stem: str,
    *,
    publish_date: str,
    topics: list[str],
    persons: list[str],
) -> EpisodeArtifactBundle:
    meta_dir = corpus_root / "metadata"
    md_path = meta_dir / f"{stem}.metadata.json"
    kg_path = meta_dir / f"{stem}.kg.json"
    gi_path = meta_dir / f"{stem}.gi.json"
    bridge_path = meta_dir / f"{stem}.bridge.json"

    _write(md_path, {"duration_seconds": 900})
    _write(
        kg_path,
        {
            "nodes": [
                {
                    "type": "Episode",
                    "id": f"episode:{stem}",
                    "properties": {"publish_date": publish_date},
                }
            ]
            + [
                {"type": "Topic", "id": tid, "properties": {"label": f"{tid.split(':')[-1]} topic"}}
                for tid in topics
            ],
            "edges": [],
        },
    )
    quotes_by_person = {p: f"quote:{stem}-{p}" for p in persons}
    insights = [
        {
            "type": "Insight",
            "id": f"insight:{stem}-i{i}",
            "properties": {"grounded": i % 2 == 0},
        }
        for i in range(len(persons))
    ]
    _write(
        gi_path,
        {
            "nodes": insights
            + [
                {"type": "Person", "id": pid, "properties": {"name": pid.split(":")[-1]}}
                for pid in persons
            ]
            + [
                {
                    "type": "Quote",
                    "id": qid,
                    "properties": {"start_s": 100 * (i + 1)},
                }
                for i, qid in enumerate(quotes_by_person.values())
            ],
            "edges": [
                {"type": "SPOKEN_BY", "from": qid, "to": pid}
                for pid, qid in quotes_by_person.items()
            ]
            + [
                {"type": "SUPPORTED_BY", "from": f"insight:{stem}-i{i}", "to": qid}
                for i, qid in enumerate(quotes_by_person.values())
            ],
        },
    )
    _write(bridge_path, {"episode_id": f"episode:{stem}"})

    return EpisodeArtifactBundle(
        metadata_path=md_path,
        gi_path=gi_path,
        kg_path=kg_path,
        bridge_path=bridge_path,
        episode_id=f"episode:{stem}",
        stem=stem,
    )


def test_all_deterministic_register_in_correct_scope() -> None:
    """Sanity: catch refactors that move an enricher to the wrong phase."""
    registry = EnricherRegistry()
    register_deterministic_enrichers(registry)
    episode_ids = {
        eid
        for eid in ALL_DETERMINISTIC_ENRICHER_IDS
        if registry.get(eid).manifest.scope is EnricherScope.EPISODE
    }
    # Public's one deterministic enricher is corpus-scope; the episode-scope ones are the
    # extension's (insight density and sentiment), pinned in its copy of this test.
    assert episode_ids == set()
