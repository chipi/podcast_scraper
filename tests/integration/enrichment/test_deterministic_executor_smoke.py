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

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from podcast_scraper.enrichment.enrichers import (
    ALL_DETERMINISTIC_ENRICHER_IDS,
    register_deterministic_enrichers,
)
from podcast_scraper.enrichment.executor import EnrichmentExecutor
from podcast_scraper.enrichment.protocol import (
    EnricherScope,
    EnricherSet,
    EpisodeArtifactBundle,
    STATUS_OK,
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
            + [{"type": "Topic", "id": tid, "properties": {"label": tid}} for tid in topics],
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


def test_executor_runs_all_six_deterministic_enrichers(tmp_path: Path) -> None:
    bundles = [
        _episode_bundle(
            tmp_path,
            "ep1",
            publish_date="2026-06-15T00:00:00Z",
            topics=["topic:a", "topic:b"],
            persons=["person:alice", "person:bob"],
        ),
        _episode_bundle(
            tmp_path,
            "ep2",
            publish_date="2026-05-15T00:00:00Z",
            topics=["topic:a", "topic:c"],
            persons=["person:alice"],
        ),
    ]

    registry = EnricherRegistry()
    register_deterministic_enrichers(registry)
    enricher_set = EnricherSet(enabled_enrichers=list(ALL_DETERMINISTIC_ENRICHER_IDS))

    executor = EnrichmentExecutor(
        corpus_root=tmp_path, registry=registry, enricher_set=enricher_set
    )
    result = asyncio.run(executor.run(episode_bundles=bundles))

    assert result.status == STATUS_OK
    # Every enricher recorded runs_ok >= 1 (episode-scope runs once per bundle).
    for eid in ALL_DETERMINISTIC_ENRICHER_IDS:
        m = result.per_enricher_metrics[eid]
        assert m.runs_ok >= 1, (eid, m.last_run_status, m.error_samples)

    # Corpus-scope envelopes exist.
    for writes in (
        "topic_cooccurrence_corpus.json",
        "temporal_velocity.json",
        "grounding_rate.json",
        "guest_coappearance.json",
    ):
        assert (tmp_path / "enrichments" / writes).is_file(), writes

    # Episode-scope envelopes for both episodes.
    for stem in ("ep1", "ep2"):
        for writes in ("insight_density.json",):
            assert (tmp_path / "metadata" / "enrichments" / f"{stem}.{writes}").is_file(), (
                stem,
                writes,
            )

    # Sanity: validate one of the corpus-scope JSON payloads.
    payload = json.loads((tmp_path / "enrichments" / "guest_coappearance.json").read_text())
    assert "pairs" in payload["data"]
    # ep1 had alice+bob → 1 pair counted once.
    pair_ids = {(p["person_a_id"], p["person_b_id"]) for p in payload["data"]["pairs"]}
    assert ("person:alice", "person:bob") in pair_ids


def test_all_deterministic_register_in_correct_scope() -> None:
    """Sanity: catch refactors that move an enricher to the wrong phase."""
    registry = EnricherRegistry()
    register_deterministic_enrichers(registry)
    episode_ids = {
        eid
        for eid in ALL_DETERMINISTIC_ENRICHER_IDS
        if registry.get(eid).manifest.scope is EnricherScope.EPISODE
    }
    assert episode_ids == {"insight_density", "insight_sentiment"}
