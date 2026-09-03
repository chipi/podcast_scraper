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
            # a AND b recur across both episodes; c appears once. #1928 put a per-topic
            # document-frequency floor on co-occurrence (``min_topic_episode_count``, default 2),
            # so this is the smallest fixture that carries BOTH cases: a real association between
            # two independently-recurring topics, and a one-off that must be excluded.
            topics=["topic:a", "topic:b", "topic:c"],
            persons=["person:alice"],
        ),
        _episode_bundle(
            tmp_path,
            "ep3",
            publish_date="2026-04-15T00:00:00Z",
            # topic:a ALONE. Without this the (a, b) pair is `df_a == df_b == episode_count == 2`
            # — the two topics appear only ever together — which every association measure
            # saturates on (NPMI is exactly 1.0) and which
            # ``require_independent_recurrence`` therefore drops. Giving `a` an episode outside
            # the pair is what makes the association real rather than a coincidence, and it is
            # the distinction the enricher exists to draw, so the fixture has to contain it.
            topics=["topic:a"],
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
        "topic_theme_clusters.json",
        "temporal_velocity.json",
        "grounding_rate.json",
        "guest_coappearance.json",
    ):
        assert (tmp_path / "enrichments" / writes).is_file(), writes

    # Episode-scope envelopes for every episode.
    for stem in ("ep1", "ep2", "ep3"):
        for writes in ("insight_density.json", "insight_sentiment.json"):
            assert (tmp_path / "metadata" / "enrichments" / f"{stem}.{writes}").is_file(), (
                stem,
                writes,
            )

    # ------------------------------------------------------------------
    # Output non-degeneracy: each deterministic enricher must emit a
    # *correct, non-empty* artifact — not just an on-disk file. This is
    # the emission contract the enricher surfaces render against.
    # (insight_sentiment / topic_theme_clusters need a richer corpus to
    # exercise their signal — their non-degeneracy is asserted on the v3
    # fixture in test_app_validation_corpus_invariants.py.)
    # ------------------------------------------------------------------
    def _corpus_data(name: str) -> dict[str, Any]:
        env = json.loads((tmp_path / "enrichments" / f"{name}.json").read_text())
        data: dict[str, Any] = env["data"]
        return data

    def _episode_data(stem: str, name: str) -> dict[str, Any]:
        p = tmp_path / "metadata" / "enrichments" / f"{stem}.{name}.json"
        data: dict[str, Any] = json.loads(p.read_text())["data"]
        return data

    # guest_coappearance: ep1's alice+bob co-appear → exactly that pair.
    guest = _corpus_data("guest_coappearance")
    pair_ids = {(p["person_a_id"], p["person_b_id"]) for p in guest["pairs"]}
    assert ("person:alice", "person:bob") in pair_ids

    # grounding_rate: per-EPISODE since #1927, and it must DISCRIMINATE — ep1 has a grounded
    # insight, ep2 an ungrounded one, so the rates have to span rather than pin to 1.0.
    #
    # The pivot is the whole point of the assertion. Per-Person the metric was 1.0 for all 689
    # people in the real corpus and could not have been anything else: an insight is grounded
    # exactly when a supporting quote exists, and the quote is what carries the speaker — so an
    # ungrounded insight has no person to attribute it to, and the denominator could only ever
    # equal the numerator. Per episode the denominator is every insight in the episode, grounded
    # or not, so the ratio can finally move.
    grounding = _corpus_data("grounding_rate")
    episodes = grounding["episodes"]
    assert len(episodes) >= 2, episodes
    rates = [e["rate"] for e in episodes]
    assert all(0.0 <= r <= 1.0 for r in rates), rates
    assert min(rates) < max(rates), f"grounding_rate does not discriminate: {rates}"
    assert grounding["corpus_grounded_insights"] < grounding["corpus_total_insights"], (
        "every insight counted as grounded — the per-Person degeneracy is back"
    )
    # Worst-first ordering is what makes it actionable: the operator opens the list at the
    # episode most in need of attention, not at a random one.
    assert rates == sorted(rates), f"episodes must be sorted worst-first: {rates}"
    assert grounding["partial_reason"] is None

    # topic_cooccurrence_corpus: co-occurring pairs discovered, with lift + pmi.
    cooc = _corpus_data("topic_cooccurrence_corpus")
    assert cooc["pairs"], "no co-occurrence pairs"
    assert all("lift" in p and "pmi" in p for p in cooc["pairs"])
    cooc_ids = {frozenset((p["topic_a_id"], p["topic_b_id"])) for p in cooc["pairs"]}
    # (a, b) survives because `a` also appears in ep3 — it recurs OUTSIDE the pair, so the
    # co-occurrence carries information rather than being two labels for one moment.
    assert frozenset(("topic:a", "topic:b")) in cooc_ids
    assert cooc["pairs_without_independent_recurrence"] == 0, (
        "with topic:a in ep3 nothing here is saturated; a non-zero count means the criterion "
        "is rejecting the fixture's genuine association"
    )
    # #1928: topic:c appears in one episode, so every pair containing it is dropped before
    # scoring. Without this floor, `lift` rewards exactly these thinnest pairs — on the real
    # corpus 99.4% of pairs co-occurred in a single episode and lift's median, p90 and max were
    # all identical (the corpus episode count), so the "strongest association" ranking surfaced
    # the weakest evidence first.
    assert not any("topic:c" in pair for pair in cooc_ids), (
        f"a single-episode topic survived the df floor: {cooc_ids}"
    )

    # temporal_velocity: ≥1 topic with a non-empty weekly series + real total.
    vtopics = _corpus_data("temporal_velocity")["topics"]
    assert vtopics, "no velocity topics"
    assert any(t["total"] >= 1 and t["weekly_counts"] for t in vtopics)

    # topic_theme_clusters: structurally sound — clustered members + singletons
    # never exceed the topic count. (Real cluster formation: v3 invariants.)
    themes = _corpus_data("topic_theme_clusters")
    assert isinstance(themes["clusters"], list)
    assert isinstance(themes["singletons"], int)
    clustered = sum(c["member_count"] for c in themes["clusters"])
    assert clustered + themes["singletons"] <= themes["topic_count"], themes

    # insight_density: ≥1 episode has timed insights binned into segments,
    # and every segment count sums back to the episode's total.
    densities = [_episode_data(stem, "insight_density") for stem in ("ep1", "ep2")]
    assert any(d["total_insights"] >= 1 for d in densities), densities
    for d in densities:
        assert sum(d["counts"].values()) == d["total_insights"]


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
