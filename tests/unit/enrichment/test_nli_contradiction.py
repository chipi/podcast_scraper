"""Unit tests for chunk-4 ``nli_contradiction`` enricher + ``FixedNliScorer``."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from podcast_scraper.enrichment.enrichers.nli_contradiction import (
    _episode_topic_insight_speaker_index,
    NliContradictionEnricher,
)
from podcast_scraper.enrichment.protocol import (
    EpisodeArtifactBundle,
    RunContext,
    STATUS_OK,
)
from podcast_scraper.enrichment.scorers.nli import FixedNliScorer
from podcast_scraper.enrichment.scorers.protocol import NliScore, NliScorer


def _bundle(meta_dir: Path, stem: str, gi: dict[str, Any]) -> EpisodeArtifactBundle:
    meta_dir.mkdir(parents=True, exist_ok=True)
    md = meta_dir / f"{stem}.metadata.json"
    md.write_text("{}", encoding="utf-8")
    gi_path = meta_dir / f"{stem}.gi.json"
    gi_path.write_text(json.dumps(gi), encoding="utf-8")
    return EpisodeArtifactBundle(
        metadata_path=md,
        gi_path=gi_path,
        kg_path=None,
        bridge_path=None,
        episode_id=f"episode:{stem}",
        stem=stem,
    )


def _ctx() -> RunContext:
    return RunContext(
        run_id="r1",
        parent_run_id=None,
        enricher_id="nli_contradiction",
        enricher_version="1.0.0",
        tier="ml",
        attempt=1,
        job_id="r1",
        cancel_event=asyncio.Event(),
    )


def _gi(
    *,
    persons: dict[str, str],
    insights: dict[str, dict[str, Any]],
    quotes: dict[str, str],
    insight_to_quote: dict[str, str],
    insight_to_topic: dict[str, list[str]],
) -> dict[str, Any]:
    """Helper to build a minimal GI from explicit relations."""
    nodes: list[dict[str, Any]] = []
    for pid, name in persons.items():
        nodes.append({"type": "Person", "id": pid, "properties": {"name": name}})
    for iid, props in insights.items():
        nodes.append({"type": "Insight", "id": iid, "properties": props})
    for qid in quotes:
        nodes.append({"type": "Quote", "id": qid})
    edges: list[dict[str, Any]] = []
    for qid, pid in quotes.items():
        edges.append({"type": "SPOKEN_BY", "from": qid, "to": pid})
    for iid, qid in insight_to_quote.items():
        edges.append({"type": "SUPPORTED_BY", "from": iid, "to": qid})
    for iid, topics in insight_to_topic.items():
        for tid in topics:
            edges.append({"type": "ABOUT", "from": iid, "to": tid})
    return {"nodes": nodes, "edges": edges}


# ---------------------------------------------------------------------------
# FixedNliScorer
# ---------------------------------------------------------------------------


def test_fixed_nli_scorer_returns_scripted_score() -> None:
    scorer = FixedNliScorer(
        scores={("A", "B"): NliScore(0.9, 0.05, 0.05)},
        default=NliScore(0.0, 1.0, 0.0),
    )
    assert asyncio.run(scorer.score("A", "B")).contradiction == 0.9
    assert asyncio.run(scorer.score("X", "Y")).neutral == 1.0


def test_fixed_nli_scorer_satisfies_protocol() -> None:
    assert isinstance(FixedNliScorer(), NliScorer)


# ---------------------------------------------------------------------------
# _episode_topic_insight_speaker_index
# ---------------------------------------------------------------------------


def test_index_resolves_topic_to_insight_speaker_chain(tmp_path: Path) -> None:
    gi = _gi(
        persons={"person:alice": "Alice", "person:bob": "Bob"},
        insights={
            "insight:i1": {"text": "AI is safe"},
            "insight:i2": {"text": "AI is dangerous"},
        },
        quotes={"quote:q1": "person:alice", "quote:q2": "person:bob"},
        insight_to_quote={"insight:i1": "quote:q1", "insight:i2": "quote:q2"},
        insight_to_topic={"insight:i1": ["topic:ai"], "insight:i2": ["topic:ai"]},
    )
    bundle = _bundle(tmp_path / "metadata", "ep1", gi)
    by_topic, labels = _episode_topic_insight_speaker_index([bundle])
    assert "topic:ai" in by_topic
    ids = {entry[0] for entry in by_topic["topic:ai"]}
    assert ids == {"insight:i1", "insight:i2"}
    assert labels == {"person:alice": "Alice", "person:bob": "Bob"}


# ---------------------------------------------------------------------------
# NliContradictionEnricher
# ---------------------------------------------------------------------------


def _run_enricher(
    enricher: NliContradictionEnricher,
    bundles: list[EpisodeArtifactBundle],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = asyncio.run(
        enricher.enrich(
            bundle=None,
            corpus_root=Path("/tmp"),
            all_bundles=bundles,
            config=config or {},
            ctx=_ctx(),
        )
    )
    assert result.status == STATUS_OK
    assert isinstance(result.data, dict)
    return result.data


def test_contradictions_above_threshold_kept(tmp_path: Path) -> None:
    gi = _gi(
        persons={"person:alice": "Alice", "person:bob": "Bob"},
        insights={
            "insight:i1": {"text": "AI is safe"},
            "insight:i2": {"text": "AI is dangerous"},
        },
        quotes={"quote:q1": "person:alice", "quote:q2": "person:bob"},
        insight_to_quote={"insight:i1": "quote:q1", "insight:i2": "quote:q2"},
        insight_to_topic={"insight:i1": ["topic:ai"], "insight:i2": ["topic:ai"]},
    )
    bundle = _bundle(tmp_path / "metadata", "ep1", gi)
    scorer = FixedNliScorer(scores={("AI is safe", "AI is dangerous"): NliScore(0.92, 0.04, 0.04)})
    enricher = NliContradictionEnricher(scorer)
    data = _run_enricher(enricher, [bundle])
    assert data["pairs_scored"] == 1
    assert len(data["contradictions"]) == 1
    row = data["contradictions"][0]
    assert row["topic_id"] == "topic:ai"
    assert row["contradiction_score"] == 0.92
    assert {row["person_a_id"], row["person_b_id"]} == {"person:alice", "person:bob"}


def test_pairs_below_threshold_dropped(tmp_path: Path) -> None:
    gi = _gi(
        persons={"person:alice": "Alice", "person:bob": "Bob"},
        insights={
            "insight:i1": {"text": "A"},
            "insight:i2": {"text": "B"},
        },
        quotes={"quote:q1": "person:alice", "quote:q2": "person:bob"},
        insight_to_quote={"insight:i1": "quote:q1", "insight:i2": "quote:q2"},
        insight_to_topic={"insight:i1": ["topic:x"], "insight:i2": ["topic:x"]},
    )
    bundle = _bundle(tmp_path / "metadata", "ep1", gi)
    scorer = FixedNliScorer(scores={("A", "B"): NliScore(0.3, 0.4, 0.3)})
    enricher = NliContradictionEnricher(scorer, threshold=0.5)
    data = _run_enricher(enricher, [bundle])
    assert data["pairs_scored"] == 1
    assert data["contradictions"] == []


def test_same_person_pairs_not_scored(tmp_path: Path) -> None:
    """Two insights from Alice on the same topic shouldn't form a contradiction pair."""
    gi = _gi(
        persons={"person:alice": "Alice"},
        insights={
            "insight:i1": {"text": "X"},
            "insight:i2": {"text": "Y"},
        },
        quotes={"quote:q1": "person:alice", "quote:q2": "person:alice"},
        insight_to_quote={"insight:i1": "quote:q1", "insight:i2": "quote:q2"},
        insight_to_topic={"insight:i1": ["topic:a"], "insight:i2": ["topic:a"]},
    )
    bundle = _bundle(tmp_path / "metadata", "ep1", gi)
    scorer = FixedNliScorer(default=NliScore(0.99, 0.005, 0.005))
    enricher = NliContradictionEnricher(scorer)
    data = _run_enricher(enricher, [bundle])
    # Same-Person pairs are filtered out before scoring.
    assert data["pairs_scored"] == 0
    assert data["contradictions"] == []


def test_threshold_override_via_config(tmp_path: Path) -> None:
    gi = _gi(
        persons={"person:a": "A", "person:b": "B"},
        insights={"insight:i1": {"text": "X"}, "insight:i2": {"text": "Y"}},
        quotes={"q1": "person:a", "q2": "person:b"},
        insight_to_quote={"insight:i1": "q1", "insight:i2": "q2"},
        insight_to_topic={"insight:i1": ["topic:t"], "insight:i2": ["topic:t"]},
    )
    bundle = _bundle(tmp_path / "metadata", "ep1", gi)
    scorer = FixedNliScorer(scores={("X", "Y"): NliScore(0.6, 0.2, 0.2)})
    enricher = NliContradictionEnricher(scorer, threshold=0.5)
    # Threshold raised to 0.7 via config → score 0.6 dropped.
    data = _run_enricher(enricher, [bundle], config={"threshold": 0.7})
    assert data["threshold"] == 0.7
    assert data["contradictions"] == []


def test_threshold_validation() -> None:
    with pytest.raises(ValueError):
        NliContradictionEnricher(FixedNliScorer(), threshold=-0.1)
    with pytest.raises(ValueError):
        NliContradictionEnricher(FixedNliScorer(), threshold=1.5)


def test_manifest_is_ml_tier() -> None:
    enricher = NliContradictionEnricher(FixedNliScorer())
    assert enricher.manifest.id == "nli_contradiction"
    assert enricher.manifest.tier.value == "ml"
    assert enricher.manifest.scope.value == "corpus"


def test_model_id_version_carry_through_output(tmp_path: Path) -> None:
    gi = _gi(
        persons={"person:a": "A", "person:b": "B"},
        insights={"insight:i1": {"text": "p"}, "insight:i2": {"text": "q"}},
        quotes={"q1": "person:a", "q2": "person:b"},
        insight_to_quote={"insight:i1": "q1", "insight:i2": "q2"},
        insight_to_topic={"insight:i1": ["topic:t"], "insight:i2": ["topic:t"]},
    )
    bundle = _bundle(tmp_path / "metadata", "ep1", gi)
    scorer = FixedNliScorer(scores={("p", "q"): NliScore(0.95, 0.025, 0.025)})
    enricher = NliContradictionEnricher(scorer, model_id="custom-nli", model_version="v9")
    data = _run_enricher(enricher, [bundle])
    assert data["model_id"] == "custom-nli"
    assert data["model_version"] == "v9"
    assert data["contradictions"][0]["model_id"] == "custom-nli"
    assert data["contradictions"][0]["model_version"] == "v9"
