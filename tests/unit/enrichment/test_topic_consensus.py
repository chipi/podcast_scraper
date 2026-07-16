"""Unit tests for the ``topic_consensus`` enricher (ADR-108 composite).

Cross-Person corroboration via **embedding cosine (shared-question gate) + low NLI contradiction
(direction gate)**. A scripted ``FixedConsensusScorer`` proves the machinery; the admission gate
keeps it dark until an eval clears precision.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from podcast_scraper.enrichment.enrichers.topic_consensus import TopicConsensusEnricher
from podcast_scraper.enrichment.eval.admission import (
    admitted_enricher_ids,
    gate_specs_from_manifests,
    known_enricher_manifests,
)
from podcast_scraper.enrichment.protocol import EpisodeArtifactBundle, RunContext, STATUS_OK
from podcast_scraper.enrichment.scorers.consensus import FixedConsensusScorer
from podcast_scraper.enrichment.scorers.protocol import ConsensusSignal


def _bundle(meta_dir: Path, stem: str, gi: dict[str, Any]) -> EpisodeArtifactBundle:
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / f"{stem}.metadata.json").write_text("{}", encoding="utf-8")
    gi_path = meta_dir / f"{stem}.gi.json"
    gi_path.write_text(json.dumps(gi), encoding="utf-8")
    return EpisodeArtifactBundle(
        metadata_path=meta_dir / f"{stem}.metadata.json",
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
        enricher_id="topic_consensus",
        enricher_version="2.0.0",
        tier="ml",
        attempt=1,
        job_id="r1",
        cancel_event=asyncio.Event(),
    )


def _two_speaker_gi() -> dict[str, Any]:
    """Alice + Bob each make one Insight on topic:risk."""
    return {
        "nodes": [
            {"type": "Person", "id": "person:alice", "properties": {"name": "Alice"}},
            {"type": "Person", "id": "person:bob", "properties": {"name": "Bob"}},
            {"type": "Insight", "id": "insight:a", "properties": {"text": "diversify to survive"}},
            {
                "type": "Insight",
                "id": "insight:b",
                "properties": {"text": "spread risk to survive"},
            },
            {"type": "Quote", "id": "quote:qa"},
            {"type": "Quote", "id": "quote:qb"},
        ],
        "edges": [
            {"type": "SPOKEN_BY", "from": "quote:qa", "to": "person:alice"},
            {"type": "SPOKEN_BY", "from": "quote:qb", "to": "person:bob"},
            {"type": "SUPPORTED_BY", "from": "insight:a", "to": "quote:qa"},
            {"type": "SUPPORTED_BY", "from": "insight:b", "to": "quote:qb"},
            {"type": "ABOUT", "from": "insight:a", "to": "topic:risk"},
            {"type": "ABOUT", "from": "insight:b", "to": "topic:risk"},
        ],
    }


_TA, _TB = "diversify to survive", "spread risk to survive"


def _scorer(cosine: float, contradiction: float) -> FixedConsensusScorer:
    return FixedConsensusScorer(
        signals={(_TA, _TB): ConsensusSignal(cosine=cosine, contradiction=contradiction)}
    )


def _run(enricher: TopicConsensusEnricher, bundles: list[EpisodeArtifactBundle]) -> dict[str, Any]:
    result = asyncio.run(
        enricher.enrich(
            bundle=None, corpus_root=Path("/tmp"), all_bundles=bundles, config={}, ctx=_ctx()
        )
    )
    assert result.status == STATUS_OK and isinstance(result.data, dict)
    return result.data


def test_high_cosine_low_contradiction_emits_consensus(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "e1", "e1", _two_speaker_gi())
    data = _run(TopicConsensusEnricher(_scorer(0.82, 0.03)), [bundle])
    assert data["pairs_scored"] == 1
    assert len(data["consensus"]) == 1
    row = data["consensus"][0]
    assert row["topic_id"] == "topic:risk"
    assert {row["person_a_id"], row["person_b_id"]} == {"person:alice", "person:bob"}
    assert row["consensus_score"] == 0.82 and row["cosine"] == 0.82
    assert row["contradiction"] == 0.03


def test_high_cosine_high_contradiction_dropped(tmp_path: Path) -> None:
    # Similar wording but they DISAGREE → contradiction gate drops it (offload-vs-intensify case).
    bundle = _bundle(tmp_path / "e1", "e1", _two_speaker_gi())
    data = _run(TopicConsensusEnricher(_scorer(0.82, 0.95)), [bundle])
    assert data["pairs_scored"] == 1 and data["consensus"] == []


def test_low_cosine_dropped(tmp_path: Path) -> None:
    # Not about the same proposition → shared-question gate drops it even at zero contradiction.
    bundle = _bundle(tmp_path / "e1", "e1", _two_speaker_gi())
    data = _run(TopicConsensusEnricher(_scorer(0.40, 0.0)), [bundle])
    assert data["pairs_scored"] == 1 and data["consensus"] == []


def test_same_speaker_not_paired(tmp_path: Path) -> None:
    gi = _two_speaker_gi()
    # Reassign Bob's quote to Alice → both insights are Alice's → no cross-Person pair.
    for e in gi["edges"]:
        if e.get("from") == "quote:qb":
            e["to"] = "person:alice"
    bundle = _bundle(tmp_path / "e1", "e1", gi)
    data = _run(TopicConsensusEnricher(_scorer(0.9, 0.0)), [bundle])
    assert data["pairs_scored"] == 0 and data["consensus"] == []


def test_unresolved_speaker_not_paired(tmp_path: Path) -> None:
    """An unresolved diarization voice must not form a consensus pair (#1167)."""
    gi = _two_speaker_gi()
    # Bob's quote is attributed to an anonymous diarization voice → excluded, so the
    # only remaining speaker is Alice and there is no cross-Person pair to score.
    gi["nodes"].append(
        {"type": "Person", "id": "person:speaker-00", "properties": {"name": "SPEAKER_00"}}
    )
    for e in gi["edges"]:
        if e.get("from") == "quote:qb":
            e["to"] = "person:speaker-00"
    bundle = _bundle(tmp_path / "e1", "e1", gi)
    data = _run(TopicConsensusEnricher(_scorer(0.9, 0.0)), [bundle])
    assert data["pairs_scored"] == 0 and data["consensus"] == []


def test_manifest_ml_tier_and_gate() -> None:
    manifest = TopicConsensusEnricher(FixedConsensusScorer()).manifest
    assert manifest.id == "topic_consensus"
    assert manifest.tier.value == "ml" and manifest.scope.value == "corpus"
    assert manifest.provider_requirement.protocol == "ConsensusScorer"  # type: ignore[union-attr]
    rule = manifest.accuracy_gate.rules[0]  # type: ignore[union-attr]
    assert (rule.metric_name, rule.min_value) == ("precision", 0.5)


def test_threshold_validation() -> None:
    with pytest.raises(ValueError):
        TopicConsensusEnricher(FixedConsensusScorer(), cos_threshold=1.5)
    with pytest.raises(ValueError):
        TopicConsensusEnricher(FixedConsensusScorer(), contra_threshold=-0.1)


def test_admission_gates_dark_until_cleared() -> None:
    specs = gate_specs_from_manifests(known_enricher_manifests())
    assert "topic_consensus" in specs
    assert "topic_consensus" not in admitted_enricher_ids(["topic_consensus"], specs, {}).admitted
    passing = admitted_enricher_ids(
        ["topic_consensus"], specs, {"topic_consensus": {"precision": 0.7}}
    )
    assert "topic_consensus" in passing.admitted
