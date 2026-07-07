"""Unit tests for the #1144 ``stance_disagreement`` enricher (no-LLM, gated dark).

The enricher machinery is correct (these tests, with a scripted ``FixedNliScorer``,
prove it) even though the *real* DeBERTa scorer's precision is 0% — that is measured by
``scripts/eval/score/disagreement_stance_eval_v1.py`` against ``gold_v1.jsonl`` and
recorded in ``gate_metrics.json``, and the admission gate keeps the enricher dark. The
last two tests assert exactly that gate behaviour.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from podcast_scraper.enrichment.enrichers.stance_disagreement import (
    StanceDisagreementEnricher,
)
from podcast_scraper.enrichment.eval.admission import (
    admitted_enricher_ids,
    gate_specs_from_manifests,
    known_enricher_manifests,
)
from podcast_scraper.enrichment.protocol import (
    EpisodeArtifactBundle,
    RunContext,
    STATUS_OK,
)
from podcast_scraper.enrichment.scorers.nli import FixedNliScorer
from podcast_scraper.enrichment.scorers.protocol import NliScore


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
        enricher_id="stance_disagreement",
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


def _two_speaker_gi() -> dict[str, Any]:
    """Alice + Bob, two insights each on topic:risk — enough to have a stance."""
    return _gi(
        persons={"person:alice": "Alice", "person:bob": "Bob"},
        insights={
            "insight:a1": {"text": "diversify"},
            "insight:a2": {"text": "spread out"},
            "insight:b1": {"text": "concentrate"},
            "insight:b2": {"text": "focus deep"},
        },
        quotes={
            "quote:qa1": "person:alice",
            "quote:qa2": "person:alice",
            "quote:qb1": "person:bob",
            "quote:qb2": "person:bob",
        },
        insight_to_quote={
            "insight:a1": "quote:qa1",
            "insight:a2": "quote:qa2",
            "insight:b1": "quote:qb1",
            "insight:b2": "quote:qb2",
        },
        insight_to_topic={
            "insight:a1": ["topic:risk"],
            "insight:a2": ["topic:risk"],
            "insight:b1": ["topic:risk"],
            "insight:b2": ["topic:risk"],
        },
    )


# Aggregated stances (insertion-order join): Alice "diversify spread out", Bob
# "concentrate focus deep". person:alice < person:bob so alice is side A.
_STANCE_A = "diversify spread out"
_STANCE_B = "concentrate focus deep"


def _run(
    enricher: StanceDisagreementEnricher,
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


def test_symmetric_contradiction_emitted(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "metadata", "ep1", _two_speaker_gi())
    scorer = FixedNliScorer(
        scores={
            (_STANCE_A, _STANCE_B): NliScore(0.9, 0.05, 0.05),
            (_STANCE_B, _STANCE_A): NliScore(0.8, 0.1, 0.1),
        }
    )
    data = _run(StanceDisagreementEnricher(scorer, threshold=0.6), [bundle])
    assert data["pairs_scored"] == 1
    assert len(data["disagreements"]) == 1
    row = data["disagreements"][0]
    assert row["topic_id"] == "topic:risk"
    assert {row["person_a_id"], row["person_b_id"]} == {"person:alice", "person:bob"}
    assert (row["a_stance"], row["b_stance"]) == (_STANCE_A, _STANCE_B)
    # Symmetric = min(0.9, 0.8) = 0.8.
    assert row["contradiction_score"] == 0.8


def test_asymmetric_dropped(tmp_path: Path) -> None:
    """High one way, low the other → min below threshold → not a disagreement."""
    bundle = _bundle(tmp_path / "metadata", "ep1", _two_speaker_gi())
    scorer = FixedNliScorer(
        scores={
            (_STANCE_A, _STANCE_B): NliScore(0.95, 0.03, 0.02),
            (_STANCE_B, _STANCE_A): NliScore(0.2, 0.6, 0.2),
        }
    )
    data = _run(StanceDisagreementEnricher(scorer, threshold=0.6), [bundle])
    assert data["pairs_scored"] == 1
    assert data["disagreements"] == []


def test_below_threshold_dropped(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "metadata", "ep1", _two_speaker_gi())
    scorer = FixedNliScorer(default=NliScore(0.3, 0.4, 0.3))
    data = _run(StanceDisagreementEnricher(scorer, threshold=0.6), [bundle])
    assert data["pairs_scored"] == 1
    assert data["disagreements"] == []


def test_min_insights_filters_thin_stance(tmp_path: Path) -> None:
    """A speaker with only 1 insight on the topic has no stance → no pair formed."""
    gi = _gi(
        persons={"person:alice": "Alice", "person:bob": "Bob"},
        insights={
            "insight:a1": {"text": "diversify"},
            "insight:a2": {"text": "spread out"},
            "insight:b1": {"text": "concentrate"},
        },
        quotes={
            "quote:qa1": "person:alice",
            "quote:qa2": "person:alice",
            "quote:qb1": "person:bob",
        },
        insight_to_quote={
            "insight:a1": "quote:qa1",
            "insight:a2": "quote:qa2",
            "insight:b1": "quote:qb1",
        },
        insight_to_topic={
            "insight:a1": ["topic:risk"],
            "insight:a2": ["topic:risk"],
            "insight:b1": ["topic:risk"],
        },
    )
    bundle = _bundle(tmp_path / "metadata", "ep1", gi)
    scorer = FixedNliScorer(default=NliScore(0.99, 0.005, 0.005))
    data = _run(StanceDisagreementEnricher(scorer, min_insights=2), [bundle])
    # Bob has 1 insight (< min_insights) → no viable pair, nothing scored.
    assert data["pairs_scored"] == 0
    assert data["disagreements"] == []


def test_threshold_validation() -> None:
    with pytest.raises(ValueError):
        StanceDisagreementEnricher(FixedNliScorer(), threshold=-0.1)
    with pytest.raises(ValueError):
        StanceDisagreementEnricher(FixedNliScorer(), threshold=1.5)


def test_records_written_matches_disagreements(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "metadata", "ep1", _two_speaker_gi())
    scorer = FixedNliScorer(
        scores={
            (_STANCE_A, _STANCE_B): NliScore(0.9, 0.05, 0.05),
            (_STANCE_B, _STANCE_A): NliScore(0.85, 0.1, 0.05),
        }
    )
    result = asyncio.run(
        StanceDisagreementEnricher(scorer, threshold=0.6).enrich(
            bundle=None, corpus_root=tmp_path, all_bundles=[bundle], config={}, ctx=_ctx()
        )
    )
    assert result.status == STATUS_OK
    assert isinstance(result.data, dict)
    assert result.records_written == len(result.data["disagreements"]) == 1


def test_manifest_ml_tier_and_gate() -> None:
    manifest = StanceDisagreementEnricher(FixedNliScorer()).manifest
    assert manifest.id == "stance_disagreement"
    assert manifest.tier.value == "ml"
    assert manifest.scope.value == "corpus"
    assert manifest.accuracy_gate is not None
    rule = manifest.accuracy_gate.rules[0]
    assert (rule.metric_name, rule.min_value) == ("precision", 0.5)
    assert manifest.accuracy_gate.on_missing_data == "reject"


def test_admission_gates_dark_at_zero_precision() -> None:
    """The measured no-LLM result (precision 0%) must keep the enricher out."""
    specs = gate_specs_from_manifests(known_enricher_manifests())
    assert "stance_disagreement" in specs  # manifest is known to the gate

    zero = admitted_enricher_ids(
        ["stance_disagreement"], specs, {"stance_disagreement": {"precision": 0.0}}
    )
    assert "stance_disagreement" not in zero.admitted

    missing = admitted_enricher_ids(["stance_disagreement"], specs, {})
    assert "stance_disagreement" not in missing.admitted  # on_missing_data=reject


def test_admission_would_promote_if_a_future_scorer_cleared_the_bar() -> None:
    """Guards the auto-promotion contract: precision >= 0.5 admits with no code edit."""
    specs = gate_specs_from_manifests(known_enricher_manifests())
    passing = admitted_enricher_ids(
        ["stance_disagreement"], specs, {"stance_disagreement": {"precision": 0.6}}
    )
    assert "stance_disagreement" in passing.admitted
