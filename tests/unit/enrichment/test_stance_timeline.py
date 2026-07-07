"""Unit tests for the ``stance_timeline`` enricher (ADR-108 reimagining of stance_disagreement).

Same-person + same-topic makes the shared-question gate free, so the machinery is testable with a
scripted ``FixedNliScorer``: stance = entail(H+) − entail(H−) per episode, series over time, and a
deterministic deviation block. The real DeBERTa precision is measured by the eval + admission gate
(the last tests assert the gate keeps it dark until an eval clears it).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from podcast_scraper.enrichment.enrichers.stance_timeline import StanceTimelineEnricher
from podcast_scraper.enrichment.eval.admission import (
    admitted_enricher_ids,
    gate_specs_from_manifests,
    known_enricher_manifests,
)
from podcast_scraper.enrichment.protocol import EpisodeArtifactBundle, RunContext, STATUS_OK
from podcast_scraper.enrichment.scorers.nli import FixedNliScorer
from podcast_scraper.enrichment.scorers.protocol import NliScore

# topic_label "AI" → topic_human "AI"; the enricher's default anchors.
_HPLUS = "AI is good and promising."
_HMINUS = "AI is bad and overhyped."


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
        enricher_id="stance_timeline",
        enricher_version="1.0.0",
        tier="ml",
        attempt=1,
        job_id="r1",
        cancel_event=asyncio.Event(),
    )


def _ep_gi(publish: str, insight_text: str) -> dict[str, Any]:
    """One episode: Alice makes one Insight about topic:ai (label 'AI'), on ``publish`` date."""
    return {
        "nodes": [
            {"type": "Episode", "id": "ep", "properties": {"publish_date": publish}},
            {"type": "Topic", "id": "topic:ai", "properties": {"label": "AI"}},
            {"type": "Person", "id": "person:alice", "properties": {"name": "Alice"}},
            {"type": "Insight", "id": "insight:i", "properties": {"text": insight_text}},
            {"type": "Quote", "id": "quote:q"},
        ],
        "edges": [
            {"type": "SPOKEN_BY", "from": "quote:q", "to": "person:alice"},
            {"type": "SUPPORTED_BY", "from": "insight:i", "to": "quote:q"},
            {"type": "ABOUT", "from": "insight:i", "to": "topic:ai"},
        ],
    }


def _scorer(entail: dict[str, tuple[float, float]]) -> FixedNliScorer:
    """entail maps insight_text → (entail_vs_H+, entail_vs_H−) → scripted NliScores."""
    scores: dict[tuple[str, str], NliScore] = {}
    for text, (ep, en) in entail.items():
        scores[(text, _HPLUS)] = NliScore(contradiction=0.0, neutral=1.0 - ep, entailment=ep)
        scores[(text, _HMINUS)] = NliScore(contradiction=0.0, neutral=1.0 - en, entailment=en)
    return FixedNliScorer(scores=scores, default=NliScore(0.0, 1.0, 0.0))


def _run(enricher: StanceTimelineEnricher, bundles: list[EpisodeArtifactBundle]) -> dict[str, Any]:
    result = asyncio.run(
        enricher.enrich(
            bundle=None, corpus_root=Path("/tmp"), all_bundles=bundles, config={}, ctx=_ctx()
        )
    )
    assert result.status == STATUS_OK and isinstance(result.data, dict)
    return result.data


def test_stance_trajectory_and_deviation(tmp_path: Path) -> None:
    # Alice on AI across 3 episodes: opposed → skeptical → supportive.
    bundles = [
        _bundle(tmp_path / "e1", "e1", _ep_gi("2024-01-01", "ai is bad")),
        _bundle(tmp_path / "e2", "e2", _ep_gi("2025-01-01", "ai is unclear")),
        _bundle(tmp_path / "e3", "e3", _ep_gi("2026-01-01", "ai is good")),
    ]
    scorer = _scorer(
        {"ai is bad": (0.1, 0.9), "ai is unclear": (0.3, 0.4), "ai is good": (0.85, 0.05)}
    )
    data = _run(StanceTimelineEnricher(scorer), bundles)
    assert len(data["timelines"]) == 1
    tl = data["timelines"][0]
    assert tl["person_id"] == "person:alice" and tl["topic_id"] == "topic:ai"
    # stance = entail(H+) − entail(H−), oldest→newest.
    stances = [p["stance"] for p in tl["points"]]
    assert stances == [-0.8, -0.1, 0.8]
    dev = tl["deviation"]
    assert dev["shifted"] is True
    assert dev["range"] == 1.6
    assert dev["sign_flips"] == 1  # −0.1 → +0.8 crosses zero
    assert dev["slope"] > 0  # opposed → supportive


def test_flat_stance_is_not_shifted(tmp_path: Path) -> None:
    bundles = [
        _bundle(tmp_path / "e1", "e1", _ep_gi("2024-01-01", "ai is good")),
        _bundle(tmp_path / "e2", "e2", _ep_gi("2025-06-01", "ai is good")),
    ]
    scorer = _scorer({"ai is good": (0.8, 0.1)})  # steady +0.7 both episodes
    tl = _run(StanceTimelineEnricher(scorer), bundles)["timelines"][0]
    assert [p["stance"] for p in tl["points"]] == [0.7, 0.7]
    assert tl["deviation"]["shifted"] is False and tl["deviation"]["sign_flips"] == 0


def test_min_points_filters_single_episode(tmp_path: Path) -> None:
    # One episode → fewer than min_points distinct dates → no timeline.
    bundles = [_bundle(tmp_path / "e1", "e1", _ep_gi("2024-01-01", "ai is bad"))]
    scorer = _scorer({"ai is bad": (0.1, 0.9)})
    data = _run(StanceTimelineEnricher(scorer), bundles)
    assert data["timelines"] == []


def test_manifest_ml_tier_and_gate() -> None:
    manifest = StanceTimelineEnricher(FixedNliScorer()).manifest
    assert manifest.id == "stance_timeline"
    assert manifest.tier.value == "ml" and manifest.scope.value == "corpus"
    rule = manifest.accuracy_gate.rules[0]  # type: ignore[union-attr]
    assert (rule.metric_name, rule.min_value) == ("precision", 0.5)
    assert manifest.accuracy_gate.on_missing_data == "reject"  # type: ignore[union-attr]


def test_admission_gates_dark_until_an_eval_clears_it() -> None:
    specs = gate_specs_from_manifests(known_enricher_manifests())
    assert "stance_timeline" in specs  # known to the gate
    # No eval data → rejected (on_missing_data=reject).
    assert "stance_timeline" not in admitted_enricher_ids(["stance_timeline"], specs, {}).admitted
    # Passing precision → auto-promotes with no code edit.
    passing = admitted_enricher_ids(
        ["stance_timeline"], specs, {"stance_timeline": {"precision": 0.6}}
    )
    assert "stance_timeline" in passing.admitted
