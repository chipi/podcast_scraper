"""Integration smoke: ``nli_contradiction`` through the executor."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from podcast_scraper.enrichment.enrichers.nli_contradiction import NliContradictionEnricher
from podcast_scraper.enrichment.executor import EnrichmentExecutor
from podcast_scraper.enrichment.protocol import (
    EnricherSet,
    EpisodeArtifactBundle,
    STATUS_OK,
)
from podcast_scraper.enrichment.registry import EnricherRegistry
from podcast_scraper.enrichment.scorers.nli import FixedNliScorer
from podcast_scraper.enrichment.scorers.protocol import NliScore

pytestmark = pytest.mark.integration


def _bundle(corpus_root: Path, stem: str) -> EpisodeArtifactBundle:
    meta_dir = corpus_root / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    md = meta_dir / f"{stem}.metadata.json"
    md.write_text("{}", encoding="utf-8")
    gi_path = meta_dir / f"{stem}.gi.json"
    gi = {
        "nodes": [
            {"type": "Person", "id": "person:alice", "properties": {"name": "Alice"}},
            {"type": "Person", "id": "person:bob", "properties": {"name": "Bob"}},
            {"type": "Insight", "id": f"insight:{stem}-1", "properties": {"text": "AI is safe"}},
            {
                "type": "Insight",
                "id": f"insight:{stem}-2",
                "properties": {"text": "AI is dangerous"},
            },
            {"type": "Quote", "id": f"quote:{stem}-1"},
            {"type": "Quote", "id": f"quote:{stem}-2"},
        ],
        "edges": [
            {"type": "SPOKEN_BY", "from": f"quote:{stem}-1", "to": "person:alice"},
            {"type": "SPOKEN_BY", "from": f"quote:{stem}-2", "to": "person:bob"},
            {"type": "SUPPORTED_BY", "from": f"insight:{stem}-1", "to": f"quote:{stem}-1"},
            {"type": "SUPPORTED_BY", "from": f"insight:{stem}-2", "to": f"quote:{stem}-2"},
            {"type": "ABOUT", "from": f"insight:{stem}-1", "to": "topic:ai"},
            {"type": "ABOUT", "from": f"insight:{stem}-2", "to": "topic:ai"},
        ],
    }
    gi_path.write_text(json.dumps(gi), encoding="utf-8")
    return EpisodeArtifactBundle(
        metadata_path=md,
        gi_path=gi_path,
        kg_path=None,
        bridge_path=None,
        episode_id=f"episode:{stem}",
        stem=stem,
    )


def test_executor_runs_nli_contradiction_with_fixed_scorer(tmp_path: Path) -> None:
    bundles = [_bundle(tmp_path, "ep1")]
    scorer = FixedNliScorer(scores={("AI is safe", "AI is dangerous"): NliScore(0.93, 0.04, 0.03)})
    enricher = NliContradictionEnricher(scorer)
    reg = EnricherRegistry()
    reg.register(enricher)
    eset = EnricherSet(enabled_enrichers=["nli_contradiction"])
    executor = EnrichmentExecutor(corpus_root=tmp_path, registry=reg, enricher_set=eset)
    result = asyncio.run(executor.run(episode_bundles=bundles))
    assert result.status == STATUS_OK
    m = result.per_enricher_metrics["nli_contradiction"]
    assert m.runs_ok == 1
    out = tmp_path / "enrichments" / "nli_contradiction.json"
    assert out.is_file()
    payload = json.loads(out.read_text(encoding="utf-8"))
    contradictions = payload["data"]["contradictions"]
    assert len(contradictions) == 1
    assert contradictions[0]["topic_id"] == "topic:ai"
    assert contradictions[0]["contradiction_score"] >= 0.9


def test_ml_tier_retry_recovers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ML-tier policy: 2 retries. Scorer raises once then succeeds."""
    monkeypatch.setattr("podcast_scraper.enrichment.executor.compute_backoff", lambda *a, **kw: 0.0)

    state = {"calls": 0}

    class FlakyScorer:
        async def score(self, premise: str, hypothesis: str) -> NliScore:
            state["calls"] += 1
            if state["calls"] == 1:
                from podcast_scraper.enrichment.resilience import ScorerTimeoutError

                raise ScorerTimeoutError("transient NLI timeout")
            return NliScore(0.8, 0.1, 0.1)

    enricher = NliContradictionEnricher(FlakyScorer())
    reg = EnricherRegistry()
    reg.register(enricher)
    eset = EnricherSet(enabled_enrichers=["nli_contradiction"])
    bundles = [_bundle(tmp_path, "ep1")]
    result = asyncio.run(
        EnrichmentExecutor(corpus_root=tmp_path, registry=reg, enricher_set=eset).run(
            episode_bundles=bundles
        )
    )
    assert result.status == STATUS_OK
    events = [
        json.loads(line)
        for line in (tmp_path / "enrichments" / "run.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    retries = [e for e in events if e["event_type"] == "enrichment.enricher.retry"]
    assert len(retries) >= 1
