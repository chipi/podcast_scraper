"""Integration smoke: ``topic_consensus`` through the executor.

The ML tier of ``topic_consensus`` (embedding cosine + NLI contradiction)
can't run in CI, so this exercises the *emission* path with a scripted
:class:`FixedConsensusScorer` — the same CI-safe stub the unit tests use —
driven end-to-end through :class:`EnrichmentExecutor`: register the enricher,
run it over a two-speaker corpus, and assert the ``topic_consensus.json``
envelope lands on disk with a real cross-person corroboration pair (not just
``runs_ok``). Companion to ``test_topic_similarity_executor_smoke.py``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from podcast_scraper.enrichment.enrichers.topic_consensus import TopicConsensusEnricher
from podcast_scraper.enrichment.executor import EnrichmentExecutor
from podcast_scraper.enrichment.protocol import (
    EnricherSet,
    EpisodeArtifactBundle,
    STATUS_OK,
)
from podcast_scraper.enrichment.registry import EnricherRegistry
from podcast_scraper.enrichment.scorers.consensus import FixedConsensusScorer
from podcast_scraper.enrichment.scorers.protocol import ConsensusSignal

pytestmark = pytest.mark.integration

_TA, _TB = "diversify to survive", "spread risk to survive"


def _two_speaker_bundle(meta_dir: Path, stem: str) -> EpisodeArtifactBundle:
    """Alice + Bob each make one Insight on topic:risk (mirrors the unit fixture)."""
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / f"{stem}.metadata.json").write_text("{}", encoding="utf-8")
    gi_path = meta_dir / f"{stem}.gi.json"
    gi_path.write_text(
        json.dumps(
            {
                "nodes": [
                    {"type": "Person", "id": "person:alice", "properties": {"name": "Alice"}},
                    {"type": "Person", "id": "person:bob", "properties": {"name": "Bob"}},
                    {"type": "Insight", "id": "insight:a", "properties": {"text": _TA}},
                    {"type": "Insight", "id": "insight:b", "properties": {"text": _TB}},
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
        ),
        encoding="utf-8",
    )
    return EpisodeArtifactBundle(
        metadata_path=meta_dir / f"{stem}.metadata.json",
        gi_path=gi_path,
        kg_path=None,
        bridge_path=None,
        episode_id=f"episode:{stem}",
        stem=stem,
    )


def test_executor_emits_topic_consensus_end_to_end(tmp_path: Path) -> None:
    # Passing signal: cosine ≥ 0.70 + contradiction ≤ 0.5 → admitted as corroboration.
    scorer = FixedConsensusScorer(
        signals={(_TA, _TB): ConsensusSignal(cosine=0.82, contradiction=0.03)}
    )
    reg = EnricherRegistry()
    reg.register(TopicConsensusEnricher(scorer))
    eset = EnricherSet(enabled_enrichers=["topic_consensus"])

    result = asyncio.run(
        EnrichmentExecutor(corpus_root=tmp_path, registry=reg, enricher_set=eset).run(
            episode_bundles=[_two_speaker_bundle(tmp_path / "metadata", "ep1")]
        )
    )

    assert result.status == STATUS_OK
    assert result.per_enricher_metrics["topic_consensus"].runs_ok == 1

    out = tmp_path / "enrichments" / "topic_consensus.json"
    assert out.is_file()
    pairs: list[dict[str, Any]] = json.loads(out.read_text(encoding="utf-8"))["data"]["consensus"]
    assert pairs, "no consensus pair emitted for two agreeing speakers"
    p = pairs[0]
    assert {p["person_a_id"], p["person_b_id"]} == {"person:alice", "person:bob"}
    assert p["topic_id"] == "topic:risk"


def test_executor_consensus_gate_rejects_contradiction(tmp_path: Path) -> None:
    """High contradiction → the same pair is not admitted as consensus (gate works)."""
    scorer = FixedConsensusScorer(
        signals={(_TA, _TB): ConsensusSignal(cosine=0.82, contradiction=0.95)}
    )
    reg = EnricherRegistry()
    reg.register(TopicConsensusEnricher(scorer))
    eset = EnricherSet(enabled_enrichers=["topic_consensus"])

    result = asyncio.run(
        EnrichmentExecutor(corpus_root=tmp_path, registry=reg, enricher_set=eset).run(
            episode_bundles=[_two_speaker_bundle(tmp_path / "metadata", "ep1")]
        )
    )

    assert result.status == STATUS_OK
    pairs = json.loads(
        (tmp_path / "enrichments" / "topic_consensus.json").read_text(encoding="utf-8")
    )["data"]["consensus"]
    assert pairs == [], f"contradicting pair should not be consensus: {pairs}"
