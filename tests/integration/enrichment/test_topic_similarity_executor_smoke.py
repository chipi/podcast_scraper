"""Integration smoke: ``topic_similarity`` through the executor.

Exercises the EMBEDDING-tier path: registers a single enricher built
from :class:`TopicSimilarityEnricher` + a deterministic
``HashEmbedder``, runs it via :class:`EnrichmentExecutor`, asserts the
envelope is on disk and the per-enricher metrics report runs_ok=1.

A companion test injects the chunk-1 ``MockEmbeddingProvider`` with a
script that raises once then succeeds, verifying that the EMBEDDING
tier's 3-retry policy recovers the run.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from podcast_scraper.enrichment.enrichers.topic_similarity import TopicSimilarityEnricher
from podcast_scraper.enrichment.executor import EnrichmentExecutor
from podcast_scraper.enrichment.protocol import (
    EnricherSet,
    EpisodeArtifactBundle,
    STATUS_OK,
)
from podcast_scraper.enrichment.registry import EnricherRegistry
from podcast_scraper.enrichment.scorers.embedding import HashEmbedder, TopicEmbeddingProvider

pytestmark = pytest.mark.integration


def _bundle(meta_dir: Path, stem: str, topic_ids: list[str]) -> EpisodeArtifactBundle:
    meta_dir.mkdir(parents=True, exist_ok=True)
    md = meta_dir / f"{stem}.metadata.json"
    md.write_text("{}", encoding="utf-8")
    kg = meta_dir / f"{stem}.kg.json"
    kg.write_text(
        json.dumps(
            {
                "nodes": [
                    {"type": "Topic", "id": tid, "properties": {"label": tid.split(":")[-1]}}
                    for tid in topic_ids
                ],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )
    return EpisodeArtifactBundle(
        metadata_path=md,
        gi_path=None,
        kg_path=kg,
        bridge_path=None,
        episode_id=f"episode:{stem}",
        stem=stem,
    )


def test_executor_runs_topic_similarity_end_to_end(tmp_path: Path) -> None:
    bundles = [_bundle(tmp_path / "metadata", "ep1", ["topic:a", "topic:b", "topic:c"])]
    enricher = TopicSimilarityEnricher(TopicEmbeddingProvider(embed_text=HashEmbedder(dim=16)))
    reg = EnricherRegistry()
    reg.register(enricher)
    eset = EnricherSet(enabled_enrichers=["topic_similarity"])
    executor = EnrichmentExecutor(corpus_root=tmp_path, registry=reg, enricher_set=eset)
    result = asyncio.run(executor.run(episode_bundles=bundles))
    assert result.status == STATUS_OK
    m = result.per_enricher_metrics["topic_similarity"]
    assert m.runs_ok == 1
    out = tmp_path / "enrichments" / "topic_similarity.json"
    assert out.is_file()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["data"]["topic_count"] == 3


def test_embedding_tier_recovers_via_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Embedding-tier policy: 3 retries. Provider raises once → executor
    retries and the second attempt succeeds."""
    monkeypatch.setattr("podcast_scraper.enrichment.executor.compute_backoff", lambda *a, **kw: 0.0)

    state = {"fail_calls": 0}
    base = HashEmbedder(dim=16)

    def flaky_embed(text: str) -> list[float]:
        # Fail on the first call only.
        if state["fail_calls"] < 1:
            state["fail_calls"] += 1
            from podcast_scraper.enrichment.resilience import DependencyAccessError

            raise DependencyAccessError("transient embed backend")
        return base(text)

    enricher = TopicSimilarityEnricher(TopicEmbeddingProvider(embed_text=flaky_embed))
    bundles = [_bundle(tmp_path / "metadata", "ep1", ["topic:a", "topic:b"])]
    reg = EnricherRegistry()
    reg.register(enricher)
    eset = EnricherSet(enabled_enrichers=["topic_similarity"])
    result = asyncio.run(
        EnrichmentExecutor(corpus_root=tmp_path, registry=reg, enricher_set=eset).run(
            episode_bundles=bundles
        )
    )
    assert result.status == STATUS_OK
    # First attempt's enricher body raised before returning; retry succeeded.
    events_path = tmp_path / "enrichments" / "run.jsonl"
    events: list[dict[str, Any]] = [
        json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line
    ]
    retry_events = [e for e in events if e["event_type"] == "enrichment.enricher.retry"]
    assert len(retry_events) >= 1
