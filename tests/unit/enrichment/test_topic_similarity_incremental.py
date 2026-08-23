"""RFC-118 §7 reconciliation gate for ``topic_similarity`` incremental.

Incremental with a delta must be canonical-JSON identical to a full pass over the
same corpus, while re-embedding only new/relabelled topics. Label equality plus the
provider's ``model_marker`` is the invalidation; an unmarked provider never reuses.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from podcast_scraper.corpus_delta import CorpusDelta
from podcast_scraper.enrichment.enrichers.topic_similarity import (
    TopicSimilarityEnricher,
    vector_cache_path,
)
from podcast_scraper.enrichment.protocol import EpisodeArtifactBundle, RunContext, STATUS_OK
from podcast_scraper.enrichment.scorers.embedding import HashEmbedder, TopicEmbeddingProvider

pytestmark = pytest.mark.unit


class CountingEmbedder:
    """HashEmbedder that counts embed calls (the thing the vector cache must cut)."""

    def __init__(self) -> None:
        self._inner = HashEmbedder(dim=16)
        self.calls = 0

    def __call__(self, text: str) -> list[float]:
        self.calls += 1
        return self._inner(text)


def _provider(marker: str = "fake:test") -> tuple[TopicEmbeddingProvider, CountingEmbedder]:
    embedder = CountingEmbedder()
    return TopicEmbeddingProvider(embed_text=embedder, model_marker=marker), embedder


def _kg(topics: dict[str, str]) -> dict[str, Any]:
    return {
        "nodes": [
            {"type": "Topic", "id": tid, "properties": {"label": label}}
            for tid, label in topics.items()
        ],
        "edges": [],
    }


def _bundle(corpus: Path, stem: str, kg: dict[str, Any]) -> EpisodeArtifactBundle:
    meta_dir = corpus / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / f"{stem}.metadata.json").write_text("{}", encoding="utf-8")
    kg_path = meta_dir / f"{stem}.kg.json"
    kg_path.write_text(json.dumps(kg), encoding="utf-8")
    return EpisodeArtifactBundle(
        metadata_path=meta_dir / f"{stem}.metadata.json",
        gi_path=None,
        kg_path=kg_path,
        bridge_path=None,
        episode_id=f"episode:{stem}",
        stem=stem,
    )


def _episodes(corpus: Path) -> list[EpisodeArtifactBundle]:
    return [
        _bundle(corpus, "e1", _kg({"topic:ai": "AI development", "topic:risk": "Model risk"})),
        _bundle(corpus, "e2", _kg({"topic:open": "Open source"})),
        _bundle(corpus, "e3", _kg({"topic:agents": "Agentic systems"})),
    ]


def _ctx() -> RunContext:
    return RunContext(
        run_id="r1",
        parent_run_id=None,
        enricher_id="topic_similarity",
        enricher_version="1.0.0",
        tier="embedding",
        attempt=1,
        job_id="r1",
        cancel_event=asyncio.Event(),
    )


def _full(corpus: Path, bundles, provider) -> dict[str, Any]:
    result = asyncio.run(
        TopicSimilarityEnricher(provider).enrich(
            bundle=None, corpus_root=corpus, all_bundles=bundles, config={}, ctx=_ctx()
        )
    )
    assert result.status == STATUS_OK and isinstance(result.data, dict)
    return result.data


def _incremental(corpus: Path, delta: CorpusDelta, provider) -> dict[str, Any]:
    result = asyncio.run(
        TopicSimilarityEnricher(provider).enrich_incremental(
            delta=delta, prior_output=None, corpus_root=corpus, config={}, ctx=_ctx()
        )
    )
    assert result.status == STATUS_OK and isinstance(result.data, dict)
    return result.data


def _canon(data: dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True)


class TestReconciliation:
    def test_one_episode_delta_matches_full(self, tmp_path):
        full_corpus = tmp_path / "full"
        provider_full, _ = _provider()
        full_data = _full(full_corpus, _episodes(full_corpus), provider_full)

        incr_corpus = tmp_path / "incr"
        bundles = _episodes(incr_corpus)
        provider_prior, prior_embedder = _provider()
        _full(incr_corpus, bundles[:2], provider_prior)
        assert prior_embedder.calls == 3  # ai, risk, open

        delta = CorpusDelta(
            changed_ids=frozenset({"episode:e3"}),
            removed_ids=frozenset(),
            all_bundles=bundles,
        )
        provider_incr, incr_embedder = _provider()
        incr_data = _incremental(incr_corpus, delta, provider_incr)

        assert _canon(incr_data) == _canon(full_data)
        assert incr_embedder.calls == 1  # only the new topic:agents embeds

    def test_relabelled_topic_re_embeds(self, tmp_path):
        corpus = tmp_path / "c"
        bundles = _episodes(corpus)
        provider1, _ = _provider()
        _full(corpus, bundles, provider1)

        # Relabel one topic in place — same topic_id, different label text.
        assert bundles[1].kg_path is not None
        bundles[1].kg_path.write_text(
            json.dumps(_kg({"topic:open": "Open-source ecosystems"})), encoding="utf-8"
        )
        delta = CorpusDelta(
            changed_ids=frozenset({"episode:e2"}), removed_ids=frozenset(), all_bundles=bundles
        )
        provider2, embedder2 = _provider()
        _incremental(corpus, delta, provider2)
        assert embedder2.calls == 1  # only the relabelled topic

    def test_marker_mismatch_reuses_nothing(self, tmp_path):
        corpus = tmp_path / "c"
        bundles = _episodes(corpus)
        provider1, _ = _provider(marker="model-A")
        _full(corpus, bundles, provider1)

        delta = CorpusDelta(changed_ids=frozenset(), removed_ids=frozenset(), all_bundles=bundles)
        provider2, embedder2 = _provider(marker="model-B")
        _incremental(corpus, delta, provider2)
        assert embedder2.calls == 4  # all topics re-embedded

    def test_unmarked_provider_never_caches(self, tmp_path):
        corpus = tmp_path / "c"
        bundles = _episodes(corpus)
        provider1, _ = _provider(marker="")
        _full(corpus, bundles, provider1)
        assert not vector_cache_path(corpus).exists()

    def test_forced_delta_ignores_cache(self, tmp_path):
        corpus = tmp_path / "c"
        bundles = _episodes(corpus)
        provider1, _ = _provider()
        _full(corpus, bundles, provider1)

        delta = CorpusDelta(
            changed_ids=frozenset(b.episode_id for b in bundles),
            removed_ids=frozenset(),
            all_bundles=bundles,
            forced=True,
        )
        provider2, embedder2 = _provider()
        _incremental(corpus, delta, provider2)
        assert embedder2.calls == 4

    def test_manifest_declares_incremental(self):
        assert TopicSimilarityEnricher.manifest.supports_incremental is True
