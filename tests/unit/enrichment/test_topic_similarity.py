"""Unit tests for the chunk-3 ``topic_similarity`` enricher + embedding providers."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from podcast_scraper.enrichment.enrichers.topic_similarity import (
    _cosine,
    TopicSimilarityEnricher,
)
from podcast_scraper.enrichment.protocol import (
    EpisodeArtifactBundle,
    RunContext,
    STATUS_OK,
)
from podcast_scraper.enrichment.scorers.embedding import (
    AsyncTopicEmbeddingProvider,
    HashEmbedder,
    TopicEmbeddingProvider,
)
from podcast_scraper.enrichment.scorers.protocol import EmbeddingProvider


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


# ---------------------------------------------------------------------------
# _cosine
# ---------------------------------------------------------------------------


def test_cosine_identical_vectors_is_one() -> None:
    assert _cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_orthogonal_is_zero() -> None:
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_handles_zero_vector() -> None:
    assert _cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_cosine_mismatched_dims_is_zero() -> None:
    assert _cosine([1.0, 0.0], [1.0]) == 0.0


# ---------------------------------------------------------------------------
# HashEmbedder
# ---------------------------------------------------------------------------


def test_hash_embedder_is_deterministic() -> None:
    embedder = HashEmbedder(dim=16)
    a = embedder("topic-alpha")
    b = embedder("topic-alpha")
    assert a == b
    assert len(a) == 16


def test_hash_embedder_different_text_different_vector() -> None:
    embedder = HashEmbedder(dim=16)
    assert embedder("foo") != embedder("bar")


def test_hash_embedder_empty_is_zero() -> None:
    embedder = HashEmbedder(dim=8)
    assert embedder("") == [0.0] * 8


def test_hash_embedder_dim_must_be_in_range() -> None:
    with pytest.raises(ValueError):
        HashEmbedder(dim=2)
    with pytest.raises(ValueError):
        HashEmbedder(dim=4096)


# ---------------------------------------------------------------------------
# TopicEmbeddingProvider
# ---------------------------------------------------------------------------


def test_topic_provider_returns_vector_for_known_label() -> None:
    embedder = HashEmbedder(dim=8)
    provider = TopicEmbeddingProvider(
        embed_text=embedder, labels={"topic:a": "Alpha", "topic:b": "Beta"}
    )
    vec_a = asyncio.run(provider.topic_vector("topic:a"))
    vec_b = asyncio.run(provider.topic_vector("topic:b"))
    assert vec_a is not None and len(vec_a) == 8
    assert vec_a != vec_b


def test_topic_provider_caches_per_topic_id() -> None:
    calls: list[str] = []

    def counting_embed(text: str) -> list[float]:
        calls.append(text)
        return [1.0, 0.0, 0.0]

    provider = TopicEmbeddingProvider(embed_text=counting_embed, labels={"topic:a": "Alpha"})
    asyncio.run(provider.topic_vector("topic:a"))
    asyncio.run(provider.topic_vector("topic:a"))
    assert calls == ["Alpha"]  # one call total — cached


def test_topic_provider_falls_back_to_id_when_label_missing() -> None:
    embedder = HashEmbedder(dim=8)
    provider = TopicEmbeddingProvider(embed_text=embedder)
    # No labels map → uses the part after the colon (`alpha`).
    vec = asyncio.run(provider.topic_vector("topic:alpha"))
    assert vec is not None
    # Same as embedding 'alpha' directly.
    assert vec == embedder("alpha")


def test_async_topic_provider_uses_awaitable_backend() -> None:
    async def async_embed(text: str) -> list[float]:
        return [0.5 if c == "a" else 0.0 for c in text]

    provider = AsyncTopicEmbeddingProvider(embed_text=async_embed, labels={"topic:a": "aaa"})
    vec = asyncio.run(provider.topic_vector("topic:a"))
    assert vec == [0.5, 0.5, 0.5]


# Runtime protocol smoke.
def test_providers_satisfy_protocol() -> None:
    embedder = HashEmbedder(dim=4)
    p = TopicEmbeddingProvider(embed_text=embedder)
    assert isinstance(p, EmbeddingProvider)

    async def _async(text: str) -> list[float]:
        return [1.0]

    a = AsyncTopicEmbeddingProvider(embed_text=_async)
    assert isinstance(a, EmbeddingProvider)


# ---------------------------------------------------------------------------
# TopicSimilarityEnricher
# ---------------------------------------------------------------------------


def _run(enricher: TopicSimilarityEnricher, bundles: list[EpisodeArtifactBundle]) -> dict[str, Any]:
    result = asyncio.run(
        enricher.enrich(
            bundle=None,
            corpus_root=Path("/tmp"),
            all_bundles=bundles,
            config={},
            ctx=_ctx(),
        )
    )
    assert result.status == STATUS_OK, f"status={result.status!r} error={result.error}"
    assert isinstance(result.data, dict)
    return result.data


def test_topic_similarity_top_k_default_includes_every_other_topic(tmp_path: Path) -> None:
    bundles = [
        _bundle(tmp_path / "metadata", "ep1", ["topic:a", "topic:b", "topic:c"]),
    ]
    embedder = HashEmbedder(dim=16)
    provider = TopicEmbeddingProvider(
        embed_text=embedder, labels={"topic:a": "Alpha", "topic:b": "Beta", "topic:c": "Gamma"}
    )
    enricher = TopicSimilarityEnricher(provider, top_k=10)
    data = _run(enricher, bundles)
    assert data["topic_count"] == 3
    assert data["top_k"] == 10
    for row in data["topics"]:
        assert len(row["top_k"]) == 2  # 3 topics total → 2 neighbours each


def test_topic_similarity_caps_at_top_k(tmp_path: Path) -> None:
    bundles = [
        _bundle(
            tmp_path / "metadata",
            "ep1",
            [f"topic:{c}" for c in "abcdefghij"],
        )
    ]
    embedder = HashEmbedder(dim=16)
    provider = TopicEmbeddingProvider(embed_text=embedder)
    enricher = TopicSimilarityEnricher(provider, top_k=3)
    data = _run(enricher, bundles)
    for row in data["topics"]:
        assert len(row["top_k"]) == 3


def test_topic_similarity_records_missing_topics(tmp_path: Path) -> None:
    bundles = [_bundle(tmp_path / "metadata", "ep1", ["topic:a", "topic:b"])]

    class PartialProvider:
        async def topic_vector(self, topic_id: str) -> list[float] | None:
            return [1.0, 0.0, 0.0] if topic_id == "topic:a" else None

    enricher = TopicSimilarityEnricher(PartialProvider())
    data = _run(enricher, bundles)
    assert data["missing_topic_ids"] == ["topic:b"]
    # Only topic:a survives — but with one topic, it has no neighbours.
    assert len(data["topics"]) == 1
    assert data["topics"][0]["top_k"] == []


def test_topic_similarity_empty_corpus_returns_empty(tmp_path: Path) -> None:
    bundles = [_bundle(tmp_path / "metadata", "ep1", [])]
    enricher = TopicSimilarityEnricher(TopicEmbeddingProvider(embed_text=HashEmbedder()))
    data = _run(enricher, bundles)
    assert data["topics"] == []
    assert data["topic_count"] == 0


def test_enricher_feeds_labels_to_provider_not_id_slug(tmp_path: Path) -> None:
    """The enricher must feed the KG id->label map to the provider so it embeds the human
    label ("AI development"), not the id slug ("ai-development"). The profile-wired provider is
    built without corpus access, so the enricher is the only path that can supply labels."""
    meta = tmp_path / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "ep.metadata.json").write_text("{}", encoding="utf-8")
    (meta / "ep.kg.json").write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "type": "Topic",
                        "id": "topic:ai-development",
                        "properties": {"label": "AI development"},
                    },
                    {
                        "type": "Topic",
                        "id": "topic:oil-prices",
                        "properties": {"label": "Oil prices"},
                    },
                ],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )
    bundle = EpisodeArtifactBundle(
        metadata_path=meta / "ep.metadata.json",
        gi_path=None,
        kg_path=meta / "ep.kg.json",
        bridge_path=None,
        episode_id="episode:ep",
        stem="ep",
    )
    embedded: list[str] = []

    def recording_embed(text: str) -> list[float]:
        embedded.append(text)
        return [float(len(text)), 1.0]

    # Provider built WITHOUT labels — mirrors the profile-wired provider (no corpus access).
    enricher = TopicSimilarityEnricher(TopicEmbeddingProvider(embed_text=recording_embed))
    data = _run(enricher, [bundle])
    assert data["topic_count"] == 2
    assert "AI development" in embedded and "Oil prices" in embedded
    assert "ai-development" not in embedded and "topic:ai-development" not in embedded


def test_topic_similarity_top_k_config_override(tmp_path: Path) -> None:
    bundles = [_bundle(tmp_path / "metadata", "ep1", ["topic:a", "topic:b", "topic:c", "topic:d"])]
    enricher = TopicSimilarityEnricher(TopicEmbeddingProvider(embed_text=HashEmbedder()))
    # config['top_k'] overrides constructor default.
    result = asyncio.run(
        enricher.enrich(
            bundle=None,
            corpus_root=tmp_path,
            all_bundles=bundles,
            config={"top_k": 2},
            ctx=_ctx(),
        )
    )
    assert result.status == STATUS_OK
    assert isinstance(result.data, dict)
    for row in result.data["topics"]:
        assert len(row["top_k"]) == 2


def test_top_k_must_be_positive() -> None:
    with pytest.raises(ValueError):
        TopicSimilarityEnricher(TopicEmbeddingProvider(embed_text=HashEmbedder()), top_k=0)


def test_topic_similarity_manifest_is_embedding_tier() -> None:
    enricher = TopicSimilarityEnricher(TopicEmbeddingProvider(embed_text=HashEmbedder()))
    assert enricher.manifest.id == "topic_similarity"
    assert enricher.manifest.tier.value == "embedding"
    assert enricher.manifest.scope.value == "corpus"


def test_topic_similarity_sets_records_written(tmp_path: Path) -> None:
    # Regression: async enrichers return EnricherResult directly (no @sync_enricher wrapper), so
    # they must set records_written themselves — it was always 0, so the run-summary under-reported
    # the ML enrichers as producing nothing (found on prod-pilot 2026-07-01; #1127 Bug-5 twin).
    bundles = [_bundle(tmp_path / "metadata", "ep1", ["topic:a", "topic:b", "topic:c"])]
    enricher = TopicSimilarityEnricher(TopicEmbeddingProvider(embed_text=HashEmbedder(dim=16)))
    result = asyncio.run(
        enricher.enrich(
            bundle=None, corpus_root=tmp_path, all_bundles=bundles, config={}, ctx=_ctx()
        )
    )
    assert result.status == STATUS_OK
    assert isinstance(result.data, dict)
    assert result.records_written == len(result.data["topics"]) == 3
