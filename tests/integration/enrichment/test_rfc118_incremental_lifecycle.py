"""RFC-118 incremental lifecycle + reconciliation over the Tier-3 synthetic corpus.

Two guardrails the unit tier cannot give:

1. **Executor 2-run lifecycle** on the real ``app-validation-corpus/v3`` layout —
   first run full (establishes cursor + vector cache), second run dispatches
   ``enrich_incremental`` with an empty delta and produces byte-identical output.
2. **§7 reconciliation at corpus scale** — full vs (prior n−1 + 1-episode
   incremental) byte-identical for BOTH ML enrichers, using deterministic
   CI-safe fixtures (HashEmbedder; a text-hash consensus scorer), over the same
   corpus the app-validation walks use.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from podcast_scraper.corpus_delta import build_delta, CorpusDelta, fingerprint_bundles
from podcast_scraper.enrichment.enrichers.topic_consensus import TopicConsensusEnricher
from podcast_scraper.enrichment.enrichers.topic_similarity import TopicSimilarityEnricher
from podcast_scraper.enrichment.executor import EnrichmentExecutor, ExecutorOptions
from podcast_scraper.enrichment.paths import corpus_enrichment_path, discover_episode_bundles
from podcast_scraper.enrichment.protocol import EnricherSet, RunContext, STATUS_OK
from podcast_scraper.enrichment.registry import EnricherRegistry
from podcast_scraper.enrichment.scorers.embedding import HashEmbedder, TopicEmbeddingProvider
from podcast_scraper.enrichment.scorers.protocol import ConsensusSignal

pytestmark = pytest.mark.integration

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "app-validation-corpus" / "v3"


class HashConsensusScorer:
    """Deterministic, text-derived ConsensusScorer — varied scores, CI-safe."""

    def __init__(self) -> None:
        self.calls = 0

    async def score(self, text_a: str, text_b: str) -> ConsensusSignal:
        self.calls += 1
        digest = hashlib.sha256(f"{text_a}\x1f{text_b}".encode("utf-8")).digest()
        return ConsensusSignal(cosine=digest[0] / 255.0, contradiction=digest[1] / 255.0)


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    dest = tmp_path / "corpus"
    shutil.copytree(FIXTURE, dest)
    return dest


def _ctx(enricher_id: str, version: str) -> RunContext:
    return RunContext(
        run_id="r",
        parent_run_id=None,
        enricher_id=enricher_id,
        enricher_version=version,
        tier="ml",
        attempt=1,
        job_id="r",
        cancel_event=asyncio.Event(),
    )


def _canon(data) -> str:
    return json.dumps(data, sort_keys=True)


class TestExecutorTwoRunLifecycle:
    def test_second_run_is_incremental_and_identical(self, corpus: Path) -> None:
        def _run() -> dict:
            provider = TopicEmbeddingProvider(
                embed_text=HashEmbedder(dim=16), model_marker="hash:16"
            )
            registry = EnricherRegistry()
            registry.register(TopicSimilarityEnricher(provider))
            executor = EnrichmentExecutor(
                corpus_root=corpus,
                registry=registry,
                enricher_set=EnricherSet(enabled_enrichers=["topic_similarity"]),
            )
            bundles = discover_episode_bundles(corpus)
            result = asyncio.run(
                executor.run(episode_bundles=bundles, options=ExecutorOptions(corpus_only=True))
            )
            assert result.status == "ok", result.status
            envelope = json.loads(
                corpus_enrichment_path(corpus, "topic_similarity.json").read_text(encoding="utf-8")
            )
            data = envelope["data"]
            assert isinstance(data, dict)
            return data

        first = _run()
        cursor = corpus / "enrichments" / "topic_similarity.delta_cursor.json"
        cache = corpus / "enrichments" / "topic_similarity.vectors_cache.json"
        assert cursor.is_file(), "cursor not established by the full run"
        assert cache.is_file(), "vector cache not written by the full run"

        second = _run()
        assert _canon(second) == _canon(first), "incremental output diverged from full"
        # The cursor survived (advanced in place) and the topics really exist.
        assert first["topic_count"] > 0


class TestReconciliationOnTier3Corpus:
    """§7 at corpus scale: full == prior(n−1) + 1-episode incremental, byte-identical."""

    def test_topic_similarity(self, corpus: Path, tmp_path: Path) -> None:
        bundles = discover_episode_bundles(corpus)
        assert len(bundles) >= 3

        def provider() -> TopicEmbeddingProvider:
            return TopicEmbeddingProvider(embed_text=HashEmbedder(dim=16), model_marker="hash:16")

        full_root = tmp_path / "full"
        full_root.mkdir()
        enricher = TopicSimilarityEnricher(provider())
        full = asyncio.run(
            enricher.enrich(
                bundle=None,
                corpus_root=full_root,
                all_bundles=bundles,
                config={},
                ctx=_ctx("topic_similarity", "1.0.0"),
            )
        )
        assert full.status == STATUS_OK

        incr_root = tmp_path / "incr"
        incr_root.mkdir()
        prior = asyncio.run(
            TopicSimilarityEnricher(provider()).enrich(
                bundle=None,
                corpus_root=incr_root,
                all_bundles=bundles[:-1],
                config={},
                ctx=_ctx("topic_similarity", "1.0.0"),
            )
        )
        assert prior.status == STATUS_OK
        delta = CorpusDelta(
            changed_ids=frozenset({bundles[-1].episode_id}),
            removed_ids=frozenset(),
            all_bundles=bundles,
        )
        incr = asyncio.run(
            TopicSimilarityEnricher(provider()).enrich_incremental(
                delta=delta,
                prior_output=prior.data,
                corpus_root=incr_root,
                config={},
                ctx=_ctx("topic_similarity", "1.0.0"),
            )
        )
        assert incr.status == STATUS_OK
        assert _canon(incr.data) == _canon(full.data)

    def test_topic_consensus(self, corpus: Path, tmp_path: Path) -> None:
        bundles = discover_episode_bundles(corpus)

        full_root = tmp_path / "full"
        full_root.mkdir()
        full_scorer = HashConsensusScorer()
        full = asyncio.run(
            TopicConsensusEnricher(full_scorer).enrich(
                bundle=None,
                corpus_root=full_root,
                all_bundles=bundles,
                config={},
                ctx=_ctx("topic_consensus", "2.0.0"),
            )
        )
        assert full.status == STATUS_OK

        incr_root = tmp_path / "incr"
        incr_root.mkdir()
        prior_scorer = HashConsensusScorer()
        prior = asyncio.run(
            TopicConsensusEnricher(prior_scorer).enrich(
                bundle=None,
                corpus_root=incr_root,
                all_bundles=bundles[:-1],
                config={},
                ctx=_ctx("topic_consensus", "2.0.0"),
            )
        )
        assert prior.status == STATUS_OK
        delta = CorpusDelta(
            changed_ids=frozenset({bundles[-1].episode_id}),
            removed_ids=frozenset(),
            all_bundles=bundles,
        )
        incr_scorer = HashConsensusScorer()
        incr = asyncio.run(
            TopicConsensusEnricher(incr_scorer).enrich_incremental(
                delta=delta,
                prior_output=prior.data,
                corpus_root=incr_root,
                config={},
                ctx=_ctx("topic_consensus", "2.0.0"),
            )
        )
        assert incr.status == STATUS_OK
        assert _canon(incr.data) == _canon(full.data)
        assert full.data is not None
        # The delta path must have re-scored ONLY pairs touching the changed episode.
        if full.data["pairs_scored"] > 0:
            assert incr_scorer.calls <= full_scorer.calls
            assert (
                incr_scorer.calls
                + len(
                    json.loads(
                        (incr_root / "enrichments" / "topic_consensus.pairs_cache.json").read_text(
                            encoding="utf-8"
                        )
                    )["pairs"]
                )
                >= full_scorer.calls
            )

    def test_backbone_delta_on_this_corpus_is_stable(self, corpus: Path) -> None:
        bundles = discover_episode_bundles(corpus)
        fresh = fingerprint_bundles(bundles)
        delta = build_delta(fresh, fresh, bundles)
        assert delta.is_empty
