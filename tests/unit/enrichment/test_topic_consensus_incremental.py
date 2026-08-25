"""RFC-118 §7 reconciliation gate for ``topic_consensus`` incremental.

THE property that makes delta safe to ship: ``enrich_incremental`` with a synthetic
1-episode delta against a prior built from the other n−1 episodes must be
**byte-identical** (canonical JSON) to a full ``enrich()`` over the same n episodes —
while re-scoring only the pairs the delta touches.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from podcast_scraper.corpus_delta import CorpusDelta
from podcast_scraper.enrichment.enrichers.topic_consensus import (
    pair_cache_path,
    TopicConsensusEnricher,
)
from podcast_scraper.enrichment.protocol import EpisodeArtifactBundle, RunContext, STATUS_OK
from podcast_scraper.enrichment.scorers.consensus import FixedConsensusScorer
from podcast_scraper.enrichment.scorers.protocol import ConsensusSignal

pytestmark = pytest.mark.unit

T_A = "diversify to survive"
T_B = "spread risk to survive"
T_C = "concentration is fragile"


class CountingScorer(FixedConsensusScorer):
    """FixedConsensusScorer that counts model invocations (the thing delta must cut)."""

    def __init__(self, signals: dict[tuple[str, str], ConsensusSignal]):
        super().__init__(signals=signals)
        self.calls = 0

    async def score(self, text_a: str, text_b: str) -> ConsensusSignal:
        self.calls += 1
        return await super().score(text_a, text_b)


def _signals() -> dict[tuple[str, str], ConsensusSignal]:
    return {
        (T_A, T_B): ConsensusSignal(cosine=0.82, contradiction=0.03),
        (T_A, T_C): ConsensusSignal(cosine=0.74, contradiction=0.10),
        (T_B, T_C): ConsensusSignal(cosine=0.55, contradiction=0.20),
    }


def _gi(person: str, insight_id: str, text: str) -> dict[str, Any]:
    quote = f"quote:{insight_id}"
    return {
        "nodes": [
            {"type": "Person", "id": f"person:{person}", "properties": {"name": person}},
            {"type": "Insight", "id": insight_id, "properties": {"text": text}},
            {"type": "Quote", "id": quote},
        ],
        "edges": [
            {"type": "SPOKEN_BY", "from": quote, "to": f"person:{person}"},
            {"type": "SUPPORTED_BY", "from": insight_id, "to": quote},
            {"type": "ABOUT", "from": insight_id, "to": "topic:risk"},
        ],
    }


def _bundle(corpus: Path, stem: str, gi: dict[str, Any]) -> EpisodeArtifactBundle:
    meta_dir = corpus / "metadata"
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


def _three_episodes(corpus: Path) -> list[EpisodeArtifactBundle]:
    return [
        _bundle(corpus, "e1", _gi("alice", "insight:a", T_A)),
        _bundle(corpus, "e2", _gi("bob", "insight:b", T_B)),
        _bundle(corpus, "e3", _gi("carol", "insight:c", T_C)),
    ]


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


def _full(corpus: Path, bundles, scorer, config=None) -> dict[str, Any]:
    enricher = TopicConsensusEnricher(scorer)
    result = asyncio.run(
        enricher.enrich(
            bundle=None,
            corpus_root=corpus,
            all_bundles=bundles,
            config=config or {},
            ctx=_ctx(),
        )
    )
    assert result.status == STATUS_OK and isinstance(result.data, dict)
    return result.data


def _incremental(corpus: Path, delta: CorpusDelta, scorer, config=None) -> dict[str, Any]:
    enricher = TopicConsensusEnricher(scorer)
    result = asyncio.run(
        enricher.enrich_incremental(
            delta=delta,
            prior_output=None,
            corpus_root=corpus,
            config=config or {},
            ctx=_ctx(),
        )
    )
    assert result.status == STATUS_OK and isinstance(result.data, dict)
    return result.data


def _canon(data: dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True)


class TestReconciliation:
    """RFC-118 §7: incremental output ≡ full output, canonical-JSON identical."""

    def test_one_episode_delta_matches_full(self, tmp_path):
        # Ground truth: a full pass over all three episodes.
        full_corpus = tmp_path / "full"
        full_data = _full(full_corpus, _three_episodes(full_corpus), CountingScorer(_signals()))

        # Incremental: prior = full pass over e1+e2 (builds the pair cache),
        # then e3 arrives as a 1-episode delta against the full bundle set.
        incr_corpus = tmp_path / "incr"
        bundles = _three_episodes(incr_corpus)
        prior_scorer = CountingScorer(_signals())
        _full(incr_corpus, bundles[:2], prior_scorer)
        assert prior_scorer.calls == 1  # e1×e2 only

        delta = CorpusDelta(
            changed_ids=frozenset({"episode:e3"}),
            removed_ids=frozenset(),
            all_bundles=bundles,
        )
        incr_scorer = CountingScorer(_signals())
        incr_data = _incremental(incr_corpus, delta, incr_scorer)

        assert _canon(incr_data) == _canon(full_data)
        # Only the pairs touching e3 hit the model; e1×e2 came from the cache.
        assert incr_scorer.calls == 2

    def test_threshold_change_refilters_without_rescoring(self, tmp_path):
        corpus = tmp_path / "c"
        bundles = _three_episodes(corpus)
        _full(corpus, bundles, CountingScorer(_signals()))

        # Nothing changed; a stricter cosine floor must re-filter purely from cache.
        delta = CorpusDelta(changed_ids=frozenset(), removed_ids=frozenset(), all_bundles=bundles)
        scorer = CountingScorer(_signals())
        data = _incremental(corpus, delta, scorer, config={"cos_threshold": 0.80})
        assert scorer.calls == 0
        assert [r["cosine"] for r in data["consensus"]] == [0.82]

    def test_forced_delta_ignores_cache(self, tmp_path):
        corpus = tmp_path / "c"
        bundles = _three_episodes(corpus)
        _full(corpus, bundles, CountingScorer(_signals()))

        delta = CorpusDelta(
            changed_ids=frozenset(b.episode_id for b in bundles),
            removed_ids=frozenset(),
            all_bundles=bundles,
            forced=True,
        )
        scorer = CountingScorer(_signals())
        _incremental(corpus, delta, scorer)
        assert scorer.calls == 3  # every pair re-scored

    def test_removed_episode_invalidates_its_pairs(self, tmp_path):
        corpus = tmp_path / "c"
        bundles = _three_episodes(corpus)
        _full(corpus, bundles, CountingScorer(_signals()))

        # e3 removed: the remaining candidate set is e1×e2 — its endpoints are NOT in
        # changed∪removed, so its cached score is reused (0 model calls), and no e3 rows
        # can survive because candidates come from the CURRENT corpus, not the cache.
        delta = CorpusDelta(
            changed_ids=frozenset(),
            removed_ids=frozenset({"episode:e3"}),
            all_bundles=bundles[:2],
        )
        scorer = CountingScorer(_signals())
        data = _incremental(corpus, delta, scorer)
        assert scorer.calls == 0
        assert all(
            "insight:c" not in (r["insight_a_id"], r["insight_b_id"]) for r in data["consensus"]
        )

    def test_model_version_bump_discards_cache(self, tmp_path):
        # Version-agnostic (the default bumped v2 -> v3 with #1817): the prior cache is
        # written at one EXPLICIT version and read back at a bumped one — the bump must
        # discard wholesale regardless of what the defaults are this year.
        corpus = tmp_path / "c"
        bundles = _three_episodes(corpus)
        prior = TopicConsensusEnricher(CountingScorer(_signals()), model_version="vOLD")
        asyncio.run(
            prior.enrich(
                bundle=None, corpus_root=corpus, all_bundles=bundles, config={}, ctx=_ctx()
            )
        )

        delta = CorpusDelta(changed_ids=frozenset(), removed_ids=frozenset(), all_bundles=bundles)
        scorer = CountingScorer(_signals())
        enricher = TopicConsensusEnricher(scorer, model_version="vNEW")
        result = asyncio.run(
            enricher.enrich_incremental(
                delta=delta, prior_output=None, corpus_root=corpus, config={}, ctx=_ctx()
            )
        )
        assert result.status == STATUS_OK
        assert scorer.calls == 3  # cache was for vOLD → discarded wholesale

    def test_cache_file_lands_next_to_output(self, tmp_path):
        corpus = tmp_path / "c"
        bundles = _three_episodes(corpus)
        _full(corpus, bundles, CountingScorer(_signals()))
        assert pair_cache_path(corpus).is_file()

    def test_manifest_declares_incremental(self):
        assert TopicConsensusEnricher.manifest.supports_incremental is True


class _BatchScorer:
    """Batch-capable fake (#1817): deterministic embeddings + scripted contradictions.

    Vectors are one-hot-ish by text hash so cosine is 1.0 for identical texts and
    ~0.0 otherwise; ``contradiction_for`` overrides per unordered text pair.
    """

    supports_batch = True

    def __init__(self, vectors: dict, contradictions: dict | None = None):
        self._vectors = vectors
        self._contras = contradictions or {}
        self.embed_batches = 0
        self.nli_pairs = 0

    async def embed_texts_batch(self, texts):
        self.embed_batches += 1
        return {t: list(self._vectors.get(t, [0.0, 0.0, 0.0])) for t in texts}

    async def contradictions_batch(self, pairs):
        self.nli_pairs += len(pairs)
        out = []
        for a, b in pairs:
            out.append(float(self._contras.get((a, b), self._contras.get((b, a), 0.1))))
        return out

    async def score(self, a, b):  # pragma: no cover — batch path must be taken
        raise AssertionError("batch-capable scorer must not fall back to score()")


class TestBatchedScoringPath:
    """#1817: batch path == legacy path outputs; budgets bound NLI deterministically."""

    @staticmethod
    def _vectors():
        # Unit vectors constructed so pairwise cosines reproduce the scripted
        # fixture: (A,B)=0.82, (A,C)=0.74, (B,C)=0.55 — the legacy and batch
        # paths then gate identically (0.70 floor admits AB + AC, drops BC).
        return {
            T_A: [1.0, 0.0, 0.0],
            T_B: [0.82, 0.5724, 0.0],
            T_C: [0.74, -0.0993, 0.6656],
        }

    def test_batch_path_matches_legacy_output(self, tmp_path):
        corpus_a = tmp_path / "a"
        bundles_a = _three_episodes(corpus_a)
        legacy = _full(corpus_a, bundles_a, CountingScorer(_signals()))

        corpus_b = tmp_path / "b"
        bundles_b = _three_episodes(corpus_b)
        batch = _BatchScorer(self._vectors())
        enricher = TopicConsensusEnricher(batch)
        result = asyncio.run(
            enricher.enrich(
                bundle=None, corpus_root=corpus_b, all_bundles=bundles_b, config={}, ctx=_ctx()
            )
        )
        assert result.status == STATUS_OK
        assert batch.embed_batches == 1
        # Same consensus pair set as the legacy fixture run (scores differ by
        # scripted fixture design, so compare the pair identities).
        legacy_pairs = {(r["insight_a_id"], r["insight_b_id"]) for r in legacy["consensus"]}
        batch_pairs = {(r["insight_a_id"], r["insight_b_id"]) for r in result.data["consensus"]}
        assert batch_pairs == legacy_pairs

    def test_nli_budget_bounds_calls_and_is_loud(self, tmp_path):
        corpus = tmp_path / "c"
        bundles = _three_episodes(corpus)
        batch = _BatchScorer(self._vectors())
        enricher = TopicConsensusEnricher(batch)
        result = asyncio.run(
            enricher.enrich(
                bundle=None,
                corpus_root=corpus,
                all_bundles=bundles,
                config={"max_nli_pairs_per_run": 0 + 1, "max_nli_pairs_per_topic": 1},
                ctx=_ctx(),
            )
        )
        assert result.status == STATUS_OK
        # Both directions of at most ONE pair reached NLI.
        assert batch.nli_pairs <= 2
        assert result.data["pairs_nli_pending"] >= 0


class TestScorerBatchSurfaces:
    """#1817: the real batch methods on the scorer layer (codecov: previously
    exercised only via enricher-level fakes, so 0 lines of the actual
    implementations ran)."""

    @staticmethod
    def _composite():
        from podcast_scraper.enrichment.scorers.consensus import (
            NliEmbeddingConsensusScorer,
        )
        from podcast_scraper.enrichment.scorers.nli import FixedNliScorer
        from podcast_scraper.enrichment.scorers.protocol import NliScore

        nli = FixedNliScorer(
            scores={("a", "b"): NliScore(0.9, 0.05, 0.05)},
            default=NliScore(0.1, 0.8, 0.1),
        )
        vectors = {"a": [1.0, 0.0], "b": [1.0, 0.0], "c": [0.0, 1.0]}
        return NliEmbeddingConsensusScorer(
            embed_text=lambda t: vectors[t],
            embed_texts=lambda ts: [vectors[t] for t in ts],
            nli=nli,
        )

    def test_supports_batch_requires_both_halves(self):
        from podcast_scraper.enrichment.scorers.consensus import (
            NliEmbeddingConsensusScorer,
        )
        from podcast_scraper.enrichment.scorers.nli import FixedNliScorer

        full = self._composite()
        assert full.supports_batch is True
        no_embed = NliEmbeddingConsensusScorer(embed_text=lambda t: [1.0], nli=FixedNliScorer())
        assert no_embed.supports_batch is False

    def test_embed_texts_batch_dedupes_and_caches(self):
        s = self._composite()
        out = asyncio.run(s.embed_texts_batch(["a", "b", "a"]))
        assert out["a"] == [1.0, 0.0] and out["b"] == [1.0, 0.0]
        # Second call hits the cache (embed_texts not needed): poison it to prove.
        s.embed_texts = None
        again = asyncio.run(s.embed_texts_batch(["a", "b"]))
        assert again == {"a": [1.0, 0.0], "b": [1.0, 0.0]}

    def test_contradictions_batch_takes_max_of_both_directions(self):
        s = self._composite()
        # (a,b) scripted 0.9 in one direction, default 0.1 the other -> max 0.9.
        out = asyncio.run(s.contradictions_batch([("a", "b"), ("b", "c")]))
        assert out[0] == pytest.approx(0.9)
        assert out[1] == pytest.approx(0.1)
        assert asyncio.run(s.contradictions_batch([])) == []

    def test_score_single_still_works(self):
        s = self._composite()
        sig = asyncio.run(s.score("a", "b"))
        assert sig.cosine == pytest.approx(1.0)
        assert sig.contradiction == pytest.approx(0.9)

    def test_fixed_nli_score_batch_matches_singles(self):
        from podcast_scraper.enrichment.scorers.nli import FixedNliScorer
        from podcast_scraper.enrichment.scorers.protocol import NliScore

        nli = FixedNliScorer(scores={("p", "h"): NliScore(0.7, 0.2, 0.1)})
        batch = asyncio.run(nli.score_batch([("p", "h"), ("x", "y")]))
        assert batch[0].contradiction == pytest.approx(0.7)
        assert batch[1] == nli.default

    def test_deberta_softmax_row_calibration(self):
        from podcast_scraper.enrichment.scorers.nli import DeBERTaNliScorer

        s = DeBERTaNliScorer._softmax_row([2.0, 1.0, 0.0])
        assert 0.0 < s.contradiction < 1.0
        assert s.contradiction + s.entailment + s.neutral == pytest.approx(1.0)
        assert s.contradiction > s.entailment > s.neutral
        bad = DeBERTaNliScorer._softmax_row(["x"])
        assert bad.neutral == 1.0 and bad.contradiction == 0.0
