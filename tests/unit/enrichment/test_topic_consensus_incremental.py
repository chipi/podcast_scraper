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
        corpus = tmp_path / "c"
        bundles = _three_episodes(corpus)
        _full(corpus, bundles, CountingScorer(_signals()))

        delta = CorpusDelta(changed_ids=frozenset(), removed_ids=frozenset(), all_bundles=bundles)
        scorer = CountingScorer(_signals())
        enricher = TopicConsensusEnricher(scorer, model_version="v3")
        result = asyncio.run(
            enricher.enrich_incremental(
                delta=delta, prior_output=None, corpus_root=corpus, config={}, ctx=_ctx()
            )
        )
        assert result.status == STATUS_OK
        assert scorer.calls == 3  # cache was for v2 → discarded wholesale

    def test_cache_file_lands_next_to_output(self, tmp_path):
        corpus = tmp_path / "c"
        bundles = _three_episodes(corpus)
        _full(corpus, bundles, CountingScorer(_signals()))
        assert pair_cache_path(corpus).is_file()

    def test_manifest_declares_incremental(self):
        assert TopicConsensusEnricher.manifest.supports_incremental is True
