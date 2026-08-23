"""E3 — survivable hard timeouts for the corpus-scope ML enrichers.

``manifest.expected_duration_s`` is primarily the hard ``asyncio.wait_for`` cap in
``executor._execute_with_resilience``. It also feeds the heartbeat stall-warning
threshold — ``HeartbeatWatchdog.is_stalled`` is evaluated post-completion (enrichers
never call ``record_heartbeat``), so raising the cap raises that warning threshold
too; that's an accepted post-hoc log-line effect, not a live-watchdog change. The two
ML enrichers run over the WHOLE corpus in one call and, on a fresh HF cache, pay a
cold sentence-transformers / cross-encoder download inside that window. The old
120s / 180s caps killed a legitimate run over the ~678-episode prod corpus and
reported STATUS_TIMEOUT.

These are regression guards: if someone tightens the caps back toward the values
that killed the run, this fails. Concrete numbers sized with the advisor
(2026-08-23); revisit once a real warm-run wall-clock exists. (#1811 E3)
"""

from __future__ import annotations

from podcast_scraper.enrichment.enrichers.topic_consensus import TopicConsensusEnricher
from podcast_scraper.enrichment.enrichers.topic_similarity import TopicSimilarityEnricher

# Floors below which the enricher demonstrably timed out over the prod corpus.
_SIMILARITY_MIN_S = 300
_CONSENSUS_MIN_S = 600


def test_topic_similarity_hard_timeout_survives_corpus_scale() -> None:
    dur = TopicSimilarityEnricher.manifest.expected_duration_s
    assert dur is not None, "topic_similarity must set an explicit hard cap"
    assert dur >= _SIMILARITY_MIN_S, (
        f"topic_similarity expected_duration_s={dur}s < {_SIMILARITY_MIN_S}s — "
        "the hard wait_for cap; 120s killed a legit corpus-scale run (#1811 E3)"
    )


def test_topic_consensus_hard_timeout_survives_corpus_scale() -> None:
    dur = TopicConsensusEnricher.manifest.expected_duration_s
    assert dur is not None, "topic_consensus must set an explicit hard cap"
    assert dur >= _CONSENSUS_MIN_S, (
        f"topic_consensus expected_duration_s={dur}s < {_CONSENSUS_MIN_S}s — "
        "two local models (MiniLM + deberta-v3-small) pairwise per topic; "
        "180s killed a legit corpus-scale run (#1811 E3)"
    )


def test_consensus_cap_exceeds_similarity_cap() -> None:
    """topic_consensus (two models, pairwise NLI) is heavier than topic_similarity."""
    consensus = TopicConsensusEnricher.manifest.expected_duration_s
    similarity = TopicSimilarityEnricher.manifest.expected_duration_s
    assert consensus is not None and similarity is not None
    assert consensus > similarity
