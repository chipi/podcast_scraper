"""Grounding must NEVER make unbounded LLM calls on one episode. A cap, or it burns money.

THE INCIDENT: gpt-5.5's bundled `score_entailment` returned empty content, so the whole batch fell
back to scoring (insight, quote) pairs ONE AT A TIME. A big episode has hundreds of pairs, each a
live LLM call with retries, and nothing bounded the loop — it ran to ~3500 calls on a SINGLE episode
over an hour on an expensive model, for a result that came back 0-grounded anyway.

The fix caps the per-pair fallback per episode: past the cap, remaining pairs score 0 (ungrounded)
and the run logs LOUD. This test drives the exact failure — a provider whose bundled NLI returns
nothing, forcing per-pair on far more pairs than the cap — and asserts the number of live calls is
bounded, not the pair count.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pytest

from podcast_scraper.gi import pipeline as gipipe
from podcast_scraper.gi.pipeline import MAX_PER_PAIR_ENTAILMENT_FALLBACK_CALLS

pytestmark = pytest.mark.unit


class _Candidate:
    def __init__(self, text: str) -> None:
        self.text = text
        self.char_start = 0
        self.char_end = len(text)
        self.qa_score = 1.0


class _EmptyBundleProvider:
    """Bundled NLI returns NOTHING (the empty-content failure), forcing per-pair. Every insight gets
    several quote candidates, so the pair count dwarfs the cap. Counts live per-pair calls."""

    def __init__(self, candidates_per_insight: int) -> None:
        self.candidates_per_insight = candidates_per_insight
        self.per_pair_calls = 0

    def extract_quotes(self, transcript: str, insight_text: str, **kw: Any) -> List[_Candidate]:
        return [
            _Candidate(f"quote {i} for {insight_text[:8]}")
            for i in range(self.candidates_per_insight)
        ]

    def score_entailment_bundled(
        self, pairs: List[Tuple[str, str]], chunk_size: int = 15, **kw: Any
    ) -> Dict[int, float]:
        return {}  # scores NOTHING -> every pair falls to per-pair

    def score_entailment(self, premise: str, hypothesis: str, **kw: Any) -> float:
        self.per_pair_calls += 1
        return 0.9


def test_per_pair_entailment_is_capped_per_episode() -> None:
    """THE GUARDRAIL. 100 insights x 5 candidates = 500 pairs, all needing per-pair scoring. Without
    the cap that is 500 live calls; with it, no more than the cap."""
    prov = _EmptyBundleProvider(candidates_per_insight=5)
    specs = [(f"insight {i} about the economy", "insight") for i in range(100)]

    gipipe._ground_insights_with_bundled_nli(
        insight_specs=specs,
        transcript="a long transcript " * 50,
        quote_extraction_provider=prov,
        entailment_provider=prov,
        qa_score_min=0.0,
        nli_entailment_min=0.5,
        extract_retries=0,
        chunk_size=15,
        pipeline_metrics=None,
        prefetched_by_idx=None,
    )

    assert prov.per_pair_calls <= MAX_PER_PAIR_ENTAILMENT_FALLBACK_CALLS, (
        f"grounding made {prov.per_pair_calls} live entailment calls on ONE episode — the cap "
        f"({MAX_PER_PAIR_ENTAILMENT_FALLBACK_CALLS}) did not hold; a runaway can still burn money"
    )
    # And it should actually REACH the cap here (500 pairs > cap), proving it engaged.
    assert prov.per_pair_calls == MAX_PER_PAIR_ENTAILMENT_FALLBACK_CALLS


def test_a_healthy_episode_under_the_cap_scores_every_pair() -> None:
    """The cap must not clip a normal episode. 20 insights x 2 candidates = 40 pairs < cap -> all
    scored, none dropped by the guardrail."""
    prov = _EmptyBundleProvider(candidates_per_insight=2)
    specs = [(f"insight {i}", "insight") for i in range(20)]
    out = gipipe._ground_insights_with_bundled_nli(
        insight_specs=specs,
        transcript="t " * 50,
        quote_extraction_provider=prov,
        entailment_provider=prov,
        qa_score_min=0.0,
        nli_entailment_min=0.5,
        extract_retries=0,
        chunk_size=15,
        pipeline_metrics=None,
        prefetched_by_idx=None,
    )
    assert prov.per_pair_calls == 40, "every pair scored; the cap did not interfere"
    # All pairs scored 0.9 (>= 0.5) -> every insight is grounded.
    assert sum(len(q) for q in out) == 40
