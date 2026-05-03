"""Integration tests for bundled score_entailment dispatch (#698 Layer B).

Covers:
- ``_ground_insights_with_bundled_nli`` collects all (insight, quote) pairs and
  issues a single bundled call (or chunked calls).
- Per-pair fallback when bundled call omits a score.
- Whole-batch fallback when bundled call raises.
- Chunk count metric matches ceil(pairs / chunk_size).
- Provider without ``score_entailment_bundled`` falls back to staged.
"""

from __future__ import annotations

import unittest
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock

import pytest

from podcast_scraper.gi.grounding import QuoteCandidate


def _candidate(text: str, char_start: int, char_end: int, qa_score: float = 1.0) -> QuoteCandidate:
    return QuoteCandidate(
        char_start=char_start,
        char_end=char_end,
        text=text,
        qa_score=qa_score,
    )


@pytest.mark.integration
class TestBundledNliDispatch(unittest.TestCase):
    def setUp(self) -> None:
        self.transcript = (
            "Climate change drives migration. "
            "Inflation eats wages. "
            "Wages have not kept up with rent. "
            "Cities are growing faster than housing supply."
        )
        self.insight_specs: List[Tuple[str, Any]] = [
            ("Economic pressure causes movement.", None),
            ("Cost-of-living outpaces income.", None),
        ]
        # Two candidates per insight (4 pairs total).
        self.prefetched: Dict[int, List[QuoteCandidate]] = {
            0: [
                _candidate("Climate change drives migration.", 0, 32),
                _candidate("Cities are growing faster than housing supply.", 89, 134),
            ],
            1: [
                _candidate("Inflation eats wages.", 33, 54),
                _candidate("Wages have not kept up with rent.", 55, 88),
            ],
        }

    def test_bundled_called_once_per_chunk(self) -> None:
        """One chunk = one bundled call. 4 pairs at chunk_size=15 → 1 call."""
        from podcast_scraper.gi.pipeline import _ground_insights_with_bundled_nli
        from podcast_scraper.workflow.metrics import Metrics

        bundled_fn = MagicMock(
            return_value={0: 0.9, 1: 0.85, 2: 0.8, 3: 0.75},
        )
        prov = MagicMock()
        prov.score_entailment_bundled = bundled_fn
        prov.score_entailment = MagicMock(side_effect=AssertionError("staged should not run"))

        m = Metrics()
        result = _ground_insights_with_bundled_nli(
            insight_specs=self.insight_specs,
            transcript=self.transcript,
            quote_extraction_provider=prov,
            entailment_provider=prov,
            qa_score_min=0.3,
            nli_entailment_min=0.5,
            extract_retries=0,
            chunk_size=15,
            pipeline_metrics=m,
            prefetched_by_idx=self.prefetched,
        )

        bundled_fn.assert_called_once()
        # 4 pairs all above threshold → 4 grounded quotes (2 per insight)
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 2)
        self.assertEqual(len(result[1]), 2)

    def test_chunk_count_metric_matches_chunks(self) -> None:
        """4 pairs at chunk_size=2 → 2 bundled calls counted."""
        from podcast_scraper.gi.pipeline import _ground_insights_with_bundled_nli
        from podcast_scraper.workflow.metrics import Metrics

        bundled_fn = MagicMock(return_value={0: 0.9, 1: 0.85})
        prov = MagicMock()
        prov.score_entailment_bundled = bundled_fn

        m = Metrics()
        _ground_insights_with_bundled_nli(
            insight_specs=self.insight_specs,
            transcript=self.transcript,
            quote_extraction_provider=prov,
            entailment_provider=prov,
            qa_score_min=0.3,
            nli_entailment_min=0.5,
            extract_retries=0,
            chunk_size=2,
            pipeline_metrics=m,
            prefetched_by_idx=self.prefetched,
        )
        # 4 pairs / chunk_size=2 = 2 chunks expected.
        self.assertEqual(m.gi_evidence_score_entailment_bundled_calls, 2)

    def test_per_pair_fallback_when_bundled_omits_score(self) -> None:
        """Pair indices missing from bundled response trigger staged per-pair fallback."""
        from podcast_scraper.gi.pipeline import _ground_insights_with_bundled_nli
        from podcast_scraper.workflow.metrics import Metrics

        # Bundled returns scores for 0 and 2 only; 1 and 3 must fall back.
        bundled_fn = MagicMock(return_value={0: 0.9, 2: 0.85})
        staged_fn = MagicMock(return_value=0.7)
        prov = MagicMock()
        prov.score_entailment_bundled = bundled_fn
        prov.score_entailment = staged_fn

        m = Metrics()
        _ground_insights_with_bundled_nli(
            insight_specs=self.insight_specs,
            transcript=self.transcript,
            quote_extraction_provider=prov,
            entailment_provider=prov,
            qa_score_min=0.3,
            nli_entailment_min=0.5,
            extract_retries=0,
            chunk_size=15,
            pipeline_metrics=m,
            prefetched_by_idx=self.prefetched,
        )
        # 2 missing pairs → 2 staged fallback calls.
        self.assertEqual(staged_fn.call_count, 2)
        self.assertEqual(m.gi_evidence_score_entailment_calls, 2)

    def test_below_nli_threshold_dropped(self) -> None:
        """Scores below threshold → quote not in grounded list."""
        from podcast_scraper.gi.pipeline import _ground_insights_with_bundled_nli

        bundled_fn = MagicMock(return_value={0: 0.9, 1: 0.1, 2: 0.95, 3: 0.2})
        prov = MagicMock()
        prov.score_entailment_bundled = bundled_fn

        result = _ground_insights_with_bundled_nli(
            insight_specs=self.insight_specs,
            transcript=self.transcript,
            quote_extraction_provider=prov,
            entailment_provider=prov,
            qa_score_min=0.3,
            nli_entailment_min=0.5,
            extract_retries=0,
            chunk_size=15,
            pipeline_metrics=None,
            prefetched_by_idx=self.prefetched,
        )
        # Insight 0: pair 0 (0.9) passes, pair 1 (0.1) drops → 1 quote.
        # Insight 1: pair 2 (0.95) passes, pair 3 (0.2) drops → 1 quote.
        self.assertEqual(len(result[0]), 1)
        self.assertEqual(len(result[1]), 1)

    def test_provider_without_bundled_falls_back_to_staged(self) -> None:
        """No ``score_entailment_bundled`` method → staged path runs end-to-end."""
        from podcast_scraper.gi.pipeline import _ground_insights_with_bundled_nli

        # Provider lacks bundled method.
        prov = MagicMock(spec=["extract_quotes", "score_entailment"])
        prov.score_entailment = MagicMock(return_value=0.9)
        prov.extract_quotes = MagicMock()

        result = _ground_insights_with_bundled_nli(
            insight_specs=self.insight_specs,
            transcript=self.transcript,
            quote_extraction_provider=prov,
            entailment_provider=prov,
            qa_score_min=0.3,
            nli_entailment_min=0.5,
            extract_retries=0,
            chunk_size=15,
            pipeline_metrics=None,
            prefetched_by_idx=self.prefetched,
        )
        # Staged path runs: 1 score_entailment call per pair (4 pairs).
        self.assertEqual(prov.score_entailment.call_count, 4)
        self.assertEqual(len(result[0]), 2)
        self.assertEqual(len(result[1]), 2)

    def test_bundled_raises_falls_through_to_staged(self) -> None:
        """Bundled raises → ``_ground_insights_dispatch`` falls back to staged + records metric."""
        from podcast_scraper.gi.pipeline import _ground_insights_dispatch
        from podcast_scraper.workflow.metrics import Metrics

        bundled_fn = MagicMock(side_effect=RuntimeError("api flapped"))
        staged_fn = MagicMock(return_value=0.9)
        prov = MagicMock()
        prov.score_entailment_bundled = bundled_fn
        prov.score_entailment = staged_fn
        prov.extract_quotes = MagicMock()

        cfg = MagicMock()
        cfg.gil_evidence_nli_mode = "bundled"
        cfg.gil_evidence_nli_chunk_size = 15

        m = Metrics()
        result = _ground_insights_dispatch(
            cfg=cfg,
            insight_specs=self.insight_specs,
            transcript=self.transcript,
            quote_extraction_provider=prov,
            entailment_provider=prov,
            qa_score_min=0.3,
            nli_entailment_min=0.5,
            extract_retries=0,
            pipeline_metrics=m,
            prefetched_by_idx=self.prefetched,
        )

        self.assertEqual(m.gi_evidence_score_entailment_bundled_fallbacks, 1)
        # Staged path produces grounded quotes.
        self.assertEqual(len(result[0]), 2)
        self.assertEqual(len(result[1]), 2)

    def test_no_pairs_returns_empty_lists(self) -> None:
        """Insights without any QA-passing candidates → no bundled call."""
        from podcast_scraper.gi.pipeline import _ground_insights_with_bundled_nli

        bundled_fn = MagicMock()
        prov = MagicMock()
        prov.score_entailment_bundled = bundled_fn
        prov.extract_quotes = MagicMock(return_value=[])

        result = _ground_insights_with_bundled_nli(
            insight_specs=self.insight_specs,
            transcript=self.transcript,
            quote_extraction_provider=prov,
            entailment_provider=prov,
            qa_score_min=0.3,
            nli_entailment_min=0.5,
            extract_retries=0,
            chunk_size=15,
            pipeline_metrics=None,
            prefetched_by_idx={0: [], 1: []},
        )

        bundled_fn.assert_not_called()
        self.assertEqual(result, [[], []])

    def test_pairs_total_metric_accumulates(self) -> None:
        """``..._pairs_total`` reflects the actual pairs sent in bundled calls."""
        from podcast_scraper.gi.pipeline import _ground_insights_with_bundled_nli
        from podcast_scraper.workflow.metrics import Metrics

        # The Gemini implementation increments ``..._pairs_total`` per chunk; our
        # pipeline-level helper does NOT (provider owns that counter). So this
        # test asserts the helper does NOT double-count.
        bundled_fn = MagicMock(return_value={0: 0.9, 1: 0.9, 2: 0.9, 3: 0.9})
        prov = MagicMock()
        prov.score_entailment_bundled = bundled_fn

        m = Metrics()
        _ground_insights_with_bundled_nli(
            insight_specs=self.insight_specs,
            transcript=self.transcript,
            quote_extraction_provider=prov,
            entailment_provider=prov,
            qa_score_min=0.3,
            nli_entailment_min=0.5,
            extract_retries=0,
            chunk_size=15,
            pipeline_metrics=m,
            prefetched_by_idx=self.prefetched,
        )
        # The pipeline-level helper should NOT increment pairs_total — provider does.
        # Since our mock bundled_fn doesn't either, the counter stays at 0.
        self.assertEqual(m.gi_evidence_score_entailment_bundled_pairs_total, 0)

    def test_dispatch_routes_to_staged_when_mode_is_staged(self) -> None:
        """``cfg.gil_evidence_nli_mode='staged'`` skips bundled even if provider supports it."""
        from podcast_scraper.gi.pipeline import _ground_insights_dispatch

        bundled_fn = MagicMock()
        staged_fn = MagicMock(return_value=0.9)
        prov = MagicMock()
        prov.score_entailment_bundled = bundled_fn
        prov.score_entailment = staged_fn
        prov.extract_quotes = MagicMock()

        cfg = MagicMock()
        cfg.gil_evidence_nli_mode = "staged"
        cfg.gil_evidence_nli_chunk_size = 15

        _ground_insights_dispatch(
            cfg=cfg,
            insight_specs=self.insight_specs,
            transcript=self.transcript,
            quote_extraction_provider=prov,
            entailment_provider=prov,
            qa_score_min=0.3,
            nli_entailment_min=0.5,
            extract_retries=0,
            pipeline_metrics=None,
            prefetched_by_idx=self.prefetched,
        )
        bundled_fn.assert_not_called()
        # Staged: 4 NLI calls.
        self.assertEqual(staged_fn.call_count, 4)
