"""Integration tests for the GIL evidence stack wiring (Issue #435).

Threshold + error-handling wiring of ``find_grounded_quotes``, exercised with
**injected** QA/NLI scores. Mocking the backends here is correct: the unit under
test is the filtering/dispatch logic, not the models.

The real embedding / QA / NLI models (the numpy2-ABI-break canary) are exercised
in ``tests/e2e/test_evidence_stack_e2e.py`` — real models belong in e2e per the
3-tier ML/AI testing policy (docs/architecture/TESTING_STRATEGY.md).
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import pytest

from podcast_scraper.gi.grounding import find_grounded_quotes, GroundedQuote
from podcast_scraper.providers.ml.extractive_qa import QASpan


@pytest.mark.integration
class TestFindGroundedQuotesWiring(unittest.TestCase):
    """Threshold + error-handling wiring of find_grounded_quotes, with injected scores."""

    @patch("podcast_scraper.providers.ml.nli_loader.entailment_score")
    @patch("podcast_scraper.providers.ml.extractive_qa.answer_candidates")
    def test_find_grounded_quotes_end_to_end(self, mock_qa_candidates, mock_nli):
        """find_grounded_quotes wires QA + NLI and returns GroundedQuotes."""
        transcript = "The capital of France is Paris. It has many museums."
        insight = "France has a capital city."

        mock_qa_candidates.return_value = [
            QASpan(answer="capital of France is Paris", start=4, end=29, score=0.88)
        ]
        mock_nli.return_value = 0.85

        quotes = find_grounded_quotes(
            transcript=transcript,
            insight_text=insight,
            qa_model_id="test-qa",
            nli_model_id="test-nli",
            qa_device="cpu",
            nli_device="cpu",
            qa_score_min=0.3,
            nli_entailment_min=0.5,
        )

        self.assertIsInstance(quotes, list)
        self.assertGreaterEqual(len(quotes), 1)
        q = quotes[0]
        self.assertIsInstance(q, GroundedQuote)
        self.assertGreaterEqual(q.char_start, 0)
        self.assertLessEqual(q.char_end, len(transcript))
        self.assertIsInstance(q.qa_score, float)
        self.assertIsInstance(q.nli_score, float)

    @patch("podcast_scraper.providers.ml.nli_loader.entailment_score")
    @patch("podcast_scraper.providers.ml.extractive_qa.answer_candidates")
    def test_find_grounded_quotes_below_threshold(self, mock_qa_candidates, mock_nli):
        """Quotes below score thresholds are filtered out."""
        mock_qa_candidates.return_value = [QASpan(answer="something", start=0, end=9, score=0.1)]
        mock_nli.return_value = 0.2

        quotes = find_grounded_quotes(
            transcript="something here",
            insight_text="a claim",
            qa_model_id="qa",
            nli_model_id="nli",
            qa_score_min=0.5,
            nli_entailment_min=0.5,
        )

        self.assertIsInstance(quotes, list)
        self.assertEqual(len(quotes), 0)

    @patch("podcast_scraper.providers.ml.nli_loader.entailment_score")
    @patch("podcast_scraper.providers.ml.extractive_qa.answer_candidates")
    def test_find_grounded_quotes_qa_failure_returns_empty(self, mock_qa_candidates, mock_nli):
        """If QA raises, find_grounded_quotes returns empty list."""
        mock_qa_candidates.side_effect = RuntimeError("model error")

        quotes = find_grounded_quotes(
            transcript="some text",
            insight_text="a claim",
            qa_model_id="qa",
            nli_model_id="nli",
        )
        self.assertEqual(quotes, [])
        mock_nli.assert_not_called()
