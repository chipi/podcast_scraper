"""Integration tests for GIL evidence stack (Issue #435).

Verifies that embedding, extractive QA, NLI, and find_grounded_quotes
integrate correctly using mocked model backends.  No real ML models
are loaded — all model calls are patched.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import pytest

from podcast_scraper.gi.grounding import find_grounded_quotes, GroundedQuote
from podcast_scraper.providers.ml.extractive_qa import QASpan


@pytest.mark.integration
class TestEvidenceStackIntegration(unittest.TestCase):
    """Evidence stack integration with mocked model backends."""

    @patch("podcast_scraper.providers.ml.embedding_loader.encode")
    def test_embedding_encode_returns_vectors(self, mock_encode):
        """encode() returns list of float vectors for a list of texts."""
        mock_encode.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

        from podcast_scraper.providers.ml import embedding_loader

        vecs = embedding_loader.encode(
            ["First sentence.", "Second sentence."],
            model_id="test-model",
            device="cpu",
        )
        self.assertEqual(len(vecs), 2)
        self.assertEqual(len(vecs[0]), len(vecs[1]))
        mock_encode.assert_called_once()

    @patch("podcast_scraper.providers.ml.extractive_qa.answer")
    def test_extractive_qa_returns_span(self, mock_answer):
        """answer() returns a QASpan with answer, offsets, and score."""
        mock_answer.return_value = QASpan(answer="Paris", start=30, end=35, score=0.95)

        from podcast_scraper.providers.ml import extractive_qa

        context = "The capital of France is Paris. It has many museums."
        span = extractive_qa.answer(
            context=context,
            question="What is the capital of France?",
            model_id="test-qa-model",
            device="cpu",
        )
        self.assertEqual(span.answer, "Paris")
        self.assertGreaterEqual(span.start, 0)
        self.assertLessEqual(span.end, len(context))
        self.assertGreater(span.score, 0.0)

    @patch("podcast_scraper.providers.ml.nli_loader.entailment_score")
    def test_nli_returns_entailment_score(self, mock_nli):
        """entailment_score() returns a float between 0 and 1."""
        mock_nli.return_value = 0.92

        from podcast_scraper.providers.ml import nli_loader

        score = nli_loader.entailment_score(
            premise="The cat sat on the mat.",
            hypothesis="A cat was on a mat.",
            model_id="test-nli-model",
            device="cpu",
        )
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    @patch("podcast_scraper.providers.ml.nli_loader.entailment_score")
    @patch("podcast_scraper.providers.ml.extractive_qa.answer")
    def test_find_grounded_quotes_end_to_end(self, mock_qa, mock_nli):
        """find_grounded_quotes wires QA + NLI and returns GroundedQuotes."""
        transcript = "The capital of France is Paris. It has many museums."
        insight = "France has a capital city."

        mock_qa.return_value = QASpan(
            answer="capital of France is Paris",
            start=4,
            end=29,
            score=0.88,
        )
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
    @patch("podcast_scraper.providers.ml.extractive_qa.answer")
    def test_find_grounded_quotes_below_threshold(self, mock_qa, mock_nli):
        """Quotes below score thresholds are filtered out."""
        mock_qa.return_value = QASpan(answer="something", start=0, end=9, score=0.1)
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
    @patch("podcast_scraper.providers.ml.extractive_qa.answer")
    def test_find_grounded_quotes_qa_failure_returns_empty(self, mock_qa, mock_nli):
        """If QA raises, find_grounded_quotes returns empty list."""
        mock_qa.side_effect = RuntimeError("model error")

        quotes = find_grounded_quotes(
            transcript="some text",
            insight_text="a claim",
            qa_model_id="qa",
            nli_model_id="nli",
        )
        self.assertEqual(quotes, [])
        mock_nli.assert_not_called()
