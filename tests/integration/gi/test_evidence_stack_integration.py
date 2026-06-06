"""Integration tests for the GIL evidence stack (Issue #435).

Two layers, deliberately separated:

1. **Real model tests** (``@pytest.mark.ml_models``) — load the *actual* embedding,
   extractive-QA, and NLI models and assert on their real outputs (vector
   dimensionality + semantic ordering, span extraction, entailment monotonicity).
   These are what catch a numpy2 / transformers / sentence-transformers ABI break;
   the previous versions mocked the very function under test and asserted on the
   mock's own return value, so they stayed green even if all three backends broke.
   They **skip** (not fail) when the models aren't provisioned, distinguishing
   "not provisioned" from a regression (``make preload-ml-models`` caches them).

2. **Wiring tests** — inject controlled QA/NLI scores to exercise
   ``find_grounded_quotes``'s threshold filtering and error handling. Mocking the
   backends here is correct: the unit under test is the wiring, not the models.
"""

from __future__ import annotations

import unittest
from typing import cast, List
from unittest.mock import patch

import pytest

from podcast_scraper import config_constants
from podcast_scraper.gi.grounding import find_grounded_quotes, GroundedQuote
from podcast_scraper.providers.ml.extractive_qa import QASpan

# Track the production defaults so these run wherever the stack is provisioned
# (``make preload-ml-models`` caches exactly these), not a sibling variant.
_EMBED_MODEL = config_constants.DEFAULT_EMBEDDING_MODEL
_QA_MODEL = config_constants.DEFAULT_EXTRACTIVE_QA_MODEL
_NLI_MODEL = config_constants.DEFAULT_NLI_MODEL

_PROVISIONING_MARKERS = (
    "offlinemode",
    "offline mode",
    "gatedrepo",
    "localentrynotfound",
    "local cache",
    "cannot reach",
    "couldn't connect",
    "connection",
    "no such file",
    "not a local folder",
    "max retries",
    "failed to import",
    "no module named",
)


def _skip_if_unprovisioned(exc: Exception) -> None:
    """Skip (not fail) when the model isn't cached / loadable offline; else re-raise."""
    haystack = f"{exc} {type(exc).__name__}".lower()
    if any(k in haystack for k in _PROVISIONING_MARKERS):
        pytest.skip(f"evidence model not provisioned: {type(exc).__name__}: {exc}")
    raise exc


@pytest.mark.integration
@pytest.mark.ml_models
class TestEvidenceStackRealModels:
    """Exercise the real embedding / QA / NLI backends — no mocks."""

    def test_embedding_encode_real_vectors_are_semantically_ordered(self) -> None:
        """Real encode() returns fixed-width vectors whose cosine similarity reflects
        meaning: paraphrases score far higher than an unrelated sentence. A numpy2 /
        sentence-transformers break surfaces here as a load error or scrambled order."""
        import math

        from podcast_scraper.providers.ml import embedding_loader

        try:
            raw = embedding_loader.encode(
                [
                    "The capital of France is Paris.",
                    "Paris is the French capital city.",
                    "I had a sandwich for lunch.",
                ],
                model_id=_EMBED_MODEL,
                device="cpu",
                allow_download=False,
            )
        except Exception as exc:  # noqa: BLE001 - provisioning vs regression
            _skip_if_unprovisioned(exc)
            return

        vecs = cast(List[List[float]], raw)

        assert len(vecs) == 3
        assert len(vecs[0]) == len(vecs[1]) == len(vecs[2])
        assert len(vecs[0]) > 0

        def _cos(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(x * x for x in b))
            return dot / (na * nb)

        related = _cos(vecs[0], vecs[1])
        unrelated = _cos(vecs[0], vecs[2])
        assert related > unrelated + 0.3, f"paraphrase {related} not >> unrelated {unrelated}"

    def test_extractive_qa_real_span_points_at_the_answer(self) -> None:
        """Real QA extracts a span from the context that contains the answer token."""
        from podcast_scraper.providers.ml import extractive_qa

        context = "The capital of France is Paris. It has many museums."
        try:
            span = extractive_qa.answer(
                context=context,
                question="What is the capital of France?",
                model_id=_QA_MODEL,
                device="cpu",
            )
        except Exception as exc:  # noqa: BLE001
            _skip_if_unprovisioned(exc)
            return

        assert isinstance(span, QASpan)
        assert 0 <= span.start <= span.end <= len(context)
        # The extracted span (or its answer text) must actually name Paris.
        assert (
            "paris" in (span.answer or "").lower()
            or "paris" in context[span.start : span.end].lower()
        )
        assert span.score > 0.0

    def test_nli_real_entailment_orders_entail_above_contradiction(self) -> None:
        """Real NLI scores a genuine entailment higher than a contradiction for the
        same premise — proves the model loaded and the head wiring is right-way-round."""
        from podcast_scraper.providers.ml import nli_loader

        premise = "The cat sat on the mat in the sun."
        try:
            entail = nli_loader.entailment_score(
                premise=premise,
                hypothesis="A cat was on a mat.",
                model_id=_NLI_MODEL,
                device="cpu",
            )
            contradict = nli_loader.entailment_score(
                premise=premise,
                hypothesis="There were no animals anywhere.",
                model_id=_NLI_MODEL,
                device="cpu",
            )
        except Exception as exc:  # noqa: BLE001
            _skip_if_unprovisioned(exc)
            return

        assert isinstance(entail, float) and isinstance(contradict, float)
        assert 0.0 <= entail <= 1.0 and 0.0 <= contradict <= 1.0
        assert entail > contradict, f"entailment {entail} not > contradiction {contradict}"

    def test_find_grounded_quotes_real_end_to_end(self) -> None:
        """The full real stack (QA + NLI) runs end-to-end and returns quotes that
        satisfy the grounding *invariants* — the integration the mocked wiring tests
        below can never prove.

        Asserts contract, not model quality: every returned quote is a *verbatim*
        transcript span (``transcript[start:end] == text``) with scores in ``[0, 1]``
        that clear the configured thresholds. It deliberately does NOT assert the
        quote is semantically the "right" span — grounding quality on a toy fixture
        is an eval concern, not an integration invariant (the real roberta/deberta
        pair happily returns an off-topic span here, which is fine for this test).
        """
        transcript = "The capital of France is Paris. It has many museums and parks."
        qa_min, nli_min = 0.05, 0.1
        # find_grounded_quotes swallows model-load errors and returns [] (see the
        # wiring test below), so probe QA + NLI first to skip-vs-fail correctly.
        from podcast_scraper.providers.ml import extractive_qa, nli_loader

        try:
            extractive_qa.answer(
                context=transcript,
                question="What is the capital?",
                model_id=_QA_MODEL,
                device="cpu",
            )
            nli_loader.entailment_score(
                premise=transcript,
                hypothesis="Paris is a capital.",
                model_id=_NLI_MODEL,
                device="cpu",
            )
        except Exception as exc:  # noqa: BLE001 - provisioning probe
            _skip_if_unprovisioned(exc)
            return

        try:
            quotes = find_grounded_quotes(
                transcript=transcript,
                insight_text="The capital of France is Paris.",
                qa_model_id=_QA_MODEL,
                nli_model_id=_NLI_MODEL,
                qa_device="cpu",
                nli_device="cpu",
                qa_score_min=qa_min,
                nli_entailment_min=nli_min,
            )
        except Exception as exc:  # noqa: BLE001
            _skip_if_unprovisioned(exc)
            return

        assert isinstance(quotes, list)
        assert quotes, "real evidence stack produced no grounded quote"
        for q in quotes:
            # Verbatim span integrity — the quote text is exactly the transcript slice.
            assert transcript[q.char_start : q.char_end] == q.text
            # Scores are real probabilities that cleared the configured thresholds.
            assert q.qa_score is not None and q.nli_score is not None
            assert 0.0 <= q.qa_score <= 1.0 and 0.0 <= q.nli_score <= 1.0
            assert q.qa_score >= qa_min and q.nli_score >= nli_min


@pytest.mark.integration
class TestFindGroundedQuotesWiring(unittest.TestCase):
    """Threshold + error-handling wiring of find_grounded_quotes, with injected scores.

    Backends are mocked on purpose: the unit under test is the filtering/dispatch
    logic, exercised deterministically by feeding it known QA/NLI scores. The real
    backends are covered by ``TestEvidenceStackRealModels`` above.
    """

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
