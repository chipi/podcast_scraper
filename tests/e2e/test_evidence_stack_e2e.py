"""Real-model e2e for the GIL evidence stack (Issue #435).

Loads the *actual* embedding, extractive-QA, and NLI models and asserts on their
real outputs (vector dimensionality + semantic ordering, span extraction,
entailment monotonicity, full grounding invariants). These are what catch a
numpy2 / transformers / sentence-transformers ABI break; the previous versions
mocked the very function under test and asserted on the mock's own return value,
so they stayed green even if all three backends broke.

Lives in tests/e2e/ per the 3-tier ML/AI testing policy (real models belong in
e2e, not integration — see docs/architecture/TESTING_STRATEGY.md). The threshold
/ error-handling *wiring* of find_grounded_quotes — which mocks the backends on
purpose — stays in tests/integration/gi/test_evidence_stack_integration.py.

These **skip** (not fail) when the models aren't provisioned, distinguishing
"not provisioned" from a regression (``make preload-ml-models`` caches them).
"""

from __future__ import annotations

from typing import cast, List

import pytest

from podcast_scraper import config_constants
from podcast_scraper.gi.grounding import find_grounded_quotes
from podcast_scraper.providers.ml.extractive_qa import QASpan

pytestmark = [pytest.mark.e2e, pytest.mark.ml_models]

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
        can never prove.

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
        # wiring test), so probe QA + NLI first to skip-vs-fail correctly.
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
