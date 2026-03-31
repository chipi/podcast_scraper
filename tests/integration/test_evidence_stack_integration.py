"""Integration tests for GIL evidence stack (Issue #435).

Loads embedding, extractive QA, and NLI with default config and runs a minimal
workflow: encode 2 strings, 1 QA call, 1 NLI pair. Requires sentence-transformers
and transformers; marked ml_models and integration.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import pytest

# Add project root for imports
PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PACKAGE_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PACKAGE_ROOT))

try:
    import sentence_transformers  # noqa: F401

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from transformers import pipeline  # noqa: F401

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


def _evidence_stack_models_available():
    """True when deps exist and default evidence models are cached (offline-safe).

    ``tests/conftest.py`` always sets HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE so tests do
    not hit the hub. Those flags must *not* force a skip here; instead we require a
    loadable HF cache (see ``is_evidence_stack_cached``).

    We do **not** short-circuit on ``ML_MODELS_VALIDATED`` alone: that flag covers
    broader CI cache checks; without the embedding/QA/NLI artifacts present locally,
    this suite would still attempt loads and fail offline.
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE or not TRANSFORMERS_AVAILABLE:
        return False

    from tests.integration.ml_model_cache_helpers import is_evidence_stack_cached

    return is_evidence_stack_cached()


@pytest.mark.integration
@pytest.mark.ml_models
@pytest.mark.slow
@pytest.mark.skipif(
    not _evidence_stack_models_available(),
    reason=(
        "evidence stack needs sentence-transformers + transformers, and default embedding/QA/NLI "
        "models cached (HF_HUB_OFFLINE=1 in conftest; run make preload-ml-models)"
    ),
)
class TestEvidenceStackLoadAndRun(unittest.TestCase):
    """Load all three evidence components with default config and run minimal workflow."""

    @staticmethod
    def _skip_if_hf_offline_load_failed(exc: BaseException) -> None:
        """Skip when cache probes passed but transformers/sentence-transformers still fail."""
        if not isinstance(exc, OSError):
            return
        low = str(exc).lower()
        if any(
            s in low
            for s in (
                "huggingface",
                "cached files",
                "outgoing traffic has been disabled",
                "local_files_only",
                "couldn't connect",
            )
        ):
            pytest.skip(
                "Evidence stack not loadable offline (incomplete HF cache or hub blocked). "
                f"Try: make preload-ml-models. Underlying error: {exc}"
            )

    def test_load_embedding_qa_nli_and_run_minimal_workflow(self):
        """Load embedding, QA, NLI; encode 2 strings, 1 QA, 1 NLI pair."""
        from podcast_scraper import config_constants
        from podcast_scraper.providers.ml import embedding_loader, extractive_qa, nli_loader

        emb_model = config_constants.DEFAULT_EMBEDDING_MODEL
        qa_model = config_constants.DEFAULT_EXTRACTIVE_QA_MODEL
        nli_model = config_constants.DEFAULT_NLI_MODEL

        try:
            # 1. Embedding: encode two strings
            vecs = embedding_loader.encode(
                ["First sentence.", "Second sentence."],
                model_id=emb_model,
                device="cpu",
            )
            self.assertIsInstance(vecs, list)
            self.assertEqual(len(vecs), 2)
            self.assertIsInstance(vecs[0], list)
            self.assertIsInstance(vecs[1], list)
            self.assertGreater(len(vecs[0]), 0)
            self.assertEqual(len(vecs[0]), len(vecs[1]))

            # 2. Extractive QA: one question on short context
            context = "The capital of France is Paris. It has many museums."
            span = extractive_qa.answer(
                context=context,
                question="What is the capital of France?",
                model_id=qa_model,
                device="cpu",
            )
            self.assertIn("paris", span.answer.lower())
            self.assertGreaterEqual(span.start, 0)
            self.assertLessEqual(span.end, len(context))
            self.assertGreater(span.score, 0.0)

            # 3. NLI: one premise/hypothesis pair
            score = nli_loader.entailment_score(
                premise="The cat sat on the mat.",
                hypothesis="A cat was on a mat.",
                model_id=nli_model,
                device="cpu",
            )
            self.assertIsInstance(score, float)
        except OSError as exc:
            self._skip_if_hf_offline_load_failed(exc)
            raise

    def test_find_grounded_quotes_integration(self):
        """Run find_grounded_quotes on short text with real QA/NLI (optional integration)."""
        from podcast_scraper import config_constants
        from podcast_scraper.gi.grounding import find_grounded_quotes

        transcript = "The capital of France is Paris. It has many museums."
        insight_text = "France has a capital city."
        try:
            quotes = find_grounded_quotes(
                transcript=transcript,
                insight_text=insight_text,
                qa_model_id=config_constants.DEFAULT_EXTRACTIVE_QA_MODEL,
                nli_model_id=config_constants.DEFAULT_NLI_MODEL,
                qa_device="cpu",
                nli_device="cpu",
            )
        except OSError as exc:
            self._skip_if_hf_offline_load_failed(exc)
            raise
        self.assertIsInstance(quotes, list)
        # May be empty or have one span depending on thresholds; just ensure no crash
        for q in quotes:
            self.assertGreaterEqual(q.char_start, 0)
            self.assertLessEqual(q.char_end, len(transcript))
            self.assertIsInstance(q.qa_score, float)
            self.assertIsInstance(q.nli_score, float)
