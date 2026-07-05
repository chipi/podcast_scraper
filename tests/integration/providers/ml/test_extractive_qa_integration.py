"""Integration tests for extractive QA (Issue #435, updated for #382 Phase E).

Post-#382 the load path goes through :class:`QAEvidenceBackend` (a subclass
of :class:`HFEvidenceBackend`). These tests mock at the backend's
``_load`` seam so no real model instantiation happens; the cache + facade
plumbing (:func:`get_qa_pipeline`, :func:`answer`, :func:`answer_multi`)
is exercised end-to-end.
"""

from __future__ import annotations

from unittest import mock

import pytest

pytest.importorskip("transformers")

from podcast_scraper.providers.ml import extractive_qa
from podcast_scraper.providers.ml.extractive_qa import QAEvidenceBackend, QASpan
from podcast_scraper.providers.ml.hf_evidence_backend import resolve_evidence_device
from podcast_scraper.providers.ml.model_registry import ModelRegistry

pytestmark = [pytest.mark.integration]


def _install_fake_backend_load(monkeypatch):
    """Replace QAEvidenceBackend._load with a no-op that installs mock
    model + tokenizer. Returns nothing — callers inspect the resulting
    backend via QAEvidenceBackend.get_or_load(...).model etc.
    """

    def fake_load(self):
        self.model = mock.Mock(name=f"model[{self.resolved_id}]")
        self.tokenizer = mock.Mock(name=f"tokenizer[{self.resolved_id}]")

    monkeypatch.setattr(QAEvidenceBackend, "_load", fake_load)
    QAEvidenceBackend.clear_cache()


class TestExtractiveQAIntegration:
    """Alias → registry → backend load → answer flow."""

    def test_answer_returns_qa_span(self, monkeypatch):
        """answer() resolves alias, gets/loads backend, returns QASpan."""
        _install_fake_backend_load(monkeypatch)

        def fake_top1(self, question, ctx):
            start = ctx.find("42")
            if start == -1:
                start = 0
            return QASpan(answer="42", start=start, end=start + 2, score=0.95)

        monkeypatch.setattr(QAEvidenceBackend, "answer_top1", fake_top1)

        span = extractive_qa.answer(
            context="The answer is 42 and nothing else.",
            question="What is the answer?",
            model_id="roberta-squad2",
            device="cpu",
        )
        assert isinstance(span, QASpan)
        assert span.answer == "42"
        assert span.score == pytest.approx(0.95)

    def test_answer_multi_returns_one_span_per_question(self, monkeypatch):
        _install_fake_backend_load(monkeypatch)
        monkeypatch.setattr(
            QAEvidenceBackend,
            "answer_top1",
            lambda self, q, c: QASpan(answer="x", start=0, end=1, score=0.5),
        )

        spans = extractive_qa.answer_multi(
            context="The answer is 42.",
            questions=["Q1?", "Q2?", "Q3?"],
            model_id="roberta-squad2",
            device="cpu",
        )
        assert len(spans) == 3
        assert all(isinstance(s, QASpan) for s in spans)

    def test_get_qa_pipeline_caches_instance(self, monkeypatch):
        """get_qa_pipeline returns the same (model, tokenizer) tuple twice —
        backing QAEvidenceBackend instance is cached by (resolved_id, device)."""
        load_calls = []

        def fake_load(self):
            load_calls.append(self.resolved_id)
            self.model = mock.Mock()
            self.tokenizer = mock.Mock()

        monkeypatch.setattr(QAEvidenceBackend, "_load", fake_load)
        QAEvidenceBackend.clear_cache()

        a = extractive_qa.get_qa_pipeline("roberta-squad2", device="cpu")
        b = extractive_qa.get_qa_pipeline("roberta-squad2", device="cpu")

        assert a[0] is b[0]  # model identity
        assert a[1] is b[1]  # tokenizer identity
        assert len(load_calls) == 1  # only loaded once

    def test_mps_device_coerced_to_cpu(self):
        """QAEvidenceBackend.mps_supported = False → MPS resolves to CPU."""
        assert resolve_evidence_device("mps", mps_supported=False) == "cpu"
        assert resolve_evidence_device(" MPS ", mps_supported=False) == "cpu"
        assert QAEvidenceBackend.mps_supported is False

    def test_resolves_registry_alias(self, monkeypatch):
        """The alias `roberta-squad2` resolves to the full HF id via ModelRegistry."""
        _install_fake_backend_load(monkeypatch)
        monkeypatch.setattr(
            QAEvidenceBackend,
            "answer_top1",
            lambda self, q, c: QASpan(answer="", start=0, end=0, score=0.0),
        )

        extractive_qa.answer(context="ctx", question="Q?", model_id="roberta-squad2", device="cpu")
        # The backend cache key should contain the resolved HF id.
        expected = ModelRegistry.resolve_evidence_model_id("roberta-squad2")
        assert any(expected in str(k) for k in QAEvidenceBackend._instances)
