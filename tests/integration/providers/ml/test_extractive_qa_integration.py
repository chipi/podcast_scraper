"""Integration tests for extractive QA pipeline (Issue #435).

Exercises the full load → answer flow with a fake transformers pipeline.
Requires ``transformers`` (``pip install -e '.[ml]'``).
"""

from __future__ import annotations

import pytest

pytest.importorskip("transformers")

from podcast_scraper.providers.ml import extractive_qa
from podcast_scraper.providers.ml.extractive_qa import QASpan
from podcast_scraper.providers.ml.model_registry import ModelRegistry

pytestmark = [pytest.mark.integration]


def _make_fake_pipeline(captured):
    """Return a fake transformers pipeline function."""

    def fake_pipeline(task, model, device=None, **kwargs):
        captured.append({"task": task, "model": model, "device": device, **kwargs})

        def pipe_fn(question, context, max_answer_len=512, **kw):
            start = context.find("42")
            if start == -1:
                start = 0
            return {
                "answer": "42",
                "start": start,
                "end": start + 2,
                "score": 0.95,
            }

        return pipe_fn

    return fake_pipeline


class TestExtractiveQAIntegration:
    """Full wiring: alias → registry → pipeline init → answer extraction."""

    def test_answer_returns_qa_span(self, monkeypatch):
        """answer() resolves alias, loads pipeline, returns QASpan."""
        captured: list = []
        monkeypatch.setattr(
            extractive_qa,
            "build_huggingface_qa_pipeline",
            lambda model_id, device, local_files_only: _make_fake_pipeline(captured)(
                "question-answering", model_id, device=device
            ),
        )
        extractive_qa._qa_pipelines.clear()

        span = extractive_qa.answer(
            context="The answer is 42 and nothing else.",
            question="What is the answer?",
            model_id="roberta-squad2",
            device="cpu",
        )

        assert isinstance(span, QASpan)
        assert span.answer == "42"
        assert span.score == pytest.approx(0.95)
        assert captured[0]["model"] == ModelRegistry.resolve_evidence_model_id("roberta-squad2")

    def test_answer_multi_returns_one_span_per_question(self, monkeypatch):
        """answer_multi() returns one QASpan per question."""
        captured: list = []
        monkeypatch.setattr(
            extractive_qa,
            "build_huggingface_qa_pipeline",
            lambda model_id, device, local_files_only: _make_fake_pipeline(captured)(
                "question-answering", model_id, device=device
            ),
        )
        extractive_qa._qa_pipelines.clear()

        spans = extractive_qa.answer_multi(
            context="The answer is 42.",
            questions=["Q1?", "Q2?", "Q3?"],
            model_id="roberta-squad2",
            device="cpu",
        )

        assert len(spans) == 3
        assert all(isinstance(s, QASpan) for s in spans)

    def test_get_qa_pipeline_caches_instance(self, monkeypatch):
        """get_qa_pipeline returns the same instance on repeated calls."""
        call_count = 0

        def counting_build(model_id, device, local_files_only):
            nonlocal call_count
            call_count += 1
            return lambda **kw: {"answer": "x", "start": 0, "end": 1, "score": 0.5}

        monkeypatch.setattr(
            extractive_qa,
            "build_huggingface_qa_pipeline",
            counting_build,
        )
        extractive_qa._qa_pipelines.clear()

        a = extractive_qa.get_qa_pipeline("roberta-squad2", device="cpu")
        b = extractive_qa.get_qa_pipeline("roberta-squad2", device="cpu")

        assert a is b
        assert call_count == 1

    def test_mps_device_coerced_to_cpu(self):
        """_get_device('mps') returns 'cpu' (MPS unsupported for QA)."""
        assert extractive_qa._get_device("mps") == "cpu"
        assert extractive_qa._get_device(" MPS ") == "cpu"
