"""Unit tests for extractive QA pipeline (Issue #435)."""

import pytest

from podcast_scraper.providers.ml.extractive_qa import answer, answer_multi, QASpan

pytestmark = [pytest.mark.unit]


class TestQASpan:
    """Tests for QASpan dataclass."""

    def test_create_span(self):
        """QASpan stores answer, start, end, score."""
        s = QASpan(answer="42", start=10, end=12, score=0.95)
        assert s.answer == "42"
        assert s.start == 10
        assert s.end == 12
        assert s.score == 0.95


class TestAnswerMocked:
    """Tests for answer() with mocked pipeline."""

    def test_answer_returns_span(self, monkeypatch):
        """answer() returns QASpan with pipeline output."""

        def fake_pipeline(*args, question=None, context=None, max_answer_len=None, **kwargs):
            return {"answer": "two", "start": 5, "end": 8, "score": 0.9}

        from podcast_scraper.providers.ml import extractive_qa

        monkeypatch.setattr(
            extractive_qa,
            "get_qa_pipeline",
            lambda *args, **kwargs: type("Pipe", (), {"__call__": fake_pipeline})(),
        )
        span = answer(
            context="one two three",
            question="What is the middle word?",
            model_id="roberta-squad2",
        )
        assert isinstance(span, QASpan)
        assert span.answer == "two"
        assert span.start == 5
        assert span.end == 8
        assert span.score == 0.9

    def test_answer_multi_returns_list(self, monkeypatch):
        """answer_multi() returns list of QASpan."""

        def fake_pipe(*args, question=None, context=None, max_answer_len=None, **kwargs):
            return {"answer": "ans", "start": 0, "end": 3, "score": 0.8}

        from podcast_scraper.providers.ml import extractive_qa

        monkeypatch.setattr(
            extractive_qa,
            "get_qa_pipeline",
            lambda *args, **kwargs: type("Pipe", (), {"__call__": fake_pipe})(),
        )
        spans = answer_multi(
            context="some context",
            questions=["Q1?", "Q2?"],
            model_id="roberta-squad2",
        )
        assert len(spans) == 2
        assert all(isinstance(s, QASpan) for s in spans)

    def test_answer_meta_tensor_score_fallback(self, monkeypatch):
        """When pipeline returns score on meta device, fallback to 0.0 (GIL + API-only)."""

        class MetaScore:
            def item(self):
                raise RuntimeError("Tensor.item() cannot be called on meta tensors")

        def fake_pipeline(*args, question=None, context=None, max_answer_len=None, **kwargs):
            return {"answer": "x", "start": 0, "end": 1, "score": MetaScore()}

        from podcast_scraper.providers.ml import extractive_qa

        monkeypatch.setattr(
            extractive_qa,
            "get_qa_pipeline",
            lambda *args, **kwargs: type("Pipe", (), {"__call__": fake_pipeline})(),
        )
        span = answer(
            context="context",
            question="Q?",
            model_id="roberta-squad2",
        )
        assert span.score == 0.0


class TestGetDevice:
    """Tests for QA device selection (avoid MPS/meta for HF QA)."""

    def test_explicit_mps_maps_to_cpu(self):
        """MPS is coerced to CPU; QA pipelines can hit meta/unsupported paths on Apple GPU."""
        from podcast_scraper.providers.ml.extractive_qa import _get_device

        assert _get_device("mps") == "cpu"
        assert _get_device(" MPS ") == "cpu"


try:
    from transformers import pipeline  # noqa: F401

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@pytest.mark.skipif(
    not TRANSFORMERS_AVAILABLE,
    reason="transformers required for load path test",
)
class TestLoadQaPipeline:
    """Tests for load_qa_pipeline (resolve + load wiring)."""

    def test_load_qa_pipeline_resolves_alias(self, monkeypatch):
        """load_qa_pipeline resolves alias via registry before loading."""
        from podcast_scraper.providers.ml.model_registry import ModelRegistry

        captured = []

        def fake_pipeline(task, model, device=None):
            captured.append(model)
            return lambda **kw: {"answer": "x", "start": 0, "end": 1, "score": 0.9}

        monkeypatch.setattr(
            "transformers.pipeline",
            fake_pipeline,
            raising=False,
        )
        from podcast_scraper.providers.ml import extractive_qa

        extractive_qa.load_qa_pipeline("roberta-squad2", device="cpu")
        assert len(captured) == 1
        assert captured[0] == ModelRegistry.resolve_evidence_model_id("roberta-squad2")
