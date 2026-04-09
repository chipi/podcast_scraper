"""Unit tests for extractive QA pipeline (Issue #435)."""

import pytest

from podcast_scraper.providers.ml.extractive_qa import (
    answer,
    answer_candidates,
    answer_multi,
    QASpan,
)

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

    def test_answer_windowed_picks_highest_score_span(self, monkeypatch):
        """Long context + windowing runs QA per window; best global span wins."""

        def fake_pipeline(*args, question=None, context=None, max_answer_len=None, **kwargs):
            if "TARGETPHRASE" in context:
                idx = context.index("TARGETPHRASE")
                end = idx + len("TARGETPHRASE")
                return {
                    "answer": context[idx:end],
                    "start": idx,
                    "end": end,
                    "score": 0.99,
                }
            return {"answer": "x", "start": 0, "end": 1, "score": 0.05}

        from podcast_scraper.providers.ml import extractive_qa

        monkeypatch.setattr(
            extractive_qa,
            "get_qa_pipeline",
            lambda *args, **kwargs: type("Pipe", (), {"__call__": fake_pipeline})(),
        )
        filler = "A" * 35
        context = filler + "TARGETPHRASE" + ("B" * 35)
        span = extractive_qa.answer(
            context,
            question="Q?",
            model_id="roberta-squad2",
            window_chars=28,
            window_overlap_chars=8,
        )
        assert span.score == 0.99
        assert context[span.start : span.end] == "TARGETPHRASE"

    def test_answer_candidates_returns_multiple_top_k(self, monkeypatch):
        """answer_candidates uses pipeline top_k when supported (#487)."""

        def fake_pipe(
            *args, question=None, context=None, max_answer_len=None, top_k=None, **kwargs
        ):
            return [
                {"answer": "one", "start": 0, "end": 3, "score": 0.9},
                {"answer": "two", "start": 4, "end": 7, "score": 0.7},
            ]

        from podcast_scraper.providers.ml import extractive_qa

        monkeypatch.setattr(
            extractive_qa,
            "get_qa_pipeline",
            lambda *args, **kwargs: type("Pipe", (), {"__call__": fake_pipe})(),
        )
        spans = answer_candidates(
            "one two three",
            "Q?",
            model_id="roberta-squad2",
            top_k=2,
        )
        assert len(spans) == 2
        assert spans[0].score == 0.9

    def test_answer_candidates_typeerror_falls_back_single(self, monkeypatch):
        """When top_k is unsupported, fall back to one span."""

        def fake_pipe(*args, **kwargs):
            if kwargs.get("top_k") is not None:
                raise TypeError("unexpected keyword top_k")
            return {"answer": "x", "start": 0, "end": 1, "score": 0.5}

        from podcast_scraper.providers.ml import extractive_qa

        monkeypatch.setattr(
            extractive_qa,
            "get_qa_pipeline",
            lambda *args, **kwargs: type("Pipe", (), {"__call__": fake_pipe})(),
        )
        spans = answer_candidates("ctx", "Q?", model_id="roberta-squad2", top_k=3)
        assert len(spans) == 1
        assert spans[0].answer == "x"


class TestGetDevice:
    """Tests for QA device selection (avoid MPS/meta for HF QA)."""

    def test_explicit_mps_maps_to_cpu(self):
        """MPS is coerced to CPU; QA pipelines can hit meta/unsupported paths on Apple GPU."""
        from podcast_scraper.providers.ml.extractive_qa import _get_device

        assert _get_device("mps") == "cpu"
        assert _get_device(" MPS ") == "cpu"
