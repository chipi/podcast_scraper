"""Integration tests for extractive QA (Issue #435, updated for #382 Phase E).

Post-#382 the module is a thin wrapper over :class:`QAEvidenceBackend`; these
tests mock at the backend's ``answer_top1`` / ``answer_top_k`` seam and at
:func:`QAEvidenceBackend.get_or_load` (the cache entrypoint) — never at the
old pipeline shape (which is gone).
"""

from unittest import mock

import pytest

from podcast_scraper.providers.ml.extractive_qa import (
    answer,
    answer_candidates,
    answer_multi,
    QAEvidenceBackend,
    QASpan,
)

pytestmark = [pytest.mark.integration]


class TestQASpan:
    def test_create_span(self):
        s = QASpan(answer="42", start=10, end=12, score=0.95)
        assert s.answer == "42"
        assert s.start == 10
        assert s.end == 12
        assert s.score == 0.95


def _stub_backend():
    """Build a Mock() that quacks like a loaded QAEvidenceBackend."""
    backend = mock.Mock(spec=QAEvidenceBackend)
    backend.model = mock.Mock()
    backend.tokenizer = mock.Mock()
    backend.device = "cpu"
    backend._loaded = True
    return backend


class TestAnswerMocked:
    """Facade tests — mock at QAEvidenceBackend.get_or_load."""

    def test_answer_returns_span(self, monkeypatch):
        backend = _stub_backend()
        backend.answer_top1.return_value = QASpan(answer="two", start=5, end=8, score=0.9)
        monkeypatch.setattr(
            QAEvidenceBackend, "get_or_load", classmethod(lambda cls, *a, **kw: backend)
        )

        span = answer(
            context="one two three",
            question="What is the middle word?",
            model_id="roberta-squad2",
        )
        assert isinstance(span, QASpan)
        assert span.answer == "two"
        assert span.score == 0.9
        backend.answer_top1.assert_called_once()

    def test_answer_multi_returns_list(self, monkeypatch):
        backend = _stub_backend()
        backend.answer_top1.return_value = QASpan(answer="ans", start=0, end=3, score=0.8)
        monkeypatch.setattr(
            QAEvidenceBackend, "get_or_load", classmethod(lambda cls, *a, **kw: backend)
        )

        spans = answer_multi(
            context="some context",
            questions=["Q1?", "Q2?"],
            model_id="roberta-squad2",
        )
        assert len(spans) == 2
        assert all(isinstance(s, QASpan) for s in spans)
        assert backend.answer_top1.call_count == 2

    def test_answer_windowed_picks_highest_score_span(self, monkeypatch):
        """Long context + windowing runs answer_top1 per window; best global span wins."""
        backend = _stub_backend()

        def top1_for_window(question, ctx):
            if "TARGETPHRASE" in ctx:
                idx = ctx.index("TARGETPHRASE")
                return QASpan(
                    answer=ctx[idx : idx + len("TARGETPHRASE")],
                    start=idx,
                    end=idx + len("TARGETPHRASE"),
                    score=0.99,
                )
            return QASpan(answer="x", start=0, end=1, score=0.05)

        backend.answer_top1.side_effect = top1_for_window
        monkeypatch.setattr(
            QAEvidenceBackend, "get_or_load", classmethod(lambda cls, *a, **kw: backend)
        )

        filler = "A" * 35
        context = filler + "TARGETPHRASE" + ("B" * 35)
        span = answer(
            context,
            question="Q?",
            model_id="roberta-squad2",
            window_chars=28,
            window_overlap_chars=8,
        )
        assert span.score == 0.99
        assert context[span.start : span.end] == "TARGETPHRASE"

    def test_answer_candidates_returns_multiple_top_k(self, monkeypatch):
        backend = _stub_backend()
        backend.answer_top_k.return_value = [
            QASpan(answer="one", start=0, end=3, score=0.9),
            QASpan(answer="two", start=4, end=7, score=0.7),
        ]
        monkeypatch.setattr(
            QAEvidenceBackend, "get_or_load", classmethod(lambda cls, *a, **kw: backend)
        )

        spans = answer_candidates(
            "one two three",
            "Q?",
            model_id="roberta-squad2",
            top_k=2,
        )
        assert len(spans) == 2
        assert spans[0].score == 0.9
        backend.answer_top_k.assert_called_once_with("Q?", "one two three", top_k=2)

    def test_answer_candidates_empty_backend_result_returns_empty_list(self, monkeypatch):
        """When answer_top_k returns [], the facade propagates the empty list."""
        backend = _stub_backend()
        backend.answer_top_k.return_value = []
        monkeypatch.setattr(
            QAEvidenceBackend, "get_or_load", classmethod(lambda cls, *a, **kw: backend)
        )

        spans = answer_candidates("ctx", "Q?", model_id="roberta-squad2", top_k=3)
        assert spans == []


class TestResolveEvidenceDevice:
    """QA MPS→CPU coercion — the ``mps_supported=False`` semantics for QAEvidenceBackend."""

    def test_explicit_mps_coerced_to_cpu(self):
        from podcast_scraper.providers.ml.hf_evidence_backend import resolve_evidence_device

        assert resolve_evidence_device("mps", mps_supported=False) == "cpu"
        assert resolve_evidence_device(" MPS ", mps_supported=False) == "cpu"

    def test_qa_class_declares_mps_unsupported(self):
        assert QAEvidenceBackend.mps_supported is False
