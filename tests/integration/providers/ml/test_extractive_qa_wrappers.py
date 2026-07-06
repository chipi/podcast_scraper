"""Additional wrapper + fallback tests for extractive_qa module-level API.

Complements ``test_extractive_qa_integration.py``. Exercises the module-level
wrappers (:func:`load_qa_model`, :func:`get_qa_model` shape, the two
deprecation aliases, :func:`answer`, :func:`answer_candidates`,
:func:`answer_multi`) — all of which lie between the caller and
:class:`QAEvidenceBackend`. All backend loads are stubbed; no real HF
model instantiation happens.
"""

from __future__ import annotations

import warnings
from unittest import mock

import pytest

pytest.importorskip("transformers")

pytestmark = [pytest.mark.integration, pytest.mark.critical_path]

from podcast_scraper.providers.ml import extractive_qa
from podcast_scraper.providers.ml.extractive_qa import QAEvidenceBackend, QASpan


def _stub_backend(monkeypatch, *, answer_top1_return=None, answer_top_k_return=None):
    """Replace QAEvidenceBackend._load with a no-op; stub top1/top_k as needed."""

    def fake_load(self):
        self.model = mock.Mock(name=f"model[{self.resolved_id}]")
        self.tokenizer = mock.Mock(name=f"tokenizer[{self.resolved_id}]")

    monkeypatch.setattr(QAEvidenceBackend, "_load", fake_load)
    QAEvidenceBackend.clear_cache()

    if answer_top1_return is not None:
        monkeypatch.setattr(
            QAEvidenceBackend,
            "answer_top1",
            lambda self, q, c: answer_top1_return,
        )
    if answer_top_k_return is not None:
        monkeypatch.setattr(
            QAEvidenceBackend,
            "answer_top_k",
            lambda self, q, c, top_k=3: answer_top_k_return,
        )


class TestModuleLevelWrappers:
    def test_load_qa_model_returns_model_tokenizer_tuple(self, monkeypatch):
        _stub_backend(monkeypatch)
        result = extractive_qa.load_qa_model("roberta-squad2", device="cpu")
        assert isinstance(result, tuple)
        assert len(result) == 2
        model, tokenizer = result
        assert model is not None
        assert tokenizer is not None

    def test_get_qa_model_returns_tuple_from_cached_backend(self, monkeypatch):
        _stub_backend(monkeypatch)
        model_a, tok_a = extractive_qa.get_qa_model("roberta-squad2", device="cpu")
        model_b, tok_b = extractive_qa.get_qa_model("roberta-squad2", device="cpu")
        # Cached across calls — identity of model + tokenizer is preserved.
        assert model_a is model_b
        assert tok_a is tok_b


class TestDeprecationAliases:
    def test_load_qa_pipeline_emits_deprecation_warning(self, monkeypatch):
        _stub_backend(monkeypatch)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = extractive_qa.load_qa_pipeline("roberta-squad2", device="cpu")
        assert any(issubclass(x.category, DeprecationWarning) for x in w)
        assert any("load_qa_pipeline is deprecated" in str(x.message) for x in w)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_get_qa_pipeline_emits_deprecation_warning(self, monkeypatch):
        _stub_backend(monkeypatch)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = extractive_qa.get_qa_pipeline("roberta-squad2", device="cpu")
        assert any(issubclass(x.category, DeprecationWarning) for x in w)
        assert any("get_qa_pipeline is deprecated" in str(x.message) for x in w)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_deprecation_aliases_delegate_to_canonical(self, monkeypatch):
        _stub_backend(monkeypatch)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Both alias + canonical should give the same (model, tok) identity
            # because the backend cache is shared.
            alias_result = extractive_qa.get_qa_pipeline("roberta-squad2", device="cpu")
            canonical_result = extractive_qa.get_qa_model("roberta-squad2", device="cpu")
        assert alias_result[0] is canonical_result[0]
        assert alias_result[1] is canonical_result[1]


class TestAnswerFunction:
    def test_answer_short_context_uses_direct_top1(self, monkeypatch):
        _stub_backend(
            monkeypatch,
            answer_top1_return=QASpan(answer="42", start=14, end=16, score=0.9),
        )
        span = extractive_qa.answer(
            context="The answer is 42.",
            question="What is the answer?",
            model_id="roberta-squad2",
            device="cpu",
        )
        assert span.answer == "42"
        assert span.score == pytest.approx(0.9)

    def test_answer_long_context_no_window_still_uses_top1(self, monkeypatch):
        """window_chars=0 → skip windowing, direct top1 even for long context."""
        _stub_backend(
            monkeypatch,
            answer_top1_return=QASpan(answer="hit", start=0, end=3, score=0.7),
        )
        long_ctx = "word " * 500  # 2500 chars
        span = extractive_qa.answer(
            context=long_ctx,
            question="Q?",
            model_id="roberta-squad2",
            device="cpu",
            window_chars=0,
        )
        assert span.answer == "hit"


class TestAnswerCandidates:
    def test_answer_candidates_short_context_returns_top_k(self, monkeypatch):
        _stub_backend(
            monkeypatch,
            answer_top_k_return=[
                QASpan(answer="A", start=0, end=1, score=0.9),
                QASpan(answer="B", start=5, end=6, score=0.5),
                QASpan(answer="C", start=10, end=11, score=0.2),
            ],
        )
        spans = extractive_qa.answer_candidates(
            context="short context",
            question="Q?",
            model_id="roberta-squad2",
            device="cpu",
            top_k=3,
        )
        assert len(spans) == 3
        assert [s.answer for s in spans] == ["A", "B", "C"]

    def test_answer_candidates_top_k_clamped_to_valid_range(self, monkeypatch):
        """top_k parameter clamped to [1, 10]."""
        seen_top_k = []

        def fake_top_k(self, q, c, top_k=3):
            seen_top_k.append(top_k)
            return [QASpan(answer="", start=0, end=0, score=0.0)] * top_k

        _stub_backend(monkeypatch)
        monkeypatch.setattr(QAEvidenceBackend, "answer_top_k", fake_top_k)

        extractive_qa.answer_candidates(
            context="c", question="q", model_id="roberta-squad2", top_k=999
        )
        extractive_qa.answer_candidates(
            context="c", question="q", model_id="roberta-squad2", top_k=0
        )
        assert seen_top_k == [10, 1]  # clamped to max, then to min


class TestAnswerMulti:
    def test_answer_multi_returns_one_span_per_question(self, monkeypatch):
        _stub_backend(
            monkeypatch,
            answer_top1_return=QASpan(answer="x", start=0, end=1, score=0.5),
        )
        spans = extractive_qa.answer_multi(
            context="ctx",
            questions=["Q1?", "Q2?", "Q3?", "Q4?"],
            model_id="roberta-squad2",
            device="cpu",
        )
        assert len(spans) == 4
        assert all(s.answer == "x" for s in spans)

    def test_answer_multi_empty_questions_returns_empty(self, monkeypatch):
        _stub_backend(monkeypatch)
        spans = extractive_qa.answer_multi(
            context="ctx",
            questions=[],
            model_id="roberta-squad2",
            device="cpu",
        )
        assert spans == []


class TestBackendAnswerTop1Fallback:
    """Empty-candidate fallback in QAEvidenceBackend.answer_top1."""

    def test_answer_top1_returns_empty_span_when_top_k_empty(self, monkeypatch):
        """When answer_top_k returns [], answer_top1 must return QASpan(0-shape)."""
        backend = QAEvidenceBackend.__new__(QAEvidenceBackend)
        monkeypatch.setattr(QAEvidenceBackend, "answer_top_k", lambda self, q, c, top_k=1: [])
        span = backend.answer_top1("q", "ctx")
        assert span.answer == ""
        assert span.start == 0
        assert span.end == 0
        assert span.score == 0.0

    def test_answer_top1_returns_first_when_top_k_nonempty(self, monkeypatch):
        """When answer_top_k returns spans, answer_top1 returns the first one."""
        backend = QAEvidenceBackend.__new__(QAEvidenceBackend)
        monkeypatch.setattr(
            QAEvidenceBackend,
            "answer_top_k",
            lambda self, q, c, top_k=1: [
                QASpan(answer="first", start=0, end=5, score=0.9),
                QASpan(answer="second", start=10, end=16, score=0.5),
            ],
        )
        span = backend.answer_top1("q", "ctx")
        assert span.answer == "first"
        assert span.score == pytest.approx(0.9)


class TestAnswerWindowing:
    """answer() windowing path (window_chars > 0 and context > window)."""

    def test_answer_windowed_iterates_multiple_windows(self, monkeypatch):
        """Covers line 340 (per-window backend.answer_top1 call) — mocked
        top1 returns a valid span so the windowing loop iterates and picks
        the best. We verify iteration count, not answer text (which the
        function re-slices from the global context)."""
        _stub_backend(monkeypatch)
        call_count = {"n": 0}

        def fake_top1(self, q, c):
            call_count["n"] += 1
            # Score increases with call number so the last window wins
            return QASpan(
                answer="span",
                start=0,
                end=4,
                score=0.1 * call_count["n"],
            )

        monkeypatch.setattr(QAEvidenceBackend, "answer_top1", fake_top1)

        long_ctx = "a" * 3000
        span = extractive_qa.answer(
            context=long_ctx,
            question="Q?",
            model_id="roberta-squad2",
            device="cpu",
            window_chars=1000,
            window_overlap_chars=100,
        )
        # answer_top1 was called at least twice (windowing fired)
        assert call_count["n"] >= 2
        # A span was returned (not the empty QASpan fallback)
        assert span.score > 0.0

    def test_answer_windowed_all_raise_falls_back_to_full_context(self, monkeypatch):
        """Covers line 375 (fallback call at end of answer()) — all window
        calls raise, then the function falls back to a single full-context
        backend.answer_top1 call."""
        _stub_backend(monkeypatch)
        state = {"call_n": 0}

        def fake_top1(self, q, c):
            state["call_n"] += 1
            # Every windowed call raises; the fallback (last) call succeeds
            if len(c) < 3000:  # windowed slice
                raise RuntimeError("window failed")
            return QASpan(answer="OK", start=0, end=2, score=0.5)

        monkeypatch.setattr(QAEvidenceBackend, "answer_top1", fake_top1)

        span = extractive_qa.answer(
            context="a" * 3000,
            question="Q?",
            model_id="roberta-squad2",
            window_chars=1000,
            window_overlap_chars=100,
        )
        # The fallback path fired: multiple windowed calls raised, then a
        # final full-context call succeeded.
        assert state["call_n"] > 1
        assert span.score == pytest.approx(0.5)
