"""The windowed QA path silently returned one candidate and one answer-fragment.

Two bugs, both invisible because every eval ran the LLM grounder instead of this one:

1. ``answer_candidates`` discarded ``top_k`` whenever windowing was active. A real transcript
   (~70k chars) always exceeds the window (1800), so the windowed branch is the ONLY one production
   ever takes — the grounding stage got exactly one candidate per insight no matter what it asked
   for.

2. A QA model answers a question; it does not quote. Its span is the ANSWER — a few words, median
   40 chars — and a fragment that short cannot serve as an NLI premise, so the gate rejected every
   one and nothing ever grounded.
"""

from __future__ import annotations

from typing import List

import pytest

from podcast_scraper.providers.ml import extractive_qa
from podcast_scraper.providers.ml.extractive_qa import expand_span_to_sentence, QASpan


class _FakeBackend:
    """Returns `top_k` spans per window so we can assert they all survive."""

    def __init__(self) -> None:
        self.calls = 0

    def answer_top_k(self, question: str, context: str, top_k: int = 3) -> List[QASpan]:
        self.calls += 1
        out = []
        for i in range(top_k):
            start = min(i * 5, max(0, len(context) - 2))
            end = min(start + 4, len(context))
            # score varies by window so ranking across windows is exercised
            out.append(
                QASpan(
                    answer=context[start:end],
                    start=start,
                    end=end,
                    score=0.9 - 0.01 * self.calls - 0.001 * i,
                )
            )
        return out

    def answer_top1(self, question: str, context: str) -> QASpan:
        return self.answer_top_k(question, context, top_k=1)[0]


class TestWindowedCandidatesHonourTopK:
    def test_windowed_path_returns_top_k_not_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = _FakeBackend()
        monkeypatch.setattr(extractive_qa, "_get_qa_backend", lambda *a, **k: backend)

        context = "x" * 10_000  # far larger than the window -> windowed branch
        spans = extractive_qa.answer_candidates(
            context=context,
            question="What evidence supports: something?",
            model_id="fake",
            window_chars=1800,
            window_overlap_chars=250,
            top_k=3,
        )

        assert backend.calls > 1, "windowing should have produced multiple windows"
        assert len(spans) == 3, "top_k was discarded on the windowed path (returned one span)"

    def test_spans_are_global_offsets(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Offsets must index the whole transcript, not the window."""
        backend = _FakeBackend()
        monkeypatch.setattr(extractive_qa, "_get_qa_backend", lambda *a, **k: backend)

        context = "y" * 10_000
        spans = extractive_qa.answer_candidates(
            context=context,
            question="q",
            model_id="fake",
            window_chars=1800,
            window_overlap_chars=250,
            top_k=3,
        )
        for s in spans:
            assert 0 <= s.start < s.end <= len(context)
            assert context[s.start : s.end] == s.answer

    def test_ranked_by_score(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = _FakeBackend()
        monkeypatch.setattr(extractive_qa, "_get_qa_backend", lambda *a, **k: backend)
        spans = extractive_qa.answer_candidates(
            context="z" * 10_000,
            question="q",
            model_id="fake",
            window_chars=1800,
            top_k=3,
        )
        scores = [s.score for s in spans]
        assert scores == sorted(scores, reverse=True)


class TestExpandSpanToSentence:
    """The QA answer is a fragment; the evidence is the sentence around it."""

    def test_expands_a_fragment_to_its_sentence(self) -> None:
        text = "Nothing here. The CFO is in conflict with the CEO over targets. Nothing after."
        frag_start = text.index("conflict")
        frag_end = frag_start + len("conflict")

        start, end = expand_span_to_sentence(text, frag_start, frag_end)
        quote = text[start:end]

        assert "The CFO is in conflict with the CEO over targets." in quote
        assert "Nothing here" not in quote
        assert len(quote) > (frag_end - frag_start)

    def test_bounded_when_there_is_no_punctuation(self) -> None:
        """A single-line blob transcript must not swallow the whole episode."""
        text = "word " * 5000  # no sentence terminators at all
        start, end = expand_span_to_sentence(text, 2000, 2010)
        assert end - start <= extractive_qa.MAX_QUOTE_CHARS

    def test_degenerate_input_is_returned_unchanged(self) -> None:
        assert expand_span_to_sentence("", 0, 0) == (0, 0)
        assert expand_span_to_sentence("abc", 2, 1) == (2, 1)
