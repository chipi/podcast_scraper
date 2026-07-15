"""The QA span score must mean something ACROSS windows, not within one.

A 70k transcript is scored in ~30 overlapping windows. If each window's score is normalised inside
that window, then the best span of *every* window comes back at ~1.0 — including a window that
contains nothing relevant. The cross-window winner is then arbitrary, and ``qa_score_min`` gates on
nothing. That is exactly what happened: the local grounder returned the single word ``"Codex"`` at
``qa_score=1.000`` as the evidence for insight after insight.

Both halves of that fix were unguarded — found by re-breaking them and watching the suite stay
green. These tests pin the two invariants that make the score comparable:

1. probabilities are absolute (softmax over the whole chunk), so a window about nothing scores LOW;
2. the distribution is masked to CLS + context first, so padding and question tokens cannot absorb
   the probability mass and make the model look like it abstains everywhere.

No model is downloaded: the tokenizer and the QA head are stubbed so the *scoring maths* — the part
that broke — is exercised directly and deterministically.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

torch = pytest.importorskip("torch")

from podcast_scraper.providers.ml.extractive_qa import QAEvidenceBackend  # noqa: E402

# Two windows over one document. Window 0 holds the answer; window 1 is filler.
CONTEXT = "The answer is insulin resistance. " + ("Filler text about nothing at all. " * 2)
_SEQ_LEN = 8

# token id -> (char_start, char_end) in CONTEXT, per chunk.
# index 0 = CLS (0,0); indices 1-2 = question tokens; 3+ = context; last = padding.
_OFFSETS = [
    [(0, 0), (0, 0), (0, 0), (0, 3), (4, 10), (11, 13), (14, 32), (0, 0)],  # window 0
    [(0, 0), (0, 0), (0, 0), (34, 40), (41, 45), (46, 51), (52, 60), (0, 0)],  # window 1
]
# sequence_ids: None = special, 0 = question, 1 = context.
_SEQ_IDS = [
    [None, 0, 0, 1, 1, 1, 1, None],
    [None, 0, 0, 1, 1, 1, 1, None],
]


class _FakeBatch(dict):
    def __init__(self, data: Dict[str, Any], seq_ids: List[List[Any]]) -> None:
        super().__init__(data)
        self._seq_ids = seq_ids

    def sequence_ids(self, idx: int) -> List[Any]:
        return self._seq_ids[idx]


class _FakeTokenizer:
    def __call__(self, *_a: Any, **_k: Any) -> _FakeBatch:
        return _FakeBatch(
            {
                "input_ids": torch.zeros((2, _SEQ_LEN), dtype=torch.long),
                "attention_mask": torch.ones((2, _SEQ_LEN), dtype=torch.long),
                "offset_mapping": torch.tensor(_OFFSETS),
                "overflow_to_sample_mapping": torch.zeros(2, dtype=torch.long),
            },
            _SEQ_IDS,
        )


class _Out:
    def __init__(self, s: Any, e: Any) -> None:
        self.start_logits = s
        self.end_logits = e


class _FakeModel:
    """A QA head that is CONFIDENT in window 0 and has no idea in window 1.

    ``outside_logit`` loads the tokens OUTSIDE the context — the question tokens (1, 2) and the
    padding (7). CLS (0) is deliberately left alone: the mask keeps CLS in the distribution on
    purpose, because it is the null-answer slot.
    """

    def __init__(self, outside_logit: float = 0.0) -> None:
        self.outside_logit = outside_logit

    def parameters(self):  # noqa: ANN201
        yield torch.zeros(1)

    def __call__(self, **_k: Any) -> _Out:
        p = self.outside_logit
        # window 0: a peak on the real answer span (tokens 3 -> 6).
        # window 1: flat — the model genuinely has nothing to say here.
        start = torch.tensor(
            [[0.0, p, p, 3.0, 0.0, 0.0, 0.0, p], [0.0, p, p, 0.0, 0.0, 0.0, 0.0, p]]
        )
        end = torch.tensor([[0.0, p, p, 0.0, 0.0, 0.0, 3.0, p], [0.0, p, p, 0.0, 0.0, 0.0, 0.0, p]])
        return _Out(start, end)


def _backend(outside_logit: float = 0.0) -> QAEvidenceBackend:
    be = object.__new__(QAEvidenceBackend)
    be.tokenizer = _FakeTokenizer()  # type: ignore[attr-defined]
    be.model = _FakeModel(outside_logit)  # type: ignore[attr-defined]
    return be


def _spans(outside_logit: float = 0.0):  # noqa: ANN202
    return _backend(outside_logit).answer_top_k("q?", CONTEXT, top_k=8, max_seq_len=_SEQ_LEN)


def test_a_window_about_nothing_scores_lower_than_the_window_with_the_evidence() -> None:
    """THE 'Codex at 1.000' BUG.

    Softmax within each window's retained spans makes the top span of EVERY window ~1.0. The filler
    window then ties with the real one and the winner is arbitrary. Absolute probabilities separate
    them.
    """
    spans = _spans()
    assert spans, "no spans returned"

    best = spans[0]
    assert "insulin resistance" in best.answer, f"the evidence window did not win: {best.answer!r}"

    filler = [s for s in spans if s.start >= 34]
    assert filler, "the filler window produced no spans — the test is not exercising the comparison"
    assert max(s.score for s in filler) < best.score / 2, (
        "the filler window scores as high as the window holding the answer — the scores are "
        "normalised per-window and mean nothing across windows"
    )


def test_the_score_is_a_probability_not_a_certainty() -> None:
    """A per-window softmax pins the best span at ~1.0. An absolute one cannot."""
    assert _spans()[0].score < 0.9, "the top span is saturated — this is the per-window softmax"


def test_padding_and_question_tokens_cannot_take_the_probability_mass() -> None:
    """THE MASK.

    Padding sits outside the context. Give it a huge logit and, unmasked, it swallows the softmax
    and every real span collapses toward zero — the model looks like it abstains everywhere. Masked
    to CLS + context, the answer is unaffected.
    """
    masked = _spans(outside_logit=50.0)
    assert masked, "no spans survived — padding absorbed the distribution"
    assert "insulin resistance" in masked[0].answer
    assert masked[0].score == pytest.approx(_spans(outside_logit=0.0)[0].score, rel=1e-6), (
        "a logit on a PADDING token changed the answer's score — the distribution is not masked to "
        "CLS + context"
    )
