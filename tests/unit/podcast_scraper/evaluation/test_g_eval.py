"""Unit tests for the G-Eval core (#932).

Covers:

- All four dimension prompts render with the expected rubric + format clauses
- Transcript truncation respects ``max_transcript_chars``
- Parser accepts well-formed JSON, strips code fences, recovers from prepended
  commentary, and rejects bad scores / wrong dimensions
- ``score_summary`` drives all four dimensions and records errors per-dim
  without aborting the rest
- ``agreement_rate`` computes the exact-or-adjacent rate correctly
"""

from __future__ import annotations

from typing import Any

import pytest

from podcast_scraper.evaluation.g_eval import (
    agreement_rate,
    build_dimension_prompt,
    DIMENSIONS,
    DimensionScore,
    parse_dimension_response,
    score_summary,
)
from podcast_scraper.evaluation.judges.base import JudgeResult, JudgeUnavailableError

# ---------------------------------------------------------------------------
# Prompt construction


def test_dimensions_constant_matches_spec() -> None:
    """The dimension tuple must match the #932 spec (order is the report's order)."""
    assert DIMENSIONS == ("faithfulness", "coverage", "coherence", "fluency")


@pytest.mark.parametrize("dim", DIMENSIONS)
def test_build_dimension_prompt_includes_rubric_and_format_for_each_dim(dim: str) -> None:
    """Each dimension prompt embeds its rubric anchors + the strict JSON spec."""
    prompt = build_dimension_prompt(
        dimension=dim, transcript="some transcript", summary="some summary"
    )
    assert "### Rubric" in prompt
    assert "### Transcript" in prompt
    assert "### Candidate summary" in prompt
    # Dimension name appears in both the rubric header and the JSON spec.
    assert dim.upper() in prompt  # rubric headers use uppercase
    assert f'"dimension": "{dim}"' in prompt
    # Strict format guard for downstream parsing
    assert "Do not include markdown code fences" in prompt


def test_build_dimension_prompt_truncates_long_transcript() -> None:
    """Transcript longer than the cap is sliced; summary is never truncated."""
    long_transcript = "x" * 50_000
    prompt = build_dimension_prompt(
        dimension="faithfulness",
        transcript=long_transcript,
        summary="short",
        max_transcript_chars=1000,
    )
    # Sliced fragment present, but not the full 50k.
    assert "x" * 1000 in prompt
    assert "x" * 1001 not in prompt


def test_build_dimension_prompt_rejects_unknown_dimension() -> None:
    """Typo-defense — unknown dimensions raise rather than silently degrade."""
    with pytest.raises(ValueError, match="Unknown G-Eval dimension"):
        build_dimension_prompt(dimension="hallucination", transcript="t", summary="s")


# ---------------------------------------------------------------------------
# Response parsing


def test_parse_dimension_response_happy_path() -> None:
    """Well-formed JSON parses cleanly."""
    text = '{"dimension": "faithfulness", "score": 4, "explanation": "mostly grounded"}'
    parsed = parse_dimension_response(
        text, expected_dimension="faithfulness", judge_model="claude-sonnet-4-6"
    )
    assert parsed.dimension == "faithfulness"
    assert parsed.score == 4
    assert parsed.explanation == "mostly grounded"
    assert parsed.judge_model == "claude-sonnet-4-6"


def test_parse_dimension_response_strips_code_fences() -> None:
    """Some judges sneak in ```json fences despite the instruction."""
    text = '```json\n{"dimension": "coverage", "score": 5, "explanation": "complete"}\n```'
    parsed = parse_dimension_response(text, expected_dimension="coverage")
    assert parsed.score == 5


def test_parse_dimension_response_recovers_from_prepended_commentary() -> None:
    """Locate the first ``{...}`` even when the judge editorializes first."""
    text = (
        "Sure, here's my assessment:\n"
        '{"dimension": "coherence", "score": 3, "explanation": "flows but bumpy"}'
    )
    parsed = parse_dimension_response(text, expected_dimension="coherence")
    assert parsed.score == 3


def test_parse_dimension_response_rejects_score_out_of_range() -> None:
    """Anything outside [1, 5] is a hard error — anchors are integers."""
    text = '{"dimension": "fluency", "score": 6, "explanation": "x"}'
    with pytest.raises(ValueError, match="out of range"):
        parse_dimension_response(text, expected_dimension="fluency")


def test_parse_dimension_response_rejects_dimension_mismatch() -> None:
    """Judge returned a different dimension → caller bug or judge confused; fail loudly."""
    text = '{"dimension": "faithfulness", "score": 4, "explanation": "x"}'
    with pytest.raises(ValueError, match="expected 'coverage'"):
        parse_dimension_response(text, expected_dimension="coverage")


def test_parse_dimension_response_rejects_non_integer_score() -> None:
    """Reject non-integer scores — G-Eval anchors are discrete ordinals."""
    text = '{"dimension": "fluency", "score": "good", "explanation": "x"}'
    with pytest.raises(ValueError, match="not an integer"):
        parse_dimension_response(text, expected_dimension="fluency")


def test_parse_dimension_response_empty_raises() -> None:
    with pytest.raises(ValueError, match="Empty"):
        parse_dimension_response("", expected_dimension="faithfulness", judge_model="m")


def test_parse_dimension_response_no_json_raises() -> None:
    with pytest.raises(ValueError, match="No JSON object"):
        parse_dimension_response("I cannot score this.", expected_dimension="fluency")


# ---------------------------------------------------------------------------
# score_summary orchestration


class _FakeJudge:
    """Programmable judge stub for ``score_summary`` tests.

    ``script`` maps dimension → callable(prompt) → JudgeResult (or raises).
    """

    def __init__(self, model: str, script: dict[str, Any]) -> None:
        self.model = model
        self._script = script
        self.calls: list[tuple[str, int]] = []

    def score(self, prompt: str, *, max_tokens: int = 512) -> JudgeResult:
        # Recover dimension from the rendered prompt (matches build_dimension_prompt).
        dim = None
        for candidate in DIMENSIONS:
            if f'"dimension": "{candidate}"' in prompt:
                dim = candidate
                break
        assert dim is not None, "Test bug: no dimension found in prompt"
        self.calls.append((dim, max_tokens))
        action = self._script[dim]
        if callable(action):
            result: JudgeResult = action(prompt)
            return result
        raise AssertionError(f"Unscripted dim={dim}")


def _ok_result(text: str, cost: float = 0.001) -> JudgeResult:
    return JudgeResult(
        text=text, model="fake", prompt_tokens=100, completion_tokens=20, cost_usd=cost
    )


def test_score_summary_drives_all_four_dimensions() -> None:
    """Happy-path: judge succeeds on all four dimensions → mean is 3.5."""
    script = {
        "faithfulness": lambda _p: _ok_result(
            '{"dimension": "faithfulness", "score": 5, "explanation": "x"}'
        ),
        "coverage": lambda _p: _ok_result(
            '{"dimension": "coverage", "score": 4, "explanation": "x"}'
        ),
        "coherence": lambda _p: _ok_result(
            '{"dimension": "coherence", "score": 3, "explanation": "x"}'
        ),
        "fluency": lambda _p: _ok_result(
            '{"dimension": "fluency", "score": 2, "explanation": "x"}'
        ),
    }
    judge = _FakeJudge("fake-model", script)
    result = score_summary(
        run_id="run-1",
        episode_id="ep-1",
        transcript="t",
        summary="s",
        judge=judge,
    )

    assert set(result.per_dimension) == set(DIMENSIONS)
    assert result.per_dimension["faithfulness"].score == 5
    assert result.per_dimension["coverage"].score == 4
    assert result.per_dimension["coherence"].score == 3
    assert result.per_dimension["fluency"].score == 2
    assert result.mean == pytest.approx(3.5)
    # 4 calls × $0.001
    assert result.total_cost_usd == pytest.approx(0.004)
    assert result.total_prompt_tokens == 400
    assert result.errors == {}
    # Confirm the judge was invoked for each of the 4 dimensions, in order.
    assert [c[0] for c in judge.calls] == list(DIMENSIONS)


def test_score_summary_records_transport_error_and_continues() -> None:
    """A JudgeUnavailableError on one dim is recorded; the rest still run."""

    def boom(_prompt: str) -> JudgeResult:
        raise JudgeUnavailableError("rate limited")

    script = {
        "faithfulness": boom,
        "coverage": lambda _p: _ok_result(
            '{"dimension": "coverage", "score": 4, "explanation": "x"}'
        ),
        "coherence": lambda _p: _ok_result(
            '{"dimension": "coherence", "score": 3, "explanation": "x"}'
        ),
        "fluency": lambda _p: _ok_result(
            '{"dimension": "fluency", "score": 2, "explanation": "x"}'
        ),
    }
    judge = _FakeJudge("fake-model", script)
    result = score_summary(
        run_id="run-1",
        episode_id="ep-1",
        transcript="t",
        summary="s",
        judge=judge,
    )
    # 3 successes + 1 error
    assert set(result.per_dimension) == {"coverage", "coherence", "fluency"}
    assert "faithfulness" in result.errors
    assert result.errors["faithfulness"].startswith("transport:")
    # mean ignores the missing dimension
    assert result.mean == pytest.approx((4 + 3 + 2) / 3)


def test_score_summary_records_parse_error_separately() -> None:
    """A judge that returns garbage JSON is recorded with ``parse:`` prefix."""
    script = {
        "faithfulness": lambda _p: _ok_result("not json"),
        "coverage": lambda _p: _ok_result(
            '{"dimension": "coverage", "score": 4, "explanation": "x"}'
        ),
        "coherence": lambda _p: _ok_result(
            '{"dimension": "coherence", "score": 3, "explanation": "x"}'
        ),
        "fluency": lambda _p: _ok_result(
            '{"dimension": "fluency", "score": 2, "explanation": "x"}'
        ),
    }
    judge = _FakeJudge("fake-model", script)
    result = score_summary(
        run_id="run-1",
        episode_id="ep-1",
        transcript="t",
        summary="s",
        judge=judge,
    )
    assert "faithfulness" in result.errors
    assert result.errors["faithfulness"].startswith("parse:")
    assert result.per_dimension["coverage"].score == 4


def test_score_summary_as_dict_round_trips_fields() -> None:
    """``as_dict`` exposes everything the finale_runner needs to persist."""
    script = {
        "faithfulness": lambda _p: _ok_result(
            '{"dimension": "faithfulness", "score": 5, "explanation": "ok"}'
        ),
        "coverage": lambda _p: _ok_result(
            '{"dimension": "coverage", "score": 4, "explanation": "ok"}'
        ),
        "coherence": lambda _p: _ok_result(
            '{"dimension": "coherence", "score": 3, "explanation": "ok"}'
        ),
        "fluency": lambda _p: _ok_result(
            '{"dimension": "fluency", "score": 2, "explanation": "ok"}'
        ),
    }
    judge = _FakeJudge("fake-model", script)
    result = score_summary(run_id="r", episode_id="e", transcript="t", summary="s", judge=judge)
    d = result.as_dict()
    assert d["run_id"] == "r"
    assert d["episode_id"] == "e"
    assert d["judge_model"] == "fake-model"
    assert d["mean"] == pytest.approx(3.5)
    assert set(d["per_dimension"]) == set(DIMENSIONS)
    assert d["per_dimension"]["faithfulness"]["score"] == 5


# ---------------------------------------------------------------------------
# Agreement rate


def _ds(dim: str, score: int) -> DimensionScore:
    return DimensionScore(dimension=dim, score=score, explanation="")


def test_agreement_rate_exact_or_adjacent_default_tolerance() -> None:
    """Default tolerance=1 → exact-or-adjacent counts as agreement."""
    a = [_ds("faithfulness", 5), _ds("coverage", 4), _ds("coherence", 3), _ds("fluency", 2)]
    b = [_ds("faithfulness", 5), _ds("coverage", 5), _ds("coherence", 1), _ds("fluency", 2)]
    rate, agree, total = agreement_rate(a, b)
    # exact: 5-5, 2-2; adjacent: 4-5; disagree: 3 vs 1 (delta 2)
    assert agree == 3
    assert total == 4
    assert rate == pytest.approx(0.75)


def test_agreement_rate_strict_tolerance_zero() -> None:
    """Tolerance=0 → only exact matches count."""
    a = [_ds("faithfulness", 5), _ds("coverage", 4)]
    b = [_ds("faithfulness", 5), _ds("coverage", 5)]
    rate, agree, total = agreement_rate(a, b, tolerance=0)
    assert agree == 1
    assert total == 2
    assert rate == 0.5


def test_agreement_rate_empty_inputs_return_zero_signal() -> None:
    """Empty in → rate=0.0, total=0 (caller treats as 'no signal')."""
    rate, agree, total = agreement_rate([], [])
    assert (rate, agree, total) == (0.0, 0, 0)


def test_agreement_rate_misaligned_dimensions_raise() -> None:
    """Order matters: caller must pre-align by dimension."""
    a = [_ds("faithfulness", 5)]
    b = [_ds("coverage", 5)]
    with pytest.raises(ValueError, match="Score alignment"):
        agreement_rate(a, b)
