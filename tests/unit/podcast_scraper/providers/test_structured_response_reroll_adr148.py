"""Repro + spec for ADR-148: one in-place re-roll for an invalid structured LLM response.

Rule #34 (repro before fix). Reproduces the p04 failure mode observed on a v2.5 mock reprocess:
a structured LLM call (summary) returns truncated/invalid JSON on the FIRST attempt and valid JSON
on a RE-ROLL (LLM is non-deterministic even at temperature 0). Today the invalid response fails the
call with no in-place retry — response-shape violations are non-retryable in place (ADR-100), and
the summary schema check runs a layer above the call so it never reaches fallover either; the whole
EPISODE fails on a one-off bad response.

The fix (ADR-148): validation co-located at the call, and a response-shape violation triggers ONE
bounded in-place re-roll on the same endpoint before falling over to another provider. These tests
pin the two invariants that fix must satisfy. They are xfail until the capability lands, so the
repro is recorded without breaking the build.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_transient_invalid_structured_response_recovers_via_one_in_place_reroll() -> None:
    """A structured call whose 1st response is invalid and 2nd is valid must recover on the SAME
    endpoint via one in-place re-roll — NOT fail, and NOT immediately fall over to another provider.

    Wired against the real seam once ADR-148 lands (a call-level validate + bounded re-roll). Until
    then this asserts the intent so the repro is on record."""
    from podcast_scraper.providers.guardrails import structured_call_with_reroll  # type: ignore

    calls = {"n": 0}

    def make_call() -> str:
        calls["n"] += 1
        # 1st: truncated/invalid JSON (the p04 shape); 2nd: valid.
        return '{"bullets": [' if calls["n"] == 1 else '{"bullets": ["a real bullet"]}'

    def validate(content: str) -> None:
        import json

        json.loads(content)  # raises on the truncated first response

    out = structured_call_with_reroll(make_call, validate, service="vllm", max_reroll=1)
    assert calls["n"] == 2  # one re-roll on the same endpoint
    assert '"bullets"' in out


def test_persistent_invalid_response_exhausts_reroll_then_signals_fallover() -> None:
    """A persistently-invalid structured response must, after the bounded in-place re-roll, raise a
    GuardrailViolation so the existing ADR-100 FallbackAware chain fallovers (then episode-fails).
    Preserves the fallover path — the re-roll is PREPENDED, not a replacement."""
    from podcast_scraper.providers.guardrails import (  # type: ignore
        GuardrailViolation,
        structured_call_with_reroll,
    )

    def make_call() -> str:
        return '{"bullets": ['  # always invalid

    def validate(content: str) -> None:
        import json

        json.loads(content)

    with pytest.raises(GuardrailViolation):
        structured_call_with_reroll(make_call, validate, service="vllm", max_reroll=1)
