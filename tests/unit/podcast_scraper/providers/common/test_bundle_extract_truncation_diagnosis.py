"""A cut-off response and a malformed one must not read identically.

extract_quotes_bundled IS the evidence stack: when it fails, insights lose their grounding
quotes, and under ``gi_require_grounding: true`` an ungrounded insight can be dropped entirely.
The 2026-08-16 acceptance run hit 4 failures in ~11 episodes, all on this one call:

    invalid JSON: Unterminated string at char 1364
    invalid JSON: Unterminated string at char 4986
    invalid JSON: Unterminated string at char 10630
    invalid JSON: Unterminated string at char 10992

Those messages are compatible with two different faults that need opposite fixes — the model
ran out of output budget mid-write (raise max_tokens) or the model emitted structurally bad
JSON (change the prompt or the strategy). Nothing in the message distinguished them, so the
run produced four failures and zero actionable information.

These tests pin the classification. They do NOT assert that the acceptance-run failures were
truncation — that is what the instrumentation is for measuring.
"""

from __future__ import annotations

import json

import pytest

from podcast_scraper.providers.common.bundle_extract_parser import (
    BundleExtractParseError,
    parse_bundled_extract_response,
)


def _parse(content: str, expected: int = 3):
    return parse_bundled_extract_response(content, expected_count=expected)


def test_a_response_cut_off_mid_string_is_diagnosed_as_truncation():
    """The exact shape seen in production: valid JSON that simply stops mid-quote."""
    cut_off = '{"0": ["a real quote that never finishes because the budget ran ou'

    with pytest.raises(BundleExtractParseError) as caught:
        _parse(cut_off)

    exc = caught.value
    assert exc.truncation_suspected is True
    assert "DOCUMENT_ENDED_EARLY" in str(exc)
    assert exc.content_length == len(cut_off)
    assert exc.error_position is not None


def test_structurally_malformed_json_is_not_blamed_on_the_budget():
    """A complete document with a mistake in the MIDDLE is the model's fault, not the budget's."""
    malformed = '{"0": ["fine"], "1": [oops not quoted], "2": ["also fine"], "3": ["padding"]}'

    with pytest.raises(BundleExtractParseError) as caught:
        _parse(malformed)

    exc = caught.value
    assert exc.truncation_suspected is False, (
        "raising max_tokens would not fix this, and saying so would send the operator "
        "chasing the wrong fix"
    )
    assert "MALFORMED_MID_DOCUMENT" in str(exc)


def test_the_message_carries_the_numbers_needed_to_act():
    """Every provider already logs the exception; the numbers must ride along in the message."""
    cut_off = '{"0": ["stops here'

    with pytest.raises(BundleExtractParseError) as caught:
        _parse(cut_off)

    text = str(caught.value)
    assert "chars=" in text
    assert "fail_at=" in text
    assert "diagnosis=" in text


def test_the_diagnosis_survives_code_fences():
    """Models fence their JSON; the offsets must describe the JSON, not the wrapper."""
    fenced = '```json\n{"0": ["a quote that gets cut off right he'

    with pytest.raises(BundleExtractParseError) as caught:
        _parse(fenced)

    exc = caught.value
    assert exc.truncation_suspected is True
    assert exc.content_length < len(fenced), "length should describe the stripped JSON"


def test_an_early_failure_in_a_long_document_is_not_truncation():
    """Position matters: a break at char 20 of a 5000-char body is not a budget cutoff."""
    body = '{"0": [BAD], ' + '"pad": ["' + ("x" * 5000) + '"]}'

    with pytest.raises(BundleExtractParseError) as caught:
        _parse(body)

    assert caught.value.truncation_suspected is False


def test_empty_content_still_reports_a_length():
    with pytest.raises(BundleExtractParseError) as caught:
        _parse("")

    assert caught.value.content_length == 0
    assert caught.value.truncation_suspected is False


def test_a_wrong_top_level_type_is_not_a_truncation():
    with pytest.raises(BundleExtractParseError) as caught:
        _parse('["not", "an", "object"]')

    exc = caught.value
    assert exc.truncation_suspected is False
    assert "top-level must be an object" in str(exc)


def test_valid_responses_are_unaffected():
    """The instrumentation must not change what a healthy parse returns."""
    parsed = _parse('{"0": ["quote a"], "1": [{"text": "quote b"}], "2": []}')

    assert parsed == {0: ["quote a"], 1: ["quote b"], 2: []}


def test_the_error_is_still_catchable_as_before():
    """Callers catch BundleExtractParseError; the new fields must not break that contract."""
    with pytest.raises(BundleExtractParseError):
        _parse("{definitely not json")

    # and it is still an ordinary Exception carrying a message
    try:
        _parse("{definitely not json")
    except BundleExtractParseError as exc:
        assert str(exc)
        assert isinstance(exc, Exception)


def test_an_unterminated_string_is_truncation_even_though_its_offset_looks_early():
    """The offset trap, pinned. Regressing this classifies every production failure backwards.

    Python reports "Unterminated string" at the position where the string STARTED, not where
    the content stopped. Here a long quote opens near the beginning and never closes, so the
    reported offset is tiny while the document is long — the exact inverse of what a naive
    "is the failure near the end?" test expects.

    A string that never closes consumes everything after it, so the document provably ended
    early no matter what the offset says. The production logs read "Unterminated string at char
    10630"; that 10630 is an opening quote, not a cutoff point.
    """
    body = '{"0": ["' + ("padding text " * 400)  # opens at char 7, runs 5000+ chars, never closes

    with pytest.raises(BundleExtractParseError) as caught:
        _parse(body)

    exc = caught.value
    assert (
        exc.error_position is not None and exc.error_position < 20
    ), "precondition: the reported offset is near the START of a long document"
    assert exc.content_length > 1000
    assert exc.truncation_suspected is True, (
        "an unterminated string proves the document ended early; a position test built on "
        "the reported offset would call this malformed and send the operator to the wrong fix"
    )


def test_json_decode_error_is_chained_for_debuggers():
    """``raise ... from exc`` — the original traceback must not be thrown away."""
    with pytest.raises(BundleExtractParseError) as caught:
        _parse('{"0": ["unterminated')

    assert isinstance(caught.value.__cause__, json.JSONDecodeError)
