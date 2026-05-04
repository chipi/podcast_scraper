"""Unit tests for ``bundle_nli_parser`` (#698 Layer B)."""

from __future__ import annotations

import pytest

from podcast_scraper.providers.common.bundle_nli_parser import (
    BundleNliParseError,
    parse_bundled_nli_response,
)


class TestParseBundledNliResponse:
    def test_zero_expected_returns_empty(self) -> None:
        assert parse_bundled_nli_response('{"0": 0.5}', expected_count=0) == {}

    def test_empty_content_raises(self) -> None:
        with pytest.raises(BundleNliParseError, match="empty"):
            parse_bundled_nli_response("", expected_count=2)

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(BundleNliParseError, match="invalid JSON"):
            parse_bundled_nli_response("{ not json }", expected_count=2)

    def test_non_object_top_level_raises(self) -> None:
        with pytest.raises(BundleNliParseError, match="top-level"):
            parse_bundled_nli_response("[0.5, 0.6]", expected_count=2)

    def test_simple_mapping(self) -> None:
        out = parse_bundled_nli_response('{"0": 0.85, "1": 0.42}', expected_count=2)
        assert out == {0: 0.85, 1: 0.42}

    def test_missing_pair_omitted_from_output(self) -> None:
        # Caller treats absence as "no score for that pair".
        out = parse_bundled_nli_response('{"0": 0.7}', expected_count=3)
        assert out == {0: 0.7}
        assert 1 not in out
        assert 2 not in out

    def test_extra_indices_outside_expected_dropped(self) -> None:
        out = parse_bundled_nli_response('{"0": 0.5, "5": 0.9}', expected_count=2)
        assert 5 not in out
        assert out[0] == 0.5

    def test_negative_index_dropped(self) -> None:
        out = parse_bundled_nli_response('{"-1": 0.5, "0": 0.7}', expected_count=1)
        assert -1 not in out
        assert out[0] == 0.7

    def test_string_score_coerced(self) -> None:
        out = parse_bundled_nli_response('{"0": "0.8"}', expected_count=1)
        assert out[0] == 0.8

    def test_int_score_coerced(self) -> None:
        out = parse_bundled_nli_response('{"0": 1, "1": 0}', expected_count=2)
        assert out == {0: 1.0, 1: 0.0}

    def test_score_above_one_clamped(self) -> None:
        out = parse_bundled_nli_response('{"0": 1.5}', expected_count=1)
        assert out[0] == 1.0

    def test_score_below_zero_clamped(self) -> None:
        out = parse_bundled_nli_response('{"0": -0.3}', expected_count=1)
        assert out[0] == 0.0

    def test_non_numeric_value_dropped(self) -> None:
        out = parse_bundled_nli_response('{"0": "high", "1": 0.5}', expected_count=2)
        assert 0 not in out
        assert out[1] == 0.5

    def test_null_value_dropped(self) -> None:
        out = parse_bundled_nli_response('{"0": null, "1": 0.5}', expected_count=2)
        assert 0 not in out
        assert out[1] == 0.5

    def test_bool_value_dropped(self) -> None:
        # Python bool is subclass of int; explicit reject keeps semantics tight.
        out = parse_bundled_nli_response('{"0": true, "1": 0.5}', expected_count=2)
        assert 0 not in out
        assert out[1] == 0.5

    def test_code_fence_stripped(self) -> None:
        out = parse_bundled_nli_response(
            '```json\n{"0": 0.7}\n```',
            expected_count=1,
        )
        assert out[0] == 0.7

    def test_envelope_key_scores(self) -> None:
        out = parse_bundled_nli_response(
            '{"scores": {"0": 0.7, "1": 0.4}}',
            expected_count=2,
        )
        assert out == {0: 0.7, 1: 0.4}

    def test_envelope_key_entailment(self) -> None:
        out = parse_bundled_nli_response(
            '{"entailment": {"0": 0.9}}',
            expected_count=1,
        )
        assert out == {0: 0.9}

    def test_envelope_key_results(self) -> None:
        out = parse_bundled_nli_response(
            '{"results": {"0": 0.6}}',
            expected_count=1,
        )
        assert out == {0: 0.6}

    def test_non_int_key_skipped(self) -> None:
        out = parse_bundled_nli_response('{"foo": 0.5, "0": 0.7}', expected_count=1)
        assert out == {0: 0.7}

    def test_realistic_envelope_with_extras(self) -> None:
        content = (
            "```json\n" '{"scores": {"0": 0.92, "1": 0.15, "2": 0.78}, "model_note": "ok"}\n' "```"
        )
        out = parse_bundled_nli_response(content, expected_count=3)
        assert out == {0: 0.92, 1: 0.15, 2: 0.78}
