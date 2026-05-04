"""Unit tests for ``bundle_extract_parser`` (#698 Layer A)."""

from __future__ import annotations

import pytest

from podcast_scraper.providers.common.bundle_extract_parser import (
    BundleExtractParseError,
    parse_bundled_extract_response,
)


class TestParseBundledExtractResponse:
    def test_zero_expected_returns_empty_dict(self) -> None:
        assert parse_bundled_extract_response('{"0": ["q"]}', expected_count=0) == {}

    def test_empty_content_raises(self) -> None:
        with pytest.raises(BundleExtractParseError, match="empty"):
            parse_bundled_extract_response("", expected_count=2)
        with pytest.raises(BundleExtractParseError, match="empty"):
            parse_bundled_extract_response("   ", expected_count=2)

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(BundleExtractParseError, match="invalid JSON"):
            parse_bundled_extract_response("{ not json }", expected_count=2)

    def test_non_object_top_level_raises(self) -> None:
        with pytest.raises(BundleExtractParseError, match="top-level"):
            parse_bundled_extract_response("[1,2,3]", expected_count=2)
        with pytest.raises(BundleExtractParseError, match="top-level"):
            parse_bundled_extract_response('"plain string"', expected_count=2)

    def test_simple_mapping(self) -> None:
        content = '{"0": ["a", "b"], "1": ["c"]}'
        out = parse_bundled_extract_response(content, expected_count=2)
        assert out == {0: ["a", "b"], 1: ["c"]}

    def test_missing_index_returns_empty_list(self) -> None:
        # Only insight 0 has quotes; insight 1 + 2 should be empty lists.
        content = '{"0": ["a"]}'
        out = parse_bundled_extract_response(content, expected_count=3)
        assert out == {0: ["a"], 1: [], 2: []}

    def test_extra_indices_outside_expected_dropped(self) -> None:
        content = '{"0": ["a"], "5": ["unexpected"]}'
        out = parse_bundled_extract_response(content, expected_count=2)
        assert 5 not in out
        assert out[0] == ["a"]
        assert out[1] == []

    def test_negative_index_dropped(self) -> None:
        content = '{"-1": ["bad"], "0": ["good"]}'
        out = parse_bundled_extract_response(content, expected_count=1)
        assert -1 not in out
        assert out[0] == ["good"]

    def test_non_int_key_skipped(self) -> None:
        content = '{"foo": ["x"], "0": ["y"]}'
        out = parse_bundled_extract_response(content, expected_count=1)
        assert out == {0: ["y"]}

    def test_int_key_as_int_works(self) -> None:
        # JSON keys are always strings, but defensive: dict mutation upstream.
        content = '{"0": ["a"], "1": ["b"]}'
        out = parse_bundled_extract_response(content, expected_count=2)
        assert out[0] == ["a"]
        assert out[1] == ["b"]

    def test_code_fence_stripped(self) -> None:
        content = '```json\n{"0": ["fenced quote"]}\n```'
        out = parse_bundled_extract_response(content, expected_count=1)
        assert out == {0: ["fenced quote"]}

    def test_bare_code_fence_stripped(self) -> None:
        content = '```\n{"0": ["bare fence"]}\n```'
        out = parse_bundled_extract_response(content, expected_count=1)
        assert out == {0: ["bare fence"]}

    def test_envelope_key_insights(self) -> None:
        content = '{"insights": {"0": ["a"], "1": ["b"]}}'
        out = parse_bundled_extract_response(content, expected_count=2)
        assert out == {0: ["a"], 1: ["b"]}

    def test_envelope_key_quotes(self) -> None:
        content = '{"quotes": {"0": ["a"]}}'
        out = parse_bundled_extract_response(content, expected_count=1)
        assert out == {0: ["a"]}

    def test_envelope_key_by_insight(self) -> None:
        content = '{"by_insight": {"0": ["a"]}}'
        out = parse_bundled_extract_response(content, expected_count=1)
        assert out == {0: ["a"]}

    def test_string_value_normalised_to_list(self) -> None:
        content = '{"0": "single quote string"}'
        out = parse_bundled_extract_response(content, expected_count=1)
        assert out == {0: ["single quote string"]}

    def test_dict_quote_with_text_field(self) -> None:
        content = '{"0": [{"text": "alpha"}, {"quote": "beta"}, {"quote_text": "gamma"}]}'
        out = parse_bundled_extract_response(content, expected_count=1)
        assert out == {0: ["alpha", "beta", "gamma"]}

    def test_dict_quote_without_text_field_skipped(self) -> None:
        content = '{"0": [{"id": 1}, "valid"]}'
        out = parse_bundled_extract_response(content, expected_count=1)
        assert out == {0: ["valid"]}

    def test_empty_strings_skipped(self) -> None:
        content = '{"0": ["", "  ", "real quote", null]}'
        out = parse_bundled_extract_response(content, expected_count=1)
        assert out == {0: ["real quote"]}

    def test_non_list_non_string_value_returns_empty(self) -> None:
        content = '{"0": 123, "1": ["a"]}'
        out = parse_bundled_extract_response(content, expected_count=2)
        assert out[0] == []
        assert out[1] == ["a"]

    def test_null_value_returns_empty(self) -> None:
        content = '{"0": null, "1": ["a"]}'
        out = parse_bundled_extract_response(content, expected_count=2)
        assert out[0] == []
        assert out[1] == ["a"]

    def test_strips_whitespace_from_strings(self) -> None:
        content = '{"0": ["  spaced  ", "trailing\\n"]}'
        out = parse_bundled_extract_response(content, expected_count=1)
        assert out == {0: ["spaced", "trailing"]}

    def test_realistic_envelope_with_extras(self) -> None:
        # Some models like to add commentary alongside the data.
        content = (
            "```json\n"
            '{"insights": {"0": ["First quote.", "Second quote."], "1": []},'
            ' "model_note": "ok"}\n'
            "```"
        )
        out = parse_bundled_extract_response(content, expected_count=2)
        assert out == {0: ["First quote.", "Second quote."], 1: []}
