"""Unit tests for the output postprocessor registry — focused on the
``extract_json_summary_field`` entry added during #1016 Round 3 for Kimi-Linear
JSON-mode experiments. Verifies the graceful-fallback semantics on malformed
or shape-mismatched input.
"""

from __future__ import annotations

import pytest

from podcast_scraper.evaluation.output_postprocess import (
    extract_json_summary_field,
    get_postprocessor,
    noop,
    REGISTRY,
)


@pytest.mark.unit
class TestExtractJsonSummaryField:
    """JSON-mode postprocessor for ``{"summary": "..."}`` candidate outputs."""

    def test_valid_json_with_summary_returns_summary_string(self) -> None:
        text = '{"summary": "This episode covers trail building."}'
        assert extract_json_summary_field(text) == "This episode covers trail building."

    def test_valid_json_with_extra_fields_still_extracts_summary(self) -> None:
        text = '{"title": "Ep 1", "summary": "Trail building.", "bullets": ["a"]}'
        assert extract_json_summary_field(text) == "Trail building."

    def test_fenced_json_block_is_stripped_before_extraction(self) -> None:
        text = '```json\n{"summary": "Fenced summary."}\n```'
        assert extract_json_summary_field(text) == "Fenced summary."

    def test_fenced_block_without_language_tag_is_stripped(self) -> None:
        text = '```\n{"summary": "No-lang fence."}\n```'
        assert extract_json_summary_field(text) == "No-lang fence."

    def test_malformed_json_returns_original_text(self) -> None:
        text = '{"summary": "missing close brace'
        # Graceful fallback: return raw text so the scorer + manual review
        # see what the model actually emitted, instead of silently losing it.
        assert extract_json_summary_field(text) == text

    def test_valid_json_without_summary_field_returns_original(self) -> None:
        text = '{"title": "Ep 1", "bullets": ["a"]}'
        assert extract_json_summary_field(text) == text

    def test_summary_field_not_a_string_returns_original(self) -> None:
        # ``{"summary": null}`` and ``{"summary": 42}`` both fall through —
        # only ``str`` summaries are extracted.
        for bad in ('{"summary": null}', '{"summary": 42}', '{"summary": [1, 2]}'):
            assert extract_json_summary_field(bad) == bad

    def test_empty_string_input_returns_empty_string(self) -> None:
        assert extract_json_summary_field("") == ""

    def test_non_json_prose_returns_unchanged(self) -> None:
        text = "This is just plain prose, not JSON at all."
        assert extract_json_summary_field(text) == text


@pytest.mark.unit
class TestRegistryWiring:
    """The registry must expose ``extract_json_summary_field`` so YAML
    configs can name it without import surgery."""

    def test_extract_json_summary_field_is_registered(self) -> None:
        assert "extract_json_summary_field" in REGISTRY
        assert REGISTRY["extract_json_summary_field"] is extract_json_summary_field

    def test_get_postprocessor_resolves_extract_json_summary_field(self) -> None:
        fn = get_postprocessor("extract_json_summary_field")
        assert fn is extract_json_summary_field

    def test_get_postprocessor_with_none_returns_noop(self) -> None:
        assert get_postprocessor(None) is noop

    def test_get_postprocessor_with_unknown_name_raises_keyerror(self) -> None:
        with pytest.raises(KeyError) as excinfo:
            get_postprocessor("does_not_exist")
        # Error message should name the missing entry + list known keys
        # so the operator can fix the YAML config quickly.
        assert "does_not_exist" in str(excinfo.value)
        assert "extract_json_summary_field" in str(excinfo.value)
