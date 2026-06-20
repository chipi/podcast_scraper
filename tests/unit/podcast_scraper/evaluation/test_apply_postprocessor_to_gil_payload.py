"""Tests for #111 — apply the eval postprocessor to GIL / KG node text fields.

Originally surfaced as the DSV2-Lite 0% coverage bug in #1016 final report
§ 6a: the configured ``decode_r1_byte_level`` postprocessor was applied to
summary text but NOT to ``gil.nodes[].properties.{label,name,description}``,
leaving byte-level BPE artifacts in node labels and making them
unrecognizable to the silver-comparison scorer.

These tests exercise the new ``apply_postprocessor_to_gil_payload`` helper
in :mod:`podcast_scraper.evaluation.output_postprocess`.
"""

from __future__ import annotations

import pytest

from podcast_scraper.evaluation.output_postprocess import (
    apply_postprocessor_to_gil_payload,
    decode_r1_byte_level,
    get_postprocessor,
    noop,
)


@pytest.mark.unit
class TestApplyPostprocessorToGilPayload:
    """Behaviour matrix for the new GI/KG postprocessor walker."""

    def test_decodes_byte_level_in_properties_label(self) -> None:
        # Realistic DSV2-Lite shape: byte-level BPE in node properties.label.
        gil = {
            "nodes": [
                {"type": "Insight", "properties": {"label": "ĠepisodeĠdiscussesĠbuilding"}},
                {"type": "Topic", "properties": {"label": "trailĠbuilding"}},
            ],
            "edges": [],
        }
        out = apply_postprocessor_to_gil_payload(gil, decode_r1_byte_level)
        assert out["nodes"][0]["properties"]["label"] == " episode discusses building"
        assert out["nodes"][1]["properties"]["label"] == "trail building"

    def test_decodes_all_text_fields_label_name_description(self) -> None:
        gil = {
            "nodes": [
                {
                    "type": "Entity",
                    "properties": {
                        "label": "ĠAcmeĠCorp",
                        "name": "ĠAcmeĠCorp",
                        "description": "AĠcompanyĊfounded in 2020",
                    },
                },
            ],
            "edges": [],
        }
        out = apply_postprocessor_to_gil_payload(gil, decode_r1_byte_level)
        props = out["nodes"][0]["properties"]
        assert props["label"] == " Acme Corp"
        assert props["name"] == " Acme Corp"
        # Newline byte-level token (Ċ) decoded too.
        assert props["description"] == "A company\nfounded in 2020"

    def test_decodes_top_level_label_on_legacy_node_shape(self) -> None:
        # Legacy shape: text fields directly on the node, not nested in properties.
        gil = {
            "nodes": [
                {"type": "Topic", "label": "trailĠbuilding"},
            ],
            "edges": [],
        }
        out = apply_postprocessor_to_gil_payload(gil, decode_r1_byte_level)
        assert out["nodes"][0]["label"] == "trail building"

    def test_noop_short_circuits_with_zero_walks(self) -> None:
        """noop postprocessor returns the payload unchanged without iterating —
        important for the common case where clean models don't declare a
        postprocessor and the get_postprocessor(None) call returns noop."""
        sentinel = object()

        class _SpyDict(dict):
            def get(self, *args, **kwargs):  # type: ignore[override]
                # If noop didn't short-circuit, we'd see iteration here.
                pytest.fail("noop should short-circuit without inspecting payload")

        result = apply_postprocessor_to_gil_payload(_SpyDict(), noop)
        assert isinstance(result, _SpyDict)
        # Also test the more realistic path — registry lookup with no name
        # returns noop, and that path should still short-circuit.
        gil = {"nodes": [{"properties": {"label": "Ġfoo"}}], "edges": []}
        out = apply_postprocessor_to_gil_payload(gil, get_postprocessor(None))
        # Untouched: noop was used, so the byte-level artifact survives.
        assert out["nodes"][0]["properties"]["label"] == "Ġfoo"
        assert sentinel is sentinel

    def test_non_dict_payload_passes_through(self) -> None:
        """Defensive: a non-dict payload (e.g. None, a list, a bare string)
        should not crash the walker."""
        assert apply_postprocessor_to_gil_payload(None, decode_r1_byte_level) is None
        assert apply_postprocessor_to_gil_payload([], decode_r1_byte_level) == []
        assert apply_postprocessor_to_gil_payload("foo", decode_r1_byte_level) == "foo"

    def test_missing_nodes_key_passes_through(self) -> None:
        """Payload without a ``nodes`` list should not crash."""
        payload = {"edges": [{"type": "SUPPORTED_BY"}]}
        out = apply_postprocessor_to_gil_payload(payload, decode_r1_byte_level)
        assert out == payload

    def test_non_dict_node_entries_skipped(self) -> None:
        """A malformed node list with non-dict entries should not crash."""
        gil = {
            "nodes": [
                {"properties": {"label": "Ġclean"}},
                None,
                "not_a_dict",
                42,
                {"properties": {"label": "Ġother"}},
            ],
            "edges": [],
        }
        out = apply_postprocessor_to_gil_payload(gil, decode_r1_byte_level)
        # Both valid nodes decoded; malformed entries untouched.
        assert out["nodes"][0]["properties"]["label"] == " clean"
        assert out["nodes"][1] is None
        assert out["nodes"][2] == "not_a_dict"
        assert out["nodes"][3] == 42
        assert out["nodes"][4]["properties"]["label"] == " other"

    def test_non_string_property_values_skipped(self) -> None:
        """Properties with non-string values (numeric scores, lists, etc.)
        should not be touched by the postprocessor."""
        gil = {
            "nodes": [
                {
                    "type": "Insight",
                    "properties": {
                        "label": "Ġreal",
                        "nli_score": 0.87,
                        "tags": ["foo", "bar"],
                        "metadata": {"nested": "dict"},
                    },
                },
            ],
            "edges": [],
        }
        out = apply_postprocessor_to_gil_payload(gil, decode_r1_byte_level)
        props = out["nodes"][0]["properties"]
        assert props["label"] == " real"
        # Non-string values untouched.
        assert props["nli_score"] == 0.87
        assert props["tags"] == ["foo", "bar"]
        assert props["metadata"] == {"nested": "dict"}

    def test_empty_string_property_skipped(self) -> None:
        """Empty string property values should be left alone — applying
        decode_r1_byte_level to "" is a no-op but we shouldn't waste cycles."""
        gil = {
            "nodes": [
                {"properties": {"label": "", "name": "Ġreal"}},
            ],
            "edges": [],
        }
        out = apply_postprocessor_to_gil_payload(gil, decode_r1_byte_level)
        props = out["nodes"][0]["properties"]
        assert props["label"] == ""
        assert props["name"] == " real"

    def test_mutation_is_in_place(self) -> None:
        """The walker mutates in place and returns the same object reference
        for convenient chaining."""
        gil = {"nodes": [{"properties": {"label": "Ġfoo"}}], "edges": []}
        out = apply_postprocessor_to_gil_payload(gil, decode_r1_byte_level)
        assert out is gil
        assert gil["nodes"][0]["properties"]["label"] == " foo"

    def test_strip_r1_then_decode_composed_postprocessor(self) -> None:
        """The strip_r1_reasoning_and_decode postprocessor (used by R1-Distill
        Magistral, etc.) composes correctly when applied to GIL nodes."""
        from podcast_scraper.evaluation.output_postprocess import (
            strip_r1_reasoning_and_decode,
        )

        # Combine R1 <summary> wrapping + byte-level artifacts in the same field.
        gil = {
            "nodes": [
                {
                    "type": "Topic",
                    "properties": {"label": "<summary>trailĠbuilding</summary>"},
                },
            ],
            "edges": [],
        }
        out = apply_postprocessor_to_gil_payload(gil, strip_r1_reasoning_and_decode)
        # strip_r1_reasoning strips the wrapping tags; decode_r1_byte_level
        # cleans the byte-level token. Both apply in one pass.
        result = out["nodes"][0]["properties"]["label"]
        assert "<summary>" not in result
        assert "</summary>" not in result
        assert "Ġ" not in result
        assert "trail building" in result


@pytest.mark.unit
class TestRealisticDSV2LiteShape:
    """Smoke-tests against the exact DSV2-Lite shape documented in #1016
    final report § 6a (the bug that motivated this fix)."""

    def test_realistic_dsv2_lite_input_is_cleaned_end_to_end(self) -> None:
        # Quoted directly from EVAL_1016_FINAL_REPORT_2026_06_17.md § 6a:
        # "gthegpodcastgepisodegdiscussesgbuilding..." — Ġ encoded as g
        # in the report's prose because the byte-level chars don't render
        # cleanly inline. We test against the actual byte-level chars (Ġ).
        gil = {
            "nodes": [
                {
                    "type": "Topic",
                    "properties": {
                        "label": "ĠtheĠpodcastĠepisodeĠdiscussesĠbuilding",
                    },
                },
                {
                    "type": "Entity",
                    "properties": {
                        "label": "ĠCascadiaĠAlliance",
                        "name": "ĠCascadiaĠAlliance",
                    },
                },
            ],
            "edges": [],
        }
        out = apply_postprocessor_to_gil_payload(gil, decode_r1_byte_level)
        topic = out["nodes"][0]["properties"]["label"]
        ent_label = out["nodes"][1]["properties"]["label"]
        ent_name = out["nodes"][1]["properties"]["name"]
        # All Ġ tokens decoded to spaces; no byte-level artifacts left.
        assert "Ġ" not in topic
        assert "Ġ" not in ent_label
        assert "Ġ" not in ent_name
        assert "the podcast episode discusses building" in topic
        assert "Cascadia Alliance" in ent_label
