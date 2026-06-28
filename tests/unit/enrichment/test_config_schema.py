"""Unit tests for ``enrichment.config_schema`` — JSON Schema validator."""

from __future__ import annotations

import pytest

from podcast_scraper.enrichment.config_schema import (
    ConfigSchemaError,
    load_schema,
    validate_enrichment_block,
)

# ---------------------------------------------------------------------------
# load_schema
# ---------------------------------------------------------------------------


def test_load_schema_returns_dict() -> None:
    schema = load_schema()
    assert isinstance(schema, dict)
    assert schema.get("title") == "Enrichment block"
    assert schema["type"] == "object"


def test_load_schema_carries_expected_properties() -> None:
    schema = load_schema()
    props = schema["properties"]
    assert set(props.keys()) >= {
        "enabled",
        "max_total_cost_usd_per_run",
        "fail_on_run_cost_cap",
        "enrichers",
    }


# ---------------------------------------------------------------------------
# validate_enrichment_block — valid cases
# ---------------------------------------------------------------------------


def test_validate_empty_block_passes() -> None:
    validate_enrichment_block({})


def test_validate_full_block_passes() -> None:
    validate_enrichment_block(
        {
            "enabled": True,
            "max_total_cost_usd_per_run": 5.00,
            "fail_on_run_cost_cap": True,
            "enrichers": {
                "topic_cooccurrence": {"enabled": True},
                "nli_contradiction": {
                    "enabled": False,
                    "opt_in": False,
                    "max_cost_usd_per_run": 0.50,
                    "expected_duration_s": 300,
                },
            },
        }
    )


def test_validate_block_with_only_enrichers_subkey_passes() -> None:
    validate_enrichment_block({"enrichers": {"x": {"enabled": True}}})


# ---------------------------------------------------------------------------
# validate_enrichment_block — invalid cases
# ---------------------------------------------------------------------------


def test_validate_enabled_wrong_type_fails() -> None:
    with pytest.raises(ConfigSchemaError):
        validate_enrichment_block({"enabled": "yes please"})


def test_validate_max_total_cost_wrong_type_fails() -> None:
    with pytest.raises(ConfigSchemaError):
        validate_enrichment_block({"max_total_cost_usd_per_run": "5.00"})


def test_validate_max_total_cost_negative_fails() -> None:
    with pytest.raises(ConfigSchemaError):
        validate_enrichment_block({"max_total_cost_usd_per_run": -1.0})


def test_validate_fail_on_run_cost_cap_wrong_type_fails() -> None:
    with pytest.raises(ConfigSchemaError):
        validate_enrichment_block({"fail_on_run_cost_cap": "true"})


def test_validate_per_enricher_enabled_wrong_type_fails() -> None:
    with pytest.raises(ConfigSchemaError):
        validate_enrichment_block({"enrichers": {"x": {"enabled": 1}}})


def test_validate_per_enricher_opt_in_wrong_type_fails() -> None:
    with pytest.raises(ConfigSchemaError):
        validate_enrichment_block({"enrichers": {"x": {"opt_in": "y"}}})


def test_validate_per_enricher_max_cost_negative_fails() -> None:
    with pytest.raises(ConfigSchemaError):
        validate_enrichment_block({"enrichers": {"x": {"max_cost_usd_per_run": -0.10}}})


def test_validate_unknown_top_level_key_fails() -> None:
    """``additionalProperties: false`` enforces no typos at the top level."""
    with pytest.raises(ConfigSchemaError):
        validate_enrichment_block({"unknownn_kkey": True})


# ---------------------------------------------------------------------------
# Minimal fallback validator (no jsonschema)
# ---------------------------------------------------------------------------


def test_minimal_validate_falls_back_when_jsonschema_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When jsonschema isn't importable, the minimal check still rejects obvious typos."""
    import sys

    monkeypatch.setitem(sys.modules, "jsonschema", None)
    with pytest.raises(ConfigSchemaError):
        validate_enrichment_block({"enabled": "yes"})


def test_minimal_validate_accepts_valid_block_without_jsonschema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "jsonschema", None)
    validate_enrichment_block(
        {
            "enabled": True,
            "enrichers": {"x": {"enabled": True, "max_cost_usd_per_run": 0.5}},
        }
    )
