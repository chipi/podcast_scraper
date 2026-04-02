"""Tests for gi.provenance.resolve_gil_artifact_model_version."""

from unittest.mock import MagicMock

import pytest

from podcast_scraper.gi.provenance import resolve_gil_artifact_model_version
from tests.conftest import create_test_config


@pytest.mark.unit
def test_resolve_stub_source_returns_stub() -> None:
    cfg = create_test_config(gi_insight_source="stub")
    assert resolve_gil_artifact_model_version(cfg, None, gi_insight_source="stub") == "stub"


@pytest.mark.unit
def test_resolve_summary_bullets_uses_transformers_summary_model() -> None:
    cfg = create_test_config(
        summary_provider="transformers",
        summary_model="sshleifer/distilbart-cnn-12-6",
        gi_insight_source="summary_bullets",
    )
    mid = resolve_gil_artifact_model_version(cfg, None, gi_insight_source="summary_bullets")
    assert "distilbart" in mid


@pytest.mark.unit
def test_resolve_provider_prefers_insight_model_when_distinct() -> None:
    cfg = create_test_config(
        gi_insight_source="provider",
        summary_provider="openai",
        openai_api_key="sk-test-key-for-unit-tests",
    )
    prov = MagicMock()
    prov.summary_model = "gpt-4o-mini"
    prov.insight_model = "gpt-4o"
    assert resolve_gil_artifact_model_version(cfg, prov, gi_insight_source="provider") == "gpt-4o"


@pytest.mark.unit
def test_resolve_provider_prefers_insight_provider_summary_model() -> None:
    cfg = create_test_config(
        gi_insight_source="provider",
        summary_provider="openai",
        openai_api_key="sk-test-key-for-unit-tests",
    )
    prov = MagicMock()
    prov.summary_model = "gpt-4o-mini"
    assert (
        resolve_gil_artifact_model_version(cfg, prov, gi_insight_source="provider") == "gpt-4o-mini"
    )


@pytest.mark.unit
def test_resolve_openai_summary_bullets_falls_back_to_openai_summary_model() -> None:
    cfg = create_test_config(
        summary_provider="openai",
        openai_summary_model="gpt-4o",
        openai_api_key="sk-test-key-for-unit-tests",
        gi_insight_source="summary_bullets",
    )
    assert (
        resolve_gil_artifact_model_version(cfg, None, gi_insight_source="summary_bullets")
        == "gpt-4o"
    )


@pytest.mark.unit
def test_resolve_summary_bullets_unknown_model_string_returns_unknown() -> None:
    prov = MagicMock()
    prov.summary_model = "unknown"
    cfg = create_test_config(
        summary_provider="openai",
        openai_api_key="sk-test-key-for-unit-tests",
        gi_insight_source="summary_bullets",
    )
    assert (
        resolve_gil_artifact_model_version(cfg, prov, gi_insight_source="summary_bullets")
        == "unknown"
    )


@pytest.mark.unit
def test_resolve_provider_unknown_insight_model_string_returns_unknown() -> None:
    cfg = create_test_config(
        gi_insight_source="provider",
        summary_provider="openai",
        openai_api_key="sk-test-key-for-unit-tests",
    )
    prov = MagicMock()
    prov.insight_model = "unknown"
    prov.summary_model = "gpt-4o-mini"
    assert resolve_gil_artifact_model_version(cfg, prov, gi_insight_source="provider") == "unknown"


@pytest.mark.unit
def test_resolve_unrecognized_insight_source_falls_back_to_stub() -> None:
    cfg = create_test_config(gi_insight_source="summary_bullets")
    assert resolve_gil_artifact_model_version(cfg, None, gi_insight_source="legacy_mode") == "stub"
