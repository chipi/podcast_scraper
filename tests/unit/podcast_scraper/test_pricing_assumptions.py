"""Unit tests for optional YAML pricing assumptions (LLM cost estimates)."""

from __future__ import annotations

import datetime as dt
import textwrap
from pathlib import Path

import pytest

from podcast_scraper import pricing_assumptions
from podcast_scraper.workflow import helpers


@pytest.mark.unit
def test_resolve_assumptions_path_relative(tmp_path: Path) -> None:
    y = tmp_path / "p.yaml"
    y.write_text("schema_version: 1\nproviders: {}\n", encoding="utf-8")
    found = pricing_assumptions.resolve_assumptions_path("p.yaml", cwd=tmp_path)
    assert found == y


@pytest.mark.unit
def test_lookup_external_pricing_openai_text() -> None:
    payload = {
        "providers": {
            "openai": {
                "text": {
                    "gpt-4o-mini": {
                        "input_cost_per_1m_tokens": 0.99,
                        "output_cost_per_1m_tokens": 0.01,
                    }
                }
            }
        }
    }
    rates = pricing_assumptions.lookup_external_pricing(
        payload, "openai", "speaker_detection", "gpt-4o-mini"
    )
    assert rates == {"input_cost_per_1m_tokens": 0.99, "output_cost_per_1m_tokens": 0.01}


@pytest.mark.unit
def test_lookup_longest_substring_gpt_models() -> None:
    payload = {
        "providers": {
            "openai": {
                "text": {
                    "gpt-4o-mini": {
                        "input_cost_per_1m_tokens": 0.15,
                        "output_cost_per_1m_tokens": 0.6,
                    },
                    "gpt-4o": {
                        "input_cost_per_1m_tokens": 2.5,
                        "output_cost_per_1m_tokens": 10.0,
                    },
                }
            }
        }
    }
    mini = pricing_assumptions.lookup_external_pricing(
        payload, "openai", "summarization", "gpt-4o-mini"
    )
    assert mini is not None
    assert mini["input_cost_per_1m_tokens"] == 0.15
    full = pricing_assumptions.lookup_external_pricing(payload, "openai", "summarization", "gpt-4o")
    assert full is not None
    assert full["input_cost_per_1m_tokens"] == 2.5


@pytest.mark.unit
def test_check_staleness_flags_old_review() -> None:
    payload = {
        "metadata": {
            "last_reviewed": "2000-01-01",
            "stale_review_after_days": 30,
        }
    }
    stale, msgs = pricing_assumptions.check_staleness(payload, today=dt.date(2026, 3, 30))
    assert stale is True
    assert msgs


@pytest.mark.unit
def test_get_provider_pricing_merges_yaml(tmp_path: Path) -> None:
    from tests.conftest import create_test_config

    pricing_assumptions.clear_pricing_assumptions_cache()
    y = tmp_path / "rates.yaml"
    y.write_text(
        textwrap.dedent("""
            schema_version: 1
            providers:
              openai:
                text:
                  gpt-4o-mini:
                    input_cost_per_1m_tokens: 0.99
                    output_cost_per_1m_tokens: 0.01
            """).strip(),
        encoding="utf-8",
    )
    cfg = create_test_config(
        openai_api_key="sk-test",
        speaker_detector_provider="openai",
        openai_speaker_model="gpt-4o-mini",
        pricing_assumptions_file=str(y),
    )
    p = helpers._get_provider_pricing(cfg, "openai", "speaker_detection", "gpt-4o-mini")
    assert p["input_cost_per_1m_tokens"] == 0.99
    assert p["output_cost_per_1m_tokens"] == 0.01
    pricing_assumptions.clear_pricing_assumptions_cache()


@pytest.mark.unit
def test_transcription_estimated_cost_per_second() -> None:
    pricing = {"cost_per_second": 0.1}
    assert helpers._transcription_estimated_cost_usd(pricing, 1.0) == pytest.approx(6.0)
    assert helpers._transcription_estimated_cost_usd(
        {"cost_per_minute": 0.006}, 10.0
    ) == pytest.approx(0.06)
