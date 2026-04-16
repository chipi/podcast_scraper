"""Unit tests for optional YAML pricing assumptions loader."""

from __future__ import annotations

import datetime as dt
from collections.abc import Generator
from pathlib import Path

import pytest
import yaml

from podcast_scraper import pricing_assumptions as pa

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clear_cache() -> Generator[None, None, None]:
    pa.clear_pricing_assumptions_cache()
    yield
    pa.clear_pricing_assumptions_cache()


def test_resolve_assumptions_path_absolute_and_relative(tmp_path: Path) -> None:
    assert pa.resolve_assumptions_path("", cwd=tmp_path) is None
    assert pa.resolve_assumptions_path("   ", cwd=tmp_path) is None
    f = tmp_path / "p.yaml"
    f.write_text("x: 1\n", encoding="utf-8")
    assert pa.resolve_assumptions_path(str(f), cwd=tmp_path) == f.resolve()
    sub = tmp_path / "sub"
    sub.mkdir()
    rel = pa.resolve_assumptions_path("../p.yaml", cwd=sub)
    assert rel is not None and rel.resolve() == f.resolve()


def test_coerce_and_match_model_section() -> None:
    entry = {"input_cost_per_1m_tokens": "2.5", "bogus": "x"}
    rates = pa._coerce_rates(entry)
    assert rates["input_cost_per_1m_tokens"] == 2.5
    assert "bogus" not in rates

    section = {
        "gpt-4o-mini": {"input_cost_per_1m_tokens": 1.0},
        "gpt-4o": {"input_cost_per_1m_tokens": 2.0},
        "default": {"input_cost_per_1m_tokens": 9.0},
    }
    assert pa._match_model_section(section, "gpt-4o-mini") == {"input_cost_per_1m_tokens": 1.0}
    assert pa._match_model_section(section, "GPT-4O-MINI") == {"input_cost_per_1m_tokens": 1.0}
    assert pa._match_model_section(section, "prefix-gpt-4o-suffix") == {
        "input_cost_per_1m_tokens": 2.0
    }
    assert pa._match_model_section(section, "unknown-model") == {"input_cost_per_1m_tokens": 9.0}
    assert pa._match_model_section({}, "m") is None
    assert pa._match_model_section({"default": {}}, "m") is None


def test_lookup_external_pricing_paths(tmp_path: Path) -> None:
    payload = yaml.safe_load("""
providers:
  openai:
    transcription:
      whisper-1:
        cost_per_minute: 0.006
    text:
      gpt-4o:
        input_cost_per_1m_tokens: 2.5
""")
    assert pa.lookup_external_pricing(payload, "openai", "transcription", "whisper-1") == {
        "cost_per_minute": 0.006
    }
    assert pa.lookup_external_pricing(payload, "openai", "summarization", "gpt-4o") == {
        "input_cost_per_1m_tokens": 2.5
    }
    assert pa.lookup_external_pricing(payload, "openai", "speaker_detection", "gpt-4o") is not None
    assert pa.lookup_external_pricing({}, "x", "summarization", "m") is None
    assert pa.lookup_external_pricing({"providers": {}}, "openai", "summarization", "m") is None
    assert pa.lookup_external_pricing(payload, "openai", "other", "m") is None


def test_load_pricing_assumptions_payload(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("[1,2]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        pa.load_pricing_assumptions_payload(bad)
    ok = tmp_path / "ok.yaml"
    ok.write_text("schema_version: '1'\nproviders: {}\n", encoding="utf-8")
    data = pa.load_pricing_assumptions_payload(ok)
    assert data["schema_version"] == "1"


def test_get_loaded_table_caches(tmp_path: Path) -> None:
    f = tmp_path / "rates.yaml"
    f.write_text("providers: {}\nmetadata: {last_reviewed: '2026-01-01'}\n", encoding="utf-8")
    p1, r1 = pa.get_loaded_table(str(f), cwd=tmp_path)
    p2, r2 = pa.get_loaded_table(str(f), cwd=tmp_path)
    assert p1 == p2 and r1 == r2
    f.write_text("providers: {x: 1}\n", encoding="utf-8")
    pa.clear_pricing_assumptions_cache()
    p3, _ = pa.get_loaded_table(str(f), cwd=tmp_path)
    assert p3 is not None and "x" in (p3.get("providers") or {})


def test_get_loaded_table_missing_returns_none(tmp_path: Path) -> None:
    payload, resolved = pa.get_loaded_table("nope.yaml", cwd=tmp_path)
    assert payload is None and resolved is None


def test_check_staleness_non_mapping_metadata_and_bad_threshold() -> None:
    assert pa.check_staleness({})[0] is False
    assert pa.check_staleness({"metadata": "nope"})[0] is False
    bad_thresh = {"metadata": {"last_reviewed": "2020-01-01", "stale_review_after_days": "x"}}
    assert pa.check_staleness(bad_thresh, today=dt.date(2026, 1, 1))[0] is False
    bad_date = {"metadata": {"last_reviewed": "not-a-date", "stale_review_after_days": 1}}
    assert pa.check_staleness(bad_date, today=dt.date(2026, 1, 1))[0] is False


def test_check_staleness_and_format_report(tmp_path: Path) -> None:
    payload = {
        "metadata": {
            "last_reviewed": "2020-01-01",
            "stale_review_after_days": 10,
        }
    }
    day = dt.date(2026, 1, 1)
    stale, msgs = pa.check_staleness(payload, today=day)
    assert stale is True
    assert msgs

    fresh = {"metadata": {"last_reviewed": "2025-12-31", "stale_review_after_days": 9999}}
    assert pa.check_staleness(fresh, today=day)[0] is False

    txt = pa.format_status_report("missing.yaml", cwd=tmp_path)
    assert "not found" in txt

    f = tmp_path / "r.yaml"
    f.write_text(
        yaml.dump(
            {
                "schema_version": 1,
                "providers": {},
                "metadata": {
                    "last_reviewed": "2026-01-01",
                    "pricing_effective_date": "2026-01-01",
                    "stale_review_after_days": 365,
                    "source_urls": {"a": "https://x"},
                },
            },
        ),
        encoding="utf-8",
    )
    rep = pa.format_status_report(str(f.name), cwd=tmp_path, today=dt.date(2026, 1, 15))
    assert "Resolved file" in rep
    assert "source_urls" in rep
