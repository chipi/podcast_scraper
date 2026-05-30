"""LLM cost monitoring (#804)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from podcast_scraper.models.entities import RssFeed
from podcast_scraper.utils.provider_metrics import ProviderCallMetrics, record_provider_call_cost
from podcast_scraper.workflow import metrics
from podcast_scraper.workflow.cost_monitoring import (
    check_cost_soft_cap_at_stage,
    CostCapExceeded,
    emit_llm_cost_event,
    enforce_cost_soft_cap,
    feed_url_for_cost_incident,
    maybe_emit_run_cost_sentry_alert,
    run_cost_usd_from_pipeline_metrics,
)
from tests.conftest import create_test_config


@pytest.mark.unit
def test_feed_url_for_cost_incident_uses_base_url() -> None:
    cfg = create_test_config(openai_api_key="sk-test", rss_url="https://cfg.example/feed.xml")
    feed = RssFeed("t", [], "https://feed.example/rss.xml", [])
    assert feed_url_for_cost_incident(feed, cfg) == "https://feed.example/rss.xml"
    assert feed_url_for_cost_incident(None, cfg) == "https://cfg.example/feed.xml"


@pytest.mark.unit
def test_run_cost_usd_sums_stage_fields() -> None:
    m = metrics.Metrics()
    m.record_llm_transcription_call(1.0, cost_usd=0.5)
    m.record_llm_summarization_call(10, 5, cost_usd=0.1)
    assert run_cost_usd_from_pipeline_metrics(m) == pytest.approx(0.6)


@pytest.mark.unit
def test_soft_cap_abort_raises() -> None:
    cfg = create_test_config(
        openai_api_key="sk-test",
        cost_soft_cap_usd_per_run=0.01,
        cost_soft_cap_action="abort",
    )
    m = metrics.Metrics()
    m.record_llm_transcription_call(5.0, cost_usd=0.05)
    with pytest.raises(CostCapExceeded):
        enforce_cost_soft_cap(cfg, m)


@pytest.mark.unit
def test_soft_cap_warn_does_not_raise() -> None:
    cfg = create_test_config(
        openai_api_key="sk-test",
        cost_soft_cap_usd_per_run=0.01,
        cost_soft_cap_action="warn",
    )
    m = metrics.Metrics()
    m.record_llm_transcription_call(5.0, cost_usd=0.05)
    enforce_cost_soft_cap(cfg, m)


@pytest.mark.unit
def test_emit_llm_cost_event_logs_pure_json(caplog: pytest.LogCaptureFixture) -> None:
    cfg = create_test_config(openai_api_key="sk-test")
    with caplog.at_level(logging.INFO, logger="podcast_scraper.workflow.cost_monitoring"):
        emit_llm_cost_event(
            cfg,
            provider="openai",
            stage="transcription",
            model="whisper-1",
            estimated_cost_usd=0.12,
        )
    assert len(caplog.records) == 1
    payload = json.loads(caplog.records[0].message)
    assert payload["event_type"] == "llm_cost"
    assert payload["estimated_cost_usd"] == 0.12


@pytest.mark.unit
def test_check_cost_soft_cap_appends_incident(tmp_path: Path) -> None:
    cfg = create_test_config(
        openai_api_key="sk-test",
        cost_soft_cap_usd_per_run=0.01,
        cost_soft_cap_action="abort",
    )
    m = metrics.Metrics()
    m.record_llm_transcription_call(5.0, cost_usd=0.05)
    incident_log = tmp_path / "corpus_incidents.jsonl"
    with pytest.raises(CostCapExceeded):
        check_cost_soft_cap_at_stage(
            cfg,
            m,
            stage="transcription",
            incident_log_path=str(incident_log),
            feed_url="https://example.com/feed.xml",
        )
    lines = incident_log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["exception_type"] == "CostCapExceeded"
    assert row["stage"] == "transcription"


@pytest.mark.unit
def test_record_provider_call_cost_emits_event(caplog: pytest.LogCaptureFixture) -> None:
    cfg = create_test_config(openai_api_key="sk-test")
    call = ProviderCallMetrics()
    with caplog.at_level(logging.INFO, logger="podcast_scraper.workflow.cost_monitoring"):
        record_provider_call_cost(
            call,
            0.25,
            cfg=cfg,
            provider_type="openai",
            capability="transcription",
            model="whisper-1",
        )
    assert call.estimated_cost == 0.25
    assert any("llm_cost" in r.message for r in caplog.records)


@pytest.mark.unit
def test_feed_url_for_cost_incident_uses_link_fallback() -> None:
    cfg = create_test_config(openai_api_key="sk-test")
    feed = type("Feed", (), {"link": "https://legacy.example/rss"})()
    assert feed_url_for_cost_incident(feed, cfg) == "https://legacy.example/rss"


@pytest.mark.unit
def test_run_cost_usd_none_metrics() -> None:
    assert run_cost_usd_from_pipeline_metrics(None) == 0.0


@pytest.mark.unit
def test_emit_llm_cost_event_skips_non_positive() -> None:
    cfg = create_test_config(openai_api_key="sk-test")
    emit_llm_cost_event(cfg, provider="openai", stage="x", model="m", estimated_cost_usd=0.0)


@pytest.mark.unit
def test_emit_llm_cost_event_jsonl_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    cfg = create_test_config(openai_api_key="sk-test", jsonl_metrics_echo_stdout=True)
    emit_llm_cost_event(
        cfg,
        provider="openai",
        stage="transcription",
        model="whisper-1",
        estimated_cost_usd=0.01,
    )
    out = capsys.readouterr().out.strip()
    assert '"event_type": "llm_cost"' in out


@pytest.mark.unit
def test_soft_cap_observe_does_not_raise() -> None:
    cfg = create_test_config(
        openai_api_key="sk-test",
        cost_soft_cap_usd_per_run=0.01,
        cost_soft_cap_action="observe",
    )
    m = metrics.Metrics()
    m.record_llm_transcription_call(5.0, cost_usd=0.05)
    enforce_cost_soft_cap(cfg, m)


@pytest.mark.unit
def test_check_cost_soft_cap_raises_without_incident_log() -> None:
    cfg = create_test_config(
        openai_api_key="sk-test",
        cost_soft_cap_usd_per_run=0.01,
        cost_soft_cap_action="abort",
    )
    m = metrics.Metrics()
    m.record_llm_transcription_call(5.0, cost_usd=0.05)
    with pytest.raises(CostCapExceeded):
        check_cost_soft_cap_at_stage(cfg, m, stage="transcription")


@pytest.mark.unit
def test_maybe_emit_run_cost_sentry_alert_fires() -> None:
    cfg = create_test_config(openai_api_key="sk-test", cost_daily_alert_usd=0.01)
    m = metrics.Metrics()
    m.record_llm_transcription_call(1.0, cost_usd=0.05)
    with patch("sentry_sdk.capture_message") as mock_cap, patch("sentry_sdk.set_tag") as mock_tag:
        maybe_emit_run_cost_sentry_alert(cfg, m)
    mock_cap.assert_called_once()
    mock_tag.assert_called_once_with("cost_anomaly", "run_threshold")


@pytest.mark.unit
def test_maybe_emit_run_cost_sentry_skips_below_threshold() -> None:
    cfg = create_test_config(openai_api_key="sk-test", cost_daily_alert_usd=100.0)
    m = metrics.Metrics()
    m.record_llm_transcription_call(1.0, cost_usd=0.01)
    maybe_emit_run_cost_sentry_alert(cfg, m)


@pytest.mark.unit
def test_enforce_cost_soft_cap_no_cap_configured() -> None:
    cfg = create_test_config(openai_api_key="sk-test", cost_soft_cap_usd_per_run=None)
    m = metrics.Metrics()
    m.record_llm_transcription_call(5.0, cost_usd=0.05)
    enforce_cost_soft_cap(cfg, m)


@pytest.mark.unit
def test_enforce_cost_soft_cap_below_threshold() -> None:
    cfg = create_test_config(
        openai_api_key="sk-test",
        cost_soft_cap_usd_per_run=10.0,
        cost_soft_cap_action="abort",
    )
    m = metrics.Metrics()
    m.record_llm_transcription_call(1.0, cost_usd=0.01)
    enforce_cost_soft_cap(cfg, m)


@pytest.mark.unit
def test_maybe_emit_run_cost_sentry_without_sdk() -> None:
    cfg = create_test_config(openai_api_key="sk-test", cost_daily_alert_usd=0.01)
    m = metrics.Metrics()
    m.record_llm_transcription_call(1.0, cost_usd=0.05)
    with patch.dict("sys.modules", {"sentry_sdk": None}):
        maybe_emit_run_cost_sentry_alert(cfg, m)
