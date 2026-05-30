"""LLM cost monitoring (#804)."""

from __future__ import annotations

import pytest

from podcast_scraper.workflow import metrics
from podcast_scraper.workflow.cost_monitoring import (
    CostCapExceeded,
    enforce_cost_soft_cap,
    run_cost_usd_from_pipeline_metrics,
)
from tests.conftest import create_test_config


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
