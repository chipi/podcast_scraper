"""Provider call cost backfill (#823)."""

from __future__ import annotations

import pytest

from podcast_scraper.utils.provider_metrics import (
    apply_estimated_cost_if_missing,
    ProviderCallMetrics,
    record_provider_call_cost,
    transcription_model_for_cfg,
)
from tests.conftest import create_test_config


@pytest.mark.unit
def test_apply_estimated_cost_if_missing_whisper() -> None:
    cfg = create_test_config(
        transcription_provider="openai",
        openai_transcription_model="whisper-1",
        openai_api_key="sk-test",
        pricing_assumptions_file="config/pricing_assumptions.yaml",
    )
    call = ProviderCallMetrics()
    apply_estimated_cost_if_missing(
        call,
        cfg=cfg,
        provider_type="openai",
        capability="transcription",
        model="whisper-1",
        audio_minutes=10.0,
    )
    assert call.estimated_cost is not None
    assert call.estimated_cost == pytest.approx(0.06)


@pytest.mark.unit
def test_transcription_model_for_cfg_whisper_and_openai() -> None:
    whisper_cfg = create_test_config(transcription_provider="whisper", whisper_model="small")
    assert transcription_model_for_cfg(whisper_cfg) == "small"
    openai_cfg = create_test_config(
        transcription_provider="openai",
        openai_transcription_model="whisper-1",
        openai_api_key="sk-test",
    )
    assert transcription_model_for_cfg(openai_cfg) == "whisper-1"


@pytest.mark.unit
def test_apply_estimated_cost_if_missing_no_op_when_cost_set() -> None:
    cfg = create_test_config(openai_api_key="sk-test")
    call = ProviderCallMetrics()
    call.set_cost(0.5)
    apply_estimated_cost_if_missing(
        call,
        cfg=cfg,
        provider_type="openai",
        capability="transcription",
        model="whisper-1",
    )
    assert call.estimated_cost == 0.5


@pytest.mark.unit
def test_record_provider_call_cost_skips_emit_when_zero() -> None:
    cfg = create_test_config(openai_api_key="sk-test")
    call = ProviderCallMetrics()
    record_provider_call_cost(
        call,
        0.0,
        cfg=cfg,
        provider_type="openai",
        capability="transcription",
        model="whisper-1",
    )
    assert call.estimated_cost == 0.0


@pytest.mark.unit
def test_apply_estimated_cost_if_missing_empty_provider() -> None:
    cfg = create_test_config(openai_api_key="sk-test")
    call = ProviderCallMetrics()
    apply_estimated_cost_if_missing(
        call,
        cfg=cfg,
        provider_type="",
        capability="transcription",
        model="whisper-1",
    )
    assert call.estimated_cost is None


@pytest.mark.unit
def test_record_provider_call_cost_backfills_when_cost_none() -> None:
    cfg = create_test_config(
        transcription_provider="openai",
        openai_transcription_model="whisper-1",
        openai_api_key="sk-test",
        pricing_assumptions_file="config/pricing_assumptions.yaml",
    )
    call = ProviderCallMetrics()
    record_provider_call_cost(
        call,
        None,
        cfg=cfg,
        provider_type="openai",
        capability="transcription",
        model="whisper-1",
        audio_minutes=5.0,
    )
    assert call.estimated_cost is not None
    assert call.estimated_cost > 0
