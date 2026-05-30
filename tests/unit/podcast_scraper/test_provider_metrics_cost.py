"""Provider call cost backfill (#823)."""

from __future__ import annotations

import pytest

from podcast_scraper.utils.provider_metrics import (
    apply_estimated_cost_if_missing,
    ProviderCallMetrics,
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
