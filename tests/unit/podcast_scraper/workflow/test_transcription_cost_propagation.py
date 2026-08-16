"""#1523: transcription cost must reach the run-level pipeline_metrics (→ manifest cost_rollup).

The orchestration backstop ``_record_transcription_metrics`` runs after every transcription and
resolves the per-call cost via ``apply_estimated_cost_if_missing`` (which also EMITS the per-episode
cost event — "the cost_monitoring event fires"). Before #1523 it filled
``call_metrics.estimated_cost`` but never recorded it onto ``pipeline_metrics``, so any path where
the provider did NOT self-record (couldn't determine audio duration, deepgram's ``audio_minutes<=0``
bail, a mocked provider) left ``llm_transcription_cost_usd == 0`` — and the manifest ``cost_rollup``
undercounted real transcription spend ("the rollup misses it").
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from podcast_scraper.utils.provider_metrics import ProviderCallMetrics
from podcast_scraper.workflow import episode_processor
from podcast_scraper.workflow.metrics import Metrics

pytestmark = pytest.mark.unit


def _job(duration_seconds: float) -> SimpleNamespace:
    # episode=None so _job_has_episode_for_metrics is False (skips the Episode-only log block);
    # _audio_sec_for_transcription_job reads episode_duration_seconds first.
    return SimpleNamespace(idx=1, episode=None, episode_duration_seconds=duration_seconds)


def test_backstop_records_transcription_cost_when_provider_did_not(tmp_path):
    """A provider that didn't self-record still gets its cost onto pipeline_metrics (#1523 gap)."""
    from tests.conftest import create_test_config

    cfg = create_test_config(
        transcription_provider="deepgram",
        deepgram_api_key="dg-test-key",
        deepgram_model="nova-3",
        pricing_assumptions_file="config/pricing_assumptions.yaml",
    )
    pm = Metrics()
    # estimated_cost=None + flag unset == the provider never recorded onto pipeline_metrics.
    call_metrics = ProviderCallMetrics()

    episode_processor._record_transcription_metrics(
        _job(600.0), cfg, 1.0, call_metrics, pipeline_metrics=pm
    )

    # 10 audio-minutes × deepgram nova-3 ($0.0043/min) = $0.043, and one recorded call.
    assert pm.llm_transcription_cost_usd == pytest.approx(0.043, rel=1e-3)
    assert pm.llm_transcription_calls == 1


def test_backstop_does_not_double_count_when_provider_already_recorded(tmp_path):
    """If the provider self-recorded (flag set), the backstop is a no-op — no double counting."""
    from tests.conftest import create_test_config

    cfg = create_test_config(
        transcription_provider="deepgram",
        deepgram_api_key="dg-test-key",
        deepgram_model="nova-3",
        pricing_assumptions_file="config/pricing_assumptions.yaml",
    )
    pm = Metrics()
    # Simulate the provider having already recorded this call's cost onto pipeline_metrics.
    pm.record_llm_transcription_call(10.0, cost_usd=0.043)
    call_metrics = ProviderCallMetrics()
    call_metrics.pipeline_transcription_recorded = True

    episode_processor._record_transcription_metrics(
        _job(600.0), cfg, 1.0, call_metrics, pipeline_metrics=pm
    )

    # Still exactly one call / one cost — the backstop did not add a second.
    assert pm.llm_transcription_calls == 1
    assert pm.llm_transcription_cost_usd == pytest.approx(0.043, rel=1e-3)
