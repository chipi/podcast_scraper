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
def test_transcription_model_for_cfg_deepgram() -> None:
    deepgram_cfg = create_test_config(
        transcription_provider="deepgram",
        deepgram_api_key="dg-test",
        deepgram_model="nova-3",
    )
    assert transcription_model_for_cfg(deepgram_cfg) == "nova-3"


@pytest.mark.unit
def test_apply_estimated_cost_if_missing_deepgram() -> None:
    cfg = create_test_config(
        transcription_provider="deepgram",
        deepgram_api_key="dg-test",
        deepgram_model="nova-3",
        pricing_assumptions_file="config/pricing_assumptions.yaml",
    )
    call = ProviderCallMetrics()
    apply_estimated_cost_if_missing(
        call,
        cfg=cfg,
        provider_type="deepgram",
        capability="transcription",
        model="nova-3",
        audio_minutes=10.0,
    )
    assert call.estimated_cost is not None
    assert call.estimated_cost == pytest.approx(0.043)


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


@pytest.mark.unit
def test_record_provider_call_cost_emits_langfuse_span(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cost choke point forwards every billable call to Langfuse tracing (#1052).

    Validates the *wiring* (record_provider_call_cost -> emit_langfuse_span), which the
    langfuse_tracing unit tests can't see. Langfuse is mocked, so no SDK/network is touched.
    """
    import podcast_scraper.utils.langfuse_tracing as lt

    captured: list[dict] = []
    monkeypatch.setattr(lt, "emit_langfuse_span", lambda **kw: captured.append(kw))

    cfg = create_test_config(output_dir="/tmp/run-x", rss_url="https://feeds/x.xml")
    record_provider_call_cost(
        ProviderCallMetrics(),
        0.02,  # explicit cost > 0 so the emit path is reached
        cfg=cfg,
        provider_type="anthropic",
        capability="summarization",
        model="claude-opus",
        prompt_tokens=100,
        completion_tokens=20,
        triggered_guardrail=True,
    )

    assert len(captured) == 1
    span = captured[0]
    assert span["provider"] == "anthropic"
    assert span["capability"] == "summarization"
    assert span["model"] == "claude-opus"
    assert span["cost"] == 0.02
    assert span["prompt_tokens"] == 100
    assert span["completion_tokens"] == 20
    assert span["run_seed"] == "/tmp/run-x"  # output dir groups the run's trace
    assert span["feed_id"] == "https://feeds/x.xml"
    assert span["triggered_guardrail"] is True


@pytest.mark.unit
def test_record_provider_call_cost_no_span_when_cost_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """No cost -> no cost event and no Langfuse span (the early return covers both)."""
    import podcast_scraper.utils.langfuse_tracing as lt

    captured: list[dict] = []
    monkeypatch.setattr(lt, "emit_langfuse_span", lambda **kw: captured.append(kw))

    cfg = create_test_config()
    record_provider_call_cost(
        ProviderCallMetrics(),
        0.0,
        cfg=cfg,
        provider_type="ollama",
        capability="summarization",
        model="llama",
    )
    assert captured == []
