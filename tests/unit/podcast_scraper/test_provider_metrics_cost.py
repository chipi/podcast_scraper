"""Provider call cost backfill (#823)."""

from __future__ import annotations

import logging

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

    Validates the *wiring* end to end: record_provider_call_cost -> emit_llm_cost_event ->
    emit_langfuse_span (the span now emits from the single cost choke point so gi/evidence/cleaning
    are covered too, not just this path). Langfuse is mocked, so no SDK/network is touched; the real
    emit_llm_cost_event runs so the delegation is exercised.
    """
    import podcast_scraper.utils.langfuse_tracing as lt
    from podcast_scraper.utils import correlation

    spans: list[dict] = []
    monkeypatch.setattr(lt, "emit_langfuse_span", lambda **kw: spans.append(kw))

    # The correlation join key (#1053): run id is process-global, episode id context-local.
    correlation._reset_for_tests()
    correlation.set_run_id("run-ABC")
    correlation.set_episode_id("ep:7")
    try:
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
    finally:
        correlation._reset_for_tests()

    assert len(spans) == 1
    span = spans[0]
    assert span["provider"] == "anthropic"
    assert span["capability"] == "summarization"
    assert span["model"] == "claude-opus"
    assert span["cost"] == 0.02
    assert span["prompt_tokens"] == 100
    assert span["completion_tokens"] == 20
    # #1053: the Langfuse trace is seeded by the run_id join key, and the span carries
    # both correlation ids so it joins with the cost event + Sentry error.
    assert span["run_seed"] == "run-ABC"
    assert span["episode_id"] == "ep:7"
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


@pytest.mark.unit
def test_record_provider_call_cost_prefers_upstream_usage_cost() -> None:
    """2026-08 finale fix: the OR/gateway route must bill from the upstream REAL cost
    (``response.usage.cost``), NOT the pricing-table estimate, which prices aliased gateway ids at
    the vendor-DIRECT rate — 3-5x the real OpenRouter bill. When the caller has no cost but the
    response carries ``usage.cost``, prefer it.
    """
    import types

    cfg = create_test_config(pricing_assumptions_file="config/pricing_assumptions.yaml")
    real_cost = 0.0123
    response = types.SimpleNamespace(usage=types.SimpleNamespace(cost=real_cost))
    call = ProviderCallMetrics()
    record_provider_call_cost(
        call,
        None,  # caller has no cost -> without the fix this hits the pricing table (direct-rate)
        cfg=cfg,
        provider_type="litellm",
        capability="summarization",
        model="podcast-flash-0731",
        prompt_tokens=1000,
        completion_tokens=500,
        response=response,
    )
    assert call.estimated_cost == pytest.approx(real_cost)


class TestNoExactDuplicateCostEvents:
    """BUG 2 guardrail — one logical provider call must emit exactly one ``llm_cost`` event.

    Root cause: ``record_provider_call_cost(cost=None, ...)`` backfilled via
    ``apply_estimated_cost_if_missing``, which (pre-fix) recursed back into
    ``record_provider_call_cost`` once it resolved a price — emitting an event from the
    recursive call AND again from the original call once it returned. A 9-episode real run
    logged 86 exact-duplicate ``stage=summarization`` events (some 4x) from exactly this
    recursion. Parametrized over the shared paths every provider funnels through: the direct
    ``record_provider_call_cost`` entry point (openai/anthropic/mistral/grok/ollama call sites)
    and the standalone ``apply_estimated_cost_if_missing`` backstop (transcription/diarization
    cost backfills) — so a regression in either layer trips this, not just the deepseek/litellm
    instance that was actually observed.
    """

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "provider_type,model,capability,call_kwargs",
        [
            (
                "openai",
                "gpt-4o-mini",
                "summarization",
                {"prompt_tokens": 5000, "completion_tokens": 200},
            ),
            (
                "anthropic",
                "claude-haiku-4-5",
                "summarization",
                {"prompt_tokens": 5000, "completion_tokens": 200},
            ),
            ("deepgram", "nova-3", "transcription", {"audio_minutes": 10.0}),
            (
                "ollama",
                "llama3.1:8b",
                "summarization",
                {"prompt_tokens": 100, "completion_tokens": 50},
            ),
        ],
    )
    def test_record_provider_call_cost_with_unresolved_cost_emits_once(
        self,
        caplog: pytest.LogCaptureFixture,
        provider_type: str,
        model: str,
        capability: str,
        call_kwargs: dict,
    ) -> None:
        """Caller has no pre-computed cost (passes ``cost=None``, forcing the backfill path) —
        this is the exact shape every provider call site uses when its own
        ``calculate_provider_cost`` call didn't run or returned unresolved."""
        cfg = create_test_config(
            openai_api_key="sk-test",
            pricing_assumptions_file="config/pricing_assumptions.yaml",
        )
        call = ProviderCallMetrics()
        with caplog.at_level(logging.INFO, logger="podcast_scraper.workflow.cost_monitoring"):
            record_provider_call_cost(
                call,
                None,
                cfg=cfg,
                provider_type=provider_type,
                capability=capability,
                model=model,
                **call_kwargs,
            )
        events = [r for r in caplog.records if r.name == "podcast_scraper.workflow.cost_monitoring"]
        assert len(events) == 1, f"expected exactly 1 llm_cost event, got {len(events)}"

    @pytest.mark.unit
    def test_apply_estimated_cost_if_missing_standalone_still_emits_once(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The 3 standalone callers (diarization/transcription cost backstops) call
        ``apply_estimated_cost_if_missing`` directly, not through ``record_provider_call_cost`` —
        the fix must not silently drop THEIR single emission while removing the duplicate."""
        cfg = create_test_config(
            openai_api_key="sk-test",
            pricing_assumptions_file="config/pricing_assumptions.yaml",
        )
        call = ProviderCallMetrics()
        with caplog.at_level(logging.INFO, logger="podcast_scraper.workflow.cost_monitoring"):
            apply_estimated_cost_if_missing(
                call,
                cfg=cfg,
                provider_type="openai",
                capability="transcription",
                model="whisper-1",
                audio_minutes=10.0,
            )
        events = [r for r in caplog.records if r.name == "podcast_scraper.workflow.cost_monitoring"]
        assert len(events) == 1
        assert call.estimated_cost is not None


@pytest.mark.unit
def test_record_provider_call_cost_table_fallback_unchanged_without_upstream() -> None:
    """The fix must NOT change behaviour when there is no upstream cost: a response without
    ``usage.cost`` (or no response) still falls back to the pricing-table estimate.
    """
    import types

    cfg = create_test_config(pricing_assumptions_file="config/pricing_assumptions.yaml")
    baseline = ProviderCallMetrics()
    apply_estimated_cost_if_missing(
        baseline,
        cfg=cfg,
        provider_type="litellm",
        capability="summarization",
        model="podcast-flash-0731",
        prompt_tokens=1000,
        completion_tokens=500,
    )
    response = types.SimpleNamespace(usage=types.SimpleNamespace(cost=None))
    call = ProviderCallMetrics()
    record_provider_call_cost(
        call,
        None,
        cfg=cfg,
        provider_type="litellm",
        capability="summarization",
        model="podcast-flash-0731",
        prompt_tokens=1000,
        completion_tokens=500,
        response=response,
    )
    assert call.estimated_cost == baseline.estimated_cost
