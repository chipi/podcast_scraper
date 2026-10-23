"""Every LLM call on every provider emits its cost event (the 2026-10-23 cost audit).

Each provider is built from config as the pipeline builds it and every LLM method is called over
HTTP against the e2e mock server, so the REAL SDK response objects reach the cost code. The test
captures ``emit_llm_cost_event`` (the one sink of the ``llm_cost`` log, Langfuse and, through
``record_provider_call_cost``, the run budget) and asserts each call emitted one event with real
token counts and the stage it belongs to. A call that raises on the mock's answer still counts:
the provider was paid for it, which is the point of the bad-JSON cases.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List
from unittest.mock import patch

import pytest

from podcast_scraper import config

pytestmark = [pytest.mark.e2e, pytest.mark.critical_path]

TRANSCRIPT = (
    "Welcome to the show. I'm Tobias Wren. Today we talk about soil, water and the people who "
    "farm them. Thanks for having me, Tobias. Farming is changing faster than ever. " * 4
)

# provider -> (summary_provider value, config prefix, E2EServerURLs method)
PROVIDERS = {
    "openai": ("openai", "openai", "openai_api_base"),
    "vllm": ("vllm", "vllm", "vllm_api_base"),
    "litellm": ("litellm", "litellm", "litellm_api_base"),
    "qwen": ("qwen", "qwen", "qwen_api_base"),
    "deepseek": ("deepseek", "deepseek", "deepseek_api_base"),
    "groq": ("groq", "groq", "groq_api_base"),
    "gemini": ("gemini", "gemini", "gemini_api_base"),
    "anthropic": ("anthropic", "anthropic", "anthropic_api_base"),
    "mistral": ("mistral", "mistral", "mistral_api_base"),
    "grok": ("grok", "grok", "grok_api_base"),
    "ollama": ("ollama", "ollama", "ollama_api_base"),
}

SELF_HOSTED = ("vllm", "litellm", "qwen")

# method -> (call, the stage its event must carry)
METHODS: Dict[str, tuple] = {
    "complete_text": (
        lambda p: p.complete_text('{"q": 1}', max_tokens=50, stage="speaker_naming"),
        "speaker_naming",
    ),
    "detect_speakers": (
        lambda p: p.detect_speakers("Ep 1 with Maria Lindqvist", "A talk.", {"Tobias Wren"}),
        "speaker_detection",
    ),
    "classify_insights": (
        lambda p: p.classify_insights(["Soil health drives yield.", "Water is scarce."]),
        "gi",
    ),
    "extract_kg_graph": (lambda p: p.extract_kg_graph(TRANSCRIPT, episode_title="Ep 1"), "kg"),
    "summarize_mega_bundled": (lambda p: p.summarize_mega_bundled(TRANSCRIPT), "summarization"),
    "summarize_extraction_bundled": (
        lambda p: p.summarize_extraction_bundled(TRANSCRIPT),
        "extraction_bundle",
    ),
    "summarize_bundled": (lambda p: p.summarize_bundled(TRANSCRIPT), "summarization"),
    "extract_quotes": (
        lambda p: p.extract_quotes(TRANSCRIPT, "Farming is changing fast."),
        "extract_quotes",
    ),
    "score_entailment": (
        lambda p: p.score_entailment(TRANSCRIPT, "Farming is changing fast."),
        "score_entailment",
    ),
}
# Ollama has no mega or extraction bundle (llm_pipeline_mode falls back to staged there).
NOT_IMPLEMENTED = {("ollama", "summarize_mega_bundled"), ("ollama", "summarize_extraction_bundled")}


def _provider(e2e_server: Any, name: str, tmp_path: Path) -> Any:
    from podcast_scraper.summarization.factory import create_summarization_provider

    summary_provider, prefix, url = PROVIDERS[name]
    kwargs: Dict[str, Any] = {
        "output_dir": str(tmp_path),
        "summary_provider": summary_provider,
        "speaker_detector_provider": summary_provider,
        f"{prefix}_api_base": getattr(e2e_server.urls, url)(),
        "preload_models": False,
    }
    if name != "ollama":
        kwargs[f"{prefix}_api_key"] = "sk-test-e2e"
    if name in SELF_HOSTED:
        # These ship no default model; a profile always names one (prod: vllm_summary_model). An
        # empty model name makes the evidence recorder skip the call, which is a test artefact.
        kwargs[f"{prefix}_summary_model"] = "podcast-flash-0731"  # one the mock advertises
    provider = create_summarization_provider(config.Config(**kwargs))
    provider.initialize()
    return provider


@pytest.fixture
def events():
    captured: List[Dict[str, Any]] = []

    def _capture(cfg: Any, **kwargs: Any) -> None:
        captured.append(kwargs)

    with patch(
        "podcast_scraper.workflow.cost_monitoring.emit_llm_cost_event", side_effect=_capture
    ):
        yield captured


def _call(fn: Callable[[Any], Any], provider: Any) -> None:
    try:
        fn(provider)
    except Exception:  # noqa: BLE001 - the mock's answer may not parse; the call was still paid
        pass


@pytest.mark.parametrize("method", sorted(METHODS))
@pytest.mark.parametrize("name", sorted(PROVIDERS))
def test_every_llm_call_emits_its_cost(
    name: str, method: str, e2e_server: Any, tmp_path: Path, events: List[Dict[str, Any]]
) -> None:
    provider = _provider(e2e_server, name, tmp_path)
    if (name, method) in NOT_IMPLEMENTED:
        assert not callable(getattr(provider, method, None)), f"{name}.{method} exists now"
        return
    fn, stage = METHODS[method]
    _call(fn, provider)
    ours = [e for e in events if e.get("stage") == stage]
    assert ours, f"{name}.{method}: no llm_cost event with stage {stage!r}; got {events}"
    assert len(ours) == 1, f"{name}.{method}: {len(ours)} events for one call: {ours}"
    event = ours[0]
    assert event.get("response") is not None or event.get("prompt_tokens"), event
    # Priced through the pricing table (0.0 for an unpriced model), never left out: an event
    # without a cost is how Gemini's resolver calls were billed at nothing.
    assert "estimated_cost_usd" in event, event
