"""``initialize()`` must bring up the LLM capability no matter which stages the run enabled (#1720).

THE DEFECT
Every cloud provider gated ``_initialize_summarization()`` on ``cfg.generate_summaries`` — a
STAGE flag — while the capability it guards serves twelve-plus methods that have nothing to do
with the summary stage: ``extract_kg_graph``, ``generate_insights``, ``clean_transcript``,
``classify_insights``, ``extract_quotes``, ``score_entailment``, ``complete_text``, …. A run
with summaries off that still extracts KG constructs the provider, calls ``initialize()``, and
then dies on "summarization not initialized. Call initialize() first." — advice the caller
already followed.

SEEN LIVE (glitchtip PODCAST-PIPELINE-19, escalated as #1720)::

    Fallback tier 'deepseek' also failed on extract_kg_graph: OpenAIProvider summarization
    not initialized

Both tiers were victims of the same gate: the primary raised it, the failover ladder built the
deepseek tier from the SAME cfg, called ``initialize()`` on it — which skipped the capability
again — and the ladder recovered nothing.

THE CONTRACT AFTER THE FIX
``initialize()`` initializes the capabilities the provider OFFERS; per-stage cfg flags decide
which stages RUN, not which capabilities exist. For five of the six providers the capability
init is literally ``self._summarization_initialized = True`` — the gate saved nothing. Ollama
additionally validates the model against the local server, which a KG-only run needs just as
much as a summary run does.

Deliberately unchanged: a provider whose ``initialize()`` was NEVER called still raises (that
contract is pinned by test_uninitialized_provider_never_fakes_a_result.py), and the local
transformers MLProvider keeps its gate — there it fences multi-GB model loading, which is what
gates are for.
"""

# mypy: disable-error-code="call-arg"
# Deliberate: Config(rss_url=...) — the field declares alias="rss", so mypy's pydantic plugin
# only knows the alias while populate-by-name accepts either at runtime (same pragma as
# test_uninitialized_provider_never_fakes_a_result.py).

from __future__ import annotations

import importlib
import json
from types import SimpleNamespace
from typing import Any, List, Tuple

import pytest

from podcast_scraper import config

pytestmark = [pytest.mark.unit]


#: (module path, class name) — the six providers that carried the ``generate_summaries`` gate.
PROVIDERS: List[Tuple[str, str]] = [
    ("podcast_scraper.providers.openai.openai_provider", "OpenAICompatibleProvider"),
    ("podcast_scraper.providers.gemini.gemini_provider", "GeminiProvider"),
    ("podcast_scraper.providers.grok.grok_provider", "GrokProvider"),
    ("podcast_scraper.providers.mistral.mistral_provider", "MistralProvider"),
    ("podcast_scraper.providers.anthropic.anthropic_provider", "AnthropicProvider"),
    ("podcast_scraper.providers.ollama.ollama_provider", "OllamaProvider"),
]

_DUMMY_KEYS = {
    "OPENAI_API_KEY": "sk-test-not-a-real-key",
    "GEMINI_API_KEY": "test-not-a-real-key",
    "GROK_API_KEY": "test-not-a-real-key",
    "MISTRAL_API_KEY": "test-not-a-real-key",
    "ANTHROPIC_API_KEY": "sk-ant-test-not-a-real-key",
    "DEEPSEEK_API_KEY": "test-not-a-real-key",
    "LITELLM_API_KEY": "test-not-a-real-key",
}


@pytest.fixture(autouse=True)
def _dummy_credentials(monkeypatch):
    for name, value in _DUMMY_KEYS.items():
        monkeypatch.setenv(name, value)


def _summaries_off_provider(module_path: str, class_name: str, monkeypatch) -> Any:
    """A provider built from a cfg whose SUMMARY STAGE is off — the #1720 run shape."""
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name, None)
    if cls is None:
        pytest.skip(f"{class_name} not present in {module_path}")
    cfg = config.Config(rss_url="https://example.com/feed.xml", generate_summaries=False)
    try:
        provider = cls(cfg)
    except Exception as exc:  # pragma: no cover - e.g. ollama needs a live server to construct
        pytest.skip(f"cannot construct {class_name} in this environment: {exc}")
    # Ollama's capability init validates the model against a live server; that side effect is
    # not what this test measures.
    if hasattr(provider, "_validate_model_available"):
        monkeypatch.setattr(provider, "_validate_model_available", lambda *_a, **_k: None)
    return provider


@pytest.mark.parametrize("module_path,class_name", PROVIDERS, ids=[p[1] for p in PROVIDERS])
def test_initialize_readies_llm_capability_even_with_summaries_off(
    module_path, class_name, monkeypatch
):
    """After ``initialize()``, the LLM capability is up regardless of ``generate_summaries``."""
    provider = _summaries_off_provider(module_path, class_name, monkeypatch)
    provider.initialize()
    assert provider._summarization_initialized, (
        f"{class_name}.initialize() left the LLM capability down because generate_summaries is "
        "False — but that flag governs the summary STAGE, and this capability also serves "
        "extract_kg_graph / generate_insights / clean_transcript / the whole evidence stack "
        "(#1720)"
    )


def _canned_kg_response() -> SimpleNamespace:
    content = json.dumps(
        {
            "topics": [{"label": "artificial intelligence"}],
            "entities": [{"name": "OpenAI", "type": "org"}],
        }
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=None,
        model="deepseek-chat",
        id="req-test-1720",
    )


class _AlwaysFailingPrimary:
    """The primary tier as the incident saw it: every LLM call raises."""

    def initialize(self) -> None:  # pragma: no cover - lifecycle no-op
        pass

    def extract_kg_graph(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("primary is down (incident shape)")


def test_fallback_tier_built_midrun_serves_extract_kg_graph(monkeypatch):
    """The exact PODCAST-PIPELINE-19 shape, proven at the consuming end.

    The failover ladder builds its tier lazily, mid-run, from the run's own cfg — summaries
    off. Before the fix the tier came up with the LLM capability down and the ladder recovered
    nothing; now the tier must actually SERVE the KG call.
    """
    from podcast_scraper.providers.openai.openai_provider import OpenAICompatibleProvider
    from podcast_scraper.summarization import factory as summ_factory
    from podcast_scraper.summarization.fallback import FallbackAwareSummarizationProvider

    cfg = config.Config(rss_url="https://example.com/feed.xml", generate_summaries=False)

    def _build_tier(cfg_arg, provider_type_override=None, **_kwargs):
        assert provider_type_override == "deepseek"
        tier = OpenAICompatibleProvider(cfg_arg)
        monkeypatch.setattr(tier, "_chat_create", lambda *a, **k: _canned_kg_response())
        return tier

    monkeypatch.setattr(summ_factory, "create_summarization_provider", _build_tier)

    wrapped = FallbackAwareSummarizationProvider(_AlwaysFailingPrimary(), ["deepseek"], cfg)
    result = wrapped.extract_kg_graph("a transcript long enough to mean something")

    assert result is not None, (
        "the fallback tier was built and initialize()d by the ladder itself, yet could not "
        "serve extract_kg_graph — the generate_summaries gate starved it (#1720)"
    )
    assert [e["name"] for e in result["entities"]] == ["OpenAI"]
