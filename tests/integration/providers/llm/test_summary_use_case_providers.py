"""The board summary is a use case of every provider's own ``summarize`` (RFC-127).

Mirrors each provider's ``test_summarize_success`` (the SDK client mocked, nothing on the network),
but renders the REAL prompt templates through the prompt store, so the test fails if a provider
resolves the board prompts anywhere but ``<ns>/summarization/board_*`` → ``shared/summarization``.

For every provider the use case must change exactly four things and nothing else:

- the prompts sent are the board templates, rendered with the material;
- the reply budget is the use case's;
- the ``llm_cost`` record carries the use case's stage;
- the metadata names the board templates.

And without a use case, ``summarize`` is the episode summary it always was.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterator, List, Tuple
from unittest.mock import Mock, patch

import pytest

from podcast_scraper import config
from podcast_scraper.providers.common.summary_use_case import SummaryUseCase

pytestmark = pytest.mark.critical_path

BUDGET = 333
STAGE = "board_summary"
SYSTEM_MARKER = "a short summary of a listener's board"
MATERIAL = {
    "episodes": [
        {
            "title": "Quokka Ledger Hour",
            "show": "Marsupial Finance",
            "summary": "Two hosts argue about small-island accounting.",
            "key_points": ["Ledgers drift when nobody reconciles them."],
            "insights": [{"speaker": "Ada Lin", "text": "Reconcile weekly, not yearly."}],
        }
    ],
    "moments": [{"text": "Counting is a moral act.", "speaker": "Ada Lin", "episode": "Quokka"}],
}
REPLY = "This board is about reconciling ledgers."


def _use_case() -> SummaryUseCase:
    return SummaryUseCase(
        system_prompt="summarization/board_system_v1",
        user_prompt="summarization/board_user_v1",
        stage=STAGE,
        max_length=BUDGET,
        params=MATERIAL,
    )


def _walk(obj: Any, depth: int = 0) -> Iterator[Tuple[str, Any]]:
    """Every (key, value) in a call's kwargs, through dicts, lists and SDK config objects."""
    if depth > 4:
        return
    if isinstance(obj, dict):
        items = list(obj.items())
    elif isinstance(obj, (list, tuple)):
        items = [("", v) for v in obj]
    elif hasattr(obj, "__dict__") and not isinstance(obj, Mock):
        items = list(vars(obj).items())
    else:
        return
    for k, v in items:
        yield str(k), v
        yield from _walk(v, depth + 1)


def _sent_text(kwargs: Dict[str, Any]) -> str:
    return " ".join(str(v) for _k, v in _walk(kwargs) if isinstance(v, str))


def _sent_budget(kwargs: Dict[str, Any]) -> List[int]:
    keys = {"max_tokens", "max_completion_tokens", "max_output_tokens"}
    return [v for k, v in _walk(kwargs) if k in keys and isinstance(v, int)]


# --- one builder per provider: (provider, the mocked SDK call that carries the request) ---


def _cfg(**overrides: Any) -> config.Config:
    base: Dict[str, Any] = {
        "rss_url": "https://example.com/feed.xml",
        "generate_summaries": True,
        "generate_metadata": True,
    }
    base.update(overrides)
    return config.Config(**base)


def _openai_chat_response() -> Mock:
    resp = Mock()
    resp.choices = [Mock()]
    resp.choices[0].message.content = REPLY
    resp.choices[0].finish_reason = "stop"
    resp.usage = Mock()
    resp.usage.prompt_tokens = 900
    resp.usage.completion_tokens = 60
    return resp


def _openai(stack: List[Any]) -> Tuple[Any, Mock]:
    from podcast_scraper.providers.openai.openai_provider import OpenAIProvider

    cfg = _cfg(
        summary_provider="openai",
        openai_api_key="sk-test123",
    )
    client = Mock()
    client.chat.completions.create.return_value = _openai_chat_response()
    provider = OpenAIProvider(cfg)
    provider.client = client
    provider.initialize()
    return provider, client.chat.completions.create


def _vllm(stack: List[Any]) -> Tuple[Any, Mock]:
    from podcast_scraper.providers.vllm import VLLMProvider

    model = "NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4"
    cfg = _cfg(
        summary_provider="vllm",
        speaker_detector_provider="vllm",
        vllm_api_base="http://dgx:8003/v1",
        vllm_summary_model=model,
        vllm_speaker_model=model,
    )
    client = Mock()
    client.chat.completions.create.return_value = _openai_chat_response()
    provider = VLLMProvider(cfg)
    provider.client = client
    provider.initialize()
    return provider, client.chat.completions.create


def _anthropic(stack: List[Any]) -> Tuple[Any, Mock]:
    from podcast_scraper.providers.anthropic.anthropic_provider import AnthropicProvider

    stack.append(patch("podcast_scraper.providers.anthropic.anthropic_provider.Anthropic"))
    mock_anthropic = stack[-1].start()
    client = Mock()
    mock_anthropic.return_value = client
    resp = Mock()
    resp.content = [Mock()]
    resp.content[0].text = REPLY
    resp.stop_reason = "end_turn"
    resp.usage = Mock()
    resp.usage.input_tokens = 900
    resp.usage.output_tokens = 60
    client.messages.create.return_value = resp
    cfg = _cfg(
        summary_provider="anthropic",
        anthropic_api_key="test-api-key-123",
    )
    provider = AnthropicProvider(cfg)
    provider.initialize()
    return provider, client.messages.create


def _gemini(stack: List[Any]) -> Tuple[Any, Mock]:
    from podcast_scraper.providers.gemini.gemini_provider import GeminiProvider

    stack.append(patch("podcast_scraper.providers.gemini.gemini_provider.genai"))
    mock_genai = stack[-1].start()
    resp = Mock()
    resp.text = REPLY
    resp.usage_metadata = Mock()
    resp.usage_metadata.prompt_token_count = 900
    resp.usage_metadata.candidates_token_count = 60
    resp.candidates = [Mock()]
    resp.candidates[0].finish_reason = "STOP"
    client = Mock()
    client.models.generate_content.return_value = resp
    mock_genai.Client.return_value = client
    cfg = _cfg(
        summary_provider="gemini",
        gemini_api_key="test-api-key-123",
    )
    provider = GeminiProvider(cfg)
    provider.initialize()
    return provider, client.models.generate_content


def _grok(stack: List[Any]) -> Tuple[Any, Mock]:
    from podcast_scraper.providers.grok.grok_provider import GrokProvider

    stack.append(patch("podcast_scraper.providers.grok.grok_provider.OpenAI"))
    mock_openai = stack[-1].start()
    client = Mock()
    client.chat.completions.create.return_value = _openai_chat_response()
    mock_openai.return_value = client
    cfg = _cfg(
        summary_provider="grok",
        grok_api_key="test-api-key-123",
    )
    provider = GrokProvider(cfg)
    provider.initialize()
    return provider, client.chat.completions.create


def _mistral(stack: List[Any]) -> Tuple[Any, Mock]:
    from podcast_scraper.providers.mistral.mistral_provider import MistralProvider

    stack.append(patch("podcast_scraper.providers.mistral.mistral_provider.Mistral"))
    mock_mistral = stack[-1].start()
    client = Mock()
    client.chat.complete.return_value = _openai_chat_response()
    mock_mistral.return_value = client
    cfg = _cfg(
        summary_provider="mistral",
        mistral_api_key="test-api-key-123",
    )
    provider = MistralProvider(cfg)
    provider.initialize()
    return provider, client.chat.complete


def _ollama(stack: List[Any]) -> Tuple[Any, Mock]:
    from podcast_scraper.providers.ollama.ollama_provider import OllamaProvider

    stack.append(patch("podcast_scraper.providers.ollama.ollama_provider.httpx"))
    mock_httpx = stack[-1].start()
    stack.append(patch("podcast_scraper.providers.ollama.ollama_provider.OpenAI"))
    mock_openai = stack[-1].start()
    ok = Mock()
    ok.raise_for_status = Mock()
    ok.json.return_value = {"models": [{"name": "llama3.1:8b"}, {"name": "llama3.2:latest"}]}
    mock_httpx.get.return_value = ok
    client = Mock()
    client.chat.completions.create.return_value = _openai_chat_response()
    mock_openai.return_value = client
    cfg = _cfg(
        summary_provider="ollama",
        ollama_api_base="http://localhost:11434/v1",
        auto_speakers=False,
    )
    provider = OllamaProvider(cfg)
    provider.initialize()
    return provider, client.chat.completions.create


PROVIDERS: Dict[str, Tuple[Callable[[List[Any]], Tuple[Any, Mock]], str]] = {
    # name: (builder, the prompt folder its templates resolve under)
    "openai": (_openai, "openai"),
    "vllm": (_vllm, "openai"),
    "anthropic": (_anthropic, "anthropic"),
    "gemini": (_gemini, "gemini"),
    "grok": (_grok, "grok"),
    "mistral": (_mistral, "mistral"),
    "ollama": (_ollama, "ollama"),
}


@pytest.fixture
def built(request: pytest.FixtureRequest) -> Iterator[Tuple[Any, Mock, Mock, str]]:
    builder, namespace = PROVIDERS[request.param]
    stack: List[Any] = [patch("podcast_scraper.utils.provider_metrics.record_provider_call_cost")]
    cost = stack[0].start()
    try:
        provider, sdk_call = builder(stack)
        yield provider, sdk_call, cost, namespace
    finally:
        for p in reversed(stack):
            p.stop()


@pytest.mark.parametrize("built", list(PROVIDERS), indirect=True)
class TestBoardUseCase:
    def test_sends_the_board_templates_rendered_with_the_material(self, built):
        provider, sdk_call, _cost, _ns = built
        result = provider.summarize(text="", params={"use_case": _use_case()})

        assert result["summary"] == REPLY
        sent = _sent_text(sdk_call.call_args.kwargs)
        assert SYSTEM_MARKER in sent
        assert "Quokka Ledger Hour" in sent and "Counting is a moral act." in sent
        assert "transcript" not in sent.lower()

    def test_reply_budget_is_the_use_cases(self, built):
        provider, sdk_call, _cost, _ns = built
        provider.summarize(text="", params={"use_case": _use_case()})

        assert BUDGET in _sent_budget(sdk_call.call_args.kwargs)

    def test_cost_is_recorded_under_the_use_case_stage(self, built):
        provider, _sdk, cost, _ns = built
        provider.summarize(text="", params={"use_case": _use_case()})

        assert cost.call_count == 1
        assert cost.call_args.kwargs["stage"] == STAGE
        assert cost.call_args.kwargs["capability"] == "summarization"

    def test_metadata_names_the_board_templates(self, built):
        provider, _sdk, _cost, ns = built
        meta = provider.summarize(text="", params={"use_case": _use_case()})["metadata"]

        prompts = meta["prompts"]
        assert prompts["system"]["name"] == f"{ns}/summarization/board_system_v1"
        assert prompts["user"]["name"] == f"{ns}/summarization/board_user_v1"
        assert prompts["system"]["sha256"] and prompts["user"]["sha256"]
        assert prompts["system"]["file"] == "shared/summarization/board_system_v1.j2"
        # The material's sizes, not the episode prompt's transcript/title params, and never the
        # material itself.
        assert prompts["user"]["params"] == {"episodes": 1, "moments": 1}
        assert "Quokka" not in str(prompts)

    def test_without_a_use_case_it_is_the_episode_summary(self, built):
        provider, sdk_call, cost, _ns = built
        provider.summarize("A long transcript about ledgers.")

        sent = _sent_text(sdk_call.call_args.kwargs)
        assert SYSTEM_MARKER not in sent
        assert "A long transcript about ledgers." in sent
        assert BUDGET not in _sent_budget(sdk_call.call_args.kwargs)
        assert cost.call_args.kwargs["stage"] is None
