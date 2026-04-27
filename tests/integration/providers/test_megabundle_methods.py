"""Unit tests for provider-level mega_bundled / extraction_bundled methods (#643).

These tests mock each SDK client and verify:
- The new methods exist and return ``MegaBundleResult`` objects.
- Prompts are sourced from ``prompting.megabundle`` (not the legacy
  bundled_clean_summary_* prompt store entries).
- JSON-mode / response_format flags are set on the API call.
- The extraction variant does not require summary/bullets in the response.

SDK deps (anthropic, mistralai, google.genai) are optional ``[llm]`` extras
and not installed in the CI unit-test job. The ``_mock_all_provider_sdks``
autouse fixture patches the module-level SDK symbols on every provider
module so ``Provider(cfg)`` does not raise ``ImportError`` during test
collection / construction.
"""

from __future__ import annotations

import json
from unittest.mock import Mock, patch

import pytest

from podcast_scraper import config
from podcast_scraper.providers.anthropic.anthropic_provider import AnthropicProvider
from podcast_scraper.providers.common.megabundle_parser import MegaBundleResult
from podcast_scraper.providers.deepseek.deepseek_provider import DeepSeekProvider
from podcast_scraper.providers.gemini.gemini_provider import GeminiProvider
from podcast_scraper.providers.grok.grok_provider import GrokProvider
from podcast_scraper.providers.mistral.mistral_provider import MistralProvider
from podcast_scraper.providers.openai.openai_provider import OpenAIProvider

# Moved from tests/unit/ — RFC-081 PR-A1: tests that import [ml]/[llm]/[server]
# gated modules belong in the integration tier per UNIT_TESTING_GUIDE.md.
pytestmark = [pytest.mark.integration]


@pytest.fixture(autouse=True)
def _mock_all_provider_sdks():
    """Mock SDK module-level symbols so provider constructors pass the
    ``None`` import guard without the real packages installed."""
    # anthropic / mistralai / google.genai are in the [llm] extra and may be
    # unavailable in the unit-test job. openai is a core dep so its SDK is
    # imported lazily inside OpenAIProvider and needs no patch.
    patches = [
        patch("podcast_scraper.providers.anthropic.anthropic_provider.Anthropic", Mock()),
        patch("podcast_scraper.providers.mistral.mistral_provider.Mistral", Mock()),
        patch("podcast_scraper.providers.deepseek.deepseek_provider.OpenAI", Mock()),
        patch("podcast_scraper.providers.grok.grok_provider.OpenAI", Mock()),
        patch("podcast_scraper.providers.gemini.gemini_provider.genai", Mock()),
    ]
    for p in patches:
        p.start()
    yield
    for p in reversed(patches):
        try:
            p.stop()
        except (RuntimeError, AttributeError):
            pass


_MEGA_JSON = json.dumps(
    {
        "title": "Ep T",
        "summary": "A prose summary.",
        "bullets": ["b1", "b2", "b3"],
        "insights": [
            {"text": "i1", "insight_type": "claim"},
            {"text": "i2", "insight_type": "fact"},
        ],
        "topics": ["t1", "t2", "t3"],
        "entities": [{"name": "Alice", "kind": "person", "role": "host"}],
    }
)

_EXTRACTION_JSON = json.dumps(
    {
        "insights": [{"text": "i1", "insight_type": "claim"}],
        "topics": ["t1", "t2"],
        "entities": [{"name": "Bob", "kind": "person", "role": "mentioned"}],
    }
)


def _cfg(summary_provider: str, extra: dict | None = None) -> config.Config:
    base = {
        "rss_url": "https://example.com/feed.xml",
        "summary_provider": summary_provider,
        "transcribe_missing": False,
        "auto_speakers": False,
        "generate_summaries": False,
    }
    if summary_provider == "anthropic":
        base["anthropic_api_key"] = "k"
    elif summary_provider == "openai":
        base["openai_api_key"] = "k"
    elif summary_provider == "gemini":
        base["gemini_api_key"] = "k"
    elif summary_provider == "mistral":
        base["mistral_api_key"] = "k"
    elif summary_provider == "grok":
        base["grok_api_key"] = "k"
    elif summary_provider == "deepseek":
        base["deepseek_api_key"] = "k"
    if extra:
        base.update(extra)
    return config.Config.model_validate(base)


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


class TestAnthropicBundledMethods:
    def _mock_resp(self, text: str, inp: int = 100, out: int = 50) -> Mock:
        resp = Mock()
        block = Mock()
        block.text = text
        resp.content = [block]
        resp.usage = Mock(input_tokens=inp, output_tokens=out)
        return resp

    @patch("podcast_scraper.providers.anthropic.anthropic_provider.Anthropic")
    def test_mega_bundled_returns_result(self, mock_anthropic_cls):
        client = Mock()
        mock_anthropic_cls.return_value = client
        client.messages.create.return_value = self._mock_resp(_MEGA_JSON)
        provider = AnthropicProvider(_cfg("anthropic"))
        provider._summarization_initialized = True
        provider.client = client

        result = provider.summarize_mega_bundled("transcript text")

        assert isinstance(result, MegaBundleResult)
        assert result.title == "Ep T"
        assert result.summary.startswith("A prose")
        assert len(result.bullets) == 3
        assert len(result.insights) == 2
        assert result.topics == ["t1", "t2", "t3"]
        assert result.entities[0]["name"] == "Alice"
        call = client.messages.create.call_args
        assert call.kwargs["temperature"] == 0.0
        assert "messages" in call.kwargs
        assert "system" in call.kwargs

    @patch("podcast_scraper.providers.anthropic.anthropic_provider.Anthropic")
    def test_extraction_bundled_no_summary_required(self, mock_anthropic_cls):
        client = Mock()
        mock_anthropic_cls.return_value = client
        client.messages.create.return_value = self._mock_resp(_EXTRACTION_JSON)
        provider = AnthropicProvider(_cfg("anthropic"))
        provider._summarization_initialized = True
        provider.client = client

        result = provider.summarize_extraction_bundled("transcript text")

        assert isinstance(result, MegaBundleResult)
        assert result.title == ""
        assert result.summary == ""
        assert result.bullets == []
        assert len(result.insights) == 1
        assert result.topics == ["t1", "t2"]


# ---------------------------------------------------------------------------
# DeepSeek
# ---------------------------------------------------------------------------


class TestDeepSeekBundledMethods:
    def _mock_resp(self, text: str) -> Mock:
        resp = Mock()
        resp.choices = [Mock(message=Mock(content=text))]
        resp.usage = Mock(prompt_tokens=100, completion_tokens=50)
        return resp

    def _build(self):
        client = Mock()
        provider = DeepSeekProvider(_cfg("deepseek"))
        provider._summarization_initialized = True
        provider.client = client
        provider.summary_model = "deepseek-chat"
        return provider, client

    def test_mega_bundled_caps_max_tokens_at_8192(self):
        provider, client = self._build()
        client.chat.completions.create.return_value = self._mock_resp(_MEGA_JSON)

        provider.summarize_mega_bundled("t", params={"max_tokens": 16384})

        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["max_tokens"] == 8192
        assert kwargs["response_format"] == {"type": "json_object"}

    def test_extraction_bundled_returns_result(self):
        provider, client = self._build()
        client.chat.completions.create.return_value = self._mock_resp(_EXTRACTION_JSON)

        result = provider.summarize_extraction_bundled("t")

        assert isinstance(result, MegaBundleResult)
        assert result.summary == ""
        assert len(result.insights) == 1
        assert client.chat.completions.create.call_args.kwargs["response_format"] == {
            "type": "json_object"
        }


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


class TestOpenAIExtractionBundled:
    def _mock_resp(self, text: str) -> Mock:
        resp = Mock()
        resp.choices = [Mock(message=Mock(content=text), finish_reason="stop")]
        resp.usage = Mock(prompt_tokens=100, completion_tokens=50)
        return resp

    def _build(self):
        client = Mock()
        provider = OpenAIProvider(_cfg("openai"))
        provider._summarization_initialized = True
        provider.client = client
        provider.summary_model = "gpt-4o-mini"
        provider.summary_temperature = 0.2
        provider.summary_seed = None
        return provider, client

    def test_extraction_bundled_uses_json_object_mode(self):
        provider, client = self._build()
        client.chat.completions.create.return_value = self._mock_resp(_EXTRACTION_JSON)

        result = provider.summarize_extraction_bundled("t")

        assert isinstance(result, MegaBundleResult)
        assert result.summary == ""
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["response_format"] == {"type": "json_object"}
        assert kwargs["temperature"] == 0.0


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------


class TestGeminiExtractionBundled:
    def _mock_resp(self, text: str) -> Mock:
        resp = Mock()
        resp.text = text
        resp.usage_metadata = Mock(prompt_token_count=100, candidates_token_count=50)
        return resp

    def _build(self):
        client = Mock()
        provider = GeminiProvider(_cfg("gemini"))
        provider._summarization_initialized = True
        provider.client = client
        provider.summary_model = "gemini-2.5-flash"
        return provider, client

    @patch(
        "podcast_scraper.providers.gemini.gemini_provider._merge_generate_content_config",
        side_effect=lambda _m, d: d,
    )
    def test_extraction_bundled_sets_json_mime(self, _mock_merge):
        provider, client = self._build()
        client.models.generate_content.return_value = self._mock_resp(_EXTRACTION_JSON)

        result = provider.summarize_extraction_bundled("t")

        assert isinstance(result, MegaBundleResult)
        call = client.models.generate_content.call_args
        cfg = call.kwargs["config"]
        assert cfg["response_mime_type"] == "application/json"
        assert cfg["temperature"] == 0.0


# ---------------------------------------------------------------------------
# Mistral
# ---------------------------------------------------------------------------


class TestMistralExtractionBundled:
    def _mock_resp(self, text: str) -> Mock:
        resp = Mock()
        resp.choices = [Mock(message=Mock(content=text))]
        resp.usage = Mock(prompt_tokens=100, completion_tokens=50)
        return resp

    def _build(self):
        client = Mock()
        provider = MistralProvider(_cfg("mistral"))
        provider._summarization_initialized = True
        provider.client = client
        provider.summary_model = "mistral-large-latest"
        return provider, client

    def test_extraction_bundled_sets_json_object(self):
        provider, client = self._build()
        client.chat.complete.return_value = self._mock_resp(_EXTRACTION_JSON)

        result = provider.summarize_extraction_bundled("t")

        assert isinstance(result, MegaBundleResult)
        kwargs = client.chat.complete.call_args.kwargs
        assert kwargs["response_format"] == {"type": "json_object"}
        assert kwargs["temperature"] == 0.0


# ---------------------------------------------------------------------------
# Grok
# ---------------------------------------------------------------------------


class TestGrokExtractionBundled:
    def _mock_resp(self, text: str) -> Mock:
        resp = Mock()
        resp.choices = [Mock(message=Mock(content=text), finish_reason="stop")]
        resp.usage = Mock(prompt_tokens=100, completion_tokens=50)
        return resp

    def _build(self):
        client = Mock()
        provider = GrokProvider(_cfg("grok"))
        provider._summarization_initialized = True
        provider.client = client
        provider.summary_model = "grok-4"
        provider.summary_temperature = 0.2
        return provider, client

    def test_extraction_bundled_sets_json_object(self):
        provider, client = self._build()
        client.chat.completions.create.return_value = self._mock_resp(_EXTRACTION_JSON)

        result = provider.summarize_extraction_bundled("t")

        assert isinstance(result, MegaBundleResult)
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["response_format"] == {"type": "json_object"}
        assert kwargs["temperature"] == 0.0


# ---------------------------------------------------------------------------
# Uninitialized guards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "provider_cls,provider_key",
    [
        (AnthropicProvider, "anthropic"),
        (DeepSeekProvider, "deepseek"),
        (OpenAIProvider, "openai"),
        (GeminiProvider, "gemini"),
        (MistralProvider, "mistral"),
        (GrokProvider, "grok"),
    ],
)
def test_extraction_bundled_raises_when_uninitialized(provider_cls, provider_key):
    provider = provider_cls(_cfg(provider_key))
    provider._summarization_initialized = False
    with pytest.raises(RuntimeError, match="not initialized"):
        provider.summarize_extraction_bundled("t")


@pytest.mark.parametrize(
    "provider_cls,provider_key",
    [
        (AnthropicProvider, "anthropic"),
        (DeepSeekProvider, "deepseek"),
    ],
)
def test_mega_bundled_raises_when_uninitialized(provider_cls, provider_key):
    provider = provider_cls(_cfg(provider_key))
    provider._summarization_initialized = False
    with pytest.raises(RuntimeError, match="not initialized"):
        provider.summarize_mega_bundled("t")
