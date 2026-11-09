"""The LLM providers' bundled GIL-evidence methods, after their move into ``*_provider_gi_naming``
modules (E04 S1), on the unit tier: CI's unit job installs ``.[dev]`` only, so the SDKs are absent.

Each provider module imports its SDK in ``try/except ImportError`` and keeps the result in a module
variable (``Anthropic``, ``Mistral``, ``genai``, ``OpenAI``). The provider is built with that
variable patched to a Mock for the constructor only — nothing in ``sys.modules`` changes, which is
what tests/conftest.py guards — and ``retry_with_metrics`` is patched to return the SDK's response
shape. Ollama's constructor also probes the server through the module's ``httpx``; patching that
name too means no Ollama has to run. The integration tier's ``test_*_bundled_methods.py`` cover the
same methods with the real SDK classes importable.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, Tuple
from unittest.mock import MagicMock, Mock, patch

import pytest

from podcast_scraper import config
from podcast_scraper.providers.common.bundle_extract_parser import BundleExtractParseError
from podcast_scraper.providers.common.bundle_nli_parser import BundleNliParseError

pytestmark = pytest.mark.unit

RETRY = "podcast_scraper.utils.provider_metrics.retry_with_metrics"


def _anthropic_response(text: str) -> Mock:
    block = Mock()
    block.text = text
    return Mock(content=[block], usage=Mock(input_tokens=100, output_tokens=50))


def _chat_response(text: str) -> Mock:
    choice = Mock()
    choice.message = Mock(content=text)
    usage = Mock(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    return Mock(choices=[choice], usage=usage)


def _gemini_response(text: str) -> Mock:
    resp = Mock()
    resp.text = text
    resp.usage_metadata = Mock(prompt_token_count=100, candidates_token_count=50)
    return resp


#: name -> (provider module, SDK variable(s) to patch, provider class name, cfg, model, response)
PROVIDERS: Dict[
    str, Tuple[str, Tuple[str, ...], str, Dict[str, Any], str, Callable[[str], Mock]]
] = {
    "anthropic": (
        "podcast_scraper.providers.anthropic.anthropic_provider",
        ("Anthropic", "anthropic"),
        "AnthropicProvider",
        {"summary_provider": "anthropic", "anthropic_api_key": "k"},
        "claude-haiku-4-5",
        _anthropic_response,
    ),
    "mistral": (
        "podcast_scraper.providers.mistral.mistral_provider",
        ("Mistral",),
        "MistralProvider",
        {"summary_provider": "mistral", "mistral_api_key": "k"},
        "mistral-small-latest",
        _chat_response,
    ),
    "grok": (
        "podcast_scraper.providers.grok.grok_provider",
        ("OpenAI",),
        "GrokProvider",
        {"summary_provider": "grok", "grok_api_key": "k"},
        "grok-3-fast",
        _chat_response,
    ),
    "ollama": (
        "podcast_scraper.providers.ollama.ollama_provider",
        ("OpenAI", "httpx"),
        "OllamaProvider",
        {"summary_provider": "ollama"},
        "llama3.1:8b",
        _chat_response,
    ),
    "gemini": (
        "podcast_scraper.providers.gemini.gemini_provider",
        ("genai",),
        "GeminiProvider",
        {"summary_provider": "gemini", "gemini_api_key": "k"},
        "gemini-2.5-flash-lite",
        _gemini_response,
    ),
}


def _provider(name: str) -> Any:
    import importlib

    module_name, sdk_vars, cls_name, cfg_kwargs, model, _resp = PROVIDERS[name]
    module = importlib.import_module(module_name)
    patches = [patch.object(module, var, MagicMock()) for var in sdk_vars]
    for p in patches:
        p.start()
    try:
        provider = getattr(module, cls_name)(config.Config(**cfg_kwargs))
    finally:
        for p in patches:
            p.stop()
    provider.client = MagicMock()
    provider._summarization_initialized = True
    provider.summary_model = model
    return provider


def _reply(name: str, text: str) -> Any:
    return patch(RETRY, return_value=PROVIDERS[name][5](text))


@pytest.mark.parametrize("name", sorted(PROVIDERS))
class TestExtractQuotesBundled:
    def test_an_uninitialized_provider_says_so(self, name: str) -> None:
        provider = _provider(name)
        provider._summarization_initialized = False
        with pytest.raises(RuntimeError, match="initial"):
            provider.extract_quotes_bundled("transcript", ["i"])

    def test_empty_inputs_return_without_a_call(self, name: str) -> None:
        provider = _provider(name)
        assert provider.extract_quotes_bundled("", ["a", "b"]) == {0: [], 1: []}
        assert provider.extract_quotes_bundled("transcript", []) == {}

    def test_quotes_resolve_to_verbatim_spans_once_each(self, name: str) -> None:
        provider = _provider(name)
        transcript = "The cat sat on the mat. The dog barked loudly."
        payload = {
            "0": ["The cat sat on the mat.", "The cat sat on the mat."],
            "1": ["The dog barked loudly."],
        }
        with _reply(name, json.dumps(payload)):
            out = provider.extract_quotes_bundled(transcript, ["cat", "dog"])
        assert [len(out[0]), len(out[1])] == [1, 1]
        for idx in (0, 1):
            qc = out[idx][0]
            assert transcript[qc.char_start : qc.char_end] == qc.text

    def test_a_reply_that_is_not_json_raises(self, name: str) -> None:
        provider = _provider(name)
        with _reply(name, "not valid JSON {"):
            with pytest.raises(BundleExtractParseError):
                provider.extract_quotes_bundled("transcript", ["i"])

    def test_an_sdk_failure_propagates(self, name: str) -> None:
        provider = _provider(name)
        with patch(RETRY, side_effect=RuntimeError("upstream 503")):
            with pytest.raises(RuntimeError, match="upstream 503"):
                provider.extract_quotes_bundled("transcript", ["i"])


@pytest.mark.parametrize("name", sorted(PROVIDERS))
class TestScoreEntailmentBundled:
    def test_an_uninitialized_provider_says_so(self, name: str) -> None:
        provider = _provider(name)
        provider._summarization_initialized = False
        with pytest.raises(RuntimeError, match="initial"):
            provider.score_entailment_bundled([("p", "h")])

    def test_no_pairs_no_call(self, name: str) -> None:
        assert _provider(name).score_entailment_bundled([]) == {}

    def test_chunks_are_scored_and_merged_in_order(self, name: str) -> None:
        provider = _provider(name)
        pairs = [(f"p{i}", f"h{i}") for i in range(3)]
        with patch.object(
            provider, "_score_entailment_bundled_chunk", side_effect=[{0: 0.7, 1: 0.8}, {0: 0.6}]
        ):
            assert provider.score_entailment_bundled(pairs, chunk_size=2) == {
                0: 0.7,
                1: 0.8,
                2: 0.6,
            }

    def test_a_chunk_reply_is_parsed(self, name: str) -> None:
        provider = _provider(name)
        with _reply(name, json.dumps({"0": 0.65, "1": 0.40})):
            scores = provider._score_entailment_bundled_chunk(
                chunk_pairs=[("p1", "h1"), ("p2", "h2")], pipeline_metrics=None
            )
        assert scores == {0: 0.65, 1: 0.40}

    def test_a_chunk_reply_that_is_not_json_raises(self, name: str) -> None:
        provider = _provider(name)
        with _reply(name, "not JSON"):
            with pytest.raises(BundleNliParseError):
                provider._score_entailment_bundled_chunk(
                    chunk_pairs=[("p", "h")], pipeline_metrics=None
                )

    def test_a_chunk_sdk_failure_propagates(self, name: str) -> None:
        provider = _provider(name)
        with patch(RETRY, side_effect=RuntimeError("503")):
            with pytest.raises(RuntimeError, match="503"):
                provider._score_entailment_bundled_chunk(
                    chunk_pairs=[("p", "h")], pipeline_metrics=None
                )
