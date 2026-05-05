"""Unit tests for OpenAI-compat providers' GIL-evidence bundled methods (#698).

Covers ``extract_quotes_bundled`` + ``score_entailment_bundled`` +
``_score_entailment_bundled_chunk`` across the 5 providers that use the
OpenAI Chat Completions SDK shape:

- OpenAI
- DeepSeek
- Grok
- Mistral
- Ollama

All share the ``client.chat.completions.create`` call pattern and
``response.choices[0].message.content`` response shape, so one parameterized
test suite covers all 5 with minimal duplication. Provider-specific glue
(retry-class, token helper, provider name) differs but is bypassed by patching
``retry_with_metrics`` directly.
"""

from __future__ import annotations

import importlib.util
import json
from typing import Tuple, Type
from unittest.mock import MagicMock, Mock, patch

import pytest

# Mock the LLM SDKs whose module-level imports run when the provider modules
# load. ``httpx`` is intentionally NOT mocked — integration tier has it
# installed for real, and OpenAIProvider's timeout_config does
# ``isinstance(base, httpx.Timeout)`` which requires the real type. Same
# pattern as tests/integration/providers/llm/test_openai_provider.py.
mock_openai = MagicMock()
mock_openai.OpenAI = MagicMock()


class _MockOpenAIError(Exception):
    """Stand-in for openai.APIError so retry_with_metrics' except clauses parse."""


class _MockOpenAIRateLimit(Exception):
    """Stand-in for openai.RateLimitError."""


mock_openai.APIError = _MockOpenAIError
mock_openai.RateLimitError = _MockOpenAIRateLimit
mock_anthropic = MagicMock()
mock_mistralai = MagicMock()
for name, mod in {
    "openai": mock_openai,
    "anthropic": mock_anthropic,
    "mistralai": mock_mistralai,
}.items():
    mod.__spec__ = importlib.util.spec_from_loader(name, loader=None)
_patch_sdks = patch.dict(
    "sys.modules",
    {
        "openai": mock_openai,
        "anthropic": mock_anthropic,
        "mistralai": mock_mistralai,
    },
)


def setUpModule():
    # Scope the SDK mocks to this module only — otherwise they leak into other
    # integration test files (test_e2e_server.py, etc.) that need the real SDK
    # to construct working OpenAI/Anthropic/Mistral provider instances. Same
    # pattern as test_gemini_provider.py / test_openai_provider.py.
    _patch_sdks.start()


def tearDownModule():
    _patch_sdks.stop()


from podcast_scraper import config
from podcast_scraper.providers.common.bundle_extract_parser import BundleExtractParseError
from podcast_scraper.providers.common.bundle_nli_parser import BundleNliParseError

pytestmark = [pytest.mark.integration]


# ---------------------------------------------------------------------------
# Per-provider provisioning


def _provider_factory(name: str) -> Tuple[Type, dict, str]:
    """Return ``(provider_class, cfg_kwargs, provider_module_name)`` for ``name``.

    The provider_module_name is the import path to use when patching
    retry_with_metrics — providers re-import the helper from inside their
    methods, so we patch the original site (provider_metrics) and let it
    flow through.
    """
    if name == "openai":
        from podcast_scraper.providers.openai.openai_provider import OpenAIProvider

        cfg_kwargs = dict(
            summary_provider="openai",
            openai_api_key="test-key",
            openai_summary_model="gpt-4o-mini",
        )
        return OpenAIProvider, cfg_kwargs, "gpt-4o-mini"
    if name == "deepseek":
        from podcast_scraper.providers.deepseek.deepseek_provider import DeepSeekProvider

        cfg_kwargs = dict(
            summary_provider="deepseek",
            deepseek_api_key="test-key",
            deepseek_summary_model="deepseek-chat",
        )
        return DeepSeekProvider, cfg_kwargs, "deepseek-chat"
    if name == "grok":
        from podcast_scraper.providers.grok.grok_provider import GrokProvider

        cfg_kwargs = dict(
            summary_provider="grok",
            grok_api_key="test-key",
            grok_summary_model="grok-3-fast",
        )
        return GrokProvider, cfg_kwargs, "grok-3-fast"
    if name == "mistral":
        from podcast_scraper.providers.mistral.mistral_provider import MistralProvider

        cfg_kwargs = dict(
            summary_provider="mistral",
            mistral_api_key="test-key",
            mistral_summary_model="mistral-small-latest",
        )
        return MistralProvider, cfg_kwargs, "mistral-small-latest"
    if name == "ollama":
        from podcast_scraper.providers.ollama.ollama_provider import OllamaProvider

        cfg_kwargs = dict(
            summary_provider="ollama",
            ollama_summary_model="qwen3.5:9b",
        )
        return OllamaProvider, cfg_kwargs, "qwen3.5:9b"
    raise ValueError(f"unknown provider: {name}")


def _make_provider(name: str):
    cls, cfg_kwargs, model = _provider_factory(name)
    cfg = config.Config(**cfg_kwargs)
    provider = cls(cfg)
    provider.client = MagicMock()
    provider._summarization_initialized = True
    provider.summary_model = model
    return provider


def _make_chat_completions_response(text: str) -> Mock:
    """Build a fake response matching OpenAI Chat Completions SDK shape."""
    msg = Mock()
    msg.content = text
    choice = Mock()
    choice.message = msg
    response = Mock()
    response.choices = [choice]
    # Fake usage so token-extraction helpers don't AttributeError.
    response.usage = Mock(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    return response


def _patch_retry(text: str):
    """Patch retry_with_metrics at its source so all providers see the patch."""
    return patch(
        "podcast_scraper.utils.provider_metrics.retry_with_metrics",
        return_value=_make_chat_completions_response(text),
    )


def _patch_retry_raises(exc: Exception):
    return patch(
        "podcast_scraper.utils.provider_metrics.retry_with_metrics",
        side_effect=exc,
    )


def _ollama_reachable() -> bool:
    """Probe Ollama at default port — gate the ``ollama`` parameterization.

    OllamaProvider.__init__ does a real httpx health check against
    ``http://localhost:11434/api/tags`` (httpx is NOT mocked in this module
    because OpenAIProvider's timeout_config needs the real ``httpx.Timeout``
    type). On CI runners with no Ollama service running, instantiation
    raises ``ConnectionError`` before the test can mock anything — every
    ``[ollama]``-parameterized test then errors. Skip the parameter when
    the server is unreachable; operators with local Ollama still get the
    coverage. Probe time-boxed at 1s so test collection stays fast.
    """
    try:
        import httpx as _real_httpx

        with _real_httpx.Client(timeout=1.0) as c:
            return c.get("http://localhost:11434/api/tags").status_code == 200
    except Exception:
        return False


PROVIDERS = ["openai", "deepseek", "grok", "mistral"]
if _ollama_reachable():
    PROVIDERS.append("ollama")


# ---------------------------------------------------------------------------
# extract_quotes_bundled — early-return paths


@pytest.mark.parametrize("name", PROVIDERS)
class TestExtractQuotesBundledEarlyReturn:
    def test_returns_empty_dict_for_each_insight_when_not_initialized(self, name: str) -> None:
        provider = _make_provider(name)
        provider._summarization_initialized = False
        out = provider.extract_quotes_bundled("transcript", ["i1", "i2"])
        assert out == {0: [], 1: []}

    def test_returns_empty_dict_for_each_insight_when_transcript_empty(self, name: str) -> None:
        provider = _make_provider(name)
        out = provider.extract_quotes_bundled("", ["i1", "i2"])
        assert out == {0: [], 1: []}

    def test_returns_empty_dict_when_no_insights(self, name: str) -> None:
        provider = _make_provider(name)
        out = provider.extract_quotes_bundled("transcript", [])
        assert out == {}


# ---------------------------------------------------------------------------
# extract_quotes_bundled — happy path


@pytest.mark.parametrize("name", PROVIDERS)
class TestExtractQuotesBundledHappy:
    def test_parses_response_and_resolves_quote_spans(self, name: str) -> None:
        provider = _make_provider(name)
        transcript = "The cat sat on the mat. The dog barked loudly."
        bundled_payload = json.dumps(
            {
                "0": ["The cat sat on the mat."],
                "1": ["The dog barked loudly."],
            }
        )
        with _patch_retry(bundled_payload):
            out = provider.extract_quotes_bundled(transcript, ["cat insight", "dog insight"])
        assert set(out.keys()) == {0, 1}
        for idx in (0, 1):
            assert len(out[idx]) == 1
            qc = out[idx][0]
            assert transcript[qc.char_start : qc.char_end] == qc.text

    def test_dedups_repeated_quotes_per_insight(self, name: str) -> None:
        provider = _make_provider(name)
        transcript = "Alpha. Beta. Gamma."
        bundled_payload = json.dumps({"0": ["Alpha.", "Alpha.", "Beta."]})
        with _patch_retry(bundled_payload):
            out = provider.extract_quotes_bundled(transcript, ["insight"])
        texts = [qc.text for qc in out[0]]
        assert texts.count("Alpha.") == 1
        assert "Beta." in texts


# ---------------------------------------------------------------------------
# extract_quotes_bundled — error paths


@pytest.mark.parametrize("name", PROVIDERS)
class TestExtractQuotesBundledErrors:
    def test_parser_failure_raises(self, name: str) -> None:
        provider = _make_provider(name)
        with _patch_retry("not valid JSON {"):
            with pytest.raises(BundleExtractParseError):
                provider.extract_quotes_bundled("transcript", ["i"])

    def test_sdk_exception_propagates(self, name: str) -> None:
        provider = _make_provider(name)
        with _patch_retry_raises(RuntimeError("upstream 503")):
            with pytest.raises(RuntimeError, match="upstream 503"):
                provider.extract_quotes_bundled("transcript", ["i"])


# ---------------------------------------------------------------------------
# score_entailment_bundled


@pytest.mark.parametrize("name", PROVIDERS)
class TestScoreEntailmentBundled:
    def test_returns_empty_dict_when_not_initialized(self, name: str) -> None:
        provider = _make_provider(name)
        provider._summarization_initialized = False
        out = provider.score_entailment_bundled([("p", "h")])
        assert out == {}

    def test_returns_empty_dict_for_no_pairs(self, name: str) -> None:
        provider = _make_provider(name)
        out = provider.score_entailment_bundled([])
        assert out == {}

    def test_chunks_pairs_and_merges_indices(self, name: str) -> None:
        provider = _make_provider(name)
        pairs = [(f"p{i}", f"h{i}") for i in range(3)]
        with patch.object(
            provider,
            "_score_entailment_bundled_chunk",
            side_effect=[{0: 0.7, 1: 0.8}, {0: 0.6}],
        ):
            out = provider.score_entailment_bundled(pairs, chunk_size=2)
        assert out == {0: 0.7, 1: 0.8, 2: 0.6}

    def test_chunk_size_floored_at_1(self, name: str) -> None:
        provider = _make_provider(name)
        with patch.object(
            provider,
            "_score_entailment_bundled_chunk",
            return_value={0: 0.5},
        ) as chunk_fn:
            provider.score_entailment_bundled([("p", "h")], chunk_size=0)
        assert chunk_fn.call_count == 1


# ---------------------------------------------------------------------------
# _score_entailment_bundled_chunk


@pytest.mark.parametrize("name", PROVIDERS)
class TestScoreEntailmentBundledChunk:
    def test_parses_chunk_response(self, name: str) -> None:
        provider = _make_provider(name)
        pairs = [("p1", "h1"), ("p2", "h2")]
        with _patch_retry(json.dumps({"0": 0.65, "1": 0.40})):
            scores = provider._score_entailment_bundled_chunk(
                chunk_pairs=pairs, pipeline_metrics=None
            )
        assert scores == {0: 0.65, 1: 0.40}

    def test_parser_failure_raises(self, name: str) -> None:
        provider = _make_provider(name)
        with _patch_retry("not JSON"):
            with pytest.raises(BundleNliParseError):
                provider._score_entailment_bundled_chunk(
                    chunk_pairs=[("p", "h")], pipeline_metrics=None
                )

    def test_sdk_exception_propagates(self, name: str) -> None:
        provider = _make_provider(name)
        with _patch_retry_raises(RuntimeError("503")):
            with pytest.raises(RuntimeError, match="503"):
                provider._score_entailment_bundled_chunk(
                    chunk_pairs=[("p", "h")], pipeline_metrics=None
                )
