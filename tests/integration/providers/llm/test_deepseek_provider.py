"""Sibling-contract tests for the first-class DeepSeek provider (ADR-144).

DeepSeekProvider is a thin SIBLING of OpenAIProvider/VLLMProvider/LiteLLMProvider over the shared
:class:`OpenAICompatibleProvider` transport base. These tests cover ONLY what is DeepSeek-specific —
identity/namespace, the default (direct) endpoint that also works via a LiteLLM gateway, key-required
auth, reasoning-model token headroom, open-model heuristics, and factory dispatch. The inherited
stage methods (summary/cleaning/GI/KG/grounding) are covered by the base's own tests
(``test_openai_provider*``); re-testing them here would just duplicate the base.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from podcast_scraper.config import Config
from podcast_scraper.providers.deepseek.deepseek_provider import _model_reasons, DeepSeekProvider
from podcast_scraper.providers.openai.openai_provider import (
    OpenAICompatibleProvider,
    OpenAIProvider,
)

_KEY = "sk-" + "x" * 40
_REASONING_MODEL = "deepseek-v4-flash"
_CHAT_MODEL = "deepseek-chat"


def _ds_cfg(**overrides: Any) -> Config:
    base: Dict[str, Any] = dict(
        rss_url="https://example.com/feed.xml",
        summary_provider="deepseek",
        speaker_detector_provider="deepseek",
        generate_summaries=True,
        generate_metadata=True,
        deepseek_api_key=_KEY,
        deepseek_summary_model=_REASONING_MODEL,
        deepseek_speaker_model=_REASONING_MODEL,
    )
    base.update(overrides)
    return Config(**base)


class TestIdentity:
    def test_is_sibling_not_openai_subclass(self):
        # Shares the transport base, but is NOT an OpenAIProvider (ADR-144).
        assert issubclass(DeepSeekProvider, OpenAICompatibleProvider)
        assert not issubclass(DeepSeekProvider, OpenAIProvider)
        assert not issubclass(OpenAIProvider, DeepSeekProvider)

    def test_namespace_and_telemetry(self):
        p = DeepSeekProvider(_ds_cfg())
        assert p._CONFIG_NS == "deepseek"
        assert p._TELEMETRY_PROVIDER == "deepseek"
        assert p._PROVIDER_LABEL == "DeepSeek"


class TestDirectAndGatewayEndpoint:
    def test_default_endpoint_is_deepseek_direct(self):
        # "direct, no gateway" needs zero config: the sibling ships its own default endpoint.
        assert DeepSeekProvider._DEFAULT_API_BASE == "https://api.deepseek.com"

    def test_api_base_override_routes_via_gateway(self):
        # Same class, config-only switch: pointing deepseek_api_base at a LiteLLM gateway routes
        # DeepSeek THROUGH the gateway instead of direct (ADR-144). Asserting the live client's
        # base_url is fragile (a sibling test installs a global openai mock); assert the config.
        p = DeepSeekProvider(_ds_cfg(deepseek_api_base="http://homelab:4001/v1"))
        assert p.cfg.deepseek_api_base == "http://homelab:4001/v1"


class TestKeyRequiredAuth:
    def test_missing_key_raises(self):
        with pytest.raises(ValueError, match="DeepSeek API key required"):
            DeepSeekProvider(
                Config(
                    rss_url="https://example.com/feed.xml",
                    summary_provider="deepseek",
                    generate_summaries=True,
                    deepseek_summary_model=_CHAT_MODEL,
                )
            )

    def test_resolves_deepseek_namespaced_key(self):
        p = DeepSeekProvider(_ds_cfg())
        assert p._resolve_api_key(p.cfg) == _KEY


class TestReasoningHeadroom:
    def test_marker_detection(self):
        assert _model_reasons("deepseek-v4-flash") is True
        assert _model_reasons("deepseek-v4-flash-2026-07-31") is True
        assert _model_reasons("deepseek-reasoner") is True
        assert _model_reasons("deepseek-chat") is False

    def test_reasoning_model_gets_headroom(self):
        p = DeepSeekProvider(_ds_cfg(deepseek_summary_model=_REASONING_MODEL))
        # A tight evidence budget (10) would be consumed entirely by reasoning_content on a v4
        # model — headroom (2048) keeps the answer from truncating to empty.
        assert p._token_kwarg(10) == {"max_tokens": 2058}

    def test_headroom_capped_at_deepseek_limit(self):
        p = DeepSeekProvider(_ds_cfg(deepseek_summary_model=_REASONING_MODEL))
        assert p._token_kwarg(8000) == {"max_tokens": 8192}  # min(8000+2048, 8192)

    def test_chat_model_is_passthrough(self):
        p = DeepSeekProvider(_ds_cfg(deepseek_summary_model=_CHAT_MODEL))
        assert p._token_kwarg(10) == {"max_tokens": 10}
        # No o1/o3/gpt-5 max_completion_tokens rename even for an OpenAI-lookalike name.
        assert p._token_kwarg(256, model="gpt-5-lookalike") == {"max_tokens": 256}


class TestOpenModelHeuristics:
    def test_temperature_not_fixed(self):
        p = DeepSeekProvider(_ds_cfg())
        assert p._temp_fixed_at_default == set()

    def test_cleaning_cap_is_deepseek_8192(self):
        assert DeepSeekProvider._CLEANING_MAX_TOKENS_CAP == 8192

    def test_cleaning_model_explicit_pin_respected(self):
        p = DeepSeekProvider(_ds_cfg(deepseek_cleaning_model="deepseek-custom-clean"))
        assert p.cleaning_model == "deepseek-custom-clean"


class TestFactoryDispatch:
    def test_summarization_factory_returns_deepseek(self):
        from podcast_scraper.summarization.factory import create_summarization_provider

        p = create_summarization_provider(_ds_cfg())
        assert isinstance(p, DeepSeekProvider)

    def test_speaker_factory_returns_deepseek(self):
        from podcast_scraper.speaker_detectors.factory import create_speaker_detector

        p = create_speaker_detector(_ds_cfg())
        assert isinstance(p, DeepSeekProvider)


class TestOpenAIProviderUnaffected:
    def test_openai_still_requires_sk_key(self, monkeypatch):
        # The base auth is unchanged for the OpenAI-native sibling.
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OpenAI API key required"):
            OpenAIProvider(
                Config(rss_url="https://example.com/feed.xml", summary_provider="openai")
            )
