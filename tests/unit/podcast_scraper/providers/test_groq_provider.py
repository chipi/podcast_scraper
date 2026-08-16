"""Unit tests for the first-class Groq provider (ADR-147).

GroqProvider is a thin SIBLING of OpenAIProvider/DeepSeekProvider/QwenProvider over the shared
:class:`OpenAICompatibleProvider` transport base. These tests cover ONLY what is Groq-specific:
identity/namespace, its OWN ``groq`` cost/telemetry namespace, the vendor default endpoint
(``api.groq.com/openai/v1`` — present, unlike Qwen), warn-not-raise bearer auth (no ``sk-`` format
assumption), the reasoning-model token-headroom guard, factory dispatch, and — uniquely among the
LLM siblings — the DUAL-USE transcription path (the same class also serves whisper-large-v3-turbo).

Do NOT confuse ``groq`` (this provider — Groq's LPU cloud) with ``grok`` (xAI). One letter apart.

The inherited stage methods (summary/cleaning/GI/KG/grounding) are covered by the base's own tests.
No network.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from podcast_scraper.config import Config
from podcast_scraper.providers.groq import GroqProvider
from podcast_scraper.providers.groq.groq_provider import (
    _extra_body_disables_thinking,
    _model_reasons,
)
from podcast_scraper.providers.openai.openai_provider import (
    OpenAICompatibleProvider,
    OpenAIProvider,
)

pytestmark = pytest.mark.unit

_MODEL = "llama-3.3-70b-versatile"
_REASONING_MODEL = "qwen/qwen3.6-27b"


def _groq_cfg(**overrides: Any) -> Config:
    base: Dict[str, Any] = dict(
        rss_url="https://example.com/feed.xml",
        summary_provider="groq",
        speaker_detector_provider="groq",
        generate_summaries=True,
        generate_metadata=True,
        groq_api_key="gsk_test-key",
        groq_summary_model=_MODEL,
        groq_speaker_model=_MODEL,
    )
    base.update(overrides)
    return Config(**base)


class TestIdentity:
    def test_is_sibling_not_openai_subclass(self):
        # Shares the transport base, but is NOT an OpenAIProvider (ADR-147).
        assert issubclass(GroqProvider, OpenAICompatibleProvider)
        assert not issubclass(GroqProvider, OpenAIProvider)
        assert not issubclass(OpenAIProvider, GroqProvider)

    def test_namespace_and_telemetry(self):
        p = GroqProvider(_groq_cfg())
        assert p._CONFIG_NS == "groq"
        assert p._TELEMETRY_PROVIDER == "groq"
        assert p._PROVIDER_LABEL == "Groq"
        assert p.get_capabilities().provider_name == "groq"


class TestVendorDefaultEndpoint:
    def test_has_default_api_base(self):
        # Unlike Qwen (None), Groq commits to one vendor endpoint so "direct" needs zero config.
        assert GroqProvider._DEFAULT_API_BASE == "https://api.groq.com/openai/v1"

    def test_gateway_override_wins(self):
        # Pointing groq_api_base at a LiteLLM gateway alias routes the same class via the gateway.
        p = GroqProvider(_groq_cfg(groq_api_base="http://homelab:4001/v1"))
        assert p.cfg.groq_api_base == "http://homelab:4001/v1"


class TestOwnCostNamespace:
    def test_pricing_reads_groq_namespace_not_openai(self):
        # ADR-147: cost is attributed to `groq`, never leaks openai's rows.
        assert GroqProvider.get_pricing("gpt-4o", "summarization") == {}

    def test_pricing_has_groq_text_row(self):
        row = GroqProvider.get_pricing(_REASONING_MODEL, "summarization")
        assert row, "expected a groq text pricing row for the bakeoff model"


class TestWarnNotRaiseBearer:
    def test_dummy_when_unset(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        p = GroqProvider(_groq_cfg(groq_api_key=None))
        assert p._resolve_api_key(p.cfg) == "EMPTY"

    def test_reads_env(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk_from_env")
        p = GroqProvider(_groq_cfg(groq_api_key=None))
        assert p._resolve_api_key(p.cfg) == "gsk_from_env"

    def test_reads_custom_env_name(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.setenv("ALT_GROQ_KEY", "gsk_alt")
        p = GroqProvider(_groq_cfg(groq_api_key=None, groq_api_key_env="ALT_GROQ_KEY"))
        assert p._resolve_api_key(p.cfg) == "gsk_alt"

    def test_explicit_key_wins(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk_from_env")
        p = GroqProvider(_groq_cfg(groq_api_key="gsk_explicit"))
        assert p._resolve_api_key(p.cfg) == "gsk_explicit"

    def test_authenticate_never_raises_without_key(self, monkeypatch):
        # Unlike OpenAIProvider/DeepSeek, a missing bearer must WARN, not fail construction.
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        GroqProvider(_groq_cfg(groq_api_key=None))  # must not raise

    def test_no_sk_prefix_assumption(self):
        # Groq issues gsk_... ids, not sk-; construction with a gsk_ key must succeed.
        GroqProvider(_groq_cfg(groq_api_key="gsk_test-key"))


class TestOpenModelHeuristics:
    def test_token_kwarg_always_max_tokens_non_reasoning(self):
        p = GroqProvider(_groq_cfg())
        assert p._token_kwarg(512) == {"max_tokens": 512}

    def test_temperature_not_fixed(self):
        p = GroqProvider(_groq_cfg())
        assert p._temp_fixed_at_default == set()

    def test_cleaning_cap_is_groq_8192(self):
        assert GroqProvider._CLEANING_MAX_TOKENS_CAP == 8192

    def test_cleaning_defaults_to_summary_model(self):
        p = GroqProvider(_groq_cfg())
        assert p.cleaning_model == _MODEL

    def test_cleaning_model_explicit_pin_respected(self):
        p = GroqProvider(_groq_cfg(groq_cleaning_model="openai/gpt-oss-20b"))
        assert p.cleaning_model == "openai/gpt-oss-20b"


class TestReasoningGuard:
    def test_model_reasons_markers(self):
        assert _model_reasons("qwen/qwen3.6-27b")
        assert _model_reasons("openai/gpt-oss-120b")
        assert _model_reasons("deepseek-r1-distill-llama-70b")
        assert _model_reasons("compound-beta")
        assert not _model_reasons("llama-3.3-70b-versatile")

    def test_reasoning_model_gets_token_headroom(self):
        p = GroqProvider(_groq_cfg(groq_summary_model=_REASONING_MODEL))
        # 512 + 2048 headroom, capped at 8192.
        assert p._token_kwarg(512) == {"max_tokens": 2560}
        assert p._token_kwarg(8000) == {"max_tokens": 8192}  # capped

    def test_thinking_left_on_warns_flag(self):
        # A reasoning model with no disable directive is flagged thinking-on (loud init warning).
        p = GroqProvider(_groq_cfg(groq_summary_model=_REASONING_MODEL))
        assert p._thinking_left_on() is True

    def test_thinking_disabled_via_extra_body(self):
        p = GroqProvider(
            _groq_cfg(
                groq_summary_model=_REASONING_MODEL,
                groq_extra_body={"reasoning_effort": "none"},
            )
        )
        assert p._thinking_left_on() is False

    def test_non_reasoning_model_never_thinking_on(self):
        p = GroqProvider(_groq_cfg())  # llama-3.3, not a reasoning model
        assert p._thinking_left_on() is False

    def test_extra_body_disable_shapes(self):
        assert _extra_body_disables_thinking({"reasoning_effort": "none"})
        assert _extra_body_disables_thinking({"thinking": {"type": "disabled"}})
        assert _extra_body_disables_thinking({"reasoning": {"enabled": False}})
        assert _extra_body_disables_thinking({"enable_thinking": False})
        assert not _extra_body_disables_thinking({"reasoning_effort": "high"})
        assert not _extra_body_disables_thinking(None)


class TestFactoryDispatch:
    def test_summarization_factory_returns_groq(self):
        from podcast_scraper.summarization.factory import create_summarization_provider

        p = create_summarization_provider(_groq_cfg())
        assert isinstance(p, GroqProvider)

    def test_speaker_factory_returns_groq(self):
        from podcast_scraper.speaker_detectors.factory import create_speaker_detector

        p = create_speaker_detector(_groq_cfg())
        assert isinstance(p, GroqProvider)


class TestDualUseTranscription:
    """Groq is the only LLM sibling that is ALSO a transcription provider (inherits transcribe).

    Do not confuse with grok (xAI), which is chat-only.
    """

    def _transcription_cfg(self, **overrides: Any) -> Config:
        base: Dict[str, Any] = dict(
            rss_url="https://example.com/feed.xml",
            transcription_provider="groq",
            groq_api_key="gsk_test-key",
        )
        base.update(overrides)
        return Config(**base)

    def test_transcription_factory_returns_groq(self):
        from podcast_scraper.transcription.factory import create_transcription_provider

        p = create_transcription_provider(self._transcription_cfg())
        assert isinstance(p, GroqProvider)

    def test_default_transcription_model_is_whisper_turbo(self):
        from podcast_scraper.transcription.factory import create_transcription_provider

        p = create_transcription_provider(self._transcription_cfg())
        assert p.transcription_model == "whisper-large-v3-turbo"

    def test_transcription_model_override(self):
        from podcast_scraper.transcription.factory import create_transcription_provider

        cfg = self._transcription_cfg(groq_transcription_model="whisper-large-v3")
        p = create_transcription_provider(cfg)
        assert p.transcription_model == "whisper-large-v3"

    def test_transcription_pricing_row_exists(self):
        row = GroqProvider.get_pricing("whisper-large-v3-turbo", "transcription")
        assert row, "expected a groq transcription pricing row for whisper-large-v3-turbo"

    def test_config_accepts_groq_transcription_provider(self):
        cfg = self._transcription_cfg()
        assert cfg.transcription_provider == "groq"


class TestOpenAIProviderUnaffected:
    def test_openai_still_requires_sk_key(self, monkeypatch):
        # The base auth is unchanged for the OpenAI-native sibling: a missing key still raises.
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OpenAI API key required"):
            OpenAIProvider(
                Config(rss_url="https://example.com/feed.xml", summary_provider="openai")
            )
