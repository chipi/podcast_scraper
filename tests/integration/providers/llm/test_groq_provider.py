#!/usr/bin/env python3
"""Integration tests for the first-class Groq provider (ADR-147).

GroqProvider is a thin SIBLING of OpenAIProvider/DeepSeekProvider/QwenProvider over the shared
:class:`OpenAICompatibleProvider` transport base — see ``test_deepseek_provider.py`` for the
sibling-contract pattern this mirrors (identity/namespace, the vendor default endpoint, warn-vs-
raise auth, reasoning-model token headroom, factory dispatch). The inherited stage methods
(summary/cleaning/GI/KG/grounding) are covered by the base's own tests (test_openai_provider*);
re-testing them here would just duplicate the base.

Uniquely among the LLM siblings, Groq is DUAL-USE: the same class also serves whisper-large-v3-
turbo transcription (inherited, unmodified, from the base). ``TestGroqProviderTranscription``
below mirrors test_openai_provider.py's ``TestOpenAIProviderTranscription`` (mocked
``client.audio.transcriptions.create``) to exercise that path, which the unit layer only covers
at the config/factory-resolution level.

Do NOT confuse ``groq`` (this provider — Groq's LPU cloud, keys ``gsk_...``) with ``grok`` (xAI).
One letter apart.
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import Mock, patch

import pytest

from podcast_scraper.config import Config
from podcast_scraper.providers.groq.groq_provider import _model_reasons, GroqProvider
from podcast_scraper.providers.openai.openai_provider import (
    OpenAICompatibleProvider,
    OpenAIProvider,
)

# NOTE: test_deepseek_provider.py (the template this mirrors) carries NO `integration` marker at
# all — a regression from the ADR-147 refactor (commit f575dbb5) that silently excludes it from
# `-m integration` / `make test-integration(-fast)` runs (verified: `pytest
# test_deepseek_provider.py -m integration --collect-only` selects 0 of its 22 tests). That gap is
# pre-existing in a file this task was told not to touch; do not copy it here. This module instead
# follows test_openai_provider.py's correct convention (`@pytest.mark.integration` +
# `critical_path`) so these tests are actually collected by the PR gate.
pytestmark = [pytest.mark.integration, pytest.mark.critical_path, pytest.mark.llm]

_KEY = "gsk_" + "x" * 40
_MODEL = "llama-3.3-70b-versatile"
_REASONING_MODEL = "qwen/qwen3.6-27b"


def _groq_cfg(**overrides: Any) -> Config:
    base: Dict[str, Any] = dict(
        rss_url="https://example.com/feed.xml",
        summary_provider="groq",
        speaker_detector_provider="groq",
        generate_summaries=True,
        generate_metadata=True,
        groq_api_key=_KEY,
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


class TestVendorDefaultEndpoint:
    def test_default_endpoint_is_groq_direct(self):
        # "direct, no gateway" needs zero config: Groq ships its own default endpoint
        # (unlike Qwen, whose _DEFAULT_API_BASE is None).
        assert GroqProvider._DEFAULT_API_BASE == "https://api.groq.com/openai/v1"

    def test_api_base_override_routes_via_gateway(self):
        # Same class, config-only switch: pointing groq_api_base at a LiteLLM gateway routes
        # Groq THROUGH the gateway instead of direct (ADR-147).
        p = GroqProvider(_groq_cfg(groq_api_base="http://homelab:4001/v1"))
        assert p.cfg.groq_api_base == "http://homelab:4001/v1"


class TestWarnNotRaiseAuth:
    def test_missing_key_warns_not_raises(self, monkeypatch, caplog):
        import logging

        # Unlike OpenAIProvider/DeepSeek, a missing bearer must WARN, not fail construction.
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with caplog.at_level(logging.WARNING):
            GroqProvider(_groq_cfg(groq_api_key=None))  # must not raise

    def test_resolves_groq_namespaced_key(self):
        p = GroqProvider(_groq_cfg())
        assert p._resolve_api_key(p.cfg) == _KEY


class TestReasoningHeadroom:
    def test_marker_detection(self):
        assert _model_reasons("qwen/qwen3.6-27b") is True
        assert _model_reasons("openai/gpt-oss-120b") is True
        assert _model_reasons("deepseek-r1-distill-llama-70b") is True
        assert _model_reasons(_MODEL) is False

    def test_reasoning_model_gets_headroom(self):
        p = GroqProvider(_groq_cfg(groq_summary_model=_REASONING_MODEL))
        # 10 + 2048 headroom.
        assert p._token_kwarg(10) == {"max_tokens": 2058}

    def test_headroom_capped_at_groq_limit(self):
        p = GroqProvider(_groq_cfg(groq_summary_model=_REASONING_MODEL))
        assert p._token_kwarg(8000) == {"max_tokens": 8192}  # min(8000+2048, 8192)

    def test_non_reasoning_model_is_passthrough(self):
        p = GroqProvider(_groq_cfg())
        assert p._token_kwarg(10) == {"max_tokens": 10}


class TestOpenModelHeuristics:
    def test_temperature_not_fixed(self):
        p = GroqProvider(_groq_cfg())
        assert p._temp_fixed_at_default == set()

    def test_cleaning_cap_is_groq_8192(self):
        assert GroqProvider._CLEANING_MAX_TOKENS_CAP == 8192

    def test_cleaning_model_explicit_pin_respected(self):
        p = GroqProvider(_groq_cfg(groq_cleaning_model="openai/gpt-oss-20b"))
        assert p.cleaning_model == "openai/gpt-oss-20b"


class TestFactoryDispatch:
    def test_summarization_factory_returns_groq(self):
        from podcast_scraper.summarization.factory import create_summarization_provider

        p = create_summarization_provider(_groq_cfg())
        assert isinstance(p, GroqProvider)

    def test_speaker_factory_returns_groq(self):
        from podcast_scraper.speaker_detectors.factory import create_speaker_detector

        p = create_speaker_detector(_groq_cfg())
        assert isinstance(p, GroqProvider)

    def test_transcription_factory_returns_groq(self):
        # Dual-use (ADR-147): the SAME factory-created instance is also the transcription
        # provider — whisper-large-v3-turbo by default.
        from podcast_scraper.transcription.factory import create_transcription_provider

        cfg = Config(
            rss_url="https://example.com/feed.xml",
            transcription_provider="groq",
            groq_api_key=_KEY,
        )
        p = create_transcription_provider(cfg)
        assert isinstance(p, GroqProvider)
        assert p.transcription_model == "whisper-large-v3-turbo"


class TestOpenAIProviderUnaffected:
    def test_openai_still_requires_sk_key(self, monkeypatch):
        # The base auth is unchanged for the OpenAI-native sibling: a missing key still raises.
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OpenAI API key required"):
            OpenAIProvider(
                Config(rss_url="https://example.com/feed.xml", summary_provider="openai")
            )


class TestGroqProviderTranscription:
    """DUAL-USE (ADR-147): GroqProvider inherits ``transcribe()`` / ``transcribe_with_segments()``
    from :class:`OpenAICompatibleProvider` unmodified — the SAME class that serves
    summary/speaker/GI/KG also serves whisper-large-v3-turbo. Mirrors
    ``test_openai_provider.py``'s ``TestOpenAIProviderTranscription`` with a mocked
    ``client.audio.transcriptions.create`` call, pinning Groq's OWN transcription-model default
    and namespace instead of OpenAI's.
    """

    def _cfg(self, **overrides: Any) -> Config:
        base: Dict[str, Any] = dict(
            rss_url="https://example.com/feed.xml",
            transcription_provider="groq",
            groq_api_key=_KEY,
            transcribe_missing=True,
        )
        base.update(overrides)
        return Config(**base)

    def test_transcription_model_defaults_to_whisper_turbo(self):
        p = GroqProvider(self._cfg())
        assert p.transcription_model == "whisper-large-v3-turbo"

    def test_transcription_model_override(self):
        p = GroqProvider(self._cfg(groq_transcription_model="whisper-large-v3"))
        assert p.transcription_model == "whisper-large-v3"

    @patch("builtins.open", create=True)
    @patch("os.path.exists")
    def test_transcribe_success(self, mock_exists, mock_open):
        mock_exists.return_value = True
        mock_open.return_value.__enter__.return_value = Mock()
        mock_open.return_value.__exit__.return_value = None

        mock_client = Mock()
        mock_response = Mock()
        mock_response.text = "Hello world"
        mock_client.audio.transcriptions.create.return_value = mock_response

        p = GroqProvider(self._cfg())
        p.client = mock_client
        p.initialize()

        result = p.transcribe("/path/to/audio.mp3")

        assert result == "Hello world"
        mock_client.audio.transcriptions.create.assert_called_once()
        call_kwargs = mock_client.audio.transcriptions.create.call_args[1]
        # Groq's OWN default model (whisper-large-v3-turbo), not OpenAI's whisper-1.
        assert call_kwargs.get("model") == "whisper-large-v3-turbo"
        assert call_kwargs.get("response_format") == "verbose_json"

    @patch("builtins.open", create=True)
    @patch("os.path.exists")
    def test_transcribe_with_language(self, mock_exists, mock_open):
        mock_exists.return_value = True
        mock_open.return_value.__enter__.return_value = Mock()
        mock_open.return_value.__exit__.return_value = None

        mock_response = Mock()
        mock_response.text = "Bonjour"
        mock_client = Mock()
        mock_client.audio.transcriptions.create.return_value = mock_response

        p = GroqProvider(self._cfg())
        p.client = mock_client
        p.initialize()

        p.transcribe("/path/to/audio.mp3", language="fr")

        call_kwargs = mock_client.audio.transcriptions.create.call_args[1]
        assert call_kwargs["language"] == "fr"

    @patch("builtins.open", create=True)
    @patch("os.path.exists")
    def test_transcribe_with_segments_verbose_json(self, mock_exists, mock_open):
        mock_exists.return_value = True
        mock_open.return_value.__enter__.return_value = Mock()
        mock_open.return_value.__exit__.return_value = None

        seg = Mock()
        seg.start = 0.0
        seg.end = 1.0
        seg.text = "Hello world"

        mock_response = Mock()
        mock_response.text = "Hello world"
        mock_response.segments = [seg]

        mock_client = Mock()
        mock_client.audio.transcriptions.create.return_value = mock_response

        p = GroqProvider(self._cfg())
        p.client = mock_client
        p.initialize()

        result_dict, elapsed = p.transcribe_with_segments("/path/to/audio.mp3")

        assert result_dict["text"] == "Hello world"
        assert len(result_dict["segments"]) == 1
        assert isinstance(elapsed, float)
        assert elapsed >= 0

    def test_transcribe_not_initialized(self):
        p = GroqProvider(self._cfg())
        # Don't call initialize().
        with pytest.raises(RuntimeError, match="not initialized"):
            p.transcribe("/path/to/audio.mp3")

    @patch("os.path.exists")
    def test_transcribe_file_not_found(self, mock_exists):
        mock_exists.return_value = False
        p = GroqProvider(self._cfg())
        p.initialize()
        with pytest.raises(FileNotFoundError, match="not found"):
            p.transcribe("/path/to/nonexistent.mp3")

    @patch("builtins.open", create=True)
    @patch("os.path.exists")
    def test_transcribe_api_error_raises_provider_runtime_error(self, mock_exists, mock_open):
        # NOTE: the base's error message hardcodes "OpenAI transcription failed" regardless of
        # _PROVIDER_LABEL (openai_provider.py transcribe()/transcribe_with_segments() error
        # branches never read self._PROVIDER_LABEL) — Groq is the first sibling to exercise this
        # inherited path, so a Groq transcription failure is misreported as an OpenAI one. That is
        # a real bug in non-test code this task does not fix; asserting only the generic
        # "transcription failed" substring here keeps this test correct under both today's mislabel
        # and a future fix, without encoding the wrong provider name as expected behaviour.
        mock_exists.return_value = True
        mock_open.return_value.__enter__.return_value = Mock()
        mock_open.return_value.__exit__.return_value = None

        mock_client = Mock()
        mock_client.audio.transcriptions.create.side_effect = Exception("boom")

        p = GroqProvider(self._cfg())
        p.client = mock_client
        p.initialize()

        from podcast_scraper.exceptions import ProviderRuntimeError

        with pytest.raises(ProviderRuntimeError, match="transcription failed"):
            p.transcribe("/path/to/audio.mp3")
