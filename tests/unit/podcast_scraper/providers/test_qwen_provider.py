"""Unit tests for the first-class Qwen provider (ADR-147).

QwenProvider is a thin SIBLING of OpenAIProvider/VLLMProvider/LiteLLMProvider/DeepSeekProvider over
the shared :class:`OpenAICompatibleProvider` transport base. These tests cover ONLY what is
Qwen-specific — identity/namespace, its OWN ``qwen`` cost/telemetry namespace, the absence of a
vendor default endpoint (DashScope out of scope), optional-bearer auth, open-model heuristics, the
fail-closed served-model check, and factory dispatch. The inherited stage methods
(summary/cleaning/GI/KG/grounding) are covered by the base's own tests; re-testing them here would
just duplicate the base. No network.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from podcast_scraper.config import Config
from podcast_scraper.providers.openai.openai_provider import (
    OpenAICompatibleProvider,
    OpenAIProvider,
)
from podcast_scraper.providers.qwen import QwenProvider

_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"


def _qwen_cfg(**overrides: Any) -> Config:
    base: Dict[str, Any] = dict(
        rss_url="https://example.com/feed.xml",
        summary_provider="qwen",
        speaker_detector_provider="qwen",
        generate_summaries=True,
        generate_metadata=True,
        qwen_api_base="http://dgx:8003/v1",
        qwen_summary_model=_MODEL,
        qwen_speaker_model=_MODEL,
    )
    base.update(overrides)
    return Config(**base)


class TestIdentity:
    def test_is_sibling_not_openai_subclass(self):
        # Shares the transport base, but is NOT an OpenAIProvider (ADR-147).
        assert issubclass(QwenProvider, OpenAICompatibleProvider)
        assert not issubclass(QwenProvider, OpenAIProvider)
        assert not issubclass(OpenAIProvider, QwenProvider)

    def test_namespace_and_telemetry(self):
        p = QwenProvider(_qwen_cfg())
        assert p._CONFIG_NS == "qwen"
        assert p._TELEMETRY_PROVIDER == "qwen"
        assert p._PROVIDER_LABEL == "Qwen"
        assert p.get_capabilities().provider_name == "qwen"

    def test_names_real_model_id_on_the_wire(self):
        # No served-name alias: the wire model is the real Qwen id (reproducibility, ADR-143/144).
        p = QwenProvider(_qwen_cfg())
        assert p.summary_model == _MODEL
        assert p.cfg.qwen_api_base == "http://dgx:8003/v1"


class TestNoVendorDefaultEndpoint:
    def test_no_default_api_base(self):
        # Qwen has no single vendor endpoint we commit to (DashScope is explicitly out of scope):
        # the profile always names qwen_api_base, exactly like vllm.
        assert QwenProvider._DEFAULT_API_BASE is None


class TestOwnCostNamespace:
    def test_pricing_reads_qwen_namespace_not_openai(self):
        # ADR-147: cost is attributed to `qwen`, never leaks openai's rows. A known OpenAI model
        # name has no row under the qwen provider namespace.
        assert QwenProvider.get_pricing("gpt-4o", "summarization") == {}


class TestOptionalBearer:
    def test_dummy_when_unset(self, monkeypatch):
        monkeypatch.delenv("QWEN_API_KEY", raising=False)
        p = QwenProvider(_qwen_cfg())
        assert p._resolve_api_key(p.cfg) == "EMPTY"

    def test_reads_env(self, monkeypatch):
        monkeypatch.setenv("QWEN_API_KEY", "tok-from-env")
        p = QwenProvider(_qwen_cfg())
        assert p._resolve_api_key(p.cfg) == "tok-from-env"

    def test_reads_custom_env_name(self, monkeypatch):
        # A DeepInfra-hosted Qwen sets qwen_api_key_env=DEEPINFRA_API_KEY.
        monkeypatch.delenv("QWEN_API_KEY", raising=False)
        monkeypatch.setenv("DEEPINFRA_API_KEY", "di-tok")
        p = QwenProvider(_qwen_cfg(qwen_api_key_env="DEEPINFRA_API_KEY"))
        assert p._resolve_api_key(p.cfg) == "di-tok"

    def test_explicit_key_wins(self, monkeypatch):
        monkeypatch.setenv("QWEN_API_KEY", "tok-from-env")
        p = QwenProvider(_qwen_cfg(qwen_api_key="explicit-tok"))
        assert p._resolve_api_key(p.cfg) == "explicit-tok"

    def test_authenticate_never_raises_without_key(self, monkeypatch):
        # Unlike OpenAIProvider, a missing/blank bearer must not fail construction (local vLLM).
        monkeypatch.delenv("QWEN_API_KEY", raising=False)
        QwenProvider(_qwen_cfg())  # must not raise


class TestOpenModelHeuristics:
    def test_token_kwarg_always_max_tokens(self):
        p = QwenProvider(_qwen_cfg())
        # Even a model whose NAME looks like an OpenAI reasoning model must not get the rename.
        assert p._token_kwarg(512, model="gpt-5-lookalike") == {"max_tokens": 512}
        assert p._token_kwarg(256) == {"max_tokens": 256}

    def test_temperature_not_fixed(self):
        p = QwenProvider(_qwen_cfg())
        assert p._temp_fixed_at_default == set()

    def test_cleaning_cap_is_qwen_8192(self):
        assert QwenProvider._CLEANING_MAX_TOKENS_CAP == 8192

    def test_cleaning_defaults_to_summary_model(self):
        p = QwenProvider(_qwen_cfg())
        assert p.cleaning_model == _MODEL

    def test_cleaning_model_explicit_pin_respected(self):
        p = QwenProvider(_qwen_cfg(qwen_cleaning_model="Qwen/other-model"))
        assert p.cleaning_model == "Qwen/other-model"


class TestOpenAIProviderUnaffected:
    def test_openai_still_requires_sk_key(self, monkeypatch):
        # The base auth is unchanged for the OpenAI-native sibling.
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OpenAI API key required"):
            OpenAIProvider(
                Config(rss_url="https://example.com/feed.xml", summary_provider="openai")
            )


class TestFactoryDispatch:
    def test_summarization_factory_returns_qwen(self):
        from podcast_scraper.summarization.factory import create_summarization_provider

        p = create_summarization_provider(_qwen_cfg())
        assert isinstance(p, QwenProvider)

    def test_speaker_factory_returns_qwen(self):
        from podcast_scraper.speaker_detectors.factory import create_speaker_detector

        p = create_speaker_detector(_qwen_cfg())
        assert isinstance(p, QwenProvider)


class TestServedModelVerification:
    """ADR-147 B3: fail-closed served-model check.

    A wrong model behind the endpoint stops the run; an unreachable endpoint only warns.
    """

    @staticmethod
    def _fake_urlopen(ids):
        payload = __import__("json").dumps({"data": [{"id": i} for i in ids]}).encode()

        class _Resp:
            def read(self_inner):
                return payload

            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *a):
                return False

        def _open(req, timeout=10):
            return _Resp()

        return _open

    def test_match_passes(self, monkeypatch):
        import podcast_scraper.providers.qwen.qwen_provider as m

        monkeypatch.setattr(m.urllib.request, "urlopen", self._fake_urlopen([_MODEL]))
        QwenProvider(_qwen_cfg())._verify_served_model()  # must not raise

    def test_dated_suffix_tolerated(self, monkeypatch):
        import podcast_scraper.providers.qwen.qwen_provider as m

        monkeypatch.setattr(m.urllib.request, "urlopen", self._fake_urlopen([_MODEL + "-20260101"]))
        QwenProvider(_qwen_cfg())._verify_served_model()  # startswith tolerance

    def test_mismatch_raises(self, monkeypatch):
        import podcast_scraper.providers.qwen.qwen_provider as m
        from podcast_scraper.providers.qwen.qwen_provider import QwenServedModelMismatch

        monkeypatch.setattr(m.urllib.request, "urlopen", self._fake_urlopen(["someoneelse/other"]))
        with pytest.raises(QwenServedModelMismatch):
            QwenProvider(_qwen_cfg())._verify_served_model()

    def test_unreachable_warns_not_raises(self, monkeypatch):
        import podcast_scraper.providers.qwen.qwen_provider as m

        def _boom(req, timeout=10):
            raise OSError("connection refused")

        monkeypatch.setattr(m.urllib.request, "urlopen", _boom)
        QwenProvider(_qwen_cfg())._verify_served_model()  # must NOT raise

    def test_initialize_gated_by_flag(self, monkeypatch):
        # Flag off -> initialize() must not run the served-model check; on -> it must.
        monkeypatch.setattr(OpenAICompatibleProvider, "initialize", lambda self: None)
        for flag, expected in ((False, 0), (True, 1)):
            called = {"n": 0}
            p = QwenProvider(_qwen_cfg(qwen_verify_served_model=flag))
            monkeypatch.setattr(p, "_verify_served_model", lambda: called.__setitem__("n", 1))
            p.initialize()
            assert called["n"] == expected
