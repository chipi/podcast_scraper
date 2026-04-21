"""Same-class regression tests for Literal-with-incomplete-dispatch bugs
(post-#646 audit).

The #646 audit found several places where a Config Literal enumerated N
values but the dispatch covered fewer. Tests here lock in the full
coverage so new Literal values don't silently ship without wiring.
"""

from __future__ import annotations

import pytest

from podcast_scraper import config as cfg_mod


@pytest.fixture
def _fake_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "DEEPSEEK_API_KEY",
        "MISTRAL_API_KEY",
        "GROK_API_KEY",
    ):
        monkeypatch.setenv(name, "test-" + name.lower().replace("_", "-") + "-dummy-key")


class TestTranscriptionProviderInfoCompleteness:
    """_build_transcription_provider_info must record a model for every
    Literal value in ``transcription_provider``."""

    @pytest.mark.parametrize(
        "provider,expected_model_key",
        [
            ("whisper", "whisper_model"),
            ("openai", "openai_model"),
            ("gemini", "gemini_model"),
            ("mistral", "mistral_model"),
        ],
    )
    def test_every_literal_value_records_its_model(
        self, _fake_keys: None, provider: str, expected_model_key: str
    ) -> None:
        from podcast_scraper.workflow.metadata_generation import (
            _build_transcription_provider_info,
        )

        cfg = cfg_mod.Config.model_validate(
            {
                "rss_url": "https://example.com/feed.xml",
                "transcription_provider": provider,
                "transcribe_missing": True,
            }
        )
        info = _build_transcription_provider_info(cfg)
        assert info is not None
        assert (
            expected_model_key in info
        ), f"{provider}: expected {expected_model_key!r} in {list(info.keys())}"
        assert info[expected_model_key], f"{provider}: {expected_model_key} should be non-empty"


class TestMLRunSuffixCompleteness:
    """_build_provider_model_suffix must produce a non-empty prefix for every
    Literal value in both transcription_provider and summary_provider so
    run directories don't collide across providers."""

    @pytest.mark.parametrize("provider", ["whisper", "openai", "gemini", "mistral"])
    def test_every_transcription_literal_gets_a_suffix(
        self, _fake_keys: None, provider: str
    ) -> None:
        from podcast_scraper.utils.filesystem import _build_provider_model_suffix

        cfg = cfg_mod.Config.model_validate(
            {
                "rss_url": "https://example.com/feed.xml",
                "transcription_provider": provider,
                "transcribe_missing": True,
                "generate_summaries": False,
            }
        )
        suffix = _build_provider_model_suffix(cfg)
        assert suffix, f"{provider}: empty suffix"
        # Each provider should contribute a distinct prefix.
        expected_prefix = {
            "whisper": "w_",
            "openai": "oa_",
            "gemini": "gm_",
            "mistral": "ms_",
        }[provider]
        assert expected_prefix in suffix, f"{provider}: expected {expected_prefix!r} in {suffix!r}"

    @pytest.mark.parametrize(
        "provider",
        ["openai", "gemini", "anthropic", "deepseek", "mistral", "grok", "ollama"],
    )
    def test_every_cloud_summary_literal_gets_a_suffix(
        self, _fake_keys: None, provider: str
    ) -> None:
        from podcast_scraper.utils.filesystem import _build_provider_model_suffix

        cfg = cfg_mod.Config.model_validate(
            {
                "rss_url": "https://example.com/feed.xml",
                "generate_summaries": True,
                "summary_provider": provider,
                "transcribe_missing": False,
            }
        )
        suffix = _build_provider_model_suffix(cfg)
        assert suffix, f"{provider}: empty suffix"
        expected_prefix = {
            "openai": "oa_",
            "gemini": "gm_",
            "anthropic": "an_",
            "deepseek": "ds_",
            "mistral": "ms_",
            "grok": "gk_",
            "ollama": "ol_",
        }[provider]
        assert expected_prefix in suffix, f"{provider}: expected {expected_prefix!r} in {suffix!r}"

    def test_suffixes_distinguish_providers_on_same_model_name(self, _fake_keys: None) -> None:
        """Two different providers must not produce identical run-dir suffixes
        when the model name is similar (prior bug: empty suffix on both →
        collision)."""
        from podcast_scraper.utils.filesystem import _build_provider_model_suffix

        cfg_gemini = cfg_mod.Config.model_validate(
            {
                "rss_url": "https://example.com/feed.xml",
                "generate_summaries": True,
                "summary_provider": "gemini",
            }
        )
        cfg_anthropic = cfg_mod.Config.model_validate(
            {
                "rss_url": "https://example.com/feed.xml",
                "generate_summaries": True,
                "summary_provider": "anthropic",
            }
        )
        assert _build_provider_model_suffix(cfg_gemini) != _build_provider_model_suffix(
            cfg_anthropic
        )
