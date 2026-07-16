"""Continuous validation that every shipped profile builds a valid Config, plus OpenAI base/key
routing precedence.

The class of bug this guards: a profile carried ``openai_base_url`` (not a Config field — the field
is ``openai_api_base``), and because Config is ``extra="forbid"`` those three DGX profiles could not
construct at all. The eval harness reads the YAML raw, so it never noticed; the failure hid for
months (#1032) until every unit skip was removed (#1173). A parametrised "does it construct" guard
means any future mistyped/removed key trips a unit test immediately, for every profile.
"""

from __future__ import annotations

import pathlib

import pytest

from podcast_scraper.config import Config

PROFILE_DIR = pathlib.Path("config/profiles")
_ALL_PROFILES = sorted(p.stem for p in PROFILE_DIR.glob("*.yaml")) if PROFILE_DIR.is_dir() else []

pytestmark = pytest.mark.unit


@pytest.fixture
def _dummy_provider_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dummy (never real) keys so every provider-key validator passes — this checks construction
    and routing, not live auth."""
    for var in (
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "ANTHROPIC_API_KEY",
        "DEEPSEEK_API_KEY",
        "GROK_API_KEY",
        "XAI_API_KEY",
        "MISTRAL_API_KEY",
        "DEEPGRAM_API_KEY",
        "VLLM_API_KEY",
    ):
        monkeypatch.setenv(var, "test-key")


@pytest.mark.parametrize("profile", _ALL_PROFILES)
def test_every_shipped_profile_constructs(
    profile: str, _dummy_provider_keys: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every shipped profile must build a valid Config: no mistyped/non-field keys (extra=forbid),
    no unmet prerequisites. generate_metadata satisfies the generate_gi prerequisite so even the
    no-LLM reprocess profile is exercised rather than skipped."""
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    Config.model_validate({"profile": profile, "generate_gi": True, "generate_metadata": True})


class TestOpenAIRoutingPrecedence:
    """openai_api_base / openai_api_key precedence — guards the DGX routing footgun."""

    def test_profile_base_url_wins_over_ambient_env(
        self, _dummy_provider_keys: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A profile's explicit openai_api_base must beat a stale OPENAI_API_BASE env var — else a
        DGX profile silently routes to real OpenAI."""
        monkeypatch.setenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        cfg = Config.model_validate(
            {"profile": "prod_dgx_balanced", "generate_gi": True, "generate_metadata": True}
        )
        assert "dgx" in (cfg.openai_api_base or ""), cfg.openai_api_base

    def test_env_base_url_is_fallback_when_unset(
        self, _dummy_provider_keys: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When nothing sets openai_api_base, OPENAI_API_BASE still fills it (e2e mock servers)."""
        monkeypatch.setenv("OPENAI_API_BASE", "http://mock:9999/v1")
        cfg = Config.model_validate(
            {"rss_url": "https://example.com/f.xml", "summary_provider": "openai"}
        )
        assert cfg.openai_api_base == "http://mock:9999/v1"

    def test_openai_api_key_env_resolves_named_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """openai_api_key_env names the env var to read the key from (VLLM_API_KEY for a DGX slot),
        without hardcoding a secret in the profile."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("VLLM_API_KEY", "vllm-secret")
        cfg = Config.model_validate(
            {
                "rss_url": "https://example.com/f.xml",
                "summary_provider": "openai",
                "openai_api_key_env": "VLLM_API_KEY",
            }
        )
        assert cfg.openai_api_key == "vllm-secret"

    def test_explicit_key_beats_key_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A directly provided openai_api_key wins over openai_api_key_env."""
        monkeypatch.setenv("VLLM_API_KEY", "from-env")
        cfg = Config.model_validate(
            {
                "rss_url": "https://example.com/f.xml",
                "summary_provider": "openai",
                "openai_api_key": "direct",
                "openai_api_key_env": "VLLM_API_KEY",
            }
        )
        assert cfg.openai_api_key == "direct"
