"""``${VAR}`` / ``${VAR:-default}`` expansion in YAML configs (#907 follow-up).

Operator-facing YAMLs use this syntax to keep hostnames / API keys / secrets
out of git. The expansion runs at YAML-load time inside ``Config``'s profile
loader and inside ``_load_config_file`` so every YAML entry point gets the
same substitution semantics.
"""

from __future__ import annotations

import pytest

from podcast_scraper.config import (
    _expand_env_in_string,
    _expand_env_vars,
    Config,
)


class TestExpandEnvInString:
    def test_set_var_substitutes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FOO", "bar")
        assert _expand_env_in_string("prefix-${FOO}-suffix") == "prefix-bar-suffix"

    def test_unset_var_with_default_uses_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MISSING_VAR", raising=False)
        assert _expand_env_in_string("${MISSING_VAR:-fallback}") == "fallback"

    def test_set_var_overrides_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOST", "real-host.example")
        result = _expand_env_in_string("${HOST:-default-host.example}")
        assert result == "real-host.example"

    def test_empty_var_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Bash-style ``:-`` treats empty same as unset; matches operator
        intuition (an env file with ``HOST=`` should fall back, not produce
        an empty hostname)."""
        monkeypatch.setenv("HOST", "")
        assert _expand_env_in_string("${HOST:-fallback}") == "fallback"

    def test_unset_var_no_default_keeps_literal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unset variable with no default leaves the literal in place so
        downstream failures surface loudly rather than collapsing to ``""``."""
        monkeypatch.delenv("UNSET_VAR", raising=False)
        assert _expand_env_in_string("${UNSET_VAR}") == "${UNSET_VAR}"

    def test_multiple_vars_per_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("USER", "marko")
        monkeypatch.setenv("DOMAIN", "example.com")
        assert _expand_env_in_string("${USER}@${DOMAIN}") == "marko@example.com"

    def test_non_var_string_passes_through(self) -> None:
        assert _expand_env_in_string("plain text") == "plain text"
        assert _expand_env_in_string("path/with/$dollars/but/no/braces") == (
            "path/with/$dollars/but/no/braces"
        )


class TestExpandEnvVarsRecursive:
    def test_nested_dict_and_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOST", "h.example")
        monkeypatch.setenv("PORT", "8002")
        data = {
            "endpoint": "http://${HOST}:${PORT}/v1",
            "fallbacks": ["${HOST}", "static-value"],
            "nested": {"inner": "${HOST:-fallback}"},
            "untouched_int": 42,
            "untouched_bool": True,
        }
        expanded = _expand_env_vars(data)
        assert expanded == {
            "endpoint": "http://h.example:8002/v1",
            "fallbacks": ["h.example", "static-value"],
            "nested": {"inner": "h.example"},
            "untouched_int": 42,
            "untouched_bool": True,
        }
        # Original not mutated.
        assert data["endpoint"] == "http://${HOST}:${PORT}/v1"


class TestEnvVarExpansionInConfigLoad:
    @pytest.fixture
    def _fake_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # DEEPGRAM: cloud_with_dgx_primary's ASR fallback moved openai -> deepgram
        # (2026-08-28), and the LLM stages moved gemini -> the litellm gateway.
        for name in ("GEMINI_API_KEY", "OPENAI_API_KEY", "DEEPGRAM_API_KEY"):
            monkeypatch.setenv(name, "test-dummy")

    def test_both_host_vars_set_expand_in_profile_yaml(
        self, _fake_keys: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With both host variables set, the profile's ``${...}`` templates expand."""
        monkeypatch.setenv("DGX_TAILNET_HOST", "dgx.test")
        monkeypatch.setenv("LITELLM_API_BASE", "http://gateway.test:4001/v1")
        cfg = Config.model_validate({"profile": "cloud_with_dgx_primary"})
        assert cfg.dgx_tailnet_host == "dgx.test"
        assert cfg.litellm_api_base == "http://gateway.test:4001/v1"

    def test_unset_dgx_tailnet_host_fails_loudly(
        self, _fake_keys: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No built-in default: a placeholder that passes validation once let a DGX-pinned run
        start, fail its health probe and degrade to cloud ASR while reporting the DGX profile.
        An unset ``DGX_TAILNET_HOST`` must stop at Config validation, naming the variable."""
        monkeypatch.delenv("DGX_TAILNET_HOST", raising=False)
        monkeypatch.setenv("LITELLM_API_BASE", "http://gateway.test:4001/v1")
        with pytest.raises(ValueError, match=r"DGX_TAILNET_HOST.*is not set.*\.env"):
            Config.model_validate({"profile": "cloud_with_dgx_primary"})

    def test_empty_dgx_tailnet_host_fails_loudly(
        self, _fake_keys: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DGX_TAILNET_HOST", "")
        monkeypatch.setenv("LITELLM_API_BASE", "http://gateway.test:4001/v1")
        with pytest.raises(ValueError, match="DGX_TAILNET_HOST"):
            Config.model_validate({"profile": "cloud_with_dgx_primary"})

    def test_unset_litellm_api_base_fails_loudly(
        self, _fake_keys: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same contract for a gateway base carried by an ``*_api_base`` field."""
        monkeypatch.setenv("DGX_TAILNET_HOST", "dgx.test")
        monkeypatch.delenv("LITELLM_API_BASE", raising=False)
        with pytest.raises(ValueError, match=r"LITELLM_API_BASE.*is not set.*\.env"):
            Config.model_validate({"profile": "cloud_with_dgx_primary"})

    def test_unset_var_in_api_base_field_names_the_field(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("LITELLM_API_BASE", raising=False)
        with pytest.raises(ValueError, match=r"vllm_api_base.*LITELLM_API_BASE"):
            Config.model_validate({"vllm_api_base": "${LITELLM_API_BASE}"})

    def test_none_and_empty_host_fields_do_not_raise(self) -> None:
        """Only an unexpanded ``${...}`` is an error; a field with no value is legitimate."""
        cfg = Config.model_validate({"litellm_api_base": "", "dgx_tailnet_host": None})
        assert cfg.dgx_tailnet_host is None
        assert not cfg.litellm_api_base
