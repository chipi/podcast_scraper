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

    def test_dgx_tailnet_host_env_var_expands_in_profile_yaml(
        self, _fake_keys: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When DGX_TAILNET_HOST is set, the profile's
        ``${DGX_TAILNET_HOST:-…}`` template substitutes to the env value."""
        monkeypatch.setenv("DGX_TAILNET_HOST", "my-real-dgx.tailnet.ts.net")
        cfg = Config.model_validate({"profile": "cloud_with_dgx_primary"})
        assert cfg.dgx_tailnet_host == "my-real-dgx.tailnet.ts.net"

    def test_dgx_tailnet_host_default_when_env_unset(
        self, _fake_keys: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When DGX_TAILNET_HOST is unset, the YAML's ``:-`` default kicks in — and that
        default must be a REACHABLE host, not a placeholder.

        It used to be ``your-dgx.tailnet.ts.net``. That satisfied the "is it set" validator,
        so a DGX-pinned run started, failed its health probe against a host that does not
        exist, and degraded to cloud ASR while every signal still reported the DGX profile.
        A placeholder that passes validation is worse than no default at all, so the default
        is now the real MagicDNS name and Config rejects the example value outright.
        """
        from podcast_scraper.providers.ml.model_registry import DEFAULT_DGX_TAILNET_HOST

        monkeypatch.delenv("DGX_TAILNET_HOST", raising=False)
        cfg = Config.model_validate({"profile": "cloud_with_dgx_primary"})
        assert cfg.dgx_tailnet_host == DEFAULT_DGX_TAILNET_HOST
        assert "your-dgx" not in str(
            cfg.dgx_tailnet_host
        ), "the shipped default is a placeholder again — a DGX pin would silently fall back"
