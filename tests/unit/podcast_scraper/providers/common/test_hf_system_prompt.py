"""Tests for #109 — load + cache vendor SYSTEM_PROMPT.txt from HF model repo.

Helper lives in :mod:`podcast_scraper.providers.common.hf_system_prompt`.
Provider-side wiring (using the result as the system content on Mistral
family models) is out of scope for these tests — they cover the I/O
helper's behaviour only.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from podcast_scraper.providers.common.hf_system_prompt import (
    _safe_cache_key,
    load_hf_system_prompt,
)


@pytest.fixture()
def tmp_cache(tmp_path: Path) -> Path:
    """Isolated per-test cache dir under tmp_path."""
    return tmp_path / "hf_system_prompts"


@pytest.mark.unit
class TestSafeCacheKey:
    """Path-traversal + slash-normalization on cache filenames."""

    def test_basic_model_id_slashes_become_double_underscore(self) -> None:
        key = _safe_cache_key("mistralai/Mistral-Small-3.2-24B", "SYSTEM_PROMPT.txt")
        assert key == "mistralai__Mistral-Small-3.2-24B__SYSTEM_PROMPT.txt"

    def test_path_traversal_attempt_is_neutralized(self) -> None:
        """A malicious or buggy model_id like ``../etc/passwd`` mustn't
        escape the cache root."""
        key = _safe_cache_key("../etc/passwd", "SYSTEM_PROMPT.txt")
        # No raw ``..`` segments survive.
        assert ".." not in key
        # No path separator.
        assert "/" not in key


@pytest.mark.unit
class TestLoadHfSystemPromptHttpFlow:
    """HTTP fetch happy + sad paths."""

    def test_200_response_returns_content_and_caches(self, tmp_cache: Path) -> None:
        body = "You are Mistral. Be terse."
        mock_resp = MagicMock(status_code=200, text=body)
        with patch(
            "podcast_scraper.providers.common.hf_system_prompt.requests.get",
            return_value=mock_resp,
        ) as get_mock:
            result = load_hf_system_prompt("mistralai/Mistral-Small-3.2", cache_dir=tmp_cache)
        assert result == body
        assert get_mock.call_count == 1
        cached = tmp_cache / _safe_cache_key("mistralai/Mistral-Small-3.2", "SYSTEM_PROMPT.txt")
        assert cached.exists()
        assert cached.read_text(encoding="utf-8") == body

    def test_cache_hit_skips_http(self, tmp_cache: Path) -> None:
        """A second call for the same model reads from cache — no HTTP."""
        body = "cached body"
        cached = tmp_cache / _safe_cache_key("mistralai/Mistral-Small-3.2", "SYSTEM_PROMPT.txt")
        tmp_cache.mkdir(parents=True, exist_ok=True)
        cached.write_text(body, encoding="utf-8")
        with patch("podcast_scraper.providers.common.hf_system_prompt.requests.get") as get_mock:
            result = load_hf_system_prompt("mistralai/Mistral-Small-3.2", cache_dir=tmp_cache)
        assert result == body
        get_mock.assert_not_called()

    def test_404_returns_none_and_writes_sentinel(self, tmp_cache: Path) -> None:
        """A model without SYSTEM_PROMPT.txt returns None and caches a
        sentinel so the next call doesn't refetch."""
        mock_resp = MagicMock(status_code=404, text="")
        with patch(
            "podcast_scraper.providers.common.hf_system_prompt.requests.get",
            return_value=mock_resp,
        ) as get_mock:
            result = load_hf_system_prompt("Qwen/Qwen3-30B-A3B-Instruct-2507", cache_dir=tmp_cache)
        assert result is None
        assert get_mock.call_count == 1
        sentinel = tmp_cache / "Qwen__Qwen3-30B-A3B-Instruct-2507____NONE__.sentinel"
        assert sentinel.exists()

    def test_sentinel_short_circuits_subsequent_calls(self, tmp_cache: Path) -> None:
        """Once a model is sentineled as 'no system prompt', no HTTP fetch
        happens on subsequent calls."""
        tmp_cache.mkdir(parents=True, exist_ok=True)
        sentinel = tmp_cache / "Qwen__Qwen3____NONE__.sentinel"
        sentinel.write_text("", encoding="utf-8")
        with patch("podcast_scraper.providers.common.hf_system_prompt.requests.get") as get_mock:
            result = load_hf_system_prompt("Qwen/Qwen3", cache_dir=tmp_cache)
        assert result is None
        get_mock.assert_not_called()

    def test_network_error_returns_none_without_sentinel(self, tmp_cache: Path) -> None:
        """A transient network error should NOT poison the cache with a
        sentinel — the next call must retry."""
        import requests

        with patch(
            "podcast_scraper.providers.common.hf_system_prompt.requests.get",
            side_effect=requests.ConnectionError("simulated"),
        ):
            result = load_hf_system_prompt("mistralai/Mistral-Small-3.2", cache_dir=tmp_cache)
        assert result is None
        sentinel = tmp_cache / "mistralai__Mistral-Small-3.2____NONE__.sentinel"
        assert not sentinel.exists()

    def test_unexpected_status_code_returns_none(self, tmp_cache: Path) -> None:
        """A 5xx or any non-200/404 status returns None and does NOT cache
        a sentinel (the issue might be transient on HF's side)."""
        mock_resp = MagicMock(status_code=503, text="Service Unavailable")
        with patch(
            "podcast_scraper.providers.common.hf_system_prompt.requests.get",
            return_value=mock_resp,
        ):
            result = load_hf_system_prompt("mistralai/Mistral-Small-3.2", cache_dir=tmp_cache)
        assert result is None
        sentinel = tmp_cache / "mistralai__Mistral-Small-3.2____NONE__.sentinel"
        assert not sentinel.exists()


@pytest.mark.unit
class TestLoadHfSystemPromptAuth:
    """HF_TOKEN handling for gated repos."""

    def test_explicit_hf_token_used_as_bearer(self, tmp_cache: Path) -> None:
        mock_resp = MagicMock(status_code=200, text="ok")
        with patch(
            "podcast_scraper.providers.common.hf_system_prompt.requests.get",
            return_value=mock_resp,
        ) as get_mock:
            load_hf_system_prompt("gated/Model", cache_dir=tmp_cache, hf_token="hf_test_token_123")
        call = get_mock.call_args
        assert call.kwargs["headers"]["Authorization"] == "Bearer hf_test_token_123"

    def test_hf_token_env_var_used_when_arg_missing(
        self, tmp_cache: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HF_TOKEN", "hf_env_token_456")
        mock_resp = MagicMock(status_code=200, text="ok")
        with patch(
            "podcast_scraper.providers.common.hf_system_prompt.requests.get",
            return_value=mock_resp,
        ) as get_mock:
            load_hf_system_prompt("gated/Model", cache_dir=tmp_cache)
        call = get_mock.call_args
        assert call.kwargs["headers"]["Authorization"] == "Bearer hf_env_token_456"

    def test_no_auth_header_when_no_token_available(
        self, tmp_cache: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("HF_TOKEN", raising=False)
        mock_resp = MagicMock(status_code=200, text="ok")
        with patch(
            "podcast_scraper.providers.common.hf_system_prompt.requests.get",
            return_value=mock_resp,
        ) as get_mock:
            load_hf_system_prompt("public/Model", cache_dir=tmp_cache)
        call = get_mock.call_args
        assert "Authorization" not in call.kwargs["headers"]


@pytest.mark.unit
class TestLoadHfSystemPromptValidation:
    """Input validation."""

    def test_empty_model_id_raises(self) -> None:
        with pytest.raises(ValueError, match="model_id"):
            load_hf_system_prompt("")

    def test_cache_dir_arg_overrides_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env_dir = tmp_path / "env_cache"
        arg_dir = tmp_path / "arg_cache"
        monkeypatch.setenv("HF_SYSTEM_PROMPT_CACHE", str(env_dir))
        mock_resp = MagicMock(status_code=200, text="hello")
        with patch(
            "podcast_scraper.providers.common.hf_system_prompt.requests.get",
            return_value=mock_resp,
        ):
            load_hf_system_prompt("vendor/Model", cache_dir=arg_dir)
        assert any(arg_dir.glob("*SYSTEM_PROMPT.txt"))
        assert not env_dir.exists() or not any(env_dir.glob("*SYSTEM_PROMPT.txt"))

    def test_cache_dir_env_var_used_when_arg_omitted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env_dir = tmp_path / "env_cache"
        monkeypatch.setenv("HF_SYSTEM_PROMPT_CACHE", str(env_dir))
        mock_resp = MagicMock(status_code=200, text="hello")
        with patch(
            "podcast_scraper.providers.common.hf_system_prompt.requests.get",
            return_value=mock_resp,
        ):
            load_hf_system_prompt("vendor/Model")
        assert any(env_dir.glob("*SYSTEM_PROMPT.txt"))
