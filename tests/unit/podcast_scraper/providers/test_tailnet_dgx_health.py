"""Unit tests for tailnet DGX Ollama health helpers (RFC-089)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from podcast_scraper.providers.tailnet_dgx.health import (
    _extract_model_names,
    check_ollama_health,
    dgx_ollama_base_url,
)


def test_dgx_ollama_base_url_plain_host() -> None:
    assert dgx_ollama_base_url("dgx-llm-1.tail-test.ts.net", 11434) == (
        "http://dgx-llm-1.tail-test.ts.net:11434"
    )


def test_dgx_ollama_base_url_preserves_http_prefix() -> None:
    assert dgx_ollama_base_url("http://dgx:11434/", 9999) == "http://dgx:11434"


def test_extract_model_names_parses_tags_payload() -> None:
    payload = {"models": [{"name": "whisper-large-v3"}, {"name": "llama3.3:70b"}]}
    assert _extract_model_names(payload) == ["whisper-large-v3", "llama3.3:70b"]


def test_extract_model_names_rejects_non_dict() -> None:
    assert _extract_model_names([]) == []


@patch("httpx.Client")
def test_check_ollama_health_ok_without_model_filter(mock_client_cls: MagicMock) -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"models": [{"name": "whisper-large-v3"}]}
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.get.return_value = mock_resp
    mock_client_cls.return_value = mock_client

    assert check_ollama_health("dgx-host") is True
    mock_client.get.assert_called_once_with("http://dgx-host:11434/api/tags")


@patch("httpx.Client")
def test_check_ollama_health_requires_model_substring(mock_client_cls: MagicMock) -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"models": [{"name": "llama3.3:70b"}]}
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.get.return_value = mock_resp
    mock_client_cls.return_value = mock_client

    assert check_ollama_health("dgx-host", require_model_substring="whisper") is False
    assert check_ollama_health("dgx-host", require_model_substring="llama3") is True


@patch("httpx.Client")
def test_check_ollama_health_non_200(mock_client_cls: MagicMock) -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 503
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.get.return_value = mock_resp
    mock_client_cls.return_value = mock_client

    assert check_ollama_health("dgx-host") is False


@patch("httpx.Client")
def test_check_ollama_health_empty_models(mock_client_cls: MagicMock) -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"models": []}
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.get.return_value = mock_resp
    mock_client_cls.return_value = mock_client

    assert check_ollama_health("dgx-host") is False


@patch("httpx.Client")
def test_check_ollama_health_network_error(mock_client_cls: MagicMock) -> None:
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.get.side_effect = OSError("connection refused")
    mock_client_cls.return_value = mock_client

    assert check_ollama_health("dgx-host") is False


def test_check_ollama_health_without_httpx(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "httpx":
            raise ImportError("no httpx")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert check_ollama_health("dgx-host") is False
