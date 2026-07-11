"""Unit tests for tailnet DGX health helpers (RFC-089 / #814).

Two services share the tailnet host:

- Ollama on :11434 → ``check_ollama_health``.
- faster-whisper-server on :8000 → ``check_faster_whisper_health`` (#814).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from podcast_scraper.providers.tailnet_dgx.health import (
    _bare_host,
    _extract_model_names,
    check_faster_whisper_health,
    check_ollama_health,
    check_pyannote_diarize_health,
    dgx_diarize_base_url,
    dgx_ollama_base_url,
    dgx_whisper_base_url,
    DgxEndpointStatus,
    probe_dgx_endpoint,
    tcp_endpoint_listening,
)

# Patch target for the TCP-connect liveness primitive (#956).
_TCP = "podcast_scraper.providers.tailnet_dgx.health.tcp_endpoint_listening"


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


# --- faster-whisper-server (#814) ----------------------------------------


def test_dgx_whisper_base_url_default_port() -> None:
    assert dgx_whisper_base_url("dgx-llm-1.tail-test.ts.net") == (
        "http://dgx-llm-1.tail-test.ts.net:8000"
    )


def test_dgx_whisper_base_url_preserves_scheme() -> None:
    assert dgx_whisper_base_url("https://dgx.example.com:9000", 8000) == (
        "https://dgx.example.com:9000"
    )


@patch("httpx.Client")
@patch(_TCP, return_value=True)
def test_check_faster_whisper_health_up_no_model_filter(
    mock_tcp: MagicMock, mock_client_cls: MagicMock
) -> None:
    # Listening + no model filter → UP via TCP liveness alone; no /v1/models ping (no queue).
    assert check_faster_whisper_health("dgx-host") is True
    mock_client_cls.assert_not_called()


@patch("httpx.Client")
@patch(_TCP, return_value=True)
def test_check_faster_whisper_health_required_model_present(
    mock_tcp: MagicMock, mock_client_cls: MagicMock
) -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "object": "list",
        "data": [
            {"id": "Systran/faster-whisper-large-v3"},
            {"id": "Systran/faster-whisper-small"},
        ],
    }
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.get.return_value = mock_resp
    mock_client_cls.return_value = mock_client

    assert check_faster_whisper_health("dgx-host", require_model_substring="large-v3") is True


@patch("httpx.Client")
@patch(_TCP, return_value=True)
def test_check_faster_whisper_health_required_model_absent(
    mock_tcp: MagicMock, mock_client_cls: MagicMock
) -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"data": [{"id": "Systran/faster-whisper-small"}]}
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.get.return_value = mock_resp
    mock_client_cls.return_value = mock_client

    assert check_faster_whisper_health("dgx-host", require_model_substring="large-v3") is False


@patch("httpx.Client")
@patch(_TCP, return_value=True)
def test_check_faster_whisper_health_busy_non_200_is_up(
    mock_tcp: MagicMock, mock_client_cls: MagicMock
) -> None:
    # Listening but /v1/models returns 503 (queued behind a job) → UP, not a false down (#956).
    mock_resp = MagicMock()
    mock_resp.status_code = 503
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.get.return_value = mock_resp
    mock_client_cls.return_value = mock_client

    assert check_faster_whisper_health("dgx-host", require_model_substring="large-v3") is True


@patch("httpx.Client")
@patch(_TCP, return_value=True)
def test_check_faster_whisper_health_busy_read_timeout_is_up(
    mock_tcp: MagicMock, mock_client_cls: MagicMock
) -> None:
    # TCP up but /v1/models read times out (single-flight, queued) → UP (busy), not down.
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.get.side_effect = Exception("read timeout")
    mock_client_cls.return_value = mock_client

    assert check_faster_whisper_health("dgx-host", require_model_substring="large-v3") is True


@patch(_TCP, return_value=False)
def test_check_faster_whisper_health_down_when_tcp_refused(mock_tcp: MagicMock) -> None:
    # Connection refused / unreachable → genuinely DOWN.
    assert check_faster_whisper_health("dgx-host") is False
    assert check_faster_whisper_health("dgx-host", require_model_substring="large-v3") is False


# --- DGX pyannote diarize service (#926) ---------------------------------


def test_dgx_diarize_base_url_default_port() -> None:
    assert dgx_diarize_base_url("dgx-llm-1.tail-test.ts.net") == (
        "http://dgx-llm-1.tail-test.ts.net:8001"
    )


def test_dgx_diarize_base_url_preserves_scheme() -> None:
    assert dgx_diarize_base_url("https://dgx.example.com:9000", 8001) == (
        "https://dgx.example.com:9000"
    )


@patch("httpx.Client")
@patch(_TCP, return_value=True)
def test_check_pyannote_diarize_health_up_no_model_filter(
    mock_tcp: MagicMock, mock_client_cls: MagicMock
) -> None:
    assert check_pyannote_diarize_health("dgx-host") is True
    mock_client_cls.assert_not_called()


@patch("httpx.Client")
@patch(_TCP, return_value=True)
def test_check_pyannote_diarize_health_required_model_present(
    mock_tcp: MagicMock, mock_client_cls: MagicMock
) -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "object": "list",
        "data": [{"id": "pyannote/speaker-diarization-community-1"}],
    }
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.get.return_value = mock_resp
    mock_client_cls.return_value = mock_client

    assert (
        check_pyannote_diarize_health(
            "dgx-host", require_model_substring="speaker-diarization-community-1"
        )
        is True
    )


@patch("httpx.Client")
@patch(_TCP, return_value=True)
def test_check_pyannote_diarize_health_required_model_absent(
    mock_tcp: MagicMock, mock_client_cls: MagicMock
) -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"data": [{"id": "pyannote/speaker-diarization-3.0"}]}
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.get.return_value = mock_resp
    mock_client_cls.return_value = mock_client

    assert check_pyannote_diarize_health("dgx-host", require_model_substring="3.1") is False


@patch("httpx.Client")
@patch(_TCP, return_value=True)
def test_check_pyannote_diarize_health_busy_non_200_is_up(
    mock_tcp: MagicMock, mock_client_cls: MagicMock
) -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 503
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.get.return_value = mock_resp
    mock_client_cls.return_value = mock_client

    assert check_pyannote_diarize_health("dgx-host", require_model_substring="3.1") is True


@patch(_TCP, return_value=False)
def test_check_pyannote_diarize_health_down_when_tcp_refused(mock_tcp: MagicMock) -> None:
    assert check_pyannote_diarize_health("dgx-host") is False


# --- down-vs-busy primitives (#956) --------------------------------------


def test_bare_host_strips_scheme_and_port() -> None:
    assert _bare_host("http://dgx-host:8002/x") == "dgx-host"
    assert _bare_host("dgx-host") == "dgx-host"
    assert _bare_host("https://dgx.example.com:9000") == "dgx.example.com"


@patch("socket.create_connection")
def test_tcp_endpoint_listening_up(mock_conn: MagicMock) -> None:
    mock_conn.return_value.__enter__.return_value = MagicMock()
    assert tcp_endpoint_listening("dgx-host", 8002) is True
    # connects to the bare host:port, never an HTTP request on the job queue
    assert mock_conn.call_args[0][0] == ("dgx-host", 8002)


@patch("socket.create_connection", side_effect=OSError("refused"))
def test_tcp_endpoint_listening_down(mock_conn: MagicMock) -> None:
    assert tcp_endpoint_listening("dgx-host", 8002) is False


@patch("httpx.Client")
@patch(_TCP, return_value=True)
def test_probe_dgx_endpoint_ready(mock_tcp: MagicMock, mock_client_cls: MagicMock) -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.get.return_value = mock_resp
    mock_client_cls.return_value = mock_client

    assert probe_dgx_endpoint("dgx-host", 8002) is DgxEndpointStatus.READY


@patch("httpx.Client")
@patch(_TCP, return_value=True)
def test_probe_dgx_endpoint_busy_on_read_timeout(
    mock_tcp: MagicMock, mock_client_cls: MagicMock
) -> None:
    # Listening (TCP up) but the HTTP read is stuck behind the GPU job → BUSY, not DOWN.
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.get.side_effect = Exception("read timeout")
    mock_client_cls.return_value = mock_client

    assert probe_dgx_endpoint("dgx-host", 8002) is DgxEndpointStatus.BUSY


@patch(_TCP, return_value=False)
def test_probe_dgx_endpoint_down(mock_tcp: MagicMock) -> None:
    assert probe_dgx_endpoint("dgx-host", 8002) is DgxEndpointStatus.DOWN
