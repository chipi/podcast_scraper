"""Unit tests for the Ollama embedding client (ADR-098 / #897)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from podcast_scraper.providers.ml.embedding_ollama import encode_via_ollama


def _mock_httpx_client(mock_client_cls: MagicMock, *, status_code: int, payload: dict) -> MagicMock:
    """Wire MagicMock for `with httpx.Client(timeout=...) as client: client.post(...)`."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = payload
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.post.return_value = mock_resp
    mock_client_cls.return_value = mock_client
    return mock_client


@patch("httpx.Client")
def test_encode_via_ollama_posts_and_parses(mock_client_cls: MagicMock) -> None:
    mock_client = _mock_httpx_client(
        mock_client_cls,
        status_code=200,
        payload={"embeddings": [[3.0, 4.0], [0.0, 1.0]]},
    )

    rows = encode_via_ollama(
        ["a", "b"],
        "http://dgx:11434",
        model_id="nomic-embed-text",
        normalize=True,
    )

    # 3,4 normalizes to 0.6,0.8 (5-3-4 triangle); 0,1 stays 0,1.
    assert rows[0] == pytest.approx([0.6, 0.8])
    assert rows[1] == pytest.approx([0.0, 1.0])
    mock_client.post.assert_called_once()
    posted_url = mock_client.post.call_args[0][0]
    assert posted_url == "http://dgx:11434/api/embed"
    body = mock_client.post.call_args[1]["json"]
    assert body == {"model": "nomic-embed-text", "input": ["a", "b"]}


@patch("httpx.Client")
def test_encode_via_ollama_strips_trailing_slash_in_base(mock_client_cls: MagicMock) -> None:
    mock_client = _mock_httpx_client(
        mock_client_cls,
        status_code=200,
        payload={"embeddings": [[1.0]]},
    )
    encode_via_ollama("x", "http://dgx:11434/", model_id="m", normalize=False)
    assert mock_client.post.call_args[0][0] == "http://dgx:11434/api/embed"


def test_encode_via_ollama_rejects_empty_base_url() -> None:
    with pytest.raises(ValueError, match="base_url is empty"):
        encode_via_ollama("x", "   ", model_id="m")


@patch("httpx.Client")
def test_encode_via_ollama_returns_empty_for_no_texts(mock_client_cls: MagicMock) -> None:
    # An empty list of inputs is a no-op; the HTTP layer is never touched.
    assert encode_via_ollama([], "http://dgx:11434", model_id="m") == []
    mock_client_cls.assert_not_called()


@patch("httpx.Client")
def test_encode_via_ollama_normalize_off_returns_raw(mock_client_cls: MagicMock) -> None:
    _mock_httpx_client(
        mock_client_cls,
        status_code=200,
        payload={"embeddings": [[3.0, 4.0]]},
    )
    rows = encode_via_ollama("a", "http://dgx:11434", model_id="m", normalize=False)
    # No normalization → raw row passes through.
    assert rows == [[3.0, 4.0]]


@patch("httpx.Client")
def test_encode_via_ollama_rejects_row_count_mismatch(mock_client_cls: MagicMock) -> None:
    _mock_httpx_client(
        mock_client_cls,
        status_code=200,
        payload={"embeddings": [[1.0]]},
    )
    with pytest.raises(ValueError, match="row count mismatch"):
        encode_via_ollama(["a", "b"], "http://dgx:11434", model_id="m")


@patch("httpx.Client")
def test_encode_via_ollama_rejects_missing_embeddings_field(mock_client_cls: MagicMock) -> None:
    _mock_httpx_client(mock_client_cls, status_code=200, payload={"oops": True})
    with pytest.raises(ValueError, match="missing 'embeddings'"):
        encode_via_ollama("x", "http://dgx:11434", model_id="m")


@patch("httpx.Client")
def test_encode_via_ollama_rejects_invalid_row_type(mock_client_cls: MagicMock) -> None:
    _mock_httpx_client(
        mock_client_cls,
        status_code=200,
        payload={"embeddings": ["not-a-list"]},
    )
    with pytest.raises(ValueError, match="invalid embedding row"):
        encode_via_ollama("x", "http://dgx:11434", model_id="m")


@patch("httpx.Client")
def test_encode_via_ollama_falls_back_to_legacy_on_404(mock_client_cls: MagicMock) -> None:
    """Older Ollama (< 0.1.30) lacks /api/embed; we retry per-text on /api/embeddings."""
    # First call hits /api/embed → 404. Subsequent calls hit /api/embeddings.
    resp_404 = MagicMock()
    resp_404.status_code = 404
    resp_legacy_a = MagicMock()
    resp_legacy_a.status_code = 200
    resp_legacy_a.json.return_value = {"embedding": [3.0, 4.0]}
    resp_legacy_b = MagicMock()
    resp_legacy_b.status_code = 200
    resp_legacy_b.json.return_value = {"embedding": [0.0, 1.0]}

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.post.side_effect = [resp_404, resp_legacy_a, resp_legacy_b]
    mock_client_cls.return_value = mock_client

    rows = encode_via_ollama(
        ["a", "b"], "http://dgx:11434", model_id="nomic-embed-text", normalize=True
    )

    assert rows[0] == pytest.approx([0.6, 0.8])
    assert rows[1] == pytest.approx([0.0, 1.0])
    # 1 call to /api/embed (404) + 1 call per text on /api/embeddings.
    assert mock_client.post.call_count == 3
    legacy_url = mock_client.post.call_args_list[1][0][0]
    assert legacy_url == "http://dgx:11434/api/embeddings"
    assert mock_client.post.call_args_list[1][1]["json"] == {
        "model": "nomic-embed-text",
        "prompt": "a",
    }


def test_encode_via_ollama_requires_httpx(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing httpx dep surfaces as a clear RuntimeError, not a bare ImportError."""
    import builtins

    real_import = builtins.__import__

    def fake_import(  # type: ignore[no-untyped-def]
        name, globals=None, locals=None, fromlist=(), level=0
    ):
        if name == "httpx":
            raise ImportError("no httpx")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="httpx required"):
        encode_via_ollama("x", "http://dgx:11434", model_id="m")
