"""Unit tests for DGX remote embedding client (RFC-089)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from podcast_scraper.providers.ml.embedding_remote import encode_via_endpoint


@patch("httpx.Client")
def test_encode_via_endpoint_posts_and_parses(mock_client_cls: MagicMock) -> None:
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.post.return_value = mock_resp
    mock_client_cls.return_value = mock_client

    rows = encode_via_endpoint(
        ["a", "b"],
        "http://dgx:8001",
        model_id="nomic-embed-text",
        normalize=True,
    )
    assert rows == [[0.1, 0.2], [0.3, 0.4]]
    mock_client.post.assert_called_once()
    call_kwargs = mock_client.post.call_args
    assert call_kwargs[0][0] == "http://dgx:8001/embed"
    assert call_kwargs[1]["json"]["texts"] == ["a", "b"]
    assert call_kwargs[1]["json"]["model"] == "nomic-embed-text"


@patch("httpx.Client")
def test_encode_via_endpoint_appends_embed_path(mock_client_cls: MagicMock) -> None:
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"embeddings": [[1.0]]}
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.post.return_value = mock_resp
    mock_client_cls.return_value = mock_client

    encode_via_endpoint("hello", "http://dgx:8001/", model_id="m")
    assert mock_client.post.call_args[0][0] == "http://dgx:8001/embed"


def test_encode_via_endpoint_rejects_empty_url() -> None:
    with pytest.raises(ValueError, match="empty"):
        encode_via_endpoint("x", "  ", model_id="m")


@patch("httpx.Client")
def test_encode_via_endpoint_rejects_row_count_mismatch(mock_client_cls: MagicMock) -> None:
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"embeddings": [[1.0]]}
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.post.return_value = mock_resp
    mock_client_cls.return_value = mock_client

    with pytest.raises(ValueError, match="mismatch"):
        encode_via_endpoint(["a", "b"], "http://dgx:8001/embed", model_id="m")
