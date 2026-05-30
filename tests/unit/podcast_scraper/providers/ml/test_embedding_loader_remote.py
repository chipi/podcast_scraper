"""Unit tests for embedding_loader remote DGX path (RFC-089)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from podcast_scraper.providers.ml import embedding_loader


@patch("podcast_scraper.providers.ml.embedding_remote.encode_via_endpoint")
def test_encode_single_string_via_remote_endpoint(mock_remote: MagicMock) -> None:
    mock_remote.return_value = [[0.1, 0.2, 0.3]]
    out = embedding_loader.encode(
        "hello",
        "nomic-embed-text",
        remote_endpoint="http://dgx:8001/embed",
    )
    assert out == [0.1, 0.2, 0.3]
    mock_remote.assert_called_once()


@patch("podcast_scraper.providers.ml.embedding_remote.encode_via_endpoint")
def test_encode_batch_via_remote_endpoint(mock_remote: MagicMock) -> None:
    mock_remote.return_value = [[1.0], [2.0]]
    out = embedding_loader.encode(
        ["a", "b"],
        "nomic-embed-text",
        remote_endpoint="http://dgx:8001/",
        return_numpy=True,
    )
    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 1)


@patch("podcast_scraper.providers.ml.embedding_remote.encode_via_endpoint")
def test_encode_single_via_remote_return_numpy(mock_remote: MagicMock) -> None:
    mock_remote.return_value = [[0.5, 0.6]]
    out = embedding_loader.encode(
        "x",
        "nomic-embed-text",
        remote_endpoint="http://dgx:8001/embed",
        return_numpy=True,
    )
    assert isinstance(out, np.ndarray)
    assert out.tolist() == [0.5, 0.6]
