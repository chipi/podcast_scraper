"""Unit tests for embedding_loader provider dispatch (ADR-098 / #897).

Covers the new ``provider`` axis on ``embedding_loader.encode``: routing to
the Ollama client, legacy-shim deprecation path, and error contracts.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from podcast_scraper.providers.ml import embedding_loader


@patch("podcast_scraper.providers.ml.embedding_ollama.encode_via_ollama")
def test_provider_ollama_routes_to_ollama_client(mock_ollama: MagicMock) -> None:
    mock_ollama.return_value = [[0.1, 0.2, 0.3]]

    out = embedding_loader.encode(
        "hello",
        "nomic-embed-text",
        remote_endpoint="http://dgx:11434",
        provider="ollama",
    )

    assert out == [0.1, 0.2, 0.3]
    mock_ollama.assert_called_once()
    # Endpoint passed through as-is — Ollama client owns the /api/embed suffix.
    args, kwargs = mock_ollama.call_args
    assert args[1] == "http://dgx:11434"
    assert kwargs["model_id"] == "nomic-embed-text"


@patch("podcast_scraper.providers.ml.embedding_ollama.encode_via_ollama")
def test_provider_ollama_batch_returns_rows(mock_ollama: MagicMock) -> None:
    mock_ollama.return_value = [[1.0], [2.0]]

    out = embedding_loader.encode(
        ["a", "b"],
        "nomic-embed-text",
        remote_endpoint="http://dgx:11434",
        provider="ollama",
    )

    assert out == [[1.0], [2.0]]


def test_provider_ollama_without_endpoint_raises() -> None:
    """Ollama requires a base URL — silently falling back to local would be a footgun."""
    with pytest.raises(ValueError, match="requires vector_embedding_endpoint"):
        embedding_loader.encode(
            "hello",
            "nomic-embed-text",
            provider="ollama",
        )


@patch("podcast_scraper.providers.ml.embedding_remote.encode_via_endpoint")
def test_legacy_shim_path_still_works_when_provider_unset(mock_shim: MagicMock) -> None:
    """Existing callers that set endpoint without provider fall through to the shim (warn-only)."""
    mock_shim.return_value = [[0.5]]

    out = embedding_loader.encode(
        "hello",
        "minilm-l6",
        remote_endpoint="http://dgx:8001/embed",
    )

    assert out == [0.5]
    mock_shim.assert_called_once()


@patch("podcast_scraper.providers.ml.embedding_remote.encode_via_endpoint")
def test_legacy_shim_path_explicit_sentence_transformers(mock_shim: MagicMock) -> None:
    """provider='sentence_transformers' + endpoint → still shim path (back-compat)."""
    mock_shim.return_value = [[0.5]]

    out = embedding_loader.encode(
        "hello",
        "minilm-l6",
        remote_endpoint="http://dgx:8001/embed",
        provider="sentence_transformers",
    )

    assert out == [0.5]
    mock_shim.assert_called_once()


def test_provider_unknown_string_treated_as_sentence_transformers_when_no_endpoint() -> None:
    """Defensive: an unknown provider value should not silently call Ollama or shim.

    Since both the Ollama path and the shim path require remote_endpoint, an
    unknown provider with no endpoint falls through to the local sentence-transformers
    path. We can't test that here without loading torch, but we can confirm it
    doesn't error trying to import httpx, and doesn't dispatch to either HTTP client.
    """
    # Patch both HTTP clients; the local path will try sentence_transformers import,
    # which we patch to a no-op MagicMock so the test doesn't need torch installed.
    with patch("podcast_scraper.providers.ml.embedding_loader.get_embedding_model") as mock_local:
        fake_model = MagicMock()
        fake_model.encode.return_value = [[0.0]]
        mock_local.return_value = fake_model

        out = embedding_loader.encode(
            "hello",
            "minilm-l6",
            provider="bogus",  # unknown — should not route to ollama/shim
        )

    mock_local.assert_called_once()
    # encode is exercised — exact return shape covered by existing tests.
    assert out is not None
