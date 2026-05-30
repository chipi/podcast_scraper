"""CLI handler tests for DGX embedding endpoint wiring (RFC-089)."""

from __future__ import annotations

from podcast_scraper.search import cli_handlers


def test_minimal_vector_config_includes_embedding_endpoint() -> None:
    cfg = cli_handlers._minimal_vector_config(
        "/tmp/out",
        vector_embedding_model="nomic-embed-text",
        vector_embedding_endpoint="http://dgx:8001/embed",
    )
    assert cfg.vector_embedding_model == "nomic-embed-text"
    assert cfg.vector_embedding_endpoint == "http://dgx:8001/embed"
    assert cfg.vector_search is True
