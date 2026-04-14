"""Unit tests for GIL edge type string normalisation."""

from __future__ import annotations

from podcast_scraper.gi.edge_normalization import normalize_gil_edge_type


def test_normalize_gil_edge_type() -> None:
    assert normalize_gil_edge_type("supported_by") == "SUPPORTED_BY"
    assert normalize_gil_edge_type("  about ") == "ABOUT"
    assert normalize_gil_edge_type(None) == ""
