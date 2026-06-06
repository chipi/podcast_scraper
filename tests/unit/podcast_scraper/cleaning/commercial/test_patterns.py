"""Unit tests for commercial sponsor patterns (B8)."""

from __future__ import annotations

import pytest

from podcast_scraper.cleaning.commercial.patterns import (
    BLOCK_END_PATTERNS,
    BLOCK_START_PATTERNS,
    BRAND_NAMES,
    SPONSOR_PATTERNS,
)

pytestmark = pytest.mark.unit


def test_every_pattern_has_valid_metadata() -> None:
    for sp in SPONSOR_PATTERNS:
        assert 0.0 < sp.confidence <= 1.0
        assert sp.boundary_hint in {"block_start", "block_end", "inline"}
        assert sp.category


def test_block_partitions_are_consistent() -> None:
    assert all(p.boundary_hint == "block_start" for p in BLOCK_START_PATTERNS)
    assert all(p.boundary_hint == "block_end" for p in BLOCK_END_PATTERNS)


def test_intro_sponsor_phrase_matches() -> None:
    assert any(
        sp.pattern.search("this episode is brought to you by Acme") for sp in SPONSOR_PATTERNS
    )


def test_over_broad_okay_so_no_longer_matches() -> None:
    """Generic conversational filler 'okay, so' must not be a sponsor boundary (B7)."""
    assert not any(sp.pattern.search("okay, so anyway as I was saying") for sp in SPONSOR_PATTERNS)


def test_host_name_not_in_brand_names() -> None:
    """Host/person names must not be sponsor brands (B7 — 'lenny' collision)."""
    assert "lenny" not in BRAND_NAMES
