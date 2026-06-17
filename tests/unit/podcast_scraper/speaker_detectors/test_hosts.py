"""Isolated unit tests for speaker_detectors.hosts (E1, RFC-059).

Covers the pure network/organisation classifiers and the transcript self-intro
extractor (#876) directly, without a spaCy model.
"""

from __future__ import annotations

import pytest

from podcast_scraper.speaker_detectors.hosts import (
    extract_self_introduced_host,
    has_org_markers,
    is_known_network,
    is_network_or_org_author,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Pushkin", True),  # whole name is a known network
        ("Pushkin Industries", True),  # first token is a known network
        ("Oprah", False),  # real-person mononym, not a known network
        ("Patrick O'Shaughnessy", False),
        ("", False),
    ],
)
def test_is_known_network(name: str, expected: bool) -> None:
    assert is_known_network(name) is expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Acme Media", True),  # explicit org marker word
        ("News & Friends", True),  # ampersand marker
        ("Oprah", False),  # trusted mononym person — no org markers
        ("Patrick O'Shaughnessy", False),
        ("", True),  # empty is treated as non-person
    ],
)
def test_has_org_markers(name: str, expected: bool) -> None:
    assert has_org_markers(name) is expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("NPR", True),  # mononym/acronym rejected for RSS author tags
        ("Oprah", True),  # lone token → treated as network for author tags
        ("Acme Media", True),  # org marker
        ("Patrick O'Shaughnessy", False),  # First Last → a real host
        ("", True),
    ],
)
def test_is_network_or_org_author(name: str, expected: bool) -> None:
    assert is_network_or_org_author(name) is expected


def test_extract_self_introduced_host_basic() -> None:
    text = "Hello and welcome to the show. I'm Patrick O'Shaughnessy and today we dig in."
    assert extract_self_introduced_host(text) == "Patrick O'Shaughnessy"


def test_extract_self_introduced_host_skips_network_bumper() -> None:
    # Network shows open with a publisher bumper in the same "I'm <X>" shape; the
    # known-network bumper is skipped and the real host name is returned (#876).
    text = "This is Unhedged from the FT. I'm Pushkin. I'm Katie Martin here with you."
    assert extract_self_introduced_host(text) == "Katie Martin"


def test_extract_self_introduced_host_none_when_absent() -> None:
    assert extract_self_introduced_host("No introductions in this clip.") is None
    assert extract_self_introduced_host(None) is None


def test_extract_self_introduced_host_only_scans_intro_window() -> None:
    # A self-intro past the intro window is ignored (a later guest "I'm …").
    text = "x" * 2100 + " I'm Late Guest"
    assert extract_self_introduced_host(text, intro_chars=2000) is None
