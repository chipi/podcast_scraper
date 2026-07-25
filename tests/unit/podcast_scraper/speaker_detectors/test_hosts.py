"""Isolated unit tests for speaker_detectors.hosts (E1, RFC-059).

Covers the pure network/organisation classifiers and the transcript self-intro
extractor (#876) directly, without a spaCy model.
"""

from __future__ import annotations

import pytest

from podcast_scraper.speaker_detectors.hosts import (
    detect_hosts_from_feed,
    extract_self_introduced_host,
    guests_introduced_by_the_host,
    has_org_markers,
    is_known_network,
    is_network_or_org_author,
    is_plausible_mononym,
    looks_like_publisher,
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


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("The New York Times", True),  # multi-token publisher on the known list
        ("Reuters", True),  # single-token publisher on the known list
        ("The Economist", True),
        ("Bloomberg Businessweek", True),  # first token is a known network
        ("Chicago Tribune", True),  # news-outlet suffix (no known-list entry needed)
        ("The Daily Gazette", True),
        ("Rolling Stone Magazine", True),
        ("Oprah", False),  # real-person mononym is kept (unlike is_network_or_org_author)
        ("Sting", False),
        ("Patrick O'Shaughnessy", False),
        ("Emily Post", False),  # a person whose surname collides with a publisher word
        ("", True),
    ],
)
def test_looks_like_publisher(name: str, expected: bool) -> None:
    # Unlike is_network_or_org_author, a lone real-person token is NOT flagged: this guard
    # strips publishers from already-resolved person surfaces without dropping mononym people.
    assert looks_like_publisher(name) is expected


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


# --- Regression: the catastrophic-backtracking hang + the IGNORECASE case-sensitivity bug ---
# The guest-intro name pattern compiles with re.IGNORECASE for its lowercase cue words. Before the
# `(?-i:...)` fix, IGNORECASE made `[A-Z]` match lowercase too, so the name pattern matched every
# multi-word lowercase phrase in the transcript — crowning non-names AND backtracking for minutes
# on a 77k-char episode (found via a faulthandler stack dump).


def test_guest_intro_name_pattern_is_case_sensitive() -> None:
    """A capitalised introduction is caught; the same words as lowercase prose are not a name."""
    assert guests_introduced_by_the_host({"v": "Jia Li is with us today."}) == {"Jia Li"}
    # Under the IGNORECASE bug this lowercase clause matched as a two-word "name". It must not.
    assert guests_introduced_by_the_host({"v": "the plan is with us today, it is."}) == set()


def test_guest_intro_no_catastrophic_backtracking_on_long_prose() -> None:
    """A long capitalisation-heavy lowercase transcript resolves fast, not in minutes.

    Budget is deliberately generous (the fix runs in ~ms; the bug took minutes) so the test guards
    the O(n^2)/backtracking regression without being flaky under CI load.
    """
    import time

    text = "the market is with us today and the future is with us now " * 2000  # ~58k chars
    start = time.perf_counter()
    guests_introduced_by_the_host({"v": text})
    assert time.perf_counter() - start < 5.0, "guest-intro scan backtracked catastrophically"


def test_detect_hosts_from_feed_extracts_journalists_phrase() -> None:
    """The feed_hosts wiring depends on this: the show blurb's "journalists X and Y" names hosts."""
    hosts = detect_hosts_from_feed(
        "Hard Fork",
        "Each week, journalists Kevin Roose and Casey Newton explore the world of tech.",
        ["The New York Times"],
    )
    assert hosts == {"Kevin Roose", "Casey Newton"}


# --- single-name self-intro recovery + introduced-guest greeting (no-anchor feed recall) ---


@pytest.mark.parametrize(
    ("token", "ok"),
    [
        ("Brandon", True),  # real mononym self-intro (Latent Space host)
        ("Neeraj", True),
        ("Oprah", True),
        ("Sting", True),
        ("American", False),  # the "I'm American" class the guard exists for
        ("Republican", False),
        ("Christian", False),
        ("Here", False),  # ordinary word, capitalised by ASR
        ("sorry", False),
        ("", False),
        ("A", False),
    ],
)
def test_is_plausible_mononym(token, ok) -> None:
    assert is_plausible_mononym(token) is ok


def test_guest_greeting_name_then_welcome() -> None:
    """The host greets a just-introduced guest by name: 'Jody Rosen, welcome …'."""
    assert guests_introduced_by_the_host({"v": "Jody Rosen, welcome to the show."}) == {
        "Jody Rosen"
    }
    assert guests_introduced_by_the_host({"v": "Nic Harrigan, thanks so much for coming on."}) == {
        "Nic Harrigan"
    }
    # a bare greeting with no preceding name is NOT a guest introduction
    assert guests_introduced_by_the_host({"v": "welcome to Hard Fork this week"}) == set()


def test_transcript_intro_does_not_capture_a_lowercase_run_as_a_host(monkeypatch) -> None:
    # N3: under a blanket re.IGNORECASE the [A-Z][a-z]+ name classes matched any letter, so
    # "I'm going to explain how this works" captured "going to explain..." as a host name. The name
    # capture is now case-SENSITIVE via (?-i:...); the cue stays case-insensitive.
    from podcast_scraper.speaker_detectors import hosts

    monkeypatch.setattr(hosts, "_extract_person_entities", lambda text, nlp: [])
    nlp = object()  # truthy so the function does not early-return; NER is patched out above
    assert (
        hosts.detect_hosts_from_transcript_intro(
            "I'm going to explain how this works today before we begin.", nlp
        )
        == set()
    )
    assert "Noah Kravitz" in hosts.detect_hosts_from_transcript_intro(
        "Welcome to the show. I'm Noah Kravitz and today we go deep.", nlp
    )
