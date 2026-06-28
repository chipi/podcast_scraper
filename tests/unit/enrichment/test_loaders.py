"""Unit tests for ``enrichment.enrichers._loaders`` shared helpers."""

from __future__ import annotations

import pytest

from podcast_scraper.enrichment.enrichers._loaders import (
    episode_duration_seconds,
    is_unresolved_speaker_placeholder,
)

# ---------------------------------------------------------------------------
# is_unresolved_speaker_placeholder
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pid",
    [
        "SPEAKER_00",
        "SPEAKER_3",
        "SPEAKER_18",
        "speaker_00",
        "person:speaker-03",
        "person:speaker-05",
        "person:SPEAKER_07",
        "person:speaker_99",
    ],
)
def test_is_unresolved_speaker_placeholder_matches_known_shapes(pid: str) -> None:
    assert is_unresolved_speaker_placeholder(pid)


@pytest.mark.parametrize(
    "pid",
    [
        "person:alice",
        "person:brandon-scott",
        "person:robert-armstrong",
        "person:speaker-jones",  # "speaker" but not followed by digits
        "person:speakers-united",
        "",
    ],
)
def test_is_unresolved_speaker_placeholder_passes_real_persons(pid: str) -> None:
    assert not is_unresolved_speaker_placeholder(pid)


def test_is_unresolved_speaker_placeholder_matches_by_name() -> None:
    # An id slugged from a real name + name field is "SPEAKER_NN" — still a
    # placeholder. (Defensive: if upstream changes the slug strategy.)
    assert is_unresolved_speaker_placeholder("person:unknown-08", "SPEAKER_08")


def test_is_unresolved_speaker_placeholder_empty_inputs_pass() -> None:
    # Empty string id + no name = nothing to flag.
    assert not is_unresolved_speaker_placeholder("", None)


# ---------------------------------------------------------------------------
# episode_duration_seconds
# ---------------------------------------------------------------------------


def test_episode_duration_top_level() -> None:
    assert episode_duration_seconds({"duration_seconds": 1800}) == 1800.0


def test_episode_duration_nested_under_episode() -> None:
    """The metadata writer stores duration under ``episode.``."""
    assert episode_duration_seconds({"episode": {"duration_seconds": 2962}}) == 2962.0


def test_episode_duration_top_level_wins_when_both_present() -> None:
    """Top-level value wins (legacy shape stays authoritative when both exist)."""
    meta = {"duration_seconds": 100, "episode": {"duration_seconds": 200}}
    assert episode_duration_seconds(meta) == 100.0


def test_episode_duration_missing_returns_zero() -> None:
    assert episode_duration_seconds({}) == 0.0
    assert episode_duration_seconds({"episode": {}}) == 0.0
    assert episode_duration_seconds({"episode": None}) == 0.0


def test_episode_duration_zero_or_negative_treated_as_missing() -> None:
    assert episode_duration_seconds({"duration_seconds": 0}) == 0.0
    assert episode_duration_seconds({"duration_seconds": -5}) == 0.0
    assert episode_duration_seconds({"episode": {"duration_seconds": 0}}) == 0.0
