"""Isolated unit tests for speaker_detectors.detection (E1, RFC-059).

Imports the submodule directly and patches the ``_extract_person_entities`` seam
(the deliberate patch point) so orchestration is tested without a real spaCy model.
"""

from __future__ import annotations

import pytest

from podcast_scraper.speaker_detectors import detection
from podcast_scraper.speaker_detectors.constants import (
    DEFAULT_SPEAKER_NAMES,
    MIN_SPEAKERS_REQUIRED,
)
from podcast_scraper.speaker_detectors.detection import (
    _build_speaker_names_list,
    detect_speaker_names,
)
from tests.conftest import create_test_config

pytestmark = pytest.mark.unit


def test_build_names_hosts_only() -> None:
    names, succeeded, used_defaults = _build_speaker_names_list({"Alice"}, [], max_names=2)
    assert names == ["Alice"]
    assert succeeded is True
    assert used_defaults is False


def test_build_names_no_hosts_no_guests_uses_defaults() -> None:
    names, succeeded, used_defaults = _build_speaker_names_list(set(), [], max_names=2)
    assert names == DEFAULT_SPEAKER_NAMES
    assert succeeded is False
    assert used_defaults is True


def test_build_names_hosts_and_guests() -> None:
    names, succeeded, used_defaults = _build_speaker_names_list({"Alice"}, ["Bob"], max_names=2)
    assert names == ["Alice", "Bob"]
    assert succeeded is True
    assert used_defaults is False


def test_build_names_single_guest_extended_to_minimum() -> None:
    names, succeeded, used_defaults = _build_speaker_names_list(set(), ["Bob"], max_names=2)
    assert names[0] == "Bob"
    assert len(names) >= MIN_SPEAKERS_REQUIRED
    assert succeeded is True
    assert used_defaults is True


def test_detect_returns_defaults_when_auto_speakers_disabled() -> None:
    cfg = create_test_config(auto_speakers=False)
    names, hosts, succeeded, used_defaults = detect_speaker_names(
        "Title", None, nlp=object(), cfg=cfg
    )
    assert names == DEFAULT_SPEAKER_NAMES
    assert hosts == set()
    assert succeeded is False
    assert used_defaults is True


def test_detect_returns_defaults_when_nlp_missing() -> None:
    names, _hosts, succeeded, used_defaults = detect_speaker_names("Title", None, nlp=None)
    assert names == DEFAULT_SPEAKER_NAMES
    assert succeeded is False
    assert used_defaults is True


def test_detect_combines_known_host_and_detected_guest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        detection, "_extract_person_entities", lambda _text, _nlp: [("Bob Guest", 1.0)]
    )
    cfg = create_test_config(auto_speakers=True)
    names, hosts, succeeded, used_defaults = detect_speaker_names(
        "Interview with Bob Guest",
        episode_description=None,
        nlp=object(),
        cfg=cfg,
        known_hosts={"Alice Host"},
    )
    assert names == ["Alice Host", "Bob Guest"]
    assert hosts == {"Alice Host"}
    assert succeeded is True
    assert used_defaults is False


def test_detect_drops_guest_that_is_a_known_host(monkeypatch: pytest.MonkeyPatch) -> None:
    # A detected name matching a known host is not double-counted as a guest.
    monkeypatch.setattr(
        detection, "_extract_person_entities", lambda _text, _nlp: [("Alice Host", 1.0)]
    )
    cfg = create_test_config(auto_speakers=True)
    names, hosts, _succeeded, _used_defaults = detect_speaker_names(
        "Interview with Alice Host",
        episode_description=None,
        nlp=object(),
        cfg=cfg,
        known_hosts={"Alice Host"},
    )
    assert names == ["Alice Host"]
    assert hosts == {"Alice Host"}
