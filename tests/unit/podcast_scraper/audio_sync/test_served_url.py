"""drift-check and build-maps measure the URL a listener's player gets, not the feed's (#2359)."""

from __future__ import annotations

import pytest

from podcast_scraper.audio_sync.drift import served_url
from podcast_scraper.server import app_audio_bridge

pytestmark = pytest.mark.unit


def test_an_http_enclosure_is_measured_where_the_api_sends_players(monkeypatch) -> None:
    monkeypatch.setattr(
        app_audio_bridge, "https_audio_url", lambda url, **_: "https://cdn.example/ep.mp3"
    )
    assert served_url("http://feed.example/ep.mp3") == "https://cdn.example/ep.mp3"


def test_without_an_https_version_the_feed_url_is_what_plays(monkeypatch) -> None:
    monkeypatch.setattr(app_audio_bridge, "https_audio_url", lambda url, **_: None)
    assert served_url("http://feed.example/ep.mp3") == "http://feed.example/ep.mp3"


def test_a_failed_lookup_falls_back_to_the_feed_url(monkeypatch) -> None:
    def boom(url: str, **_: object) -> str:
        raise RuntimeError("network")

    monkeypatch.setattr(app_audio_bridge, "https_audio_url", boom)
    assert served_url("http://feed.example/ep.mp3") == "http://feed.example/ep.mp3"
