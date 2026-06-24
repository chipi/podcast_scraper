"""Unit tests for the audio-bridge resolver (#1070).

``_head_request`` is monkeypatched so no real network I/O runs in CI.
"""

from __future__ import annotations

import httpx

from podcast_scraper.server import app_audio_bridge
from podcast_scraper.server.app_audio_bridge import resolve_audio


def test_resolve_success(monkeypatch) -> None:
    monkeypatch.setattr(
        app_audio_bridge,
        "_head_request",
        lambda url, timeout: (
            200,
            "https://cdn.example/final.mp3",
            {"content-type": "audio/mpeg", "content-length": "12345"},
        ),
    )
    res = resolve_audio("https://host.example/track/ep.mp3")
    assert res.verified is True
    assert res.final_url == "https://cdn.example/final.mp3"
    assert res.content_type == "audio/mpeg"
    assert res.content_length == 12345


def test_resolve_non_2xx_is_unverified(monkeypatch) -> None:
    monkeypatch.setattr(
        app_audio_bridge,
        "_head_request",
        lambda url, timeout: (404, "https://host.example/ep.mp3", {}),
    )
    res = resolve_audio("https://host.example/ep.mp3")
    assert res.verified is False
    assert res.final_url == "https://host.example/ep.mp3"
    assert res.content_length is None


def test_resolve_network_error_falls_back_to_original(monkeypatch) -> None:
    def boom(url: str, timeout: float):
        raise httpx.ConnectError("unreachable")

    monkeypatch.setattr(app_audio_bridge, "_head_request", boom)
    res = resolve_audio("https://host.example/ep.mp3")
    assert res.verified is False
    assert res.final_url == "https://host.example/ep.mp3"
    assert res.content_type is None
