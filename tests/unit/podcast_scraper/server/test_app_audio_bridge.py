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


BBC_RSS = (
    "http://open.live.bbc.co.uk/mediaselector/6/redir/version/2.0/mediaset/"
    "audio-nondrm-download-rss/proto/http/vpid/p0p09f1b.mp3"
)


def _no_network(*_a, **_k):
    raise AssertionError("no network call expected")


def test_https_url_is_returned_unchanged_without_a_lookup(monkeypatch) -> None:
    monkeypatch.setattr(app_audio_bridge, "_probe", _no_network)
    monkeypatch.setattr(app_audio_bridge, "_get_json", _no_network)
    assert app_audio_bridge.https_audio_url("https://cdn.example/ep.mp3") == (
        "https://cdn.example/ep.mp3"
    )


def test_http_url_switches_to_its_https_form_when_that_plays(monkeypatch) -> None:
    app_audio_bridge._https_cache.clear()
    monkeypatch.setattr(
        app_audio_bridge, "_probe", lambda url, timeout: (206, "https://cdn.example/x.mp3")
    )
    assert app_audio_bridge.https_audio_url("http://host.example/ep.mp3") == (
        "https://host.example/ep.mp3"
    )


def test_https_form_that_redirects_back_to_http_does_not_count(monkeypatch) -> None:
    app_audio_bridge._https_cache.clear()
    monkeypatch.setattr(
        app_audio_bridge, "_probe", lambda url, timeout: (206, "http://ads.example/x.mp3")
    )
    assert app_audio_bridge.https_audio_url("http://host.example/ep.mp3") is None


def test_bbc_enclosure_resolves_through_the_media_selector(monkeypatch) -> None:
    app_audio_bridge._https_cache.clear()
    # BBC's https form ends on Triton's http ad server (measured 2026-10-16), so it must not win.
    monkeypatch.setattr(
        app_audio_bridge, "_probe", lambda url, timeout: (206, "http://bbc.pdn.tritondigital.com/v")
    )
    seen = []

    def selector(url, timeout):
        seen.append(url)
        return {
            "media": [
                {
                    "connection": [
                        {"protocol": "http", "href": "http://aod.example/a.mp3"},
                        {"protocol": "https", "href": "https://aod.example/a.mp3?tok=1"},
                    ]
                }
            ]
        }

    monkeypatch.setattr(app_audio_bridge, "_get_json", selector)
    assert app_audio_bridge.https_audio_url(BBC_RSS) == "https://aod.example/a.mp3?tok=1"
    assert "/mediaset/audio-nondrm-download/vpid/p0p09f1b/" in seen[0]


def test_lookup_is_cached(monkeypatch) -> None:
    app_audio_bridge._https_cache.clear()
    calls = []

    def probe(url, timeout):
        calls.append(url)
        return 206, "https://host.example/ep.mp3"

    monkeypatch.setattr(app_audio_bridge, "_probe", probe)
    for _ in range(3):
        app_audio_bridge.https_audio_url("http://host.example/ep.mp3")
    assert len(calls) == 1


def test_nothing_found_on_a_network_error(monkeypatch) -> None:
    app_audio_bridge._https_cache.clear()

    def boom(url, timeout):
        raise httpx.ConnectError("unreachable")

    monkeypatch.setattr(app_audio_bridge, "_probe", boom)
    monkeypatch.setattr(app_audio_bridge, "_get_json", boom)
    assert app_audio_bridge.https_audio_url(BBC_RSS) is None
