"""SSRF-target classification for feed-body URLs (review 2026-07-17 M22).

``_is_ssrf_target`` is the literal-IP guard applied to untrusted transcript /
enclosure URLs before a fetch: it must block the cloud metadata API and RFC-1918
literals while *allowing* loopback (the e2e / acceptance mock server binds
127.0.0.1) and plain hostnames (which the host-level egress iptables covers).
"""

from __future__ import annotations

import pytest

from podcast_scraper.rss.downloader import _is_ssrf_target

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata API (link-local)
        "http://10.0.0.5/feed.mp3",  # RFC-1918
        "http://192.168.1.10/ep.mp3",  # RFC-1918
        "http://172.16.0.1/ep.mp3",  # RFC-1918
        "http://[fe80::1]/ep.mp3",  # IPv6 link-local
    ],
)
def test_blocks_private_and_linklocal_literals(url: str) -> None:
    assert _is_ssrf_target(url) is True


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8080/feed.xml",  # loopback — mock server / local dev
        "http://[::1]:9000/feed.xml",  # IPv6 loopback
        "http://example.com/feed.xml",  # hostname, not a literal IP
        "http://8.8.8.8/feed.mp3",  # public literal
        "not-a-url",  # no host
        "",  # empty
    ],
)
def test_allows_loopback_hostnames_and_public(url: str) -> None:
    assert _is_ssrf_target(url) is False
