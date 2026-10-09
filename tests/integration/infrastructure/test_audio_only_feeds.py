"""The fixture server can serve every feed AUDIO-ONLY, so a pipeline run measures ASR (#2187).

WHY THIS EXISTS. Every corpus feed item carries a ``<podcast:transcript>`` (the speaker-labelled
VTT), the pipeline always takes a publisher transcript when one is offered, and no config knob
says otherwise. So the fixture audio — fifteen non-English episodes among it — can only reach
Whisper through a feed that does not offer the transcript. `--no-transcripts` on
`scripts/tools/run_e2e_mock_server.py` serves exactly that; these tests pin that the strip removes
the transcript and nothing else.
"""

from __future__ import annotations

import urllib.request
import xml.etree.ElementTree as ET  # nosec B405 - parses our own fixture feeds
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

try:
    from tests.e2e.fixtures.e2e_http_server import (
        E2EHTTPRequestHandler,
        strip_feed_language,
        strip_podcast_transcripts,
    )
except ImportError:  # pragma: no cover - mirrors this directory's conftest fallback
    E2EHTTPRequestHandler = None  # type: ignore[assignment,misc]
    strip_feed_language = None  # type: ignore[assignment]
    strip_podcast_transcripts = None  # type: ignore[assignment]

requires_handler = pytest.mark.skipif(
    E2EHTTPRequestHandler is None, reason="E2E handler unavailable"
)

RSS_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "rss"
CORPUS_FEEDS = sorted(RSS_DIR.glob("p*_corpus.xml"))
NS = {"podcast": "https://podcastindex.org/namespace/1.0"}


def _items(xml: bytes) -> list[ET.Element]:
    return ET.fromstring(xml).findall("./channel/item")  # nosec B314 - own fixtures


def _enclosure(item: ET.Element) -> ET.Element:
    enclosure = item.find("enclosure")
    assert enclosure is not None, f"item {item.findtext('guid')!r} has no enclosure"
    return enclosure


@requires_handler
@pytest.mark.parametrize("feed", CORPUS_FEEDS, ids=lambda p: p.stem)
def test_the_strip_removes_the_transcript_and_nothing_else(feed: Path) -> None:
    original = feed.read_bytes()
    stripped = strip_podcast_transcripts(original)

    before, after = _items(original), _items(stripped)
    assert len(after) == len(before)
    assert all(
        item.find("podcast:transcript", NS) is not None for item in before
    ), f"{feed.name}: every corpus item is expected to offer a transcript to strip"
    assert all(item.find("podcast:transcript", NS) is None for item in after)
    for b, a in zip(before, after):
        assert ET.tostring(_enclosure(b)) == ET.tostring(_enclosure(a))
        assert b.findtext("guid") == a.findtext("guid")
        assert b.findtext("title") == a.findtext("title")


@requires_handler
def test_the_served_feed_is_audio_only_only_when_asked(e2e_server) -> None:
    allowed = E2EHTTPRequestHandler.get_allowed_podcasts()
    strip = E2EHTTPRequestHandler.get_strip_transcripts()
    try:
        E2EHTTPRequestHandler.set_allowed_podcasts(None)

        E2EHTTPRequestHandler.set_strip_transcripts(False)
        with urllib.request.urlopen(e2e_server.urls.feed("corpus_p13")) as resp:  # nosec B310
            offered = resp.read()

        E2EHTTPRequestHandler.set_strip_transcripts(True)
        with urllib.request.urlopen(e2e_server.urls.feed("corpus_p13")) as resp:  # nosec B310
            audio_only = resp.read()
            length = int(resp.headers["Content-Length"])
    finally:
        E2EHTTPRequestHandler.set_allowed_podcasts(allowed)
        E2EHTTPRequestHandler.set_strip_transcripts(strip)

    assert b"<podcast:transcript" in offered
    assert b"<podcast:transcript" not in audio_only
    assert length == len(audio_only)
    assert [_enclosure(i).get("url") for i in _items(audio_only)] == [
        _enclosure(i).get("url") for i in _items(offered)
    ]


@requires_handler
@pytest.mark.parametrize("feed", CORPUS_FEEDS, ids=lambda p: p.stem)
def test_the_language_strip_removes_the_language_and_nothing_else(feed: Path) -> None:
    original = feed.read_bytes()
    stripped = strip_feed_language(original)

    channel_before = ET.fromstring(original).find("channel")  # nosec B314 - own fixtures
    channel_after = ET.fromstring(stripped).find("channel")  # nosec B314 - own fixtures
    assert channel_before is not None and channel_after is not None
    assert channel_before.findtext("language"), f"{feed.name}: expected a <language> to strip"
    assert channel_after.find("language") is None
    assert [ET.tostring(i) for i in _items(stripped)] == [ET.tostring(i) for i in _items(original)]


@requires_handler
def test_both_strips_compose_on_the_served_feed(e2e_server) -> None:
    allowed = E2EHTTPRequestHandler.get_allowed_podcasts()
    strip_t = E2EHTTPRequestHandler.get_strip_transcripts()
    strip_l = E2EHTTPRequestHandler.get_strip_language()
    try:
        E2EHTTPRequestHandler.set_allowed_podcasts(None)
        E2EHTTPRequestHandler.set_strip_transcripts(True)
        E2EHTTPRequestHandler.set_strip_language(True)
        with urllib.request.urlopen(e2e_server.urls.feed("corpus_p10")) as resp:  # nosec B310
            body = resp.read()
            length = int(resp.headers["Content-Length"])
    finally:
        E2EHTTPRequestHandler.set_allowed_podcasts(allowed)
        E2EHTTPRequestHandler.set_strip_transcripts(strip_t)
        E2EHTTPRequestHandler.set_strip_language(strip_l)

    assert b"<language>" not in body
    assert b"<podcast:transcript" not in body
    assert length == len(body)
