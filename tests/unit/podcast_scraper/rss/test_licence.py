"""Licence and copyright declarations in a feed, recorded as evidence (#2325).

Each test is one requirement of the brief: namespace-aware parsing, channel and episode kept
apart, known licences with their conditions, unknown text kept verbatim, conflicts flagged, links
kept for review, "no explicit licence found" when there is nothing, and the audio/video scope of
``<podcast:license>`` never extended. Shapes from the production survey of 2026-10-11 (80 feeds:
79 ``<copyright>`` notices, three ``<podcast:license>`` holding a name, no Creative Commons).
"""

from __future__ import annotations

from xml.etree import ElementTree as ET

import pytest
from defusedxml.ElementTree import fromstring as safe_fromstring

from podcast_scraper.rss import licence as L

pytestmark = pytest.mark.unit

BASE = "https://example.com/feed.xml"


def _channel(inner: str, ns: str = "") -> ET.Element:
    xml = f"<rss version='2.0' {ns}><channel><title>T</title>{inner}</channel></rss>"
    channel: ET.Element | None = safe_fromstring(xml).find("channel")
    assert channel is not None
    return channel


P = "xmlns:podcast='https://podcastindex.org/namespace/1.0'"
CC = "xmlns:creativeCommons='http://backend.userland.com/creativeCommonsRssModule'"


def test_rss_copyright_is_kept_verbatim_and_is_not_a_licence() -> None:
    rec = L.channel_licence(
        _channel(
            "<copyright>© 2020-2021 THE NEW YORK TIMES COMPANY; The New York Times encourages "
            "the use of RSS feeds for personal use in a news reader.</copyright>"
        ),
        BASE,
    )
    (d,) = rec.declarations
    assert d.mechanism == "rss_copyright" and d.known is None
    assert d.raw.startswith("© 2020-2021 THE NEW YORK TIMES COMPANY;")
    assert d.raw.endswith("in a news reader.")
    assert rec.status == "declared"
    label = L.summary_label(rec)
    assert label is not None and label.startswith("© 2020-2021 THE NEW YORK TIMES")


def test_a_namespace_is_matched_by_uri_not_by_prefix() -> None:
    rec = L.channel_licence(
        _channel(
            "<p2:license url='https://creativecommons.org/licenses/by-nc/4.0/'>cc-by-nc-4.0"
            "</p2:license>",
            "xmlns:p2='https://podcastindex.org/namespace/1.0'",
        ),
        BASE,
    )
    (d,) = rec.declarations
    assert d.mechanism == "podcast_license"
    assert d.known is not None and d.known.id == "CC-BY-NC-4.0"


def test_a_known_licence_carries_its_conditions() -> None:
    rec = L.channel_licence(_channel("<podcast:license>cc-by-sa-4.0</podcast:license>", P), BASE)
    k = rec.declarations[0].known
    assert k is not None
    assert (k.commercial_use, k.adaptations, k.attribution, k.share_alike) == (
        True,
        True,
        True,
        True,
    )
    nd = L.recognise("https://creativecommons.org/licenses/by-nc-nd/3.0/")
    assert nd is not None
    assert (nd.commercial_use, nd.adaptations) == (False, False)
    zero = L.recognise("cc0-1.0")
    assert zero is not None
    assert zero.attribution is False


def test_a_name_in_podcast_license_is_kept_and_noted_not_guessed() -> None:
    """Disrupting Japan, Made In Africa, Africa Tech Summit: a holder's name, no url."""
    rec = L.channel_licence(_channel("<podcast:license>Tim Romero</podcast:license>", P), BASE)
    (d,) = rec.declarations
    assert d.raw == "Tim Romero" and d.known is None and d.url is None
    assert d.notes == ["custom licence without a url (the Podcasting 2.0 spec requires one)"]
    assert L.summary_label(rec) == "Tim Romero"


def test_podcast_license_speaks_for_audio_video_only() -> None:
    rec = L.channel_licence(
        _channel(
            "<copyright>Example Media</copyright><podcast:license>cc-by-4.0</podcast:license>", P
        ),
        BASE,
    )
    scopes = {d.mechanism: d.asset_scope for d in rec.declarations}
    assert scopes == {"rss_copyright": "unspecified", "podcast_license": "audio_video"}


def test_creative_commons_module_url() -> None:
    rec = L.channel_licence(
        _channel(
            "<creativeCommons:license>http://creativecommons.org/licenses/by-nd/2.0/"
            "</creativeCommons:license>",
            CC,
        ),
        BASE,
    )
    (d,) = rec.declarations
    assert d.mechanism == "creative_commons_license"
    assert d.known is not None and d.known.id == "CC-BY-ND-2.0"
    assert rec.documents_for_review == ["http://creativecommons.org/licenses/by-nd/2.0/"]


def test_nothing_declared_is_no_explicit_licence_found() -> None:
    rec = L.channel_licence(_channel("<description>A show about tea.</description>"), BASE)
    assert rec.status == "no_explicit_licence_found" and rec.declarations == []
    assert L.summary_label(rec) == "No explicit licence found"
    assert L.summary_label(None) is None  # not checked: no line at all


def test_conflicting_declarations_are_flagged_not_resolved() -> None:
    rec = L.channel_licence(
        _channel(
            "<copyright>All rights reserved</copyright>"
            "<podcast:license>cc-by-4.0</podcast:license>"
            "<creativeCommons:license>https://creativecommons.org/licenses/by-nc/4.0/"
            "</creativeCommons:license>",
            f"{P} {CC}",
        ),
        BASE,
    )
    assert len(rec.declarations) == 3
    assert "different licences declared: CC-BY-4.0, CC-BY-NC-4.0" in rec.conflicts
    assert any(c.startswith("'All rights reserved' alongside a licence") for c in rec.conflicts)


def test_notices_and_licence_links_in_the_description_are_kept_for_review() -> None:
    rec = L.channel_licence(
        _channel(
            "<description><![CDATA[A show. This podcast is licensed under a Creative Commons "
            "licence. <a href='/terms-of-use'>Terms</a> "
            "<a href='https://example.com/about'>About</a>]]></description>"
        ),
        BASE,
    )
    (n,) = rec.declarations
    assert n.mechanism == "notice" and "licensed under a Creative Commons" in n.raw
    assert rec.documents_for_review == ["https://example.com/terms-of-use"]


def test_an_episode_keeps_its_own_declarations_apart_from_the_channel() -> None:
    xml = (
        f"<rss {P}><channel><title>T</title><podcast:license>cc-by-4.0</podcast:license>"
        "<item><title>E1</title><podcast:license>cc-by-nc-4.0</podcast:license></item>"
        "<item><title>E2</title></item></channel></rss>"
    )
    channel: ET.Element | None = safe_fromstring(xml).find("channel")
    assert channel is not None
    chan = L.channel_licence(channel, BASE)
    first, second = channel.findall("item")
    ep1 = L.episode_licence(first, BASE, channel=chan)
    ep2 = L.episode_licence(second, BASE, channel=chan)
    assert [d.level for d in ep1.declarations] == ["episode"]
    assert ep1.conflicts == [
        "episode licence differs from the channel's: CC-BY-NC-4.0 vs CC-BY-4.0"
    ]
    assert ep2.status == "no_explicit_licence_found" and ep2.declarations == []


def test_copyright_is_a_channel_element_only() -> None:
    item = safe_fromstring("<item><title>E</title><copyright>Someone</copyright></item>")
    assert L.episode_licence(item, BASE).declarations == []


def test_a_record_names_its_source_and_read_time() -> None:
    rec = L.channel_licence_from_bytes(
        b"<rss><channel><title>T</title><copyright>X</copyright></channel></rss>",
        BASE,
        "https://feeds.example.com/x.rss",
    )
    assert rec is not None
    assert rec.source_feed_url == "https://feeds.example.com/x.rss"
    assert rec.retrieved_at and rec.retrieved_at.endswith("+00:00")


def test_long_notices_are_shortened_for_the_line_only() -> None:
    long = "Copyright " + "x" * 200
    rec = L.channel_licence(_channel(f"<copyright>{long}</copyright>"), BASE)
    assert rec.declarations[0].raw == long
    label = L.summary_label(rec)
    assert label is not None and len(label) == 90 and label.endswith("…")
