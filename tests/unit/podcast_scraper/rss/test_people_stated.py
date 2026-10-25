"""What an RSS item states about who is on the episode (synthetic items)."""

from __future__ import annotations

import xml.etree.ElementTree as ET  # nosec B405

import pytest
from defusedxml.ElementTree import fromstring as safe_fromstring

from podcast_scraper.rss.parser import extract_episode_people_stated

pytestmark = [pytest.mark.unit]

NS = (
    'xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd" '
    'xmlns:dc="http://purl.org/dc/elements/1.1/" '
    'xmlns:podcast="https://podcastindex.org/namespace/1.0"'
)


def _item(body: str) -> ET.Element:
    item: ET.Element | None = safe_fromstring(
        f"<rss {NS}><channel><item>{body}</item></channel></rss>"
    ).find("channel/item")
    assert item is not None
    return item


def test_an_item_that_states_nobody_gives_nothing() -> None:
    assert extract_episode_people_stated(_item("<title>x</title>")) == []


def test_a_byline_is_split_into_people_with_no_role() -> None:
    got = extract_episode_people_stated(
        _item("<itunes:author>Ada Wren, Tobias Holt &amp; Mira Lane</itunes:author>")
    )
    assert [(p["name"], p["source"], p["role"]) for p in got] == [
        ("Ada Wren", "itunes_author", None),
        ("Tobias Holt", "itunes_author", None),
        ("Mira Lane", "itunes_author", None),
    ]
    assert got[0]["raw"] == "Ada Wren, Tobias Holt & Mira Lane"


def test_a_show_prefix_and_an_rss_author_email_are_split_off() -> None:
    got = extract_episode_people_stated(
        _item(
            "<author>show@example.org (Ada Wren)</author>"
            "<itunes:author>Talks: Ada Wren</itunes:author>"
        )
    )
    assert [(p["name"], p["source"]) for p in got] == [
        ("Ada Wren", "author"),
        ("Talks", "itunes_author"),
        ("Ada Wren", "itunes_author"),
    ]


def test_podcast_person_keeps_its_role_and_defaults_to_host() -> None:
    got = extract_episode_people_stated(
        _item(
            '<podcast:person role="Guest">Mira Lane</podcast:person>'
            "<podcast:person>Ada Wren</podcast:person>"
        )
    )
    assert [(p["name"], p["source"], p["role"]) for p in got] == [
        ("Mira Lane", "podcast_person", "guest"),
        ("Ada Wren", "podcast_person", "host"),
    ]


def test_the_episode_metadata_stores_what_the_item_states() -> None:
    from podcast_scraper.workflow.metadata_generation import EpisodeMetadata

    people = extract_episode_people_stated(
        _item('<podcast:person role="host">Ada Wren</podcast:person>')
    )
    doc = EpisodeMetadata(title="T", episode_id="e1", people_stated=people).model_dump(mode="json")
    assert doc["people_stated"] == [
        {"name": "Ada Wren", "source": "podcast_person", "role": "host", "raw": "Ada Wren"}
    ]
    assert (
        EpisodeMetadata(title="T", episode_id="e1").model_dump(mode="json")["people_stated"] is None
    )
