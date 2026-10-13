"""``use_publisher_transcripts: false`` routes an episode to the audio, whatever the feed offers.

D6 (operator, 2026-10-13): non-English feeds go through our ASR, because that is the path the
language gates measured (publisher transcripts vary, and skip invented-line removal, gap recovery
and punctuation repair). It is a per-feed switch, not a language branch (D-39).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET  # nosec B405 - builds an element, parses nothing

import pytest

from podcast_scraper import config as cfg_mod
from podcast_scraper.models.entities import Episode
from podcast_scraper.workflow.episode_processor import publisher_transcript_choice

pytestmark = pytest.mark.unit

_VTT = ("https://pub.example/ep1.vtt", "text/vtt")


def _episode() -> Episode:
    return Episode(
        idx=0,
        title="ep1",
        title_safe="ep1",
        item=ET.Element("item"),
        transcript_urls=[_VTT],
        media_url="https://pub.example/ep1.mp3",
        media_type="audio/mpeg",
    )


def test_publisher_transcript_is_chosen_by_default() -> None:
    cfg = cfg_mod.Config(rss="https://feed.example/rss")
    assert publisher_transcript_choice(_episode(), cfg) == _VTT


def test_switch_off_sends_the_episode_to_the_audio() -> None:
    cfg = cfg_mod.Config(rss="https://feed.example/rss", use_publisher_transcripts=False)
    assert publisher_transcript_choice(_episode(), cfg) is None
