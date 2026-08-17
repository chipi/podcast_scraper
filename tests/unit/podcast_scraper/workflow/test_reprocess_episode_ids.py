"""Selecting a repair set by EXPLICIT episode list (#32).

WHY THIS EXISTS RATHER THAN REUSING --reprocess-source
Measured 2026-08-17 on a corpus carrying #18 unpreprocessed-audio damage: all 9 damaged episodes
had ``transcript_source: whisper_transcription`` — and so did all 6 healthy ones. Selecting by
source would have re-transcribed 6 healthy episodes to reach 9 damaged ones, at real ASR cost.

A detector that can only produce a LIST is useless without a selector that consumes one. That was
exactly the gap that made the placeholder gate a dead end until ``gi-repair`` existed.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pytest

from podcast_scraper import config
from podcast_scraper.workflow.episode_processor import _force_reprocess_for_source

pytestmark = [pytest.mark.unit]


class _Episode:
    def __init__(self, guid: str, episode_id: str | None = None) -> None:
        self.item = ET.fromstring(f"<item><guid>{guid}</guid></item>")
        self.guid = guid
        self.episode_id = episode_id or guid
        self.title = "An Episode"
        self.title_safe = "An Episode"
        self.idx = 1


def _cfg(root: Path, **kw: Any) -> config.Config:
    return config.Config(
        rss_url="https://example.com/feed.xml",
        output_dir=str(root),
        single_feed_uses_corpus_layout=True,
        **kw,
    )


def _corpus(cfg: config.Config, *, guid: str, episode_id: str, source: str) -> str:
    run = Path(str(cfg.output_dir)) / "run_20260815-120000"
    (run / "metadata").mkdir(parents=True, exist_ok=True)
    name = "0001 - An Episode"
    (run / "metadata" / f"{name}.metadata.json").write_text(
        json.dumps(
            {
                "episode": {"episode_id": episode_id, "guid": guid, "title": "An Episode"},
                "content": {"transcript_source": source},
            }
        ),
        encoding="utf-8",
    )
    fresh = Path(str(cfg.output_dir)) / "run_20260816-090000"
    (fresh / "metadata").mkdir(parents=True, exist_ok=True)
    return str(fresh)


def test_an_episode_on_the_list_is_forced(tmp_path):
    cfg = _cfg(tmp_path, reprocess_episode_ids=["ep-damaged"])
    fresh = _corpus(cfg, guid="ep-damaged", episode_id="ep-damaged", source="whisper_transcription")

    assert _force_reprocess_for_source(_Episode("ep-damaged"), fresh, None, cfg) is True


def test_an_episode_NOT_on_the_list_is_left_alone(tmp_path):
    """The whole point: a healthy episode sharing the same transcript_source must be untouched."""
    cfg = _cfg(tmp_path, reprocess_episode_ids=["ep-damaged"])
    fresh = _corpus(cfg, guid="ep-healthy", episode_id="ep-healthy", source="whisper_transcription")

    assert _force_reprocess_for_source(_Episode("ep-healthy"), fresh, None, cfg) is False


def test_matching_works_on_the_guid_when_the_list_holds_guids(tmp_path):
    """Detectors emit whatever the artifact carries — guid or episode_id. Matching only one of
    them makes an operator's list silently miss episodes."""
    cfg = _cfg(tmp_path, reprocess_episode_ids=["rss-guid-1"])
    fresh = _corpus(
        cfg, guid="rss-guid-1", episode_id="different-episode-id", source="direct_download"
    )

    assert (
        _force_reprocess_for_source(
            _Episode("rss-guid-1", "different-episode-id"), fresh, None, cfg
        )
        is True
    )


def test_matching_works_on_the_episode_id_when_the_list_holds_episode_ids(tmp_path):
    cfg = _cfg(tmp_path, reprocess_episode_ids=["substack:post:12345"])
    fresh = _corpus(
        cfg, guid="rss-guid-2", episode_id="substack:post:12345", source="direct_download"
    )

    assert (
        _force_reprocess_for_source(_Episode("rss-guid-2", "substack:post:12345"), fresh, None, cfg)
        is True
    )


def test_the_list_does_not_disturb_reprocess_source(tmp_path):
    """Both selectors coexist: an empty list must not suppress the #925 source match."""
    cfg = _cfg(tmp_path, reprocess_source="whisper_transcription")
    fresh = _corpus(cfg, guid="ep-1", episode_id="ep-1", source="whisper_transcription")

    assert _force_reprocess_for_source(_Episode("ep-1"), fresh, None, cfg) is True


def test_neither_selector_means_no_forcing(tmp_path):
    cfg = _cfg(tmp_path)
    fresh = _corpus(cfg, guid="ep-1", episode_id="ep-1", source="whisper_transcription")

    assert _force_reprocess_for_source(_Episode("ep-1"), fresh, None, cfg) is False


def test_the_list_forces_even_when_the_source_does_not_match(tmp_path):
    """The list is authoritative — it exists precisely because source cannot express the set."""
    cfg = _cfg(
        tmp_path,
        reprocess_episode_ids=["ep-1"],
        reprocess_source="whisper_transcription",
    )
    fresh = _corpus(cfg, guid="ep-1", episode_id="ep-1", source="direct_download")

    assert _force_reprocess_for_source(_Episode("ep-1"), fresh, None, cfg) is True
