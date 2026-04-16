"""Integration: run index catalog (episode manifest JSON) for multi-feed / observability."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest

from podcast_scraper.utils import filesystem
from podcast_scraper.workflow.run_index import (
    build_failure_summary,
    create_run_index,
    EpisodeIndexEntry,
    find_episode_metadata_relative_path,
    RunIndex,
)

pytestmark = [pytest.mark.integration, pytest.mark.module_workflow]


def _episode_xml() -> ET.Element:
    item = ET.Element("item")
    guid = ET.SubElement(item, "guid")
    guid.text = "urn:episode:test-1"
    return item


def test_build_failure_summary_groups_errors() -> None:
    idx = RunIndex(
        episodes=[
            EpisodeIndexEntry(episode_id="a", status="failed", error_type="Timeout"),
            EpisodeIndexEntry(episode_id="b", status="failed", error_type="Timeout"),
            EpisodeIndexEntry(episode_id="c", status="ok"),
        ],
    )
    summary = build_failure_summary(idx)
    assert summary["total_failed"] == 2
    assert summary["by_error_type"]["Timeout"] == 2
    assert "a" in summary["failed_episode_ids"]


def test_create_run_index_finds_metadata_and_transcript(tmp_path: Path) -> None:
    meta_dir = tmp_path / filesystem.METADATA_SUBDIR
    trx_dir = tmp_path / filesystem.TRANSCRIPTS_SUBDIR
    meta_dir.mkdir()
    trx_dir.mkdir()

    ep = SimpleNamespace(
        idx=0,
        title="Hello World",
        title_safe="Hello World",
        item=_episode_xml(),
        transcript_urls=[("https://example.com/t.vtt", "text/vtt")],
        transcript_url="https://example.com/t.vtt",
    )

    from podcast_scraper.utils.filesystem import truncate_whisper_title

    title_for = truncate_whisper_title(getattr(ep, "title_safe", ep.title), for_log=False)
    (meta_dir / f"0000 - {title_for}.metadata.json").write_text("{}", encoding="utf-8")
    (trx_dir / f"0000 - {title_for}.txt").write_text("transcript", encoding="utf-8")

    idx = create_run_index(
        "run-1",
        "https://feeds.example.com/show.xml",
        [ep],
        str(tmp_path),
    )
    assert idx.episodes_processed >= 1
    assert idx.episodes[0].status == "ok"
    assert idx.episodes[0].metadata_path is not None
    assert idx.episodes[0].transcript_path is not None
    data = json.loads(idx.to_json())
    assert data["schema_version"] in ("1.0.0", "1.1.0")
    assert data["run_id"] == "run-1"


def test_create_run_index_skipped_without_transcript_url(tmp_path: Path) -> None:
    meta_dir = tmp_path / filesystem.METADATA_SUBDIR
    meta_dir.mkdir()
    ep = SimpleNamespace(
        idx=0,
        title="Only",
        title_safe="Only",
        item=_episode_xml(),
        transcript_urls=[],
    )
    idx = create_run_index("r2", None, [ep], str(tmp_path))
    assert idx.episodes_skipped >= 1


def test_find_episode_metadata_relative_path_glob(tmp_path: Path) -> None:
    meta_dir = tmp_path / filesystem.METADATA_SUBDIR
    meta_dir.mkdir()
    ep = SimpleNamespace(
        idx=1,
        title="Title",
        title_safe="Title",
        item=_episode_xml(),
        transcript_urls=[],
    )
    from podcast_scraper.utils.filesystem import truncate_whisper_title

    stem = truncate_whisper_title("Title", for_log=False)
    (meta_dir / f"0001 - {stem}_suffix.metadata.json").write_text("{}", encoding="utf-8")

    rel = find_episode_metadata_relative_path(ep, str(tmp_path), run_suffix="suffix")
    assert rel is not None
    assert rel.endswith(".metadata.json")
