"""Unit tests for corpus incident JSONL helper (GitHub #557)."""

import json
from pathlib import Path

import pytest

from podcast_scraper.utils.corpus_incidents import append_corpus_incident

pytestmark = [pytest.mark.unit, pytest.mark.module_utils]


def test_append_corpus_incident_writes_jsonl(tmp_path: Path) -> None:
    log_path = str(tmp_path / "corpus_incidents.jsonl")
    append_corpus_incident(
        log_path,
        scope="episode",
        category="policy",
        message="skip: too large",
        exception_type="PolicySkip",
        stage="transcription",
        feed_url="https://example.com/feed.xml",
        episode_id="ep-1",
        episode_idx=3,
    )
    append_corpus_incident(
        log_path,
        scope="feed",
        category="hard",
        message="boom",
        exception_type="RuntimeError",
        feed_url="https://example.com/other.xml",
    )
    lines = Path(log_path).read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    row0 = json.loads(lines[0])
    assert row0["scope"] == "episode"
    assert row0["category"] == "policy"
    assert row0["exception_type"] == "PolicySkip"
    assert row0["stage"] == "transcription"
    assert row0["feed_url"] == "https://example.com/feed.xml"
    assert row0["episode_id"] == "ep-1"
    assert row0["episode_idx"] == 3
    row1 = json.loads(lines[1])
    assert row1["scope"] == "feed"
    assert row1["category"] == "hard"
    assert "episode_id" not in row1


def test_append_corpus_incident_noop_when_path_empty() -> None:
    append_corpus_incident(
        "",
        scope="feed",
        category="soft",
        message="x",
        exception_type="ValueError",
    )
