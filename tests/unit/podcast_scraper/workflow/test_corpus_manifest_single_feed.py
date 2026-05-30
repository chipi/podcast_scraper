"""Single-feed corpus_manifest stamping (#807)."""

from __future__ import annotations

import json
from pathlib import Path

from podcast_scraper.workflow.corpus_operations import (
    CORPUS_MANIFEST_FILE,
    MultiFeedFeedResult,
    upsert_corpus_manifest_feed,
    utc_iso_now,
)


def test_upsert_corpus_manifest_feed_writes_produced_by(tmp_path: Path) -> None:
    parent = tmp_path / "corpus"
    parent.mkdir()
    upsert_corpus_manifest_feed(
        str(parent),
        MultiFeedFeedResult(
            "https://example.com/feed.xml",
            True,
            None,
            3,
            finished_at=utc_iso_now(),
        ),
    )
    doc = json.loads((parent / CORPUS_MANIFEST_FILE).read_text(encoding="utf-8"))
    assert doc["produced_by"]["code_version"]
    assert doc["produced_by"]["git_sha"]
    assert len(doc["feeds"]) == 1
    assert doc["feeds"][0]["episodes_processed"] == 3


def test_upsert_preserves_other_feed_rows(tmp_path: Path) -> None:
    parent = tmp_path / "corpus"
    parent.mkdir()
    upsert_corpus_manifest_feed(
        str(parent),
        MultiFeedFeedResult("https://a.example/feed.xml", True, None, 1, finished_at=utc_iso_now()),
    )
    upsert_corpus_manifest_feed(
        str(parent),
        MultiFeedFeedResult("https://b.example/feed.xml", True, None, 2, finished_at=utc_iso_now()),
    )
    doc = json.loads((parent / CORPUS_MANIFEST_FILE).read_text(encoding="utf-8"))
    urls = {f["feed_url"] for f in doc["feeds"]}
    assert urls == {"https://a.example/feed.xml", "https://b.example/feed.xml"}
