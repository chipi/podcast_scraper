"""Single-feed corpus_manifest stamping (#807)."""

from __future__ import annotations

import json
from pathlib import Path

from podcast_scraper import config
from podcast_scraper.workflow.corpus_operations import (
    _manifest_feed_rows_to_results,
    CORPUS_MANIFEST_FILE,
    corpus_parent_for_manifest_stamp_from_cfg,
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


def test_corpus_parent_for_manifest_stamp_single_feed_layout(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    (corpus / "feeds").mkdir(parents=True)
    cfg = config.Config.model_validate(
        {
            "rss_url": "https://example.com/feed.xml",
            "output_dir": str(corpus),
            "single_feed_uses_corpus_layout": True,
            "openai_api_key": "sk-test",
        }
    )
    assert corpus_parent_for_manifest_stamp_from_cfg(cfg) == str(corpus.resolve())


def test_corpus_parent_for_manifest_stamp_feeds_dir(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    (corpus / "feeds").mkdir(parents=True)
    cfg = config.Config.model_validate(
        {
            "rss_url": "https://example.com/feed.xml",
            "output_dir": str(corpus),
            "openai_api_key": "sk-test",
        }
    )
    assert corpus_parent_for_manifest_stamp_from_cfg(cfg) == str(corpus.resolve())


def test_manifest_feed_rows_to_results_parses_failure_kind() -> None:
    rows = [
        {
            "feed_url": "https://x.example/f.xml",
            "ok": False,
            "failure_kind": "soft",
            "episodes_processed": 1,
        },
        "not-a-dict",
        {"feed_url": "", "ok": True},
    ]
    out = _manifest_feed_rows_to_results(rows)
    assert len(out) == 1
    assert out[0].failure_kind == "soft"
