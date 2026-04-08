"""Unit tests for corpus manifest, summary, and status (GitHub #506)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper import config as cfg_mod
from podcast_scraper.workflow.corpus_operations import (
    collect_corpus_status,
    CORPUS_MANIFEST_FILE,
    CORPUS_RUN_SUMMARY_FILE,
    finalize_multi_feed_batch,
    MultiFeedFeedResult,
    utc_iso_now,
    write_corpus_manifest,
    write_corpus_run_summary,
)


@pytest.mark.unit
def test_write_corpus_manifest_and_summary(tmp_path: Path) -> None:
    parent = str(tmp_path / "c")
    Path(parent).mkdir()
    feeds = [
        MultiFeedFeedResult("https://a/feed", True, None, 2),
        MultiFeedFeedResult("https://b/feed", False, "boom", 0),
    ]
    write_corpus_manifest(parent, feeds)
    write_corpus_run_summary(parent, feeds, overall_ok=False)
    mf = Path(parent) / CORPUS_MANIFEST_FILE
    sf = Path(parent) / CORPUS_RUN_SUMMARY_FILE
    assert mf.is_file() and sf.is_file()
    md = json.loads(mf.read_text(encoding="utf-8"))
    assert md.get("schema_version") == "1.0.0"
    assert len(md.get("feeds") or []) == 2
    sd = json.loads(sf.read_text(encoding="utf-8"))
    assert sd.get("overall_ok") is False


@pytest.mark.unit
def test_per_feed_finished_at_in_manifest_and_summary(tmp_path: Path) -> None:
    parent = str(tmp_path / "c")
    Path(parent).mkdir()
    ts = "2026-04-08T12:00:00Z"
    feeds = [
        MultiFeedFeedResult("https://a/feed", True, None, 2, finished_at=ts),
        MultiFeedFeedResult("https://b/feed", False, "boom", 0, finished_at=ts),
    ]
    write_corpus_manifest(parent, feeds)
    write_corpus_run_summary(parent, feeds, overall_ok=False)
    md = json.loads((Path(parent) / CORPUS_MANIFEST_FILE).read_text(encoding="utf-8"))
    for row in md.get("feeds") or []:
        assert row.get("last_run_finished_at") == ts
    sd = json.loads((Path(parent) / CORPUS_RUN_SUMMARY_FILE).read_text(encoding="utf-8"))
    for row in sd.get("feeds") or []:
        assert row.get("finished_at") == ts


@pytest.mark.unit
def test_collect_corpus_status_multi_feed_tree(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    feeds = corpus / "feeds"
    (feeds / "rss_a_1" / "metadata").mkdir(parents=True)
    (feeds / "rss_a_1" / "metadata" / "x.metadata.json").write_text("{}", encoding="utf-8")
    (feeds / "rss_b_1" / "metadata").mkdir(parents=True)
    write_corpus_manifest(
        str(corpus),
        [MultiFeedFeedResult("https://a", True, None, 1)],
    )
    st = collect_corpus_status(str(corpus))
    assert st["manifest_present"] is True
    assert len(st["feeds_subdirs"]) == 2
    assert sum(int(r["metadata_files"]) for r in st["feeds_subdirs"]) >= 1


@pytest.mark.unit
def test_finalize_multi_feed_batch_skips_index_without_vector_search(tmp_path: Path) -> None:
    parent = str(tmp_path / "c")
    Path(parent).mkdir()
    cfg = cfg_mod.Config(
        rss="https://a/feed.xml",
        output_dir=parent,
        user_agent="t",
        timeout=30,
        vector_search=False,
    )
    doc = finalize_multi_feed_batch(
        parent,
        cfg,
        [MultiFeedFeedResult("https://a/feed.xml", True, None, 1, finished_at=utc_iso_now())],
    )
    assert doc.get("schema_version") == "1.0.0"
    assert len(doc.get("feeds") or []) == 1
    assert not (Path(parent) / "search" / "vectors.faiss").is_file()
