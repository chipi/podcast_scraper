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
    format_corpus_status_text,
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


@pytest.mark.unit
def test_collect_corpus_status_invalid_manifest_ignored(tmp_path: Path) -> None:
    corpus = tmp_path / "c"
    corpus.mkdir()
    (corpus / CORPUS_MANIFEST_FILE).write_text("{ not json", encoding="utf-8")
    st = collect_corpus_status(str(corpus))
    assert st["manifest_present"] is False


@pytest.mark.unit
def test_collect_corpus_status_search_meta_and_failed_index(tmp_path: Path) -> None:
    corpus = tmp_path / "c"
    (corpus / "search").mkdir(parents=True)
    (corpus / "search" / "vectors.faiss").write_bytes(b"")
    (corpus / "search" / "index_meta.json").write_text(
        json.dumps({"embedding_model": "m", "index_kind": "flat"}),
        encoding="utf-8",
    )
    feeds = corpus / "feeds" / "rss_x"
    (feeds / "metadata").mkdir(parents=True)
    (feeds / "index.json").write_text(
        json.dumps(
            {
                "episodes": [
                    {"status": "failed", "error_message": "transient boom"},
                ],
            },
        ),
        encoding="utf-8",
    )
    st = collect_corpus_status(str(corpus))
    assert st["search_index_present"] is True
    assert st["search_embedding_model"] == "m"
    assert st["search_index_kind"] == "flat"
    rows = st["feeds_subdirs"]
    assert len(rows) == 1
    assert rows[0]["index_failed_episodes"] == 1
    assert "boom" in (rows[0].get("sample_index_error") or "")


@pytest.mark.unit
def test_format_corpus_status_text_includes_optional_lines(tmp_path: Path) -> None:
    corpus = str(tmp_path / "c")
    Path(corpus).mkdir()
    st = collect_corpus_status(corpus)
    text = format_corpus_status_text(st)
    assert "Corpus parent:" in text
    assert "Per-feed directories:" in text


@pytest.mark.unit
def test_finalize_skips_index_when_vector_backend_not_faiss(tmp_path: Path) -> None:
    parent = str(tmp_path / "c")
    Path(parent).mkdir()
    cfg = cfg_mod.Config(
        rss="https://a/feed.xml",
        output_dir=parent,
        user_agent="t",
        timeout=30,
        vector_search=True,
        vector_backend="qdrant",
    )
    doc = finalize_multi_feed_batch(
        parent,
        cfg,
        [MultiFeedFeedResult("https://a/feed.xml", True, None, 1)],
    )
    assert doc.get("overall_ok") is True


@pytest.mark.unit
def test_finalize_returns_summary_when_feed_results_empty(tmp_path: Path) -> None:
    parent = str(tmp_path / "c")
    Path(parent).mkdir()
    cfg = cfg_mod.Config(
        rss="https://a/feed.xml",
        output_dir=parent,
        user_agent="t",
        timeout=30,
        vector_search=True,
    )
    doc = finalize_multi_feed_batch(parent, cfg, [])
    assert doc.get("feeds") == []
