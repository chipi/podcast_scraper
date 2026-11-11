"""Corpus walks leave ``sources/`` (other content types, RFC-134 §5) out unless they opt in.

A reader that assumes podcast episodes stays podcast-only without having to remember to filter;
the tools that must see every run (run telemetry, the cost rollup) opt in.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Iterable

import pytest

from podcast_scraper.search.corpus_scope import discover_all_metadata_files, discover_metadata_files
from podcast_scraper.server.pipeline_run_prometheus import discover_run_json_paths_in_mtime_window
from podcast_scraper.utils.corpus_walk import (
    corpus_files_with_suffixes,
    corpus_relpath_is_other_type,
    corpus_rglob,
    prune_corpus_dirs,
)
from podcast_scraper.workflow.corpus_cost_aggregation import aggregate_corpus_costs

pytestmark = pytest.mark.unit


def _run(run_dir: Path, item: str, cost: float) -> None:
    meta = run_dir / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / f"{item}.metadata.json").write_text(
        json.dumps({"feed": {"feed_id": item}, "episode": {"episode_id": item}}), encoding="utf-8"
    )
    (meta / f"{item}.kg.json").write_text("{}", encoding="utf-8")
    (run_dir / "run.json").write_text("{}", encoding="utf-8")
    (run_dir / "metrics.json").write_text(
        json.dumps({"llm_summarization_cost_usd": cost}), encoding="utf-8"
    )


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    _run(tmp_path / "feeds" / "pod" / "run_20261101-000000", "ep", 1.0)
    # A podcast feed whose slug happens to be "sources" is still a podcast.
    _run(tmp_path / "feeds" / "sources" / "run_20261101-000000", "ep2", 0.5)
    _run(tmp_path / "sources" / "test_text_item" / "texts" / "run_20261101-000000", "t1", 2.0)
    return tmp_path


def _names(paths: Iterable[Path]) -> set[str]:
    return {p.name for p in paths}


def test_rglob_skips_sources_unless_asked(corpus: Path) -> None:
    assert _names(corpus_rglob(corpus, "*.kg.json")) == {"ep.kg.json", "ep2.kg.json"}
    assert _names(corpus_rglob(corpus, "*.kg.json", include_sources=True)) == {
        "ep.kg.json",
        "ep2.kg.json",
        "t1.kg.json",
    }


def test_suffix_walk_skips_sources_unless_asked(corpus: Path) -> None:
    assert _names(corpus_files_with_suffixes(corpus, [".kg.json"])) == {
        "ep.kg.json",
        "ep2.kg.json",
    }
    assert "t1.kg.json" in _names(
        corpus_files_with_suffixes(corpus, [".kg.json"], include_sources=True)
    )


def test_prune_drops_sources_only_at_the_root(tmp_path: Path) -> None:
    at_root = ["feeds", "sources", ".trash"]
    prune_corpus_dirs(tmp_path, tmp_path, at_root)
    assert at_root == ["feeds"]
    deeper = ["sources", ".viewer"]
    prune_corpus_dirs(tmp_path, tmp_path / "feeds", deeper)
    assert deeper == ["sources"]
    opted_in = ["feeds", "sources"]
    prune_corpus_dirs(tmp_path, tmp_path, opted_in, include_sources=True)
    assert opted_in == ["feeds", "sources"]


def test_relpath_rule_reads_the_first_segment() -> None:
    assert corpus_relpath_is_other_type("sources/news_article/x/run_1/metadata/a.json")
    assert corpus_relpath_is_other_type("./sources")
    assert not corpus_relpath_is_other_type("feeds/sources/run_1/metadata/a.json")
    assert not corpus_relpath_is_other_type("")


def test_membership_is_podcast_only(corpus: Path) -> None:
    assert _names(discover_metadata_files(corpus)) == {"ep.metadata.json", "ep2.metadata.json"}
    assert _names(discover_all_metadata_files(corpus)) == {
        "ep.metadata.json",
        "ep2.metadata.json",
    }


def test_membership_asked_for_sources_keeps_the_newest_run_per_item(corpus: Path) -> None:
    source = corpus / "sources" / "test_text_item" / "texts"
    _run(source / "run_20261102-000000", "t1", 2.0)  # t1 again, in a newer run
    rel = {p.relative_to(corpus).as_posix() for p in discover_metadata_files(corpus)}
    assert not any(r.startswith("sources/") for r in rel)
    asked = {
        p.relative_to(corpus).as_posix()
        for p in discover_metadata_files(corpus, include_sources=True)
    }
    assert asked - rel == {
        "sources/test_text_item/texts/run_20261102-000000/metadata/t1.metadata.json"
    }
    every = {
        p.relative_to(corpus).as_posix()
        for p in discover_all_metadata_files(corpus, include_sources=True)
    }
    assert {r for r in every if r.startswith("sources/")} == {
        "sources/test_text_item/texts/run_20261101-000000/metadata/t1.metadata.json",
        "sources/test_text_item/texts/run_20261102-000000/metadata/t1.metadata.json",
    }


def test_the_source_layout_is_recognised() -> None:
    from podcast_scraper.search.corpus_scope import (
        classify_metadata_relpath,
        LAYOUT_SOURCE_RUN,
        source_dir_and_run_segment_from_relpath,
    )

    rel = "sources/news_article/bbc/run_20261101-000000/metadata/a.metadata.json"
    assert classify_metadata_relpath(rel) == LAYOUT_SOURCE_RUN
    assert source_dir_and_run_segment_from_relpath(rel) == (
        "sources/news_article/bbc",
        "run_20261101-000000",
    )
    assert source_dir_and_run_segment_from_relpath("sources/x/run_1/metadata/a.json") == (
        None,
        None,
    )


def test_run_telemetry_sees_every_run(corpus: Path) -> None:
    now = time.time()
    for p in corpus.rglob("run.json"):
        os.utime(p, (now, now))
    found = discover_run_json_paths_in_mtime_window(corpus, now - 60, now + 60)
    assert len(found) == 3


def test_cost_rollup_counts_every_run(corpus: Path) -> None:
    result = aggregate_corpus_costs(corpus)
    assert result["run_count"] == 3
    assert result["total_cost_usd"] == pytest.approx(3.5)
