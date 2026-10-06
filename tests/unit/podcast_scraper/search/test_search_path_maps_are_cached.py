"""The two corpus-walk maps a search reads are built once per corpus generation, not per search.

Prod 2026-10-06: each search walked the whole corpus twice (episode -> gi.json, scope ->
metadata path). That was most of a search's ~4 s, and beside the cache warmer's CPU-bound
entity-id map after a restart the walks stalled on the GIL: searches took 186-195 s and the
player post-deploy smoke failed on 504s. Cached, a search took ~0.4 s.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from podcast_scraper import perf_cache
from podcast_scraper.search import corpus_search

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _fresh_cache():
    perf_cache.clear()
    yield
    perf_cache.clear()


@pytest.fixture
def counted(monkeypatch):
    calls = {"gi": 0, "rel": 0}

    def gi(output_dir):
        calls["gi"] += 1
        return {"ep1": Path(output_dir) / "ep1.gi.json"}

    def rel(output_dir):
        calls["rel"] += 1
        return {"scope": "feeds/x/run_1/metadata/ep1.metadata.json"}

    monkeypatch.setattr(corpus_search, "merged_episode_gi_paths", gi)
    monkeypatch.setattr(corpus_search, "_metadata_relpath_by_scope_from_corpus", rel)
    return calls


def _new_generation(root: Path, stamp: float) -> None:
    summary = root / "corpus_run_summary.json"
    summary.write_text("{}", encoding="utf-8")
    os.utime(summary, (stamp, stamp))


def test_each_map_is_built_once_per_corpus_generation(tmp_path, counted) -> None:
    _new_generation(tmp_path, 1_000_000)
    for _ in range(3):
        corpus_search.cached_episode_gi_paths(tmp_path)
        corpus_search.cached_metadata_relpath_by_scope(tmp_path)
    assert counted == {"gi": 1, "rel": 1}


def test_an_ingest_rebuilds_them(tmp_path, counted) -> None:
    _new_generation(tmp_path, 1_000_000)
    corpus_search.cached_episode_gi_paths(tmp_path)
    corpus_search.cached_metadata_relpath_by_scope(tmp_path)
    _new_generation(tmp_path, 1_000_500)
    corpus_search.cached_episode_gi_paths(tmp_path)
    corpus_search.cached_metadata_relpath_by_scope(tmp_path)
    assert counted == {"gi": 2, "rel": 2}


def test_two_searches_walk_the_corpus_once(tmp_path, counted) -> None:
    _new_generation(tmp_path, 1_000_000)
    for _ in range(2):
        corpus_search._filter_and_enrich(
            [],
            tmp_path,
            types_norm=None,
            feed=None,
            since=None,
            speaker=None,
            topic=None,
            episode_id=None,
            grounded_only=False,
            top_k=5,
            dedupe_kg_surfaces=True,
            collect_cap=5,
        )
    assert counted == {"gi": 1, "rel": 1}
