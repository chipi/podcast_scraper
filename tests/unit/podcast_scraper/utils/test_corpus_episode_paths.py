"""Unit tests for multi-feed episode → GI/KG path resolution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.search.faiss_store import VECTORS_FILE
from podcast_scraper.utils.corpus_episode_paths import (
    corpus_search_parent_hint,
    list_artifact_paths_for_episode,
    pick_single_artifact_path,
)


def _write_meta(
    meta_dir: Path,
    base: str,
    *,
    feed_id: str,
    episode_id: str,
) -> None:
    meta_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "feed": {"feed_id": feed_id},
        "episode": {"episode_id": episode_id},
    }
    (meta_dir / f"{base}.metadata.json").write_text(
        json.dumps(doc),
        encoding="utf-8",
    )


@pytest.mark.unit
def test_list_gi_paths_multi_feed_disambiguate_feed_id(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    for fid, slug in (("feed_a", "rss_a"), ("feed_b", "rss_b")):
        meta = corpus / "feeds" / slug / "run_x" / "metadata"
        _write_meta(meta, "ep1", feed_id=fid, episode_id="same-ep")
        gi = meta / "ep1.gi.json"
        gi.write_text(
            json.dumps({"episode_id": "same-ep", "nodes": [], "edges": []}),
            encoding="utf-8",
        )

    all_paths = list_artifact_paths_for_episode(corpus, "same-ep", feed_id=None, kind="gi")
    assert len(all_paths) == 2

    only_a = list_artifact_paths_for_episode(corpus, "same-ep", feed_id="feed_a", kind="gi")
    assert len(only_a) == 1
    assert "rss_a" in str(only_a[0])

    assert pick_single_artifact_path(all_paths) is None
    assert pick_single_artifact_path(only_a) == only_a[0]


@pytest.mark.unit
def test_list_kg_paths_flat_layout(tmp_path: Path) -> None:
    meta = tmp_path / "metadata"
    _write_meta(meta, "x", feed_id="rss_host_h", episode_id="kg-ep")
    kg = meta / "x.kg.json"
    kg.write_text(
        json.dumps({"episode_id": "kg-ep", "nodes": [], "edges": []}),
        encoding="utf-8",
    )
    paths = list_artifact_paths_for_episode(tmp_path, "kg-ep", kind="kg")
    assert paths == [kg.resolve()]


@pytest.mark.unit
def test_corpus_search_parent_hint_detects_ancestor_index(tmp_path: Path) -> None:
    corpus = tmp_path / "mycorpus"
    (corpus / "search").mkdir(parents=True)
    (corpus / "search" / VECTORS_FILE).write_bytes(b"")
    feed_meta = corpus / "feeds" / "rss_x" / "metadata"
    feed_meta.mkdir(parents=True)

    hints = corpus_search_parent_hint(feed_meta)
    assert len(hints) == 1
    assert str(corpus.resolve()) in hints[0]

    assert corpus_search_parent_hint(corpus) == []


@pytest.mark.unit
def test_corpus_search_parent_hint_no_false_positive_outside_tree(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    (a / "search").mkdir(parents=True)
    (a / "search" / VECTORS_FILE).write_bytes(b"")
    (b / "nested").mkdir(parents=True)
    assert corpus_search_parent_hint(b / "nested") == []


@pytest.mark.unit
def test_list_gi_paths_from_yaml_metadata(tmp_path: Path) -> None:
    import yaml

    meta = tmp_path / "metadata"
    meta.mkdir(parents=True)
    doc = {"feed": {"feed_id": "feed_yaml"}, "episode": {"episode_id": "yaml-ep"}}
    (meta / "a.metadata.yaml").write_text(
        yaml.safe_dump(doc, sort_keys=False),
        encoding="utf-8",
    )
    gi = meta / "a.gi.json"
    gi.write_text(
        json.dumps({"episode_id": "yaml-ep", "nodes": [], "edges": []}),
        encoding="utf-8",
    )
    paths = list_artifact_paths_for_episode(tmp_path, "yaml-ep", kind="gi")
    assert paths == [gi.resolve()]


@pytest.mark.unit
def test_list_gi_paths_fallback_rglob_without_metadata_file(tmp_path: Path) -> None:
    """Orphan .gi.json (no sibling .metadata.*) still resolves via rglob fallback."""
    gi = tmp_path / "feeds" / "rss_z" / "run" / "metadata" / "only.gi.json"
    gi.parent.mkdir(parents=True)
    gi.write_text(
        json.dumps({"episode_id": "orphan-ep", "nodes": [], "edges": []}),
        encoding="utf-8",
    )
    paths = list_artifact_paths_for_episode(tmp_path, "orphan-ep", kind="gi")
    assert paths == [gi.resolve()]
