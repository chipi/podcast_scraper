"""RFC-118 outer skip-gate for the finalize topic-clusters build.

An EMPTY corpus delta with a current clusters artifact skips the build without
loading the index; every guard failure (missing artifact, model mismatch,
unreadable files, non-empty delta) falls through to a normal build.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from podcast_scraper.corpus_delta import CorpusDelta
from podcast_scraper.workflow import orchestration

pytestmark = pytest.mark.unit

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _search_dir(tmp_path: Path, *, clusters: bool = True, meta_model: str | None = MODEL):
    index_dir = tmp_path / "search"
    lance_dir = index_dir / "lance_index"
    lance_dir.mkdir(parents=True)
    (lance_dir / "data").mkdir()  # non-empty, so the finalize lance-dir check passes
    if clusters:
        (index_dir / "topic_clusters.json").write_text(
            json.dumps({"model": MODEL, "cluster_count": 3, "topic_count": 9, "singletons": 2}),
            encoding="utf-8",
        )
    if meta_model is not None:
        (lance_dir / "index_meta.json").write_text(
            json.dumps({"embedding_model": meta_model, "embed_dim": 384}), encoding="utf-8"
        )
    return index_dir, lance_dir


def _metrics():
    return SimpleNamespace(
        topic_clusters_built=False,
        topic_clusters_skipped_delta_empty=False,
        topic_cluster_count=0,
        topic_cluster_topic_count=0,
        topic_cluster_singletons=0,
        topic_cluster_seconds=0.0,
    )


def _empty_delta() -> CorpusDelta:
    return CorpusDelta(changed_ids=frozenset(), removed_ids=frozenset(), all_bundles=[])


class TestSkipHelper:
    def test_skips_and_reports_existing_counts(self, tmp_path):
        index_dir, lance_dir = _search_dir(tmp_path)
        m = _metrics()
        assert orchestration._skip_topic_clusters_on_empty_delta(index_dir, lance_dir, m)
        assert m.topic_clusters_built is True
        assert m.topic_clusters_skipped_delta_empty is True
        assert (m.topic_cluster_count, m.topic_cluster_topic_count) == (3, 9)

    def test_no_artifact_builds_normally(self, tmp_path):
        index_dir, lance_dir = _search_dir(tmp_path, clusters=False)
        assert not orchestration._skip_topic_clusters_on_empty_delta(
            index_dir, lance_dir, _metrics()
        )

    def test_model_mismatch_builds_normally(self, tmp_path):
        # An embedding-model change re-embeds without touching gi/kg — the one case an
        # empty delta cannot see. The gate must fall through to a rebuild.
        index_dir, lance_dir = _search_dir(tmp_path, meta_model="some/other-model")
        assert not orchestration._skip_topic_clusters_on_empty_delta(
            index_dir, lance_dir, _metrics()
        )

    def test_missing_index_meta_builds_normally(self, tmp_path):
        index_dir, lance_dir = _search_dir(tmp_path, meta_model=None)
        assert not orchestration._skip_topic_clusters_on_empty_delta(
            index_dir, lance_dir, _metrics()
        )

    def test_metrics_optional(self, tmp_path):
        # The multi-feed finalize has no pipeline_metrics object.
        index_dir, lance_dir = _search_dir(tmp_path)
        assert orchestration._skip_topic_clusters_on_empty_delta(index_dir, lance_dir, None)


class TestFinalizeClusterGate:
    def test_empty_delta_skips_the_build(self, tmp_path):
        _search_dir(tmp_path)
        m = _metrics()
        with patch(
            "podcast_scraper.search.topic_clusters.build_topic_clusters_for_corpus"
        ) as build:
            orchestration._maybe_build_topic_clusters_after_index(
                str(tmp_path), m, delta=_empty_delta()
            )
        build.assert_not_called()
        assert m.topic_clusters_skipped_delta_empty is True

    def test_nonempty_delta_builds(self, tmp_path):
        _search_dir(tmp_path)
        m = _metrics()
        delta = CorpusDelta(changed_ids=frozenset({"ep1"}), removed_ids=frozenset(), all_bundles=[])
        with patch(
            "podcast_scraper.search.topic_clusters.build_topic_clusters_for_corpus",
            return_value={"cluster_count": 1, "topic_count": 1, "singletons": 0},
        ) as build:
            orchestration._maybe_build_topic_clusters_after_index(str(tmp_path), m, delta=delta)
        build.assert_called_once()
        assert m.topic_clusters_skipped_delta_empty is False

    def test_no_delta_builds(self, tmp_path):
        _search_dir(tmp_path)
        m = _metrics()
        with patch(
            "podcast_scraper.search.topic_clusters.build_topic_clusters_for_corpus",
            return_value={"cluster_count": 1, "topic_count": 1, "singletons": 0},
        ) as build:
            orchestration._maybe_build_topic_clusters_after_index(str(tmp_path), m, delta=None)
        build.assert_called_once()
