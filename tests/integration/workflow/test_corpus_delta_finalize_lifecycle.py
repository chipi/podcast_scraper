"""Manifest-advance rules through the REAL multi-feed finalize (RFC-118).

The unit tier proves the delta math; this proves the finalize WIRING against the
Tier-3 synthetic corpus: when the manifest advances, when it must not (index or
cluster failure — review M2), and that an unchanged corpus yields an empty delta
whose cluster build is skipped outright.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from podcast_scraper import config
from podcast_scraper.corpus_delta import load_fingerprint_manifest, manifest_path
from podcast_scraper.workflow.corpus_operations import (
    finalize_multi_feed_batch,
    MultiFeedFeedResult,
)

pytestmark = pytest.mark.integration

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "app-validation-corpus" / "v3"


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    dest = tmp_path / "corpus"
    shutil.copytree(FIXTURE, dest)
    return dest


def _cfg() -> config.Config:
    return config.Config.model_validate(
        {"rss_url": "https://example.com/feed.xml", "vector_search": True, "run_id": "t"}
    )


def _results() -> list[MultiFeedFeedResult]:
    return [
        MultiFeedFeedResult(
            feed_url="https://example.com/feed.xml", ok=True, error=None, episodes_processed=1
        )
    ]


def _finalize(corpus: Path, *, index_ok: bool = True, clusters_raise: bool = False):
    with (
        patch(
            "podcast_scraper.search.reindex.run_index_in_subprocess", return_value=index_ok
        ) as idx,
        patch(
            "podcast_scraper.search.topic_clusters.build_topic_clusters_for_corpus",
            side_effect=(RuntimeError("boom") if clusters_raise else None),
            return_value={"cluster_count": 1, "topic_count": 1, "singletons": 0},
        ) as clusters,
    ):
        finalize_multi_feed_batch(str(corpus), _cfg(), _results())
    return idx, clusters


def test_successful_finalize_advances_the_manifest(corpus: Path) -> None:
    assert not manifest_path(corpus).is_file()
    idx, _clusters = _finalize(corpus)
    assert idx.called
    manifest = load_fingerprint_manifest(corpus)
    assert manifest, "manifest not written after a successful finalize"
    # The backbone scope crossed into the index call.
    assert idx.call_args.kwargs.get("backbone_changed_relpaths"), "delta scope not passed"


def test_failed_index_leaves_the_manifest_untouched(corpus: Path) -> None:
    _finalize(corpus, index_ok=False)
    assert not manifest_path(corpus).is_file(), "manifest advanced past a failed index"


def test_failed_clusters_leave_the_manifest_untouched(corpus: Path) -> None:
    # Review M2: advancing past a failed cluster build would make the empty-delta
    # skip-gate reuse a stale clusters artifact forever.
    _finalize(corpus, clusters_raise=True)
    assert not manifest_path(corpus).is_file(), "manifest advanced past failed clusters"


def test_second_finalize_sees_empty_delta_and_skips_clusters(corpus: Path) -> None:
    _finalize(corpus)
    assert load_fingerprint_manifest(corpus)

    # Make the empty-delta skip-gate satisfiable: current clusters artifact + matching
    # index meta (the model guard) — the gate reads both files directly.
    search_dir = corpus / "search"
    lance_dir = search_dir / "lance_index"
    lance_dir.mkdir(parents=True, exist_ok=True)
    (lance_dir / "data").mkdir(exist_ok=True)
    (search_dir / "topic_clusters.json").write_text(
        '{"model": "m", "cluster_count": 1, "topic_count": 1, "singletons": 0}',
        encoding="utf-8",
    )
    (lance_dir / "index_meta.json").write_text(
        '{"embedding_model": "m", "embed_dim": 16}', encoding="utf-8"
    )

    idx, clusters = _finalize(corpus)
    # Index still runs (its own fingerprint skip is the cheap inner gate) with an
    # EMPTY backbone scope; the cluster BUILD is skipped outright.
    assert idx.call_args.kwargs.get("backbone_changed_relpaths") == []
    assert not clusters.called, "cluster build ran despite an empty delta + current artifact"
