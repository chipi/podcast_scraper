"""The ``corpus_indexed`` stage hook point (Common #49): once the corpus search index is built,
by the multi-feed batch or by ``index --with-clusters``, an owner can rebuild what it derives
from the index (the insight clusters)."""

from __future__ import annotations

import logging
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Tuple
from unittest.mock import patch

import pytest

from podcast_scraper import builtin_stage_hooks, stage_hooks
from podcast_scraper.extensions import Extension, use_extensions
from podcast_scraper.search import cli_handlers
from podcast_scraper.stage_hooks import PipelineHooks
from podcast_scraper.workflow import corpus_operations
from podcast_scraper.workflow.corpus_operations import MultiFeedFeedResult

pytestmark = pytest.mark.unit


def _recorder(calls: List[Tuple[str, Path]], who: str) -> PipelineHooks:
    return PipelineHooks(name=who, corpus_indexed=(lambda root: calls.append((who, root)),))


def test_corpus_indexed_runs_the_built_in_hooks_then_each_extensions(tmp_path: Path) -> None:
    calls: List[Tuple[str, Path]] = []
    with (
        patch.object(builtin_stage_hooks, "BUILTIN", PipelineHooks(name="builtin")),
        use_extensions(
            [
                Extension(name="a", pipeline_hooks=_recorder(calls, "a")),
                Extension(name="b", pipeline_hooks=_recorder(calls, "b")),
            ]
        ),
    ):
        stage_hooks.corpus_indexed(tmp_path)
    assert calls == [("a", tmp_path), ("b", tmp_path)]


class _Cfg(SimpleNamespace):
    def model_copy(self, update: Any) -> "_Cfg":
        return _Cfg(**{**vars(self), **update})


def _finalize(corpus: Path, *, index_ok: bool) -> List[Path]:
    seen: List[Path] = []
    cfg = _Cfg(vector_search=True, rss_url="https://feeds.example/a.xml")
    result = MultiFeedFeedResult(
        feed_url="https://feeds.example/a.xml",
        ok=True,
        error=None,
        episodes_processed=1,
        finished_at="2026-11-25T12:00:00Z",
    )
    with (
        patch("podcast_scraper.workflow.orchestration._maybe_spawn_enrichment_after_pipeline"),
        patch("podcast_scraper.corpus_delta.compute_corpus_delta", side_effect=OSError("no")),
        patch("podcast_scraper.search.reindex.run_index_in_subprocess", return_value=index_ok),
        patch("podcast_scraper.search.groupings.build_topic_clusters_for_corpus"),
        patch.object(stage_hooks, "corpus_indexed", side_effect=seen.append),
    ):
        corpus_operations.finalize_multi_feed_batch(str(corpus), cfg, [result])
    return seen


def test_the_batch_runs_corpus_indexed_once_its_index_is_built(tmp_path: Path) -> None:
    assert _finalize(tmp_path, index_ok=True) == [tmp_path]


def test_the_batch_skips_corpus_indexed_when_its_index_did_not_complete(tmp_path: Path) -> None:
    assert _finalize(tmp_path, index_ok=False) == []


@pytest.mark.parametrize("with_clusters, want", [(True, 1), (False, 0)])
def test_index_with_clusters_runs_corpus_indexed(
    tmp_path: Path, with_clusters: bool, want: int
) -> None:
    seen: List[Path] = []
    stats = SimpleNamespace(
        errors=[],
        episodes_scanned=0,
        episodes_skipped_unchanged=0,
        episodes_reindexed=0,
        vectors_upserted=0,
    )
    args = Namespace(output_dir=str(tmp_path), with_clusters=with_clusters)
    with (
        patch.object(cli_handlers, "index_corpus", return_value=stats),
        patch("podcast_scraper.search.groupings.build_topic_clusters_for_corpus"),
        patch.object(stage_hooks, "corpus_indexed", side_effect=seen.append),
    ):
        assert cli_handlers.run_index_cli(args, logging.getLogger("t")) == cli_handlers.EXIT_SUCCESS
    assert seen == [Path(tmp_path)] * want
