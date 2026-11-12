"""One feed's step of a multi-feed batch leaves the corpus-root work to the batch.

THE DEFECT (prod, every nightly through 2026-11-07). Each feed's ``_finalize_pipeline`` resolves
its finalize dir to its own run dir (``<corpus>/feeds/<slug>/run_<id>/``) and then:

- enqueued a ``corpus_enrichment`` job into that run dir's ``.viewer/jobs.jsonl``, a registry the
  API never drains: 80 orphaned rows and 80 "the API server promotes it" lines a night, next to
  the one real enqueue ``finalize_multi_feed_batch`` makes at the corpus root;
- wrote ``corpus_edges_stamp.json`` and an ``identity_ledger.json`` built from that run's episodes
  alone into the run dir, where nothing reads them (the 11-07 nightly logged "identity ledger
  refreshed after enrich-edges: 0 joins" per touched feed);
- computed a corpus delta and wrote a fingerprint manifest for the run dir, for an index and
  clusters the batch builds at the corpus root instead.

The edge pass itself stays per feed: it is per-episode and scoped to the run's own episodes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from podcast_scraper import config
from podcast_scraper.workflow import metrics, orchestration
from podcast_scraper.workflow.corpus_operations import (
    is_multi_feed_batch_child,
    MultiFeedFeedResult,
)

pytestmark = pytest.mark.unit

_LOG = logging.getLogger("test-multi-feed-child")


def _cfg(output_dir: Path, *, corpus_layout: bool = False) -> config.Config:
    return config.Config.model_validate(
        {
            "rss_url": "https://example.com/feed.xml",
            "output_dir": str(output_dir),
            "single_feed_uses_corpus_layout": corpus_layout,
            "vector_search": True,
            "enrichment": {"enabled": True},
            "run_id": "t",
        }
    )


class TestWhoIsABatchChild:
    def test_a_feed_step_of_a_batch_is(self, tmp_path: Path) -> None:
        # Both batch loops rebase the step's output_dir to <corpus>/feeds/<slug>, flag off.
        assert is_multi_feed_batch_child(_cfg(tmp_path / "corpus" / "feeds" / "abc"))

    def test_a_scoped_corpus_layout_add_is_not(self, tmp_path: Path) -> None:
        """Its output_dir has the same shape (the validator wraps it), but it owns its own
        corpus-root follow-ups: nothing else enqueues its enrichment."""
        cfg = _cfg(tmp_path / "corpus", corpus_layout=True)
        assert Path(str(cfg.output_dir)).parent.name == "feeds"
        assert not is_multi_feed_batch_child(cfg)

    def test_a_legacy_single_feed_run_is_not(self, tmp_path: Path) -> None:
        assert not is_multi_feed_batch_child(_cfg(tmp_path / "out"))


def _finalize(cfg: config.Config, run_dir: Path) -> dict[str, Any]:
    calls: dict[str, Any] = {"enqueue": [], "edges": [], "delta": []}
    with (
        patch.object(
            orchestration,
            "_maybe_spawn_enrichment_after_pipeline",
            side_effect=lambda c, d: calls["enqueue"].append(d),
        ),
        patch.object(
            orchestration,
            "_finalize_enrich_edges",
            side_effect=lambda d, m, **kw: calls["edges"].append((d, kw)),
        ),
        patch.object(
            orchestration,
            "_compute_corpus_delta_safe",
            side_effect=lambda d, m: calls["delta"].append(d),
        ),
        patch("podcast_scraper.search.indexer.maybe_index_corpus"),
        patch.object(orchestration, "_maybe_build_topic_clusters_after_index"),
        patch("podcast_scraper.providers.ml.diarization.ad_signatures.write_for_corpus"),
        patch.object(orchestration.wf_helpers, "generate_pipeline_summary", return_value=(0, "")),
        patch.object(orchestration.wf_helpers, "cleanup_pipeline"),
        patch.object(orchestration, "_log_episode_results"),
    ):
        res = SimpleNamespace(temp_dir=None)
        orchestration._finalize_pipeline(
            cfg, 0, res, str(run_dir), "t", metrics.Metrics(), [], None, None, None, None
        )
    return calls


def _result() -> MultiFeedFeedResult:
    return MultiFeedFeedResult(
        feed_url="https://feeds.example/podcast.xml",
        ok=True,
        error=None,
        episodes_processed=1,
        finished_at="2026-11-07T00:00:00Z",
    )
