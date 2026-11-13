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

from pathlib import Path

import pytest

from podcast_scraper import config
from podcast_scraper.workflow.corpus_operations import is_multi_feed_batch_child

pytestmark = pytest.mark.unit


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
