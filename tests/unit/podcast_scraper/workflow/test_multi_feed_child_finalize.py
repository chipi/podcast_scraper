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
from podcast_scraper.insight_stage.edges_cli import parse_enrich_edges_argv, run_enrich_edges_cli
from podcast_scraper.workflow import corpus_operations, metrics, orchestration
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


class TestTheFinalize:
    def test_a_batch_child_enqueues_no_enrichment(self, tmp_path: Path) -> None:
        feed = tmp_path / "corpus" / "feeds" / "abc"
        run_dir = feed / "run_t"
        run_dir.mkdir(parents=True)
        cfg = _cfg(feed).model_copy(update={"skip_auto_vector_index": True})
        calls = _finalize(cfg, run_dir)
        assert calls["enqueue"] == [], "the batch enqueues once at the corpus root"

    def test_a_batch_child_defers_the_edge_stamp_and_ledger(self, tmp_path: Path) -> None:
        feed = tmp_path / "corpus" / "feeds" / "abc"
        run_dir = feed / "run_t"
        run_dir.mkdir(parents=True)
        cfg = _cfg(feed).model_copy(update={"skip_auto_vector_index": True})
        calls = _finalize(cfg, run_dir)
        assert calls["edges"] == [(str(run_dir), {"defer_corpus_writes": True})]
        assert calls["delta"] == [], "no run-dir delta or fingerprint manifest for a batch child"

    def test_a_scoped_add_still_does_all_of_it_at_the_corpus_root(self, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        cfg = _cfg(corpus, corpus_layout=True)
        run_dir = Path(str(cfg.output_dir)) / "run_t"
        run_dir.mkdir(parents=True)
        calls = _finalize(cfg, run_dir)
        assert calls["enqueue"] == [str(corpus)]
        assert calls["edges"] == [(str(corpus), {"defer_corpus_writes": False})]
        assert calls["delta"] == [str(corpus)]


class TestTheEdgePass:
    def _corpus(self, tmp_path: Path) -> Path:
        from tests.unit.search.test_enrich_edges_cli import _build_corpus

        _build_corpus(tmp_path)
        return tmp_path

    def test_deferred_it_writes_no_stamp_and_no_ledger(self, tmp_path: Path) -> None:
        corpus = self._corpus(tmp_path)
        args = parse_enrich_edges_argv(["--output-dir", str(corpus)])
        args.defer_corpus_writes = True
        assert run_enrich_edges_cli(args, _LOG) == 0
        assert not (corpus / "corpus_edges_stamp.json").exists()
        assert not (corpus / "identity_ledger.json").exists()

    def test_by_default_it_still_stamps_and_refreshes(self, tmp_path: Path) -> None:
        corpus = self._corpus(tmp_path)
        assert (
            run_enrich_edges_cli(parse_enrich_edges_argv(["--output-dir", str(corpus)]), _LOG) == 0
        )
        assert (corpus / "corpus_edges_stamp.json").is_file()
        assert (corpus / "identity_ledger.json").is_file()


def _result() -> MultiFeedFeedResult:
    return MultiFeedFeedResult(
        feed_url="https://feeds.example/podcast.xml",
        ok=True,
        error=None,
        episodes_processed=1,
        finished_at="2026-11-07T00:00:00Z",
    )


class TestTheBatchOwnsTheLedger:
    """The children no longer refresh the ledger, so the batch must, once, at the corpus root.
    With enrichment on, the queued enrichment refreshes it when it finishes; a second refresh at
    batch end would only invalidate every cache one extra time."""

    def _run(self, tmp_path: Path, enabled: bool) -> list:
        from podcast_scraper.identity import refresh_guard

        done: list = []
        cfg = SimpleNamespace(enrichment={"enabled": enabled})

        def _refresh(root: Any, **kw: Any) -> bool:
            done.append((Path(root), kw["after"]))
            return True

        with patch.object(refresh_guard, "refresh_ledger_after", side_effect=_refresh):
            assert refresh_guard.refresh_ledger_after_batch(tmp_path / "corpus", cfg)
        return done

    def test_without_enrichment_the_batch_refreshes_it(self, tmp_path: Path) -> None:
        assert self._run(tmp_path, enabled=False) == [(tmp_path / "corpus", "multi-feed batch")]

    def test_with_enrichment_the_enrichment_run_does(self, tmp_path: Path) -> None:
        assert self._run(tmp_path, enabled=True) == []

    @pytest.mark.parametrize("rel", ["cli.py", "service.py"])
    def test_both_batch_loops_call_it_after_the_finalize(self, rel: str) -> None:
        """Both loops rebase their feeds into batch children, so both must refresh."""
        src = Path(corpus_operations.__file__).resolve().parents[1] / rel
        text = src.read_text(encoding="utf-8")
        after = text[text.index("finalize_multi_feed_batch(\n") :]
        # Through the CLI hooks (E04 S7), whose built-in owner is the refresh itself.
        assert "cli_hooks.batch_completed(" in after[:800], rel
        from podcast_scraper import builtin_cli_hooks

        assert builtin_cli_hooks._refresh_ledger_after_batch in builtin_cli_hooks.BATCH_COMPLETED
