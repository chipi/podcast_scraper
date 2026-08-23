"""E4 — a multi-feed batch enqueues corpus enrichment against the CORPUS ROOT.

Regression guard for the box-confirmed defect (2026-08-23): the per-feed pipeline
finalize enqueues enrichment against the per-feed run scratch dir
(``<corpus>/feeds/<slug>/run_<id>/``), whose ``.viewer/jobs.jsonl`` the API server
never drains. So after a multi-feed reprocess, ~N enrichment enqueues were orphaned
in throwaway per-run registries and the drainable corpus-root registry gained zero
enrichment rows — the 127-incident re-enrichment silently no-op'd for days.

The fix makes ``finalize_multi_feed_batch`` (which owns corpus integration) enqueue the
enrichment follow-up against the corpus root. These tests assert exactly one queued
``corpus_enrichment`` row lands in ``<corpus_root>/.viewer/jobs.jsonl``. (#1811 E4)
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from podcast_scraper import config
from podcast_scraper.server.jobs import COMMAND_ENRICHMENT, STATUS_QUEUED
from podcast_scraper.server.pipeline_job_registry import read_jobs
from podcast_scraper.workflow import corpus_operations
from podcast_scraper.workflow.corpus_operations import MultiFeedFeedResult


def _feed_result() -> MultiFeedFeedResult:
    return MultiFeedFeedResult(
        feed_url="https://feeds.example/podcast.xml",
        ok=True,
        error=None,
        episodes_processed=1,
        finished_at="2026-08-23T12:00:00Z",
    )


def _cfg(*, enrichment_enabled: bool) -> Any:
    # finalize_multi_feed_batch only touches template_cfg.vector_search (kept False so it
    # returns before the index subprocess) and, via _maybe_spawn_enrichment_after_pipeline,
    # cfg.enrichment / cfg.profile / cfg.config_path.
    return SimpleNamespace(
        vector_search=False,
        enrichment={"enabled": enrichment_enabled},
        profile=None,
        config_path=None,
    )


@pytest.mark.unit
def test_multi_feed_batch_enqueues_one_enrichment_at_corpus_root(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    corpus_operations.finalize_multi_feed_batch(
        str(corpus), _cfg(enrichment_enabled=True), [_feed_result()]
    )

    jobs = read_jobs(corpus)
    enr = [j for j in jobs if j.get("command_type") == COMMAND_ENRICHMENT]
    assert len(enr) == 1, (
        f"expected exactly one corpus_enrichment row in the CORPUS-ROOT registry, got "
        f"{len(enr)} — the batch must own the enrichment enqueue (#1811 E4)"
    )
    assert enr[0]["status"] == STATUS_QUEUED, (
        "the row must be queued (force_queued) so the API server promotes it — a RUNNING "
        "row would promise a process this batch process cannot start"
    )


@pytest.mark.unit
def test_real_config_flows_through_and_enqueues(tmp_path: Path) -> None:
    """Prove the enqueue fires with a genuine ``config.Config`` (not just a SimpleNamespace).

    The gate is ``isinstance(cfg.enrichment, dict)`` and ``Config.enrichment`` is declared as a
    ``dict`` field — so a real Config exercises the same predicate the SimpleNamespace tests use,
    AND confirms a genuine Config object flows through finalize_multi_feed_batch ->
    _maybe_spawn_enrichment_after_pipeline without a type error (#1811 E4).
    """
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    # Dict-unpack (as create_test_config does) — the pydantic populate-by-name alias `rss_url`
    # is accepted at runtime; a dict[str, Any] unpack avoids mypy's per-field kwarg check.
    cfg_kwargs: dict[str, Any] = {
        "rss_url": "https://feeds.example/podcast.xml",
        "output_dir": str(tmp_path / "cfg_out"),
        "enrichment": {"enabled": True},
        "vector_search": False,
    }
    cfg = config.Config(**cfg_kwargs)

    corpus_operations.finalize_multi_feed_batch(str(corpus), cfg, [_feed_result()])

    enr = [j for j in read_jobs(corpus) if j.get("command_type") == COMMAND_ENRICHMENT]
    assert len(enr) == 1
    assert enr[0]["status"] == STATUS_QUEUED


@pytest.mark.unit
def test_no_enrichment_row_when_enrichment_disabled(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    corpus_operations.finalize_multi_feed_batch(
        str(corpus), _cfg(enrichment_enabled=False), [_feed_result()]
    )

    jobs = read_jobs(corpus)
    enr = [j for j in jobs if j.get("command_type") == COMMAND_ENRICHMENT]
    assert enr == [], "enrichment disabled must not enqueue anything"


@pytest.mark.unit
def test_empty_batch_enqueues_nothing(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    corpus_operations.finalize_multi_feed_batch(str(corpus), _cfg(enrichment_enabled=True), [])

    jobs = read_jobs(corpus)
    enr = [j for j in jobs if j.get("command_type") == COMMAND_ENRICHMENT]
    assert enr == [], "an empty batch (no feed results) must not enqueue enrichment"
