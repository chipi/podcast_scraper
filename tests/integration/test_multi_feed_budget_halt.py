"""A spent budget must end the BATCH, not just the feed that spent it.

Every other per-feed failure is local: feed 3's RSS being unreachable says nothing about feed 4,
so the loop continues and the batch reports a partial success. A cost cap is categorically
different — the budget belongs to the whole invocation, so once it is exhausted every remaining
feed can only overspend further.

Before this, cli's handler caught the CostCapExceeded, logged "Feed failed", and `continue`d
(cli.py, the `except Exception` around run_pipeline_fn). classify_multi_feed_feed_exception marked
it "hard", but that only changes the final exit code — it never stopped the loop. That is how a
"$5 per run" cap behaved as "$5 per feed, times fourteen" on 2026-08-18.

These tests drive cli.main with a stub run_pipeline, because the property under test belongs to
the LOOP, not to the pipeline: what matters is how many feeds get started after the cap trips.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple

import pytest

from podcast_scraper import cli
from podcast_scraper.workflow.cost_monitoring import CostCapExceeded

pytestmark = [pytest.mark.integration]


def _feeds_file(tmp_path: Path, n: int) -> Path:
    """A feeds spec, the real shape the prod reprocess workflow passes as --feeds-spec."""
    p = tmp_path / "feeds.spec.yaml"
    body = "feeds:\n" + "".join(f"  - url: https://example.com/feed{i}.xml\n" for i in range(n))
    p.write_text(body, encoding="utf-8")
    return p


def _run(tmp_path: Path, *, feeds: int, trip_on: int | None) -> Tuple[int, List[str]]:
    """Run a multi-feed batch; ``trip_on`` is the 0-based feed whose pipeline blows the budget.

    Returns (exit code, the rss urls whose pipeline actually STARTED).
    """
    started: List[str] = []

    def fake_run_pipeline(cfg: Any):
        started.append(str(getattr(cfg, "rss_url", "")))
        if trip_on is not None and len(started) - 1 == trip_on:
            raise CostCapExceeded(9.63, 5.0)
        return 1, "1 episode"

    code = cli.main(
        [
            "--feeds-spec",
            str(_feeds_file(tmp_path, feeds)),
            "--output-dir",
            str(tmp_path / "corpus"),
            "--no-transcript-cache",
        ],
        run_pipeline_fn=fake_run_pipeline,
    )
    return code, started


def test_the_batch_STOPS_at_the_feed_that_exhausts_the_budget(tmp_path) -> None:
    """THE regression test. Feed 3 of 14 trips the cap; feeds 4-14 must never start."""
    code, started = _run(tmp_path, feeds=14, trip_on=2)
    assert len(started) == 3, f"expected the batch to stop after feed 3, it ran {len(started)}"
    assert code != 0, "a halted batch must not report success"


def test_a_halted_batch_still_writes_its_manifest(tmp_path) -> None:
    """break, not return: the operator has to see which feeds ran and which never started."""
    corpus = tmp_path / "corpus"
    _run(tmp_path, feeds=8, trip_on=1)
    manifest = corpus / "corpus_manifest.json"
    summary = corpus / "corpus_run_summary.json"
    assert (
        manifest.is_file() or summary.is_file()
    ), "a halted batch wrote no manifest — the run is then invisible to any later audit"


def test_the_halt_is_reported_with_the_denominator(tmp_path, caplog) -> None:
    with caplog.at_level("ERROR"):
        _run(tmp_path, feeds=14, trip_on=2)
    assert "HALTING THE BATCH" in caplog.text
    assert "3 of 14" in caplog.text, "the operator needs to know how much of the batch ran"


def test_an_ORDINARY_feed_failure_still_lets_the_batch_continue(tmp_path) -> None:
    """The halt must be specific to the budget, not a general "stop on any error".

    A single unreachable feed ending a 14-feed nightly run would be a worse regression than the
    bug being fixed.
    """
    started: List[str] = []

    def fake_run_pipeline(cfg: Any):
        started.append(str(getattr(cfg, "rss_url", "")))
        if len(started) == 3:
            raise RuntimeError("feed 3 is unreachable")
        return 1, "1 episode"

    cli.main(
        [
            "--feeds-spec",
            str(_feeds_file(tmp_path, 10)),
            "--output-dir",
            str(tmp_path / "corpus"),
            "--no-transcript-cache",
        ],
        run_pipeline_fn=fake_run_pipeline,
    )
    assert len(started) == 10, "one bad feed must not end the batch"


def test_a_batch_that_never_trips_runs_every_feed(tmp_path) -> None:
    code, started = _run(tmp_path, feeds=6, trip_on=None)
    assert len(started) == 6
    assert code == 0
