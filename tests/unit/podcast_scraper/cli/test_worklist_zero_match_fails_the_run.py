"""A repair that was handed a work-list and matched NOTHING must fail — not exit 0.

THE INCIDENT (prod, 2026-09-29, #50). A scoped ``rederive_only`` job was given two episode ids
that happened to be RFC-4151 ``tag:`` URIs::

    tag:soundcloud,2010:tracks/2135080263

The jobs API split them on the comma, so the work-list file held four fragments, none of which is
an episode id. Selection logged ``reprocess work-list: none of ...``, derived nothing, and the CLI
returned 0 — so the job registry recorded the run as ``succeeded``. A repair that did nothing,
reporting green, on the exact path used to repair production.

``workflow/worklist_report.py`` already KNEW: its #1855 contract calls zero-matched "the incident
this guard exists to keep loud" and logs it at ERROR. Two things kept that from mattering:

1. The single-feed path — which every API-dispatched job takes — never called
   ``log_worklist_outcome()`` at all. Measured on the three repair runs that day: the outcome line
   appeared 0 times in each log. The guard only existed on the multi-feed path.
2. On either path, ERROR was a log line, not an exit code. The registry reads the exit code.

What does NOT change, deliberately: some ids matched and some did not stays exit 0 with a WARNING.
#1855 decided that on purpose — "a few stale ids alongside real repairs, nothing lost" — and a sweep
carrying one stale id must not be reported as a failed repair.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Iterator
from unittest.mock import patch

import pytest

from podcast_scraper import cli, config
from podcast_scraper.workflow.worklist_report import get_worklist_report, reset_worklist_report

pytestmark = pytest.mark.unit

#: Verbatim from the incident — the shape that cannot survive a comma separator.
_TAG_ID = "tag:soundcloud,2010:tracks/2135080263"


@pytest.fixture(autouse=True)
def _fresh_report() -> Iterator[None]:
    """The report is process-global; every test must start from an empty one."""
    reset_worklist_report()
    yield
    reset_worklist_report()


def _selection(requested: list[str], matched: list[str], completed: list[str]):
    """A stand-in pipeline that records what REAL selection records, then returns normally.

    ``scraping.py`` calls ``request`` and ``mark_matched``; ``metadata_generation`` calls
    ``mark_completed``. Returning ``(n, "ok")`` is the point: the pipeline itself succeeded, so any
    failure must come from the work-list outcome, not from a crashed run.
    """

    def fake_run(cfg: config.Config) -> tuple[int, str]:
        report = get_worklist_report()
        report.request(requested)
        report.mark_feed_searched()
        report.mark_matched(matched)
        for ep in completed:
            report.mark_completed(ep)
        return (len(completed), "ok")

    return fake_run


def _run(tmp: str, fake_run, *feeds: str) -> int:
    ids = Path(tmp) / "ids.txt"
    ids.write_text(f"{_TAG_ID}\n", encoding="utf-8")
    urls = list(feeds) or ["https://a.example/feed.xml"]
    argv = [urls[0]]
    for extra in urls[1:]:
        argv += ["--rss", extra]
    argv += [
        "--output-dir",
        tmp,
        "--pipeline-stage",
        "rederive_only",
        "--reprocess-existing-only",
        "--reprocess-episode-ids",
        str(ids),
    ]
    with patch.object(cli, "_validate_ffmpeg"), patch.object(cli, "_validate_python_version"):
        return cli.main(argv, run_pipeline_fn=fake_run)


class TestZeroMatchedIsAFailedRun:
    def test_single_feed_run_that_matched_nothing_exits_nonzero(self) -> None:
        """THE INCIDENT: every API job is single-feed, and this path returned 0."""
        with tempfile.TemporaryDirectory() as tmp:
            code = _run(tmp, _selection(requested=[_TAG_ID], matched=[], completed=[]))
        assert code != 0, (
            "a repair handed a work-list that matched NOTHING exited 0, so the job registry would "
            "record it as `succeeded` — the #50 incident, a no-op repair reporting green"
        )

    def test_multi_feed_run_that_matched_nothing_exits_nonzero(self) -> None:
        """Same contract on the batch path, where the outcome WAS logged but still exited 0."""
        with tempfile.TemporaryDirectory() as tmp:
            code = _run(
                tmp,
                _selection(requested=[_TAG_ID], matched=[], completed=[]),
                "https://a.example/feed.xml",
                "https://b.example/feed.xml",
            )
        assert code != 0

    def test_the_single_feed_path_now_SAYS_so(self, caplog: pytest.LogCaptureFixture) -> None:
        """The loud line was missing entirely from single-feed logs; an exit code alone is not
        enough for the operator reading the job log to know WHY it failed."""
        with tempfile.TemporaryDirectory() as tmp, caplog.at_level("ERROR"):
            _run(tmp, _selection(requested=[_TAG_ID], matched=[], completed=[]))
        assert "repaired 0/1" in caplog.text
        assert "NOT FOUND" in caplog.text


class TestMatchedButNothingFinishedIsAFailedRun:
    """Prod, 2026-09-29, job 7b4465dd: a ``rederive_only`` for two omnycontent episodes MATCHED
    both, then refused both ("found metadata but no transcript") — and exited 0, ``succeeded``.
    Everything was found and nothing was repaired: the same outcome as the total miss above."""

    _OMNY = ["52b31e33-08b2-4ea7-a5d0-b4b9014b34d9", "6ea24954-06b5-4daa-a250-b4980120947f"]

    def test_single_feed_run_that_finished_nothing_exits_nonzero(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp, caplog.at_level("ERROR"):
            code = _run(tmp, _selection(self._OMNY, matched=self._OMNY, completed=[]))
        assert code != 0, "a repair of 0/2 with both episodes selected exited 0 — job 7b4465dd"
        assert "repaired 0/2" in caplog.text
        assert "did NOT finish" in caplog.text

    def test_multi_feed_run_that_finished_nothing_exits_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code = _run(
                tmp,
                _selection(self._OMNY, matched=self._OMNY, completed=[]),
                "https://a.example/feed.xml",
                "https://b.example/feed.xml",
            )
        assert code != 0

    def test_a_partial_repair_still_succeeds(self, caplog: pytest.LogCaptureFixture) -> None:
        """One of two finished: a red status would hide the one that worked. Logged at ERROR."""
        with tempfile.TemporaryDirectory() as tmp, caplog.at_level("ERROR"):
            code = _run(tmp, _selection(self._OMNY, matched=self._OMNY, completed=self._OMNY[:1]))
        assert code == 0
        assert "repaired 1/2" in caplog.text and "did NOT finish" in caplog.text


class TestWhatMustStillSucceed:
    def test_a_repair_that_matched_and_finished_exits_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code = _run(tmp, _selection([_TAG_ID], matched=[_TAG_ID], completed=[_TAG_ID]))
        assert code == 0

    def test_partial_unmatched_stays_a_warning_not_a_failure(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """#1855's deliberate carve-out: one stale id beside real repairs is not a failed run."""

        def fake_run(cfg: config.Config) -> tuple[int, str]:
            report = get_worklist_report()
            report.request(["ep-real", "ep-stale"])
            report.mark_feed_searched()
            report.mark_matched(["ep-real"])
            report.mark_completed("ep-real")
            return (1, "ok")

        with tempfile.TemporaryDirectory() as tmp, caplog.at_level("WARNING"):
            code = _run(tmp, fake_run)
        assert code == 0
        assert "NOT FOUND" in caplog.text

    def test_an_ordinary_run_with_no_worklist_is_untouched(self) -> None:
        def fake_run(cfg: config.Config) -> tuple[int, str]:
            return (3, "ok")

        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch.object(cli, "_validate_ffmpeg"),
                patch.object(cli, "_validate_python_version"),
            ):
                code = cli.main(
                    ["https://a.example/feed.xml", "--output-dir", tmp], run_pipeline_fn=fake_run
                )
        assert code == 0
