"""A repair must report against the denominator it was given.

The 2026-08-18 run was asked for 32 episodes, repaired 0, and said nothing that distinguished
that from success. Establishing "0 of 32" took a separate audit against the live corpus API,
hours later and after the money was spent.

The distinction these tests protect is between three different states that a naive report would
collapse into one number:
  * NOT FOUND    — selection never located the episode. Silent, and the incident's actual outcome.
  * NOT FINISHED — selected, started, then failed. Different problem, different fix.
  * repaired     — metadata written.
"""

from __future__ import annotations

import threading

import pytest

from podcast_scraper.workflow.worklist_report import (
    get_worklist_report,
    log_worklist_outcome,
    reset_worklist_report,
)

pytestmark = [pytest.mark.unit]


@pytest.fixture(autouse=True)
def _fresh():
    reset_worklist_report()
    yield
    reset_worklist_report()


def test_no_worklist_means_nothing_to_report() -> None:
    """An ordinary ingest run must not grow a spurious 'repaired 0/0' line."""
    assert get_worklist_report().active is False
    assert log_worklist_outcome() is None


def test_a_complete_repair_says_so() -> None:
    r = get_worklist_report()
    r.request(["ep1", "ep2", "ep3"])
    r.mark_matched(["ep1", "ep2", "ep3"])
    for e in ("ep1", "ep2", "ep3"):
        r.mark_completed(e)
    line = r.summary()
    assert "repaired 3/3" in line
    assert "all requested episodes repaired" in line
    assert "NOT FOUND" not in line


def test_THE_INCIDENT_zero_repaired_is_stated_with_the_ids() -> None:
    """What the 2026-08-18 run should have printed and did not."""
    r = get_worklist_report()
    r.request([f"ep{i}" for i in range(32)])
    # selection matched nothing: the work-list never restricted anything
    line = r.summary()
    assert "repaired 0/32" in line
    assert "32 NOT FOUND" in line
    assert "ep0" in line


def test_NOT_FOUND_and_NOT_FINISHED_are_reported_SEPARATELY() -> None:
    """Collapsing them would hide which of two very different problems occurred."""
    r = get_worklist_report()
    r.request(["found-and-done", "found-but-failed", "never-found"])
    r.mark_matched(["found-and-done", "found-but-failed"])
    r.mark_completed("found-and-done")

    line = r.summary()
    assert "repaired 1/3" in line
    assert "1 NOT FOUND" in line and "never-found" in line
    assert "did NOT finish" in line and "found-but-failed" in line
    assert r.unmatched == ["never-found"]
    assert r.incomplete == ["found-but-failed"]


def test_completion_matches_on_guid_as_well_as_episode_id() -> None:
    """Detectors emit whichever identifier the artifact carries, so a list may hold either."""
    r = get_worklist_report()
    r.request(["a-guid-value"])
    r.mark_matched(["a-guid-value"])
    r.mark_completed(episode_id="some-other-id", guid="a-guid-value")
    assert "repaired 1/1" in r.summary()


def test_completing_an_episode_that_was_never_requested_is_ignored() -> None:
    """A repair run also touches nothing else; an unrelated episode must not inflate the count."""
    r = get_worklist_report()
    r.request(["ep1"])
    r.mark_matched(["ep1"])
    r.mark_completed("some-unrelated-episode")
    assert "repaired 0/1" in r.summary()


def test_ids_accumulate_ACROSS_feeds() -> None:
    """The report is process-scoped because no single feed can see the whole answer.

    A 32-episode list drawn from two feeds matches nothing in the other twelve — normal, not an
    error — so a per-feed report would call every healthy feed a failure.
    """
    r = get_worklist_report()
    for _ in range(14):  # every feed's config carries the whole list
        r.request(["ep1", "ep2"])
    r.mark_matched(["ep1"])  # feed 3 held ep1
    r.mark_matched(["ep2"])  # feed 9 held ep2
    r.mark_completed("ep1")
    r.mark_completed("ep2")
    assert "repaired 2/2" in r.summary()
    assert r.unmatched == []


def test_blank_and_whitespace_ids_are_ignored() -> None:
    r = get_worklist_report()
    r.request(["ep1", "", "   ", None])  # type: ignore[list-item]
    assert len(r.requested) == 1


def test_long_lists_are_truncated_but_the_COUNT_is_never_hidden() -> None:
    """A silently truncated list would understate the problem; the count must stay honest."""
    r = get_worklist_report()
    r.request([f"ep{i:03d}" for i in range(100)])
    line = r.summary(max_listed=5)
    assert "repaired 0/100" in line
    assert "100 NOT FOUND" in line
    assert "+95 more" in line


def test_the_outcome_is_logged_at_ERROR_when_a_repair_is_incomplete(caplog) -> None:
    """A partial repair (STARTED and did not finish) reported at INFO can be filtered out of view;
    that is how it got missed — so it stays ERROR.

    #1855: this is the INCOMPLETE case (matched but not completed). An id that was never matched at
    all is an ORPHANED reference, not a repair failure, and is a WARNING (covered separately)."""
    r = get_worklist_report()
    r.request(["ep1", "ep2"])
    r.mark_matched(["ep1", "ep2"])
    r.mark_completed("ep1")  # ep2 matched but never completed -> incomplete, not orphaned
    with caplog.at_level("ERROR"):
        line = log_worklist_outcome()
    assert line is not None
    assert "repaired 1/2" in caplog.text


def test_a_clean_outcome_is_logged_at_INFO_not_ERROR(caplog) -> None:
    r = get_worklist_report()
    r.request(["ep1"])
    r.mark_matched(["ep1"])
    r.mark_completed("ep1")
    with caplog.at_level("INFO"):
        log_worklist_outcome()
    assert "repaired 1/1" in caplog.text
    assert not [rec for rec in caplog.records if rec.levelname == "ERROR"]


def test_as_dict_carries_the_same_facts_for_machines() -> None:
    r = get_worklist_report()
    r.request(["a", "b", "c"])
    r.mark_matched(["a", "b"])
    r.mark_completed("a")
    d = r.as_dict()
    assert d["requested"] == 3 and d["matched"] == 2 and d["completed"] == 1
    assert d["unmatched_ids"] == ["c"]
    assert d["incomplete_ids"] == ["b"]


def test_concurrent_completion_recording_loses_nothing() -> None:
    """Episodes finish on the processing thread pool, so this is written to concurrently."""
    r = get_worklist_report()
    ids = [f"ep{i}" for i in range(200)]
    r.request(ids)
    r.mark_matched(ids)

    def work(chunk):
        for e in chunk:
            r.mark_completed(e)

    threads = [threading.Thread(target=work, args=(ids[i::4],)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(r.completed) == 200
    assert "repaired 200/200" in r.summary()


def test_orphan_only_worklist_logs_warning_not_error(caplog) -> None:
    """#1855: requested ids with no corpus entry anywhere (orphaned/stale references) are NOT a
    repair failure — every repairable item repaired. That must log at WARNING, not ERROR (which the
    signal-fleet auto-files as a bug on every sweep carrying one stale id)."""
    import logging

    r = get_worklist_report()
    r.request(["ep1", "ep2", "orphan"])
    r.mark_matched(["ep1", "ep2"])
    r.mark_completed("ep1")
    r.mark_completed("ep2")
    # orphan: requested, never matched -> unmatched. ep1/ep2 matched + completed -> no incomplete.

    with caplog.at_level(logging.DEBUG):
        line = log_worklist_outcome()

    assert line is not None and "NOT FOUND" in line
    errors = [rec for rec in caplog.records if rec.levelno >= logging.ERROR]
    assert not errors, f"orphan-only worklist logged at ERROR (should be WARNING) #1855: {errors}"
    assert [rec for rec in caplog.records if rec.levelno == logging.WARNING]


def test_an_episode_that_was_already_right_is_complete_but_not_called_repaired() -> None:
    """A repeat of a finished repair: nothing is left unrepaired (so the run must not fail as
    'repaired nothing'), but the line must not claim work that was not done either."""
    r = get_worklist_report()
    r.request(["ep1", "ep2"])
    r.mark_matched(["ep1", "ep2"])
    r.mark_completed("ep1")
    r.mark_already_right("ep2")

    assert r.completed == {"ep1", "ep2"} and r.already_right == {"ep2"}
    assert not r.repaired_nothing
    assert "repaired 2/2 (1 of them already right, nothing to repair)" in r.summary()
    assert r.as_dict()["already_right"] == 1


def test_a_worklist_of_only_already_right_episodes_is_not_a_failed_run() -> None:
    r = get_worklist_report()
    r.request(["ep1"])
    r.mark_matched(["ep1"])
    r.mark_already_right("ep1")

    assert not r.repaired_nothing and "all requested episodes repaired" in r.summary()


def test_already_right_matches_on_guid_and_ignores_ids_nobody_requested() -> None:
    r = get_worklist_report()
    r.request(["guid-1"])
    r.mark_already_right("derived-id-not-requested", "guid-1")
    r.mark_already_right("stray", "also-stray")

    assert r.already_right == {"guid-1"}


def test_the_summary_is_unchanged_when_nothing_was_already_right() -> None:
    r = get_worklist_report()
    r.request(["ep1"])
    r.mark_matched(["ep1"])
    r.mark_completed("ep1")

    assert r.summary() == "work-list: repaired 1/1 · all requested episodes repaired"


# --- declined: ran, and the stage left it as it was by design (2026-10-15) ----------------------


def _the_8_3_run():
    """The 2026-10-15 ``repunctuate_only``, small: 2 repaired, 3 refused, 1 crashed."""
    r = get_worklist_report()
    r.request(["ok1", "ok2", "ref1", "ref2", "ref3", "crashed"])
    r.mark_matched(["ok1", "ok2", "ref1", "ref2", "ref3", "crashed"])
    r.mark_completed("ok1")
    r.mark_completed("ok2")
    r.mark_declined("ref1", reason="NoWindowAccepted")
    r.mark_declined("ref2", reason="NoWindowAccepted")
    r.mark_declined("ref3", reason="PlainTranscript")
    return r


def test_a_refused_episode_is_declined_not_unfinished() -> None:
    """It ran and the guards said no: saying 'did NOT finish' reads as a crash."""
    r = _the_8_3_run()
    line = r.summary()
    assert "repaired 2/6" in line
    assert "1 selected but did NOT finish: [crashed]" in line
    assert (
        "3 left as they were by design (NoWindowAccepted 2, PlainTranscript 1): "
        "[ref1, ref2, ref3]" in line
    )
    assert r.incomplete == ["crashed"]
    assert r.declined_ids == ["ref1", "ref2", "ref3"]
    d = r.as_dict()
    assert d["incomplete_ids"] == ["crashed"]
    assert d["declined"] == {
        "ref1": "NoWindowAccepted",
        "ref2": "NoWindowAccepted",
        "ref3": "PlainTranscript",
    }


def test_declined_is_not_completed_so_a_run_of_only_refusals_still_repaired_nothing() -> None:
    """The episode is still wrong: a run that repaired none of what it was asked is a failed run."""
    r = get_worklist_report()
    r.request(["ref1", "ref2"])
    r.mark_matched(["ref1", "ref2"])
    r.mark_declined("ref1", reason="NoWindowAccepted")
    r.mark_declined("ref2", reason="NoWindowAccepted")
    assert r.repaired_nothing is True
    assert r.completed == set()
    assert "all requested episodes repaired" not in r.summary()


def test_refusals_alone_log_at_WARNING_and_a_crash_beside_them_still_logs_at_ERROR(
    caplog,
) -> None:
    import logging

    r = get_worklist_report()
    r.request(["ok1", "ref1"])
    r.mark_matched(["ok1", "ref1"])
    r.mark_completed("ok1")
    r.mark_declined("ref1", reason="NoWindowAccepted")
    with caplog.at_level(logging.DEBUG):
        log_worklist_outcome()
    assert not [rec for rec in caplog.records if rec.levelno >= logging.ERROR]
    assert [rec for rec in caplog.records if rec.levelno == logging.WARNING]

    caplog.clear()
    reset_worklist_report()
    _the_8_3_run()
    with caplog.at_level(logging.DEBUG):
        log_worklist_outcome()
    assert [rec for rec in caplog.records if rec.levelno >= logging.ERROR]


def test_declined_matches_on_guid_ignores_strangers_and_a_completion_wins() -> None:
    r = get_worklist_report()
    r.request(["guid-1", "ep2"])
    r.mark_matched(["guid-1", "ep2"])
    r.mark_declined("derived-id", "guid-1", reason="NoWindowAccepted")
    r.mark_declined("stray", "also-stray", reason="NoWindowAccepted")
    r.mark_declined("ep2", reason="NoWindowAccepted")
    r.mark_completed("ep2")
    assert r.declined_ids == ["guid-1"]
    assert r.incomplete == []
    assert "repaired 1/2" in r.summary()
