"""The repair work-list for episodes serving without a summary (#1686).

Three states must stay distinguishable, because collapsing any two of them is how this issue
happened in the first place:

    healthy    -> not on any list
    RETRYABLE  -> failed once; a requeue is worth the provider spend
    TERMINAL   -> failed on two or more runs; the requeue was already tried and did not work.
                  Marko: "if episode fails on re-queue then it is finally failed for pipeline
                  and we need to manually investigate."

The attempt count is DERIVED from the run dirs on disk rather than stored, so there is no
counter to keep in sync and it survives a restore from backup.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.summary_repair import (
    assess_summaries,
    check_corpus_summaries,
    MAX_SUMMARY_ATTEMPTS,
    previously_terminal,
    retryable_episode_ids,
    terminal_episode_ids,
    write_work_list,
)

pytestmark = [pytest.mark.unit]


def _episode(root: Path, run: str, episode_id: str, *, summary, ledger=None) -> None:
    """Write one episode artifact into ``root/run/metadata/``."""
    meta_dir = root / run / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "episode": {"episode_id": episode_id},
        "summary": summary,
        "processing": {"stage_ledger": ledger or {}},
    }
    (meta_dir / f"{episode_id}.metadata.json").write_text(json.dumps(doc), encoding="utf-8")


FAILED = {"summarization": {"outcome": "failed", "reason": "tokenizer_threading"}}
GOOD = {"bullets": ["a real bullet"]}


class TestItSeparatesRetryableFromTerminal:
    def test_one_failed_run_is_retryable(self, tmp_path) -> None:
        _episode(tmp_path, "run_001", "ep-a", summary=None, ledger=FAILED)
        assert retryable_episode_ids(tmp_path) == ["ep-a"]
        assert terminal_episode_ids(tmp_path) == []

    def test_a_second_failed_run_makes_it_terminal(self, tmp_path) -> None:
        """The requeue was tried and did not work. Dispatching again just spends money."""
        _episode(tmp_path, "run_001", "ep-a", summary=None, ledger=FAILED)
        _episode(tmp_path, "run_002", "ep-a", summary=None, ledger=FAILED)
        assert terminal_episode_ids(tmp_path) == ["ep-a"]
        assert retryable_episode_ids(tmp_path) == []

    def test_the_threshold_is_the_documented_one(self, tmp_path) -> None:
        for n in range(1, MAX_SUMMARY_ATTEMPTS + 1):
            _episode(tmp_path, f"run_{n:03d}", "ep-a", summary=None, ledger=FAILED)
        assert assess_summaries(tmp_path)["ep-a"]["attempts"] == MAX_SUMMARY_ATTEMPTS
        assert assess_summaries(tmp_path)["ep-a"]["terminal"] is True


class TestARecoveredEpisodeLeavesTheList:
    def test_a_later_run_with_a_summary_clears_it(self, tmp_path) -> None:
        """A work-list that keeps listing fixed episodes is a work-list nobody reads."""
        _episode(tmp_path, "run_001", "ep-a", summary=None, ledger=FAILED)
        _episode(tmp_path, "run_002", "ep-a", summary=GOOD)
        assert assess_summaries(tmp_path) == {}
        assert retryable_episode_ids(tmp_path) == []

    def test_a_healthy_episode_never_appears(self, tmp_path) -> None:
        _episode(tmp_path, "run_001", "ep-a", summary=GOOD)
        ok, report = check_corpus_summaries(tmp_path)
        assert ok is True
        assert "every episode has a summary" in report


class TestWhatCountsAsLosingTheSummary:
    def test_a_degraded_outcome_is_not_a_loss(self, tmp_path) -> None:
        """`deadline_exceeded_but_completed` is a summary that was SLOW but real.

        Counting it here would put healthy episodes on a repair list and re-summarise them at
        provider expense — the exact confusion between "adjacent" and "the thing" that the
        capability audit kept making.
        """
        _episode(
            tmp_path,
            "run_001",
            "ep-a",
            summary=GOOD,
            ledger={
                "summarization": {
                    "outcome": "degraded",
                    "reason": "deadline_exceeded_but_completed",
                }
            },
        )
        assert assess_summaries(tmp_path) == {}

    def test_an_in_flight_retry_marker_is_not_a_loss(self, tmp_path) -> None:
        _episode(
            tmp_path,
            "run_001",
            "ep-a",
            summary=GOOD,
            ledger={
                "summarization": {"outcome": "degraded", "reason": "retrying_tokenizer_threading"}
            },
        )
        assert assess_summaries(tmp_path) == {}

    def test_the_cause_slug_is_carried_through(self, tmp_path) -> None:
        """Which cause decides whether a person or a requeue is the right response."""
        _episode(tmp_path, "run_001", "ep-a", summary=None, ledger=FAILED)
        assert assess_summaries(tmp_path)["ep-a"]["reasons"] == {"tokenizer_threading": 1}


class TestTheGateRefusesToTolerateEvenOne:
    def test_a_single_missing_summary_fails_the_check(self, tmp_path) -> None:
        """Marko: "it's never acceptable to have an episode without the summary of a single one."

        A threshold here is precisely how 8 became normal, so there is no threshold.
        """
        _episode(tmp_path, "run_001", "ep-a", summary=None, ledger=FAILED)
        _episode(tmp_path, "run_001", "ep-b", summary=GOOD)
        ok, report = check_corpus_summaries(tmp_path)
        assert ok is False
        assert "1 episode(s) have NO summary" in report

    def test_the_report_separates_the_two_populations(self, tmp_path) -> None:
        _episode(tmp_path, "run_001", "ep-a", summary=None, ledger=FAILED)
        _episode(tmp_path, "run_001", "ep-b", summary=None, ledger=FAILED)
        _episode(tmp_path, "run_002", "ep-b", summary=None, ledger=FAILED)
        _ok, report = check_corpus_summaries(tmp_path)
        assert "retryable (requeue): 1" in report
        assert "TERMINAL (needs a person): 1" in report


class TestTheWorkList:
    def test_it_writes_the_retryable_ids_for_reprocess(self, tmp_path) -> None:
        _episode(tmp_path, "run_001", "ep-a", summary=None, ledger=FAILED)
        dest = tmp_path / "worklist.txt"
        assert write_work_list(tmp_path, dest) == 1
        body = dest.read_text()
        assert "ep-a" in body
        assert "--reprocess-episode-ids" in body, "the file must say what to do with it"

    def test_terminal_episodes_get_their_own_file_not_silence(self, tmp_path) -> None:
        """ "Not on the list" and "the pipeline gave up" must not look the same."""
        _episode(tmp_path, "run_001", "ep-t", summary=None, ledger=FAILED)
        _episode(tmp_path, "run_002", "ep-t", summary=None, ledger=FAILED)
        dest = tmp_path / "worklist.txt"
        assert write_work_list(tmp_path, dest) == 0
        assert "ep-t" not in dest.read_text()
        terminal = dest.with_name("worklist.txt.terminal")
        assert terminal.exists()
        assert "ep-t" in terminal.read_text()
        assert "need a person" in terminal.read_text()

    def test_no_terminal_file_when_nothing_is_terminal(self, tmp_path) -> None:
        _episode(tmp_path, "run_001", "ep-a", summary=None, ledger=FAILED)
        dest = tmp_path / "worklist.txt"
        write_work_list(tmp_path, dest)
        assert not dest.with_name("worklist.txt.terminal").exists()

    def test_the_list_is_stable_across_runs(self, tmp_path) -> None:
        """An unstable order makes every diff of the work-list unreadable."""
        for eid in ("ep-c", "ep-a", "ep-b"):
            _episode(tmp_path, "run_001", eid, summary=None, ledger=FAILED)
        assert retryable_episode_ids(tmp_path) == ["ep-a", "ep-b", "ep-c"]


class TestItSurvivesABrokenCorpus:
    def test_an_unreadable_artifact_does_not_lose_the_rest(self, tmp_path) -> None:
        _episode(tmp_path, "run_001", "ep-a", summary=None, ledger=FAILED)
        (tmp_path / "run_001" / "metadata" / "junk.metadata.json").write_text("{ not json")
        assert retryable_episode_ids(tmp_path) == ["ep-a"]

    def test_an_artifact_predating_the_ledger_is_still_a_defect(self, tmp_path) -> None:
        """No summary is no summary, whether or not anything recorded why.

        I first wrote this test asserting the opposite — that a pre-#1647 artifact is "not a
        failure" — which quietly made the LEDGER the defect condition. That is the same mistake
        as the original bug one level up: an episode that loses its summary by a path nobody
        tagged would stay invisible. The ledger supplies the CAUSE; the missing summary is the
        defect. Cause unknown is reported as `unattributed`, not dropped.
        """
        meta_dir = tmp_path / "run_001" / "metadata"
        meta_dir.mkdir(parents=True)
        (meta_dir / "old.metadata.json").write_text(
            json.dumps({"episode": {"episode_id": "ep-old"}, "summary": None}), encoding="utf-8"
        )
        assessed = assess_summaries(tmp_path)
        assert "ep-old" in assessed
        assert assessed["ep-old"]["reasons"] == {"unattributed": 1}
        assert assessed["ep-old"]["terminal"] is False, "cause unknown is still worth one try"

    def test_a_summary_lost_without_a_ledger_record_is_not_invisible(self, tmp_path) -> None:
        """The sabotage that found this: if the ledger decided membership, a degradation path
        that forgot to record itself would produce a silently healthy-looking corpus."""
        _episode(
            tmp_path,
            "run_001",
            "ep-untracked",
            summary=None,
            ledger={"summarization": {"outcome": "degraded", "reason": "something_else"}},
        )
        assessed = assess_summaries(tmp_path)
        assert "ep-untracked" in assessed
        # ...and the cause must stay honest. A `degraded` record is NOT a recorded failure, so
        # counting it as one would dress an unexplained loss up as an explained one — and the
        # `reason` slug is exactly what an operator reads to decide requeue vs investigate.
        assert assessed["ep-untracked"]["reasons"] == {"unattributed": 1}
        assert assessed["ep-untracked"]["attempts"] == 0

    def test_an_empty_corpus_is_clean_not_crashed(self, tmp_path) -> None:
        ok, _report = check_corpus_summaries(tmp_path)
        assert ok is True


class TestItGuardsHardAgainstLooping:
    """Marko, 2026-08-20: "we need to guard against looping hard."

    The derived attempt count is sound only while every requeue leaves evidence. A requeue that
    dies BEFORE writing metadata leaves the corpus byte-identical, so the audit re-derives the
    same answer and re-emits the same episode — forever, at provider expense, with an automated
    dispatcher happily obliging. No new evidence after a dispatch IS the evidence that
    dispatching is not working.
    """

    def test_an_episode_that_made_no_progress_is_escalated(self, tmp_path) -> None:
        _episode(tmp_path, "run_001", "ep-a", summary=None, ledger=FAILED)
        dest = tmp_path / "worklist.txt"

        assert write_work_list(tmp_path, dest) == 1, "first pass: dispatch it"
        # The requeue died before writing anything — the corpus is unchanged.
        assert write_work_list(tmp_path, dest) == 0, "second pass: must NOT dispatch again"
        assert "ep-a" in dest.with_name("worklist.txt.terminal").read_text()

    def test_the_escalation_says_why(self, tmp_path) -> None:
        """ "Gave up after two real attempts" and "the requeue never reached it" are different
        bugs and must not arrive looking identical."""
        _episode(tmp_path, "run_001", "ep-a", summary=None, ledger=FAILED)
        dest = tmp_path / "worklist.txt"
        write_work_list(tmp_path, dest)
        write_work_list(tmp_path, dest)
        terminal = dest.with_name("worklist.txt.terminal").read_text()
        assert "no-progress" in terminal

    def test_real_progress_is_not_mistaken_for_a_loop(self, tmp_path) -> None:
        """A requeue that ran and failed HONESTLY leaves a new run dir. That is progress toward
        the terminal state, and must reach it via the attempt count, not the loop guard."""
        _episode(tmp_path, "run_001", "ep-a", summary=None, ledger=FAILED)
        dest = tmp_path / "worklist.txt"
        write_work_list(tmp_path, dest)
        _episode(tmp_path, "run_002", "ep-a", summary=None, ledger=FAILED)
        assert write_work_list(tmp_path, dest) == 0
        assert assess_summaries(tmp_path)["ep-a"]["attempts"] == 2

    def test_a_recovered_episode_is_not_escalated(self, tmp_path) -> None:
        """The requeue WORKED. It must leave both lists entirely, not land in terminal."""
        _episode(tmp_path, "run_001", "ep-a", summary=None, ledger=FAILED)
        dest = tmp_path / "worklist.txt"
        write_work_list(tmp_path, dest)
        _episode(tmp_path, "run_002", "ep-a", summary=GOOD)
        assert write_work_list(tmp_path, dest) == 0
        assert not dest.with_name("worklist.txt.terminal").exists()
        assert "ep-a" not in dest.read_text()

    def test_the_attempt_count_is_written_so_the_next_pass_can_compare(self, tmp_path) -> None:
        """Without the count on the line, the guard cannot tell a re-listing from progress."""
        _episode(tmp_path, "run_001", "ep-a", summary=None, ledger=FAILED)
        dest = tmp_path / "worklist.txt"
        write_work_list(tmp_path, dest)
        assert "attempts=1" in dest.read_text()

    def test_a_new_episode_is_still_dispatched_alongside_an_escalated_one(self, tmp_path) -> None:
        """The guard must escalate the stuck episode WITHOUT freezing the whole list."""
        _episode(tmp_path, "run_001", "ep-stuck", summary=None, ledger=FAILED)
        dest = tmp_path / "worklist.txt"
        write_work_list(tmp_path, dest)
        _episode(tmp_path, "run_001", "ep-new", summary=None, ledger=FAILED)
        assert write_work_list(tmp_path, dest) == 1
        assert "ep-new" in dest.read_text()
        assert "ep-stuck" in dest.with_name("worklist.txt.terminal").read_text()

    def test_a_hand_edited_work_list_without_counts_does_not_crash(self, tmp_path) -> None:
        """Operators edit these files. A missing `# attempts=` must degrade, not raise."""
        _episode(tmp_path, "run_001", "ep-a", summary=None, ledger=FAILED)
        dest = tmp_path / "worklist.txt"
        dest.write_text("# hand written\nep-a\n", encoding="utf-8")
        write_work_list(tmp_path, dest)  # must not raise
        assert "ep-a" in dest.with_name("worklist.txt.terminal").read_text()


class TestTheGuardHoldsRatherThanOscillating:
    """The bug the local end-to-end run found that these unit tests had missed.

    Escalating an id moves it OUT of the main work-list and into the `.terminal` sidecar. The
    first version did not read that sidecar back, so the pass after an escalation no longer
    recognised the id as "seen before", treated it as new, and dispatched it again:

        pass 1: dispatch 1     pass 2: dispatch 0 (escalated)     pass 3: dispatch 1  <-- loop

    Two passes looked correct, which is exactly how many the tests did. Marko asked to guard
    against looping HARD, so the guard is now tested to the point of boredom.
    """

    def test_it_does_not_re_dispatch_after_escalating(self, tmp_path) -> None:
        _episode(tmp_path, "run_001", "ep-stuck", summary=None, ledger=FAILED)
        dest = tmp_path / "worklist.txt"
        assert write_work_list(tmp_path, dest) == 1
        assert write_work_list(tmp_path, dest) == 0
        assert write_work_list(tmp_path, dest) == 0, "the third pass is where it looped"

    def test_it_stays_quiet_over_many_passes(self, tmp_path) -> None:
        """A dispatcher on a timer runs this hundreds of times. One dispatch, then silence."""
        _episode(tmp_path, "run_001", "ep-stuck", summary=None, ledger=FAILED)
        dest = tmp_path / "worklist.txt"
        dispatches = [write_work_list(tmp_path, dest) for _ in range(10)]
        assert dispatches == [1] + [0] * 9, dispatches

    def test_terminal_survives_a_deleted_main_work_list(self, tmp_path) -> None:
        """The sidecar is the durable half; losing the main list must not resurrect an id."""
        _episode(tmp_path, "run_001", "ep-stuck", summary=None, ledger=FAILED)
        dest = tmp_path / "worklist.txt"
        write_work_list(tmp_path, dest)
        write_work_list(tmp_path, dest)
        dest.unlink()
        assert write_work_list(tmp_path, dest) == 0

    def test_a_terminal_episode_that_recovers_leaves_both_lists(self, tmp_path) -> None:
        """Durable is not permanent. If a human fixes it, the corpus is the source of truth."""
        _episode(tmp_path, "run_001", "ep-stuck", summary=None, ledger=FAILED)
        dest = tmp_path / "worklist.txt"
        write_work_list(tmp_path, dest)
        write_work_list(tmp_path, dest)
        assert "ep-stuck" in previously_terminal(dest)

        _episode(tmp_path, "run_002", "ep-stuck", summary=GOOD)
        assert write_work_list(tmp_path, dest) == 0
        ok, _report = check_corpus_summaries(tmp_path)
        assert ok is True, "a repaired episode must clear the gate"

    def test_a_healthy_corpus_writes_no_terminal_file_at_all(self, tmp_path) -> None:
        _episode(tmp_path, "run_001", "ep-a", summary=GOOD)
        dest = tmp_path / "worklist.txt"
        assert write_work_list(tmp_path, dest) == 0
        assert not dest.with_name("worklist.txt.terminal").exists()
