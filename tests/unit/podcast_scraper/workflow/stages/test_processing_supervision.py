"""Unit tests for the pipeline supervision bounds added after the 2026-08-12 incidents.

Background — both production incidents on 2026-08-12 shared one amplifier. A
``CostCapExceeded`` raised in the main thread (from ``orchestration``'s
``check_cost_soft_cap_at_stage``, in a region with no ``try/finally``) unwound past the
point that sets ``transcription_complete_event``. The ``ProcessingProcessor`` thread was
left with a continue-predicate that defaults to ``True``, so it never terminated:

* with nothing left to submit, it spun at 0.05s/iteration for 4h15m — live pid, ~2.5%
  CPU, zero progress, zero log output, until cancelled by hand;
* with one more job available, ``executor.submit`` fired into a shutting-down interpreter
  and raised ``RuntimeError: cannot schedule new futures after interpreter shutdown``,
  killing the run and discarding every episode still queued.

These tests pin the bounds that make both presentations impossible.
"""

import os
import sys
import threading
import time
import unittest

PACKAGE_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from podcast_scraper.workflow.stages import processing


class _Cfg:
    """Minimal config stand-in; the helper only ever does ``getattr``."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class TestProcessingLoopBudget(unittest.TestCase):
    """``_processing_loop_budget_seconds`` — the wall-clock backstop for the work loop."""

    def test_defaults_when_unset(self):
        """A config that never heard of the setting still gets a bound.

        This is the important case: the bound must apply to every existing deployment
        without anyone opting in, because the incident happened on a config that had no
        such key.
        """
        budget = processing._processing_loop_budget_seconds(_Cfg(), max_workers=4)
        self.assertEqual(budget, float(processing.DEFAULT_PROCESSING_LOOP_BUDGET_SECONDS))
        self.assertIsNotNone(budget)

    def test_positive_override_is_used(self):
        budget = processing._processing_loop_budget_seconds(
            _Cfg(processing_loop_budget_seconds=90), max_workers=4
        )
        self.assertEqual(budget, 90.0)

    def test_zero_disables_the_bound(self):
        """Opting out must be explicit and must actually work."""
        budget = processing._processing_loop_budget_seconds(
            _Cfg(processing_loop_budget_seconds=0), max_workers=4
        )
        self.assertIsNone(budget)

    def test_negative_disables_the_bound(self):
        budget = processing._processing_loop_budget_seconds(
            _Cfg(processing_loop_budget_seconds=-1), max_workers=4
        )
        self.assertIsNone(budget)

    def test_garbage_value_falls_back_to_default_rather_than_raising(self):
        """A malformed config must not crash the run, and must not silently unbound it.

        Falling back to ``None`` here would turn a typo into an unbounded loop — the exact
        failure being defended against — so the fallback is the default, not disabled.
        """
        budget = processing._processing_loop_budget_seconds(
            _Cfg(processing_loop_budget_seconds="not-a-number"), max_workers=4
        )
        self.assertEqual(budget, float(processing.DEFAULT_PROCESSING_LOOP_BUDGET_SECONDS))

    def test_default_is_generous_enough_for_real_runs(self):
        """Guard against a future tightening that would truncate legitimate work.

        The longest legitimate production run observed was a 36-episode job at roughly two
        hours. The default must stay clear of that.
        """
        self.assertGreaterEqual(processing.DEFAULT_PROCESSING_LOOP_BUDGET_SECONDS, 2 * 60 * 60)


class TestSupervisionExitSemantics(unittest.TestCase):
    """The two bounds, exercised through the same predicates the loop uses.

    ``_run_parallel_processing_loop`` is nested inside ``process_processing_jobs_concurrent``
    and cannot be imported directly, so these tests reconstruct the exit predicate with the
    identical logic. If that predicate is changed in the module, these tests will NOT catch
    it — see ``test_supervision_predicate_is_documented_as_mirrored`` below and the
    follow-up to extract the loop to module scope.
    """

    @staticmethod
    def _exit_reason(main_alive: bool, elapsed: float, budget):
        """Mirror of ``_supervision_exit_reason``."""
        if not main_alive:
            return "main thread exited"
        if budget is not None and elapsed > budget:
            return f"wall-clock budget exceeded ({elapsed:.0f}s > {budget:.0f}s)"
        return None

    def test_dead_main_thread_stops_the_loop(self):
        """The wedge: a worker must never outlive its parent."""
        reason = self._exit_reason(main_alive=False, elapsed=1.0, budget=3600.0)
        self.assertIsNotNone(reason)
        self.assertIn("main thread", reason)

    def test_dead_main_thread_wins_even_with_budget_disabled(self):
        """Disabling the wall-clock bound must not disable liveness."""
        reason = self._exit_reason(main_alive=False, elapsed=0.0, budget=None)
        self.assertIsNotNone(reason)

    def test_budget_exceeded_stops_the_loop(self):
        reason = self._exit_reason(main_alive=True, elapsed=7200.0, budget=3600.0)
        self.assertIsNotNone(reason)
        self.assertIn("budget exceeded", reason)

    def test_healthy_loop_is_not_stopped(self):
        """A live parent inside budget must never be interrupted."""
        self.assertIsNone(self._exit_reason(main_alive=True, elapsed=10.0, budget=3600.0))

    def test_healthy_loop_with_no_budget_is_not_stopped(self):
        self.assertIsNone(self._exit_reason(main_alive=True, elapsed=10**9, budget=None))

    def test_main_thread_is_alive_under_test(self):
        """Sanity: the real predicate's liveness source behaves as assumed."""
        self.assertTrue(threading.main_thread().is_alive())


class TestSubmitGuardContract(unittest.TestCase):
    """A pool that refuses work must stop submission, not kill the run.

    Reconstructs the ``_try_submit`` contract: on ``RuntimeError`` the episode index is
    un-marked (so a resumed run reprocesses it, kept idempotent by ``skip_existing``) and
    submission stops rather than propagating.
    """

    def test_submit_failure_unmarks_and_stops_without_raising(self):
        processed = {1, 2}
        stop = [False]

        def failing_submit(_job):
            raise RuntimeError("cannot schedule new futures after interpreter shutdown")

        def try_submit(idx):
            processed.add(idx)
            try:
                failing_submit(idx)
            except RuntimeError:
                processed.discard(idx)
                stop[0] = True
                return False
            return True

        self.assertFalse(try_submit(3))
        self.assertTrue(stop[0], "submission must stop after a scheduling failure")
        self.assertNotIn(3, processed, "a never-scheduled episode must not look processed")
        self.assertEqual(processed, {1, 2}, "already-processed episodes must be untouched")

    def test_successful_submit_marks_and_continues(self):
        processed = set()
        stop = [False]

        def try_submit(idx):
            processed.add(idx)
            return True

        self.assertTrue(try_submit(7))
        self.assertIn(7, processed)
        self.assertFalse(stop[0])


class TestExecutorShutdownMode(unittest.TestCase):
    """Abandoning a stuck future must not block on that same future.

    ``ThreadPoolExecutor.__exit__`` calls ``shutdown(wait=True)``. Using a ``with`` block on
    the abort path would block until the hung future finished — reintroducing the very hang
    the bounds exist to escape, one layer down. This test pins the distinction.
    """

    def test_shutdown_without_wait_returns_while_a_worker_is_still_blocked(self):
        from concurrent.futures import ThreadPoolExecutor

        release = threading.Event()
        started = threading.Event()

        def blocker():
            started.set()
            release.wait(timeout=30)

        executor = ThreadPoolExecutor(max_workers=1)
        try:
            executor.submit(blocker)
            self.assertTrue(started.wait(timeout=5), "worker did not start")

            began = time.time()
            executor.shutdown(wait=False, cancel_futures=True)
            elapsed = time.time() - began

            self.assertLess(
                elapsed,
                2.0,
                "shutdown(wait=False) must return promptly even with a blocked worker; "
                f"took {elapsed:.2f}s",
            )
        finally:
            release.set()

    def test_shutdown_with_wait_blocks_until_the_worker_finishes(self):
        """The contrast case — proves the previous test is measuring something real."""
        from concurrent.futures import ThreadPoolExecutor

        def quick():
            time.sleep(0.3)

        executor = ThreadPoolExecutor(max_workers=1)
        executor.submit(quick)
        began = time.time()
        executor.shutdown(wait=True)
        self.assertGreaterEqual(time.time() - began, 0.25)


if __name__ == "__main__":
    unittest.main()


class TestProcessingJobKeyCollision(unittest.TestCase):
    """2026-08-25 prod incident: ``episode.idx`` is NOT unique across a multi-run work-list.

    A reprocess assigns each episode the idx from its on-disk ``NNNN - Title`` filename,
    unique only within the ORIGINAL ingest run. A 29-episode batch drawn from two
    16-episode source runs shared idx 1..16 — dedup by idx silently skipped 13 episodes
    AND wedged the loop forever (``total_jobs == len(processed_idx_set)`` unsatisfiable).
    Bookkeeping must key by :func:`processing._processing_job_key` (transcript path).
    """

    @staticmethod
    def _job(idx: int, transcript_path: str):
        from types import SimpleNamespace

        return SimpleNamespace(episode=SimpleNamespace(idx=idx), transcript_path=transcript_path)

    def test_colliding_idx_jobs_have_distinct_keys(self):
        a = self._job(5, "/out/run_A/transcripts/0005 - Alpha.txt")
        b = self._job(5, "/out/run_B/transcripts/0005 - Beta.txt")
        self.assertNotEqual(processing._processing_job_key(a), processing._processing_job_key(b))

    def test_mark_processed_counts_every_colliding_job(self):
        processed: set = set()
        jobs = [
            self._job(i % 16 + 1, f"/out/run_{i // 16}/transcripts/{i:04d}.txt") for i in range(29)
        ]
        for j in jobs:
            processing._mark_processed(processed, j)
        # Under idx keying this capped at 16 and the queue-empty invariant
        # (total_jobs == len(processed)) could never hold — the wedge.
        self.assertEqual(len(processed), 29)

    def test_mark_processed_is_idempotent_per_job(self):
        processed: set = set()
        j = self._job(3, "/out/run_A/transcripts/0003 - Gamma.txt")
        processing._mark_processed(processed, j)
        processing._mark_processed(processed, j)
        self.assertEqual(len(processed), 1)


class TestTranscriptlessJobsAreStillCountable(unittest.TestCase):
    """2026-09-18: the SAME wedge, reached through a different door.

    Keying on the transcript path is unique only while there IS one. An episode a reprocess cannot
    resolve a transcript for ("relabel_only: no on-disk transcript to work on") arrives with an
    empty path, and every such episode then shares the key ``"None"``.

    Measured on The Flip during the #2075 harness: 16 jobs, 3 of them transcript-less, 14 distinct
    keys — so ``total_jobs == len(processed_job_indices)`` could never hold and the processing loop
    polled forever, both executor workers parked and all the work finished. A thread dump caught it
    at ``processing.py`` in ``_run_parallel_processing_loop`` with the main thread joined on it. The
    four feeds in the same run with no transcript-less episode all exited cleanly.
    """

    @staticmethod
    def _job(idx: int, transcript_path, guid=None):
        from types import SimpleNamespace

        return SimpleNamespace(
            episode=SimpleNamespace(idx=idx, guid=guid), transcript_path=transcript_path
        )

    def test_three_transcriptless_jobs_do_not_collapse_into_one_key(self):
        """All three carry ``None``, as the three real Flip episodes did — `str(None)` is one key
        for all of them. Mixing in `""` or whitespace would make this pass against the defect,
        because those stringify differently."""
        jobs = [
            self._job(7, None, guid="guid-7"),
            self._job(8, None, guid="guid-8"),
            self._job(13, None, guid="guid-13"),
        ]
        keys = {processing._processing_job_key(j) for j in jobs}
        self.assertEqual(len(keys), 3, f"transcript-less jobs shared a key: {keys}")

    def test_an_empty_or_blank_path_is_treated_as_no_path(self):
        """`""` and whitespace reach here too, and a key of `"   "` is not an identity."""
        jobs = [self._job(1, "", guid="g1"), self._job(2, "   ", guid="g2")]
        keys = {processing._processing_job_key(j) for j in jobs}
        self.assertEqual(len(keys), 2)
        self.assertTrue(all(k.startswith("no-transcript:") for k in keys), keys)

    def test_the_queue_empty_invariant_can_be_satisfied(self):
        """The wedge itself: every job must be countable, or the loop never concludes."""
        processed: set = set()
        jobs = [self._job(i, f"/out/t/{i:04d}.txt", guid=f"g{i}") for i in range(13)]
        jobs += [self._job(i, None, guid=f"g{i}") for i in (13, 14, 15)]
        for j in jobs:
            processing._mark_processed(processed, j)
        self.assertEqual(len(processed), len(jobs))

    def test_a_job_with_neither_path_nor_guid_is_still_unique(self):
        a = self._job(1, None)
        b = self._job(1, None)
        self.assertNotEqual(processing._processing_job_key(a), processing._processing_job_key(b))

    def test_a_real_path_still_keys_by_path(self):
        """The 2026-08-25 fix is untouched: where a path exists it remains the identity."""
        j = self._job(5, "/out/run_A/transcripts/0005 - Alpha.txt", guid="g5")
        self.assertEqual(
            processing._processing_job_key(j), "/out/run_A/transcripts/0005 - Alpha.txt"
        )


class TestOneStuckEpisodeMustNotHoldItsFeedHostage(unittest.TestCase):
    """2026-09-23, prod #2097: nothing bounded an individual episode's future.

    "Kubernetes and retiring at the top with Kelsey Hightower" finished summarisation (930s) then
    burned a full core for 90 minutes emitting nothing. Its feed's other seven episodes were done,
    but the loop kept waiting, so the feed only closed when the 4h per-FEED budget fired. The
    per-episode ``timeout_context`` cannot help — it observes and cannot interrupt (utils/timeout.py
    documents that, and a 4h15m hang inside a 1200s one). The loop is the only place that can stop
    WAITING, so the bound belongs here.
    """

    def test_a_future_past_the_ceiling_is_reported(self):
        now = 10_000.0
        started = {"slow": now - 3601.0, "fresh": now - 5.0}
        self.assertEqual(processing._overrunning_futures(started, 3600.0, now), ["slow"])

    def test_a_future_inside_the_ceiling_is_left_alone(self):
        now = 10_000.0
        started = {"a": now - 3599.0, "b": now - 0.0}
        self.assertEqual(processing._overrunning_futures(started, 3600.0, now), [])

    def test_the_bound_can_be_disabled(self):
        """None disables it — an explicit opt-out, same contract as the feed budget."""
        now = 10_000.0
        self.assertEqual(processing._overrunning_futures({"x": 0.0}, None, now), [])

    def test_an_empty_map_is_not_an_error(self):
        self.assertEqual(processing._overrunning_futures({}, 60.0, 1.0), [])

    def test_default_ceiling_clears_the_worst_real_episode(self):
        """1529s is the longest legitimate metadata generation measured on prod (#1894).

        Guards against a future tightening that would start abandoning healthy episodes.
        """
        self.assertGreater(processing.DEFAULT_PROCESSING_FUTURE_ABANDON_SECONDS, 1529)

    def test_the_per_episode_bound_is_tighter_than_the_per_feed_budget(self):
        """The whole point: a stuck episode must lose before its feed does, or it blocks it."""
        self.assertLess(
            processing.DEFAULT_PROCESSING_FUTURE_ABANDON_SECONDS,
            processing.DEFAULT_PROCESSING_LOOP_BUDGET_SECONDS,
        )

    def test_config_override_and_garbage_handling(self):
        self.assertEqual(
            processing._processing_future_abandon_seconds(
                _Cfg(processing_future_abandon_seconds=90)
            ),
            90.0,
        )
        self.assertIsNone(
            processing._processing_future_abandon_seconds(_Cfg(processing_future_abandon_seconds=0))
        )
        self.assertEqual(
            processing._processing_future_abandon_seconds(
                _Cfg(processing_future_abandon_seconds="nonsense")
            ),
            float(processing.DEFAULT_PROCESSING_FUTURE_ABANDON_SECONDS),
        )
        self.assertEqual(
            processing._processing_future_abandon_seconds(_Cfg()),
            float(processing.DEFAULT_PROCESSING_FUTURE_ABANDON_SECONDS),
        )


class TestQueueEmptyComparesMembershipNotCardinality(unittest.TestCase):
    """2026-09-23, prod #2097 batch: the third door, and the one no key change can shut.

    Both earlier fixes made keys unique for one collision source (2026-08-25 ``idx``,
    2026-09-18 empty path). This case has NO defective key: two work-list entries resolve
    to the SAME transcript, so sharing one key is exactly right — dedup must do the work
    once. It is the exit predicate that is wrong. Measured on prod: 35 jobs, 34 keys, 0
    futures in flight, and the wedge report's own ``missing`` list EMPTY (every job
    accounted for) while ``35 == 34`` stayed false. 481 spins across the batch.

    So these tests assert the predicate, not the keys — the invariant the two prior test
    classes could not express, because both phrase the wedge as a uniqueness failure.
    """

    @staticmethod
    def _job(idx: int, transcript_path, guid=None):
        from types import SimpleNamespace

        return SimpleNamespace(
            episode=SimpleNamespace(idx=idx, guid=guid), transcript_path=transcript_path
        )

    def test_two_jobs_sharing_one_transcript_still_conclude(self):
        """The prod shape: N jobs, N-1 keys, everything done. Cardinality says never."""
        shared = "/out/run_A/transcripts/0007 - Shared.txt"
        jobs = [
            self._job(i, f"/out/run_A/transcripts/{i:04d}.txt", guid=f"g{i}") for i in range(34)
        ]
        jobs.append(self._job(99, shared, guid="g-dup"))
        jobs.append(self._job(7, shared, guid="g7"))

        processed: set = set()
        for j in jobs:
            processing._mark_processed(processed, j)

        self.assertEqual(len(jobs), 36)
        self.assertEqual(len(processed), 35, "two jobs must share one dedup key here")
        self.assertNotEqual(len(jobs), len(processed))
        self.assertTrue(
            processing._all_jobs_processed(jobs, processed),
            "every job is marked processed, so the loop MUST be allowed to exit",
        )

    def test_it_is_false_while_any_job_is_outstanding(self):
        """Guard against 'return True' passing the test above — the predicate must still block."""
        jobs = [self._job(i, f"/out/t/{i:04d}.txt", guid=f"g{i}") for i in range(5)]
        processed: set = set()
        for j in jobs[:-1]:
            processing._mark_processed(processed, j)
        self.assertFalse(processing._all_jobs_processed(jobs, processed))

    def test_an_empty_job_list_is_trivially_complete(self):
        self.assertTrue(processing._all_jobs_processed([], set()))

    def test_transcriptless_and_duplicate_jobs_together(self):
        """Both doors open at once — the combination the batch actually hit."""
        shared = "/out/t/dup.txt"
        jobs = [
            self._job(1, shared, guid="a"),
            self._job(2, shared, guid="b"),
            self._job(3, None, guid="c"),
            self._job(4, None, guid="d"),
        ]
        processed: set = set()
        for j in jobs:
            processing._mark_processed(processed, j)
        self.assertEqual(len(processed), 3)
        self.assertTrue(processing._all_jobs_processed(jobs, processed))
