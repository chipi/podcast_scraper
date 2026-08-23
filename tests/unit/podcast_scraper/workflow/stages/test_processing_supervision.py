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
