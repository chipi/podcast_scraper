"""The transcription loop must be unable to wait forever.

WHY THIS EXISTS. The processing loop got termination bounds after the 2026-08-12 incident
(#1180). The transcription loop — same shape, same hazard — did not, and on 2026-08-19 prod was
found with a container "Up 7 days" caused by exactly this class of wedge in the sibling thread.

The transcription loop's only exit is ``downloads_complete_event`` plus an empty queue. That
event is set by the main thread AFTER ``process_episodes``, which deliberately re-raises
``CostCapExceeded`` (processing.py:1605) and ``ResilienceFuseOpenError`` (processing.py:1607, the
ADR-122 reprocess-mode halt). An escape there left the event unset; the loop is non-daemon, so the
process could never exit.

orchestration now sets both events from a finally, which closes the known escape. These bounds are
the backstop that does NOT depend on the caller getting that right: no matter which line the main
thread dies on, a worker must not outlive its parent. Two independent conditions, mirroring #1180:
main-thread liveness, and a wall-clock budget.
"""

from __future__ import annotations

import ast
import inspect
import threading
import time
from types import SimpleNamespace

import pytest

from podcast_scraper.workflow.stages import transcription

pytestmark = [pytest.mark.unit]


# -- the budget -------------------------------------------------------------------------------


def test_a_config_that_never_heard_of_the_setting_still_gets_a_bound() -> None:
    """The important case: the bound must apply to every existing deployment without opt-in.

    The incident happened on a config that had no such key, so a bound that requires
    configuration is a bound that would not have fired.
    """
    budget = transcription._transcription_loop_budget_seconds(SimpleNamespace())
    assert budget == float(transcription.DEFAULT_TRANSCRIPTION_LOOP_BUDGET_SECONDS)
    assert budget > 0


def test_an_explicit_budget_is_honoured() -> None:
    cfg = SimpleNamespace(transcription_loop_budget_seconds=120)
    assert transcription._transcription_loop_budget_seconds(cfg) == 120.0


@pytest.mark.parametrize("bad", [0, -1, "nonsense", None])
def test_a_nonsense_budget_falls_back_to_the_default_rather_than_disabling_the_bound(bad) -> None:
    """Fail SAFE: a typo must not silently remove the only thing preventing an immortal loop."""
    cfg = SimpleNamespace(transcription_loop_budget_seconds=bad)
    assert transcription._transcription_loop_budget_seconds(cfg) == float(
        transcription.DEFAULT_TRANSCRIPTION_LOOP_BUDGET_SECONDS
    )


def test_the_default_is_generous_enough_not_to_kill_real_work() -> None:
    """ASR is minutes per episode; this bound exists to make 'forever' impossible, not to
    second-guess scheduling. A too-tight bound would abandon legitimate long batches."""
    assert transcription.DEFAULT_TRANSCRIPTION_LOOP_BUDGET_SECONDS >= 4 * 60 * 60


# -- the exit conditions ----------------------------------------------------------------------


def test_a_healthy_loop_is_told_to_continue() -> None:
    assert (
        transcription._transcription_supervision_exit_reason(time.time(), 3600.0) is None
    ), "supervision must not stop a loop that is fine"


def test_an_overrunning_loop_is_stopped() -> None:
    started = time.time() - 10_000
    reason = transcription._transcription_supervision_exit_reason(started, 3600.0)
    assert reason is not None and "wall-clock budget exceeded" in reason


def test_a_worker_must_not_outlive_a_DEAD_MAIN_THREAD() -> None:
    """The condition that makes the wedge impossible regardless of any event.

    Checked from a worker thread with a stubbed main_thread that reports dead — the real main
    thread is alive during tests, and the property under test is precisely what happens when it
    is not.
    """
    captured = {}

    class _DeadMain:
        @staticmethod
        def is_alive() -> bool:
            return False

    real = threading.main_thread

    def worker():
        threading.main_thread = _DeadMain  # type: ignore[assignment]
        try:
            captured["reason"] = transcription._transcription_supervision_exit_reason(
                time.time(), 3600.0
            )
        finally:
            threading.main_thread = real  # type: ignore[assignment]

    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout=5)

    assert captured.get("reason") == "main thread exited"


def test_main_thread_death_wins_even_when_the_budget_is_untouched() -> None:
    """The two bounds are independent; neither may mask the other."""

    class _DeadMain:
        @staticmethod
        def is_alive() -> bool:
            return False

    real = threading.main_thread
    threading.main_thread = _DeadMain  # type: ignore[assignment]
    try:
        # a budget of a century — only liveness can stop this
        reason = transcription._transcription_supervision_exit_reason(time.time(), 3.15e9)
    finally:
        threading.main_thread = real  # type: ignore[assignment]
    assert reason == "main thread exited"


# -- the loops actually consult it --------------------------------------------------------------


def test_BOTH_loops_consult_supervision_not_just_one() -> None:
    """Structural. Guarding only one loop is the mistake already made once on this branch:
    the first version of the orchestration fix released the processing thread and left the
    transcription thread exposed to the identical wedge."""
    src = inspect.getsource(transcription.process_transcription_jobs_concurrent)
    tree = ast.parse(src.lstrip())

    # Only `while True:` loops. A `while <condition>:` has an intrinsic exit — the submit helper's
    # `while len(futures) < max_workers` drains a queue and stops on queue.Empty, so it cannot
    # wedge and does not need a bound. `while True` has no exit at all except the ones written
    # inside it, which is exactly the shape that hung for 7 days.
    whiles = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.While) and isinstance(n.test, ast.Constant) and n.test.value is True
    ]
    assert len(whiles) >= 2, f"expected the sequential and parallel loops, found {len(whiles)}"

    def consults(node) -> bool:
        return any(
            isinstance(sub, ast.Call)
            and (getattr(sub.func, "id", None) or getattr(sub.func, "attr", None))
            in ("_transcription_supervision_exit_reason", "_tx_must_stop")
            for sub in ast.walk(node)
        )

    unguarded = [w for w in whiles if not consults(w)]
    assert not unguarded, (
        f"{len(unguarded)} transcription while-loop(s) never consult supervision — such a loop "
        "can only exit via downloads_complete_event, so a main thread that dies before setting "
        "it leaves this non-daemon thread running forever."
    )
