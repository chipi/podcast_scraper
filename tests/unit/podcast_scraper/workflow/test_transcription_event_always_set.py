"""A tripped cost cap must not leave the process unable to exit.

FOUND IN PRODUCTION, 2026-08-19. A container was discovered "Up 7 days". Its process had raised
``CostCapExceeded: cost soft cap exceeded: $12.4599 > $5.0000`` on 2026-08-12 09:28Z and then
never exited, so the container never died and nobody noticed for a week.

The mechanism, in ``orchestration.run_pipeline``:

  Step 9    joins the transcription thread, then calls check_cost_soft_cap_at_stage  -> RAISES
  Step 9.5  sets transcription_complete_event                                        <- SKIPPED

The ProcessingProcessor thread's continue-predicate waits on that event, so it never terminated.
The #1180 supervision work bounds that spin to DEFAULT_PROCESSING_LOOP_BUDGET_SECONDS (4 hours),
which downgrades a permanent zombie to a four-hour one rather than removing it.

WHY THIS MATTERS MORE NOW THAN IT DID. The cost work on this branch makes the cap ACTUALLY TRIP —
before it, the cap could barely fire at all, so the unwind was rarely reached. A working cap
sitting above an unwind hazard would have turned every successful cost-abort into a zombie: a fix
that stops the spending and hangs the run is not a fix.

These tests assert the invariant directly — the event is set on the exception path — because that
is the property the process's ability to exit depends on. They are deliberately about the SHAPE of
the code (try/finally) rather than about any one exception type: a provider error or a disk
failure escaping Step 9 wedges identically.
"""

from __future__ import annotations

import ast
import inspect
import sys
import threading
import time
from pathlib import Path

import pytest

from podcast_scraper.workflow import orchestration
from podcast_scraper.workflow.cost_monitoring import CostCapExceeded

pytestmark = [pytest.mark.unit]


def _run_pipeline_source() -> str:
    """Source of the function that actually owns Step 9.

    NOT ``run_pipeline``: Step 9 lives in ``_process_episodes_with_threading``, which
    ``run_pipeline`` calls. My first version of this test read run_pipeline, found no try/finally
    because there is none to find there, and reported a hazard that had already been fixed — a
    test asserting against the wrong function is indistinguishable from a real failure.
    """
    return inspect.getsource(orchestration._process_episodes_with_threading)


def test_the_completion_event_is_set_from_a_FINALLY_not_just_the_happy_path() -> None:
    """The structural invariant: no path through Step 9 may skip the event.

    Asserted on the AST rather than by string matching, so reformatting cannot silently
    "fix" the test while the hazard returns.
    """
    tree = ast.parse(_run_pipeline_source().lstrip())
    fn = tree.body[0]

    def sets_the_event(nodes) -> bool:
        for node in nodes:
            for sub in ast.walk(node):
                if (
                    isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr == "set"
                    and isinstance(sub.func.value, ast.Name)
                    and sub.func.value.id == "transcription_complete_event"
                ):
                    return True
        return False

    protected = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Try) and node.finalbody and sets_the_event(node.finalbody)
    ]
    assert protected, (
        "transcription_complete_event.set() is not in any finally: block. An exception from "
        "the transcription stage — a tripped cost cap above all — will unwind past it and leave "
        "the processing thread waiting forever. This is the 2026-08-12 wedge; a container was "
        "found Up 7 days from it on 2026-08-19."
    )


def test_the_cost_cap_CHECK_is_inside_that_protected_block() -> None:
    """It is not enough that a finally exists — the raising call must be under it."""
    tree = ast.parse(_run_pipeline_source().lstrip())

    def calls_cap_check(node) -> bool:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                fname = getattr(sub.func, "id", None) or getattr(sub.func, "attr", None)
                if fname == "check_cost_soft_cap_at_stage":
                    return True
        return False

    def sets_event(nodes) -> bool:
        return any(
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and sub.func.attr == "set"
            and isinstance(sub.func.value, ast.Name)
            and sub.func.value.id == "transcription_complete_event"
            for node in nodes
            for sub in ast.walk(node)
        )

    covered = [
        t
        for t in ast.walk(tree)
        if isinstance(t, ast.Try)
        and t.finalbody
        and sets_event(t.finalbody)
        and any(calls_cap_check(stmt) for stmt in t.body)
    ]
    assert covered, (
        "check_cost_soft_cap_at_stage is NOT inside the try whose finally sets the completion "
        "event, so a tripped cap still unwinds past it."
    )


def test_the_event_semantics_this_relies_on() -> None:
    """threading.Event.set() is idempotent, which is why Step 9.5 can keep its own call."""
    ev = threading.Event()
    ev.set()
    ev.set()
    assert ev.is_set() is True


def test_a_finally_really_does_run_when_CostCapExceeded_propagates() -> None:
    """The behaviour the structural tests stand in for, demonstrated concretely."""
    event = threading.Event()

    def step9():
        try:
            raise CostCapExceeded(12.4599, 5.0)
        finally:
            event.set()

    with pytest.raises(CostCapExceeded):
        step9()
    assert event.is_set(), "the processing thread would never have been released"


def test_the_incident_is_recorded_where_the_next_reader_will_be() -> None:
    """A structural guard nobody understands gets deleted by the next person who reformats.

    The prod evidence lives in the source comment, not only in a commit message, because the
    commit message is not what someone reads when they are looking at this code.
    """
    src = Path(orchestration.__file__).read_text(encoding="utf-8")
    assert "Up 7 DAYS" in src or "Up 7 days" in src
    assert "2026-08-12" in src


# ---------------------------------------------------------------------------
# #1570/#1564: the same finally must ALSO join the workers on the exception path,
# so provider cleanup (run_pipeline's finally) cannot reset the shared summary
# provider's init flags while the released ProcessingProcessor is still draining
# (the "OpenAIProvider not initialized" clean_transcript storm, 12x/52s).
# ---------------------------------------------------------------------------


def test_the_exception_path_joins_workers_before_returning_to_cleanup() -> None:
    """Structural: the finally that sets the completion event also joins the workers.

    Asserted on the AST — the Step-9.5 join sits AFTER the try and is skipped when Step 9 raises,
    so without a join inside the finally the exception unwinds to provider cleanup with a live
    worker still using that provider.
    """
    tree = ast.parse(_run_pipeline_source().lstrip())

    def sets_event(nodes) -> bool:
        return any(
            isinstance(sub, ast.Call)
            and isinstance(sub.func, ast.Attribute)
            and sub.func.attr == "set"
            and isinstance(sub.func.value, ast.Name)
            and sub.func.value.id == "transcription_complete_event"
            for node in nodes
            for sub in ast.walk(node)
        )

    def joins_workers(nodes) -> bool:
        return any(
            isinstance(sub, ast.Call)
            and getattr(sub.func, "id", None) == "_join_worker_threads_before_cleanup"
            for node in nodes
            for sub in ast.walk(node)
        )

    protected = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Try)
        and node.finalbody
        and sets_event(node.finalbody)
        and joins_workers(node.finalbody)
    ]
    assert protected, (
        "_join_worker_threads_before_cleanup() is not called in the finally that releases the "
        "workers. On the exception path the Step-9.5 join is skipped, so cleanup resets the shared "
        "summary provider while the ProcessingProcessor still drains -> clean_transcript raises "
        "'not initialized' (#1570/#1564)."
    )


def test_the_join_is_gated_on_the_exception_path_only() -> None:
    """The join must be under a ``sys.exc_info()`` guard — the success path is Step 9.5's job,
    and joining unconditionally here would double the work (harmless) but muddy the contract."""
    src = _run_pipeline_source()
    assert "sys.exc_info()" in src and "_join_worker_threads_before_cleanup" in src, (
        "the exception-path join should be guarded by sys.exc_info() so it only runs while an "
        "exception is propagating, not on the normal success path."
    )


def test_join_helper_blocks_until_a_live_worker_finishes() -> None:
    """Behavioural: the helper actually waits for a still-running thread (the property cleanup-
    ordering depends on)."""
    done = threading.Event()

    def worker() -> None:
        time.sleep(0.05)
        done.set()

    t = threading.Thread(target=worker, name="ProcessingProcessor")
    t.start()
    orchestration._join_worker_threads_before_cleanup([None, t], num_episodes=1)
    assert done.is_set() and not t.is_alive(), "helper returned before the worker finished"


def test_cleanup_runs_only_after_the_worker_is_joined_on_the_exception_path() -> None:
    """Behavioural end-to-end of the fix's shape: a worker still running when CostCapExceeded
    propagates must be joined BEFORE the outer cleanup runs — i.e. no cleanup-during-live-worker."""
    order: list[str] = []

    def worker() -> None:
        time.sleep(0.05)
        order.append("worker_done")

    t = threading.Thread(target=worker, name="ProcessingProcessor")
    t.start()

    def guarded_step9() -> None:
        try:
            raise CostCapExceeded(9.0, 5.0)
        finally:
            # Mirrors the orchestration finally: release + (exception path) join before unwinding.
            if sys.exc_info()[0] is not None:
                orchestration._join_worker_threads_before_cleanup([t], num_episodes=1)

    with pytest.raises(CostCapExceeded):
        try:
            guarded_step9()
        finally:
            order.append("cleanup")  # stands in for run_pipeline's _cleanup_providers

    assert order == ["worker_done", "cleanup"], (
        "provider cleanup ran while the worker was still alive — the #1570 use-after-cleanup "
        f"window is open (order was {order})."
    )
