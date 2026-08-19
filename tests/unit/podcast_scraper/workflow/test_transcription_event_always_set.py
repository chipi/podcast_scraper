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
import threading
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
