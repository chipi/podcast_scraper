"""One spend ledger for the whole CLI invocation — the number that bounds what a run can cost.

WHY THIS EXISTS (2026-08-18). ``cost_soft_cap_usd_per_run: 5.0`` with ``action: abort`` was
configured and active while a single repair spent roughly $48. Three separate properties of the
old design each made that possible, and all three are properties of *where the number lived*:

1.  It lived on ``Metrics``, which ``_setup_pipeline_environment`` rebuilds per ``run_pipeline``
    call (orchestration.py), and ``cli`` calls ``run_pipeline`` once per feed. "Per run" therefore
    meant "per feed": a 14-feed batch had a $70 ceiling, not $5.
2.  It was read by summing seven hardcoded ``llm_*`` attributes. ``diarization_cost_usd`` is
    accumulated and was not among them, so an entire paid stage was invisible — and so is every
    future stage that someone forgets to add to that list.
3.  Nothing consulted it until after a stage had finished. Transcription runs to completion in a
    background thread before the first check is reachable, so ASR — the dominant cost — could
    never be stopped mid-feed.

This module fixes (1) and (2) structurally. The ledger is scoped to the PROCESS, which is exactly
one CLI invocation and therefore exactly what an operator means by "this run"; and it is fed from
``record_provider_call_cost``, the one choke point every provider already routes through
(all eight provider namespaces including ml/diarization), so it counts money by observing spend
rather than by knowing field names. Adding a paid stage cannot silently escape it.

(3) is fixed by the ledger's *callers* — the pre-flight gate in the scraping stage and the
per-call authorisation in the transcription stage — which this module deliberately does not know
about. Its only job is to hold the number.

PROCESS-SCOPED STATE. A module-level singleton is the pragmatic choice, not a shortcut: the
recording choke point is called deep inside providers that hold no reference to any run object,
and the batch spans several ``run_pipeline`` calls that share no object either. The repo already
does exactly this for run correlation and the #562 gates, both of which reset between invocations
for the same reason ``reset()`` exists here.

THREAD SAFETY IS LOAD-BEARING. Transcription runs in a background thread while the processing
thread also records LLM spend, so every mutation is under a lock. ``check_and_reserve`` is a
single atomic operation for that reason: two threads that each separately asked "is there room?"
and then both spent would each be individually correct and jointly over the cap.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Literal, Optional

logger = logging.getLogger(__name__)

BudgetAction = Literal["abort", "warn", "observe"]


class RunBudget:
    """Accumulated spend for one CLI invocation, with an optional cap.

    ``cap_usd`` of ``None`` or ``<= 0`` means unbounded — the ledger still counts, so the run
    can report what it spent, but nothing is ever refused. That matches the previous behaviour of
    an unset cap and keeps this module safe to enable everywhere.
    """

    def __init__(self, cap_usd: Optional[float] = None, action: BudgetAction = "observe") -> None:
        self._lock = threading.RLock()
        self._spent = 0.0
        self._cap = float(cap_usd) if cap_usd is not None and float(cap_usd) > 0 else None
        self._action: BudgetAction = action
        self._tripped = False
        self._trip_reason: Optional[str] = None

    # -- configuration ---------------------------------------------------------------------

    @property
    def cap_usd(self) -> Optional[float]:
        return self._cap

    @property
    def action(self) -> BudgetAction:
        return self._action

    @property
    def enforced(self) -> bool:
        """True when a breach actually stops work (a cap is set AND the action is abort)."""
        return self._cap is not None and self._action == "abort"

    # -- accounting ------------------------------------------------------------------------

    def record(self, usd: Optional[float]) -> None:
        """Add an actually-incurred cost. Never raises — telemetry must not break a call."""
        if usd is None:
            return
        try:
            amount = float(usd)
        except (TypeError, ValueError):
            return
        if amount <= 0:
            return
        with self._lock:
            self._spent += amount

    @property
    def spent_usd(self) -> float:
        with self._lock:
            return round(self._spent, 6)

    @property
    def remaining_usd(self) -> float:
        """Budget left. ``inf`` when uncapped, and never negative once overspent."""
        with self._lock:
            if self._cap is None:
                return float("inf")
            return max(0.0, round(self._cap - self._spent, 6))

    # -- authorisation ---------------------------------------------------------------------

    def would_exceed(self, estimated_usd: float) -> bool:
        """Would spending ``estimated_usd`` on top of what is already spent breach the cap?

        False whenever there is no cap, so callers need no special case for the unbounded run.
        """
        with self._lock:
            if self._cap is None:
                return False
            return (self._spent + max(0.0, float(estimated_usd))) > self._cap

    def check_and_reserve(self, estimated_usd: float) -> bool:
        """Atomically authorise ``estimated_usd`` of about-to-happen spend.

        Returns True when the caller may proceed. Returns False — and latches the ledger as
        tripped — when the cap is ENFORCED and this spend would breach it. Under ``warn`` /
        ``observe`` the caller is always allowed to proceed; the breach is only reported.

        This does NOT add to ``spent_usd``: the actual cost arrives through ``record`` when the
        call completes, and double-counting an estimate against its own actual would make the
        ledger drift upward with every call. What "reserve" means here is the atomicity — the
        decision and the trip latch happen together under one lock, so two threads cannot both
        be told yes for the same remaining dollar.
        """
        with self._lock:
            if self._cap is None:
                return True
            projected = self._spent + max(0.0, float(estimated_usd))
            if projected <= self._cap:
                return True
            reason = (
                f"projected spend ${projected:.4f} would exceed the "
                f"${self._cap:.4f} per-run cap (already spent ${self._spent:.4f})"
            )
            if self._action == "abort":
                self._tripped = True
                self._trip_reason = reason
                logger.error("run budget: %s — refusing this work", reason)
                return False
            # warn / observe: say so, then let it through. An operator who chose not to abort
            # gets the signal without the stop.
            log = logger.warning if self._action == "warn" else logger.info
            log("run budget: %s (action=%s — allowed)", reason, self._action)
            return True

    def trip(self, reason: str) -> None:
        """Latch the ledger as tripped from outside (e.g. a post-hoc check found a breach)."""
        with self._lock:
            self._tripped = True
            self._trip_reason = reason

    @property
    def tripped(self) -> bool:
        """True once the cap has refused work.

        This is the signal a worker thread sets and the MAIN thread reads. Raising
        ``CostCapExceeded`` inside a worker only kills that worker — the main thread blocks on
        ``transcription_thread.join()`` and never learns, which is how a background thread was
        wedged on 2026-08-12. State crosses threads safely; exceptions do not.
        """
        with self._lock:
            return self._tripped

    @property
    def trip_reason(self) -> Optional[str]:
        with self._lock:
            return self._trip_reason

    def summary(self) -> str:
        """One line for logs: what was spent, against what."""
        with self._lock:
            if self._cap is None:
                return f"run budget: spent ${self._spent:.4f} (no cap configured)"
            return (
                f"run budget: spent ${self._spent:.4f} of ${self._cap:.4f} "
                f"(action={self._action}{', TRIPPED' if self._tripped else ''})"
            )


# -- module-level singleton -----------------------------------------------------------------

_budget = RunBudget()
_budget_lock = threading.Lock()


def get_run_budget() -> RunBudget:
    """The ledger for this process. Always returns one; an unconfigured ledger is uncapped."""
    return _budget


def configure_run_budget(cfg: Any) -> RunBudget:
    """(Re)configure the process ledger from a config, preserving spend already recorded.

    Called once per ``run_pipeline``. Spend is deliberately NOT reset: the multi-feed batch calls
    ``run_pipeline`` once per feed, and resetting here would restore precisely the per-feed
    accounting that let $48 through a $5 cap. Use ``reset_run_budget`` to start a fresh batch.
    """
    global _budget
    cap = getattr(cfg, "cost_soft_cap_usd_per_run", None)
    action = getattr(cfg, "cost_soft_cap_action", "observe") or "observe"
    if action not in ("abort", "warn", "observe"):
        action = "observe"
    with _budget_lock:
        carried = _budget.spent_usd
        was_tripped = _budget.tripped
        reason = _budget.trip_reason
        _budget = RunBudget(cap_usd=cap, action=action)  # type: ignore[arg-type]
        _budget.record(carried)
        if was_tripped and reason:
            _budget.trip(reason)
    return _budget


def reset_run_budget(
    cap_usd: Optional[float] = None, action: BudgetAction = "observe"
) -> RunBudget:
    """Start a fresh ledger. For tests, and for a genuinely new batch within one process."""
    global _budget
    with _budget_lock:
        _budget = RunBudget(cap_usd=cap_usd, action=action)
    return _budget
