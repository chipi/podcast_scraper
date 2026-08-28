"""Did the repair repair what it was asked to repair?

WHY THIS EXISTS (2026-08-18). A run was dispatched to fix 32 named episodes. It ran for six
hours, re-transcribed roughly 181 OTHER episodes, repaired none of the 32, and reported nothing
that made any of that visible. Establishing "it repaired zero of thirty-two" took a separate
audit against the live corpus API, after the money was gone.

The denominator is the whole point. A run that says ``repaired 32/32`` and a run that says
``repaired 0/32, unmatched: [...]`` are the same log line with different numbers, and one of them
is an incident. Without the line, the two are indistinguishable without an audit.

SCOPE, and why this is process-scoped like the run budget: a work-list is corpus-wide but the
pipeline runs once per feed, each with its own config and output directory. A 32-episode list
drawn from two feeds matches nothing in the other twelve — normal, not an error — so no single
feed can tell whether the batch as a whole succeeded. Only the process can.

WHAT "MATCHED" MEANS, precisely. It means selection found the episode on disk and included it in
the run. It does NOT mean the episode was successfully reprocessed: an episode can be selected
and then fail in transcription or enrichment. Matched-but-not-completed is reported separately
rather than folded in, because a report that overstates what it verified is how the last one
became invisible.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class WorklistReport:
    """What was asked for, what selection found, and what actually finished."""

    requested: Set[str] = field(default_factory=set)
    matched: Set[str] = field(default_factory=set)
    completed: Set[str] = field(default_factory=set)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    @staticmethod
    def _clean(ids: Iterable[Any]) -> Set[str]:
        """Normalise a list of ids, dropping empties.

        The None check is explicit and NOT `if str(i).strip()`: str(None) is "None", a perfectly
        truthy five-character string, so the naive form silently invents an episode called
        "None" and then reports it forever as NOT FOUND.
        """
        out: Set[str] = set()
        for i in ids:
            if i is None:
                continue
            key = str(i).strip()
            if key:
                out.add(key)
        return out

    def request(self, ids: Iterable[Any]) -> None:
        """Record the ids this run was asked to repair. Idempotent across feeds."""
        with self._lock:
            self.requested.update(self._clean(ids))

    def mark_matched(self, ids: Iterable[Any]) -> None:
        """Record ids that selection found on disk and included in the run."""
        with self._lock:
            self.matched.update(self._clean(ids))

    def mark_completed(self, episode_id: Optional[str], guid: Optional[str] = None) -> None:
        """Record that a requested episode finished its pipeline.

        Accepts either identifier because the work-list is matched on both: detectors emit
        whichever the artifact carries, so a list can name episode_ids, guids, or a mix.
        """
        with self._lock:
            for candidate in (episode_id, guid):
                key = str(candidate or "").strip()
                if key and key in self.requested:
                    self.completed.add(key)

    @property
    def active(self) -> bool:
        """True only when this run was given a work-list; otherwise there is nothing to report."""
        with self._lock:
            return bool(self.requested)

    @property
    def unmatched(self) -> List[str]:
        """Requested episodes that selection never found — the silent-failure case."""
        with self._lock:
            return sorted(self.requested - self.matched)

    @property
    def incomplete(self) -> List[str]:
        """Selected but never finished: started and then failed, rather than never started."""
        with self._lock:
            return sorted(self.matched - self.completed)

    def summary(self, *, max_listed: int = 20) -> str:
        """The line an operator should be able to read instead of running an audit."""
        with self._lock:
            total = len(self.requested)
            done = len(self.completed)
            line = f"work-list: repaired {done}/{total}"
            unmatched = sorted(self.requested - self.matched)
            incomplete = sorted(self.matched - self.completed)
            if unmatched:
                shown = ", ".join(unmatched[:max_listed])
                more = (
                    f" (+{len(unmatched) - max_listed} more)" if len(unmatched) > max_listed else ""
                )
                line += f" · {len(unmatched)} NOT FOUND in any feed's corpus: [{shown}]{more}"
            if incomplete:
                shown = ", ".join(incomplete[:max_listed])
                more = (
                    f" (+{len(incomplete) - max_listed} more)"
                    if len(incomplete) > max_listed
                    else ""
                )
                line += f" · {len(incomplete)} selected but did NOT finish: [{shown}]{more}"
            if not unmatched and not incomplete and total:
                line += " · all requested episodes repaired"
            return line

    def as_dict(self) -> Dict[str, Any]:
        """Machine-readable form for the run summary document."""
        with self._lock:
            return {
                "requested": len(self.requested),
                "matched": len(self.matched),
                "completed": len(self.completed),
                "unmatched_ids": sorted(self.requested - self.matched),
                "incomplete_ids": sorted(self.matched - self.completed),
            }


_report = WorklistReport()
_report_lock = threading.Lock()


def get_worklist_report() -> WorklistReport:
    """The report for this process."""
    return _report


def reset_worklist_report() -> WorklistReport:
    """Start a fresh report. For tests, and for a genuinely new batch within one process."""
    global _report
    with _report_lock:
        _report = WorklistReport()
    return _report


def log_worklist_outcome() -> Optional[str]:
    """Emit the outcome line at the end of a run. Returns it, or None when no work-list was given.

    Severity tracks WHAT is missing, not merely that something is (#1855):

    - ``incomplete`` — an episode whose repair STARTED and then failed. This is the partial-repair
      the loud level exists to catch, so ERROR (survives a log level that hides INFO).
    - ``unmatched`` with ZERO matched — the work-list matched NOTHING in any feed. This is the
      2026-08-18 incident: a run that quietly does nothing while costing money, learned about from a
      corpus audit two days later. It must be LOUD, so ERROR (the ``THE_INCIDENT`` tests capture at
      ERROR precisely to pin this).
    - ``unmatched`` while SOME matched — a few stale / orphaned ids alongside real repairs. Nothing
      lost: every repairable item repaired. So WARNING — visible, but not an error the signal-fleet
      auto-files as a bug on every sweep that carries one stale id.

    The zero-matched carve-out is why this is not simply "any unmatched → WARNING": collapsing the
    total miss into the stale-id case re-buries the exact incident this guard exists to keep loud.
    """
    report = get_worklist_report()
    if not report.active:
        return None
    line = report.summary()
    if report.incomplete or (report.unmatched and not report.matched):
        logger.error("%s", line)
    elif report.unmatched:
        logger.warning("%s", line)
    else:
        logger.info("%s", line)
    return line
