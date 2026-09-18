"""The single list of digest enqueuers, and the one loop that drives them (#2119).

WHY THIS EXISTS
---------------
There are two processes that fire digests, and they must stay in step:

* ``server/scheduler.py`` — the in-process APScheduler, used by the operator surface.
* ``infra/deploy/digest_scheduler.py`` — the sidecar container, which is the ONLY path in
  production for the player (it runs ``PODCAST_SERVE_APP_ONLY=1``, ADR-116, which force-disables
  the in-process scheduler).

They were maintained separately, and they drifted. ``daily_recap`` (#2039) and the monthly
``recommendations`` digest (wave-H) were added to the in-process scheduler and not to the
sidecar, so both shipped dead to production and stayed dead for their entire lives — silently,
because the sidecar logged a healthy ``tick: enqueued 0 envelope(s)`` every hour.

Adding an enqueuer to :data:`ENQUEUERS` now wires BOTH callers at once. There is no second list
to forget.

DESIGN NOTES
------------
* Enqueuers are declared by module/function NAME and imported lazily inside
  :func:`enqueue_all_due`. The sidecar is a standalone script that must not drag the FastAPI app
  in at import time, and the scheduler's digest branch is already a lazy-import site.
* Each enqueuer is isolated. The cadences are independent; a broken assembler in one must not
  silence every notification the product has. A failure is recorded against that enqueuer and
  the rest still run.
* The result is per-enqueuer, never a bare total. A single number across several enqueuers
  cannot distinguish "nobody is due" from "this enqueuer is never called", and that ambiguity
  is exactly what hid #2119 for months.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

#: label -> (module under ``podcast_scraper.server``, callable name).
#: Every entry is driven by BOTH the in-process scheduler and the production sidecar.
#: Each callable takes ``(corpus_root: Path, data_dir: Path)`` and returns enqueued envelope ids.
ENQUEUERS: tuple[tuple[str, str, str], ...] = (
    # "Your Week" — weekly slot gate, per-ISO-week envelope id (#1412).
    ("weekly", "app_digest_personal", "enqueue_due_digests"),
    # Monthly recommendations (wave-H) — 1st of the month at the user's hour, per-month id.
    ("recommendations", "app_digest_recommendations", "enqueue_due_recommendations"),
    # Daily post-episode recap (#2039, RFC-122) — daily hour slot, per-day id.
    ("daily_recap", "app_digest_daily_recap", "enqueue_due_daily_recaps"),
)


@dataclass(frozen=True)
class DispatchResult:
    """Outcome of one dispatch pass across every enqueuer."""

    #: label -> enqueued envelope ids (empty when nobody was due).
    ids: dict[str, list[str]] = field(default_factory=dict)
    #: label -> repr of the exception, for enqueuers that raised.
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def all_ids(self) -> list[str]:
        out: list[str] = []
        for label, _, _ in ENQUEUERS:
            out.extend(self.ids.get(label, []))
        return out

    @property
    def total(self) -> int:
        return len(self.all_ids)

    def summary(self) -> str:
        """``weekly=2 recommendations=0 daily_recap=ERR`` — per-enqueuer, never a bare total."""
        parts = []
        for label, _, _ in ENQUEUERS:
            if label in self.errors:
                parts.append(f"{label}=ERR")
            else:
                parts.append(f"{label}={len(self.ids.get(label, []))}")
        return " ".join(parts)


def enqueue_all_due(corpus_root: Path, data_dir: Path, now: int | None = None) -> DispatchResult:
    """Run every enqueuer in :data:`ENQUEUERS`, isolated from one another.

    Each enqueuer applies its own consent + cadence-slot gate and dedupes on a per-period
    envelope id, so calling this hourly is safe and idempotent.

    Never raises for an enqueuer failure — the failure is recorded in
    :attr:`DispatchResult.errors` and the remaining enqueuers still run.
    """
    ids: dict[str, list[str]] = {}
    errors: dict[str, str] = {}

    for label, module_name, func_name in ENQUEUERS:
        try:
            module = importlib.import_module(f"podcast_scraper.server.{module_name}")
            func = getattr(module, func_name)
            # ``now`` is threaded through only when the enqueuer accepts it, so this stays
            # tolerant of enqueuers with the shorter two-arg signature.
            result = (
                func(corpus_root, data_dir) if now is None else func(corpus_root, data_dir, now)
            )
            ids[label] = list(result)
        except Exception as exc:  # noqa: BLE001 — one bad enqueuer must not mute the rest
            logger.exception("digest dispatch: enqueuer %r failed", label)
            errors[label] = repr(exc)

    return DispatchResult(ids=ids, errors=errors)
