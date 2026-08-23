"""Budget HEADROOM — how much room is left, not how much was spent (#1651).

Every cost signal in this estate is spend-triggered: ``cost_soft_cap_usd_per_run`` compares
accumulated spend, the ops card reports spend, the metrics report spend. All of them answer
"what happened". None answers "how much is left". So an account about to run dry looks
identical to a healthy one, and the first notice of exhaustion is a production failure.

That is not hypothetical. Job ``8645ecd0`` (2026-08-13 02:12Z) died mid-corpus with
``no budget/credit left on this key — this is NOT retryable, so the run is hard-stopping``,
and twelve auto-filed escalations (#1622-#1629, #1634-#1636, #1638) trace to the same
condition.

The generalisable shape, recorded after the 2026-08-12/13 outages and confirmed again by
#1646: *every detection signal is exception-triggered, so a process that stops working
invisibly looks healthy; every cost signal is spend-triggered, so an account about to run dry
looks healthy.* The estate measures what happened and never measures what remains.

This module is the "what remains" half, deliberately kept free of provider SDKs: callers
supply the numbers they can observe, and this decides whether a batch may proceed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

#: Refuse to start a batch projected to consume more than this share of remaining budget.
#: Below 1.0 on purpose: finishing a batch with nothing left means the NEXT run hard-stops
#: mid-corpus, which is the failure this exists to prevent, merely deferred by one job.
DEFAULT_MAX_CONSUMPTION_RATIO = 0.8


class HeadroomVerdict(str, Enum):
    """Whether a planned batch may proceed."""

    OK = "ok"
    TIGHT = "tight"  # proceeds, but the operator is told
    INSUFFICIENT = "insufficient"  # refuse BEFORE starting
    UNKNOWN = "unknown"  # headroom could not be determined


@dataclass(frozen=True)
class HeadroomCheck:
    """The decision plus every number behind it.

    The inputs travel with the verdict because "insufficient budget" with no arithmetic is
    indistinguishable from a bug, and an operator who cannot check the reasoning will override
    it — which is how a soft cap becomes decorative.
    """

    verdict: HeadroomVerdict
    remaining_usd: Optional[float]
    projected_cost_usd: Optional[float]
    consumption_ratio: Optional[float]
    reason: str

    @property
    def may_proceed(self) -> bool:
        """UNKNOWN proceeds: refusing on unmeasurable headroom would block every run on a
        gateway that does not report budgets, trading a real failure for a certain one."""
        return self.verdict is not HeadroomVerdict.INSUFFICIENT

    def explain(self) -> str:
        """One line for the run log / job record."""
        if self.remaining_usd is None or self.projected_cost_usd is None:
            return f"budget headroom: {self.verdict.value} — {self.reason}"
        return (
            f"budget headroom: {self.verdict.value} — projected ${self.projected_cost_usd:.2f} "
            f"against ${self.remaining_usd:.2f} remaining "
            f"({(self.consumption_ratio or 0) * 100:.0f}% of what is left) — {self.reason}"
        )


def check_headroom(
    *,
    remaining_usd: Optional[float],
    projected_cost_usd: Optional[float],
    max_consumption_ratio: float = DEFAULT_MAX_CONSUMPTION_RATIO,
) -> HeadroomCheck:
    """Decide whether a batch projected to cost *projected_cost_usd* may start.

    Called BEFORE the batch is queued. A cap checked only during a run cannot prevent the
    failure it exists to prevent — it can only interrupt it halfway, leaving a partially
    ingested corpus, which is exactly what ``8645ecd0`` did.
    """
    if remaining_usd is None:
        return HeadroomCheck(
            HeadroomVerdict.UNKNOWN,
            None,
            projected_cost_usd,
            None,
            "remaining budget could not be read from the gateway",
        )
    if projected_cost_usd is None:
        return HeadroomCheck(
            HeadroomVerdict.UNKNOWN,
            remaining_usd,
            None,
            None,
            "no cost projection was supplied for this batch",
        )

    if remaining_usd <= 0:
        return HeadroomCheck(
            HeadroomVerdict.INSUFFICIENT,
            remaining_usd,
            projected_cost_usd,
            None,
            "no budget remaining — the next provider call will hard-stop the run",
        )

    ratio = projected_cost_usd / remaining_usd
    if ratio > 1.0:
        verdict, reason = (
            HeadroomVerdict.INSUFFICIENT,
            "projected cost exceeds remaining budget; the run would hard-stop mid-corpus",
        )
    elif ratio > max_consumption_ratio:
        verdict, reason = (
            HeadroomVerdict.INSUFFICIENT,
            f"projected cost would consume more than {max_consumption_ratio:.0%} of remaining "
            "budget, leaving the next run to hard-stop instead",
        )
    elif ratio > max_consumption_ratio / 2:
        verdict, reason = (
            HeadroomVerdict.TIGHT,
            "proceeding, but this batch consumes a large share of what is left",
        )
    else:
        verdict, reason = HeadroomVerdict.OK, "sufficient headroom"

    return HeadroomCheck(verdict, remaining_usd, projected_cost_usd, round(ratio, 4), reason)


def project_batch_cost(
    episode_count: int,
    median_episode_minutes: float,
    usd_per_audio_minute: float,
) -> float:
    """Project a batch's modelled cost from AUDIO MINUTES (#1658).

    Minutes, not episode count: modelled cost is dominated by transcription minutes, so a
    15-episode window means something completely different on a 49-minute feed than on a
    92-minute one. Sizing by episode count is what tripped the cap and caused both the G1
    wedge and the G2 crash (#1620).
    """
    if episode_count <= 0 or median_episode_minutes <= 0 or usd_per_audio_minute <= 0:
        return 0.0
    return episode_count * median_episode_minutes * usd_per_audio_minute
