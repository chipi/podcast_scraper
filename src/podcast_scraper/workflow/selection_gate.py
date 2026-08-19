"""Price a selection BEFORE the run spends anything, and refuse work that cannot be afforded.

WHY THIS EXISTS (2026-08-18). A repair asked for 32 episodes, selected 678, and spent ~$48 under
an active $5 cap. The cap could not have stopped it: ASR runs to completion in a background
thread before any enforcement point is reachable, so by the time spend is checked the money is
already gone. Everything downstream of selection is therefore too late.

This module is the one place that is early enough. Transcription cost is a function of audio
duration, which is known from the feed BEFORE a single byte is downloaded — so a selection can be
priced and refused while it still costs nothing. That asymmetry is why a pre-flight gate works
here at all: LLM cost is only knowable after the call, but ASR — the dominant cost, and the one
that emptied the account — is knowable in advance.

It also prints the thing whose absence made the incident invisible for six hours: a line saying
how many episodes were selected out of how many exist. "32 of 678" and "678 of 678" differ by one
character of operator attention and by $48.

WHAT THIS DELIBERATELY DOES NOT DO. It does not track scope continuously. The episode set is
built once per feed and cannot grow afterwards, so re-checking it would add machinery that can
only ever confirm what the gate already decided. What CAN grow is cost per episode — retries and
failover re-transcribe the same audio — and that is the run budget's job, not the gate's. The
gate bounds scope; the ledger bounds spend. Neither replaces the other, and the estimate below is
a FLOOR for exactly that reason.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

logger = logging.getLogger(__name__)

# Retries and speech-coverage failover re-transcribe audio the estimate counted once, so actual
# billed minutes exceed selected minutes. Measured on the 2026-08-18 incident: Deepgram billed
# 187.6 audio-hours for a selection totalling roughly 155 corpus-hours (and 271 /listen requests
# for ~181 episodes) — about 1.21x. Rounded up to 1.25 because an estimate that under-predicts is
# the failure mode that costs money, while one that over-predicts only asks the operator to run a
# smaller batch. This is an allowance on the ESTIMATE only; the ledger still records real spend.
RETRY_ALLOWANCE = 1.25

#: Transcription providers that incur NO PER-CALL BILL because the model runs on hardware we
#: already own — local Whisper, and the self-hosted tailnet/DGX endpoints.
#:
#: None of them has a row in the pricing table, and without this they were indistinguishable from
#: a CLOUD provider whose price simply could not be resolved. That distinction matters: "cost
#: could NOT be priced, the cap cannot be applied" is alarming and correct for an unpriced cloud
#: provider, and alarming and WRONG for local Whisper, where there is no bill to cap. Found by
#: running a real local corpus through the gate, not by reading the code.
#:
#: NOT a claim that these are free in every sense — GPU time and electricity are real. They are
#: free of the thing this cap governs: a per-call charge from a third party.
NO_PER_CALL_CHARGE_PROVIDERS = frozenset({"whisper", "tailnet_dgx_whisper", "moss"})


@dataclass(frozen=True)
class SelectionEstimate:
    """What a selection contains and what it is expected to cost."""

    selected: int
    """Episodes this run will process."""

    available: int
    """Episodes it could have processed — the denominator that makes 32-vs-678 visible."""

    priced: int
    """Selected episodes whose duration was known, so their cost is computed rather than assumed."""

    unpriced: int
    """Selected episodes with NO known duration. Counted and reported — never treated as free."""

    audio_seconds: float
    """Total audio of the priced episodes."""

    asr_usd: Optional[float]
    """Estimated ASR cost incl. the retry allowance, or None when pricing cannot be resolved."""

    self_hosted: bool = False
    """True when transcription runs on hardware we own, so ``asr_usd`` of 0.0 means FREE OF A
    PER-CALL BILL rather than 'not yet priced'."""

    @property
    def audio_hours(self) -> float:
        return self.audio_seconds / 3600.0

    @property
    def fully_priced(self) -> bool:
        """True when every selected episode contributed a real duration to the estimate."""
        return self.unpriced == 0 and self.asr_usd is not None

    def describe(self) -> str:
        """The selection manifest line. Read this in the log before letting a repair run."""
        if self.self_hosted:
            cost = "no per-call charge (self-hosted ASR)"
        elif self.asr_usd is not None:
            cost = f"est. ${self.asr_usd:.2f}"
        else:
            cost = "est. UNKNOWN (no price)"
        line = (
            f"selection: {self.selected} of {self.available} episodes · "
            f"{self.audio_hours:.1f} audio-hours · {cost}"
        )
        if self.unpriced:
            line += (
                f" · {self.unpriced} episode(s) have NO known duration and are NOT in that "
                f"estimate — the real cost is higher"
            )
        return line


def _episode_duration_seconds(episode: Any) -> Optional[float]:
    """Duration for one selected episode, from its feed item. None when the feed omits it."""
    item = getattr(episode, "item", None)
    if item is None:
        return None
    try:
        from ..rss.parser import extract_episode_metadata

        _, _, _, duration, _, _ = extract_episode_metadata(item, "")
    except Exception:  # noqa: BLE001 - a malformed item must not break selection
        return None
    if duration is None or not isinstance(duration, (int, float)):
        return None
    value = float(duration)
    return value if value > 0 else None


def estimate_selection(
    episodes: Sequence[Any],
    cfg: Any,
    *,
    available: Optional[int] = None,
) -> SelectionEstimate:
    """Price ``episodes`` for transcription without contacting any provider.

    Provider-agnostic by construction: the transcription provider and model come off ``cfg`` and
    the price comes from ``calculate_provider_cost`` — the same function that prices the real
    calls, so the estimate and the bill are derived from one pricing table rather than two.
    """
    durations: List[float] = []
    unpriced = 0
    for ep in episodes:
        seconds = _episode_duration_seconds(ep)
        if seconds is None:
            unpriced += 1
        else:
            durations.append(seconds)

    audio_seconds = sum(durations)
    asr_usd: Optional[float] = None

    if str(getattr(cfg, "transcription_provider", None) or "whisper") in (
        NO_PER_CALL_CHARGE_PROVIDERS
    ):
        # PRICED, AT ZERO — not "unpriceable". The gate must not warn about a cap it has no
        # reason to apply.
        return SelectionEstimate(
            selected=len(episodes),
            available=int(available) if available is not None else len(episodes),
            priced=len(durations),
            unpriced=unpriced,
            audio_seconds=audio_seconds,
            asr_usd=0.0,
            self_hosted=True,
        )

    if durations:
        try:
            from ..utils.provider_metrics import transcription_model_for_cfg
            from .helpers import calculate_provider_cost

            provider = str(getattr(cfg, "transcription_provider", None) or "whisper")
            unit = calculate_provider_cost(
                cfg=cfg,
                provider_type=provider,
                capability="transcription",
                model=transcription_model_for_cfg(cfg),
                audio_minutes=audio_seconds / 60.0,
            )
            if unit is not None:
                asr_usd = round(float(unit) * RETRY_ALLOWANCE, 4)
        except Exception:  # noqa: BLE001 - an unpriceable selection is reported, not fatal
            logger.debug("selection pricing unavailable", exc_info=True)

    return SelectionEstimate(
        selected=len(episodes),
        available=int(available) if available is not None else len(episodes),
        priced=len(durations),
        unpriced=unpriced,
        audio_seconds=audio_seconds,
        asr_usd=asr_usd,
    )


def affordable_episode_count(episodes: Sequence[Any], cfg: Any, remaining_usd: float) -> int:
    """How many of ``episodes``, in order, fit inside ``remaining_usd``.

    Answers the operator's actual next question — "then how many CAN I do?" — so a refusal is
    actionable instead of merely a stop. Episodes with no known duration stop the count: an
    unpriceable episode cannot be shown to fit.
    """
    if remaining_usd == float("inf"):
        return len(episodes)
    fits = 0
    for i in range(1, len(episodes) + 1):
        est = estimate_selection(episodes[:i], cfg)
        if not est.fully_priced or est.asr_usd is None or est.asr_usd > remaining_usd:
            break
        fits = i
    return fits


def enforce_selection_budget(
    episodes: Sequence[Any],
    cfg: Any,
    *,
    available: Optional[int] = None,
    scope: str = "",
) -> SelectionEstimate:
    """Log the selection manifest, and refuse the run when it cannot be afforded.

    Returns the estimate when the work may proceed. Raises
    :class:`~podcast_scraper.workflow.cost_monitoring.CostCapExceeded` when the cap is ENFORCED
    (``cost_soft_cap_action=abort``) and this selection would breach it.

    Raising here is safe in a way it is not deeper in the pipeline: selection runs on the main
    thread before any worker starts, so the exception reaches the caller instead of quietly
    killing a background thread while the main thread waits on a join.
    """
    from .cost_monitoring import CostCapExceeded
    from .run_budget import configure_run_budget, get_run_budget

    # Configure from THIS cfg rather than trusting an earlier caller to have done it. The gate is
    # the thing standing between a selection and the money, so it must not depend on call order to
    # be armed — a cap that silently fails to apply because a step ran in the wrong sequence is the
    # same failure mode as the incident. configure carries spend and trip state forward, so calling
    # it here as well as in run_pipeline is idempotent.
    configure_run_budget(cfg)

    estimate = estimate_selection(episodes, cfg, available=available)
    where = f" [{scope}]" if scope else ""
    logger.info("%s%s", estimate.describe(), where)

    if not episodes:
        return estimate

    budget = get_run_budget()
    if budget.cap_usd is None:
        return estimate

    if estimate.self_hosted:
        # Nothing to bill, so nothing to refuse. Stated once, quietly, rather than warned about.
        logger.info(
            "transcription is self-hosted (%s) — no per-call ASR charge for this selection",
            getattr(cfg, "transcription_provider", None),
        )
        return estimate

    if estimate.asr_usd is None:
        # Unpriceable is NOT free. Say so loudly and continue: refusing every run whose provider
        # has no pricing row would ground the pipeline on a config gap rather than a cost problem.
        logger.warning(
            "selection cost could NOT be priced (provider=%s) — the $%.2f cap cannot be applied "
            "to this selection in advance; only spend already recorded is bounded",
            getattr(cfg, "transcription_provider", None),
            budget.cap_usd,
        )
        return estimate

    if budget.check_and_reserve(estimate.asr_usd):
        logger.info(
            "run budget: %s of the $%.2f cap remains before this selection",
            f"${budget.remaining_usd:.2f}",
            budget.cap_usd,
        )
        return estimate

    fits = affordable_episode_count(episodes, cfg, budget.remaining_usd)
    message = (
        f"selection of {estimate.selected} episode(s) is estimated at ${estimate.asr_usd:.2f}, "
        f"which exceeds the ${budget.cap_usd:.2f} per-run cap with "
        f"${budget.remaining_usd:.2f} remaining (already spent ${budget.spent_usd:.2f}). "
        f"About {fits} episode(s) would fit — split the work-list and re-run."
    )
    logger.error("REFUSING TO START: %s%s", message, where)
    raise CostCapExceeded(estimate.asr_usd + budget.spent_usd, float(budget.cap_usd))
