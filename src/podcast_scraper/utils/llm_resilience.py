"""Per-model resilience profiles: how hard to retry, how long to back off, when to break.

Resilience is a property of the MODEL/ENDPOINT, not the pipeline stage — gemini-2.5-flash-lite is
overloaded whether it is summarising or grounding. So the knobs live here, keyed by (provider,
model), with a DEFAULT and per-model overrides. Nothing is hardcoded at the call sites any more;
``retry_with_metrics`` reads the resolved profile off the metrics object.

WHY gemini-2.5-flash-lite is special: it is cheap and very good, so everyone hammers it, so it
throws 503/UNAVAILABLE far more than heavier models. Hammering it back with tight retries makes the
storm worse. Its profile is deliberately CONSERVATIVE — more patience, longer backoff, a breaker
that trips sooner and cools down longer — so when Google says "not now" we wait our turn.

Registry note (ADR-112): ideally these per-model values would be registry-materialised like every
other tuned param. They live in utils for now to avoid a utils -> providers.ml import cycle; moving
them into the registry (or having the registry override this table) is a clean follow-up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class ResilienceProfile:
    """The retry/backoff/breaker posture for one model."""

    max_retries: int
    initial_delay: float  # seconds before the first retry
    max_delay: float  # ceiling on exponential backoff
    breaker_failure_threshold: int  # overload failures within the window before the breaker opens
    breaker_cooldown_seconds: float  # how long the breaker stays open before a probe

    def with_overrides(self, **kw: object) -> "ResilienceProfile":
        """A copy of this profile with the given fields replaced (per-model tuning)."""
        from dataclasses import replace

        return replace(self, **kw)  # type: ignore[arg-type]


# The posture every model gets unless it has an override. Matches the historical retry_with_metrics
# defaults (3 retries, 1s→30s backoff) plus the breaker defaults, so behaviour is unchanged for
# models without a specific profile.
DEFAULT_PROFILE = ResilienceProfile(
    max_retries=3,
    initial_delay=1.0,
    max_delay=30.0,
    breaker_failure_threshold=3,
    breaker_cooldown_seconds=30.0,
)

# gemini-2.5-flash-lite: cheap, popular, frequently 503s under load. Back off HARDER and be patient.
_FLASH_LITE_CONSERVATIVE = ResilienceProfile(
    max_retries=6,  # more patience — its 503s clear if you wait
    initial_delay=2.0,  # start backing off further out
    max_delay=60.0,  # allow a longer wait for the storm to pass
    breaker_failure_threshold=2,  # open the breaker sooner so we stop hammering
    breaker_cooldown_seconds=60.0,  # and stay off it longer before probing
)

# Keyed by (provider, model). Matched exact-first, then by provider default.
_PROFILES: Dict[Tuple[str, str], ResilienceProfile] = {
    ("gemini", "gemini-2.5-flash-lite"): _FLASH_LITE_CONSERVATIVE,
}


def resolve_resilience(provider: Optional[str], model: Optional[str]) -> ResilienceProfile:
    """The resilience profile for a (provider, model). Exact match wins; otherwise the default."""
    if provider and model:
        exact = _PROFILES.get((provider, model))
        if exact is not None:
            return exact
    return DEFAULT_PROFILE
