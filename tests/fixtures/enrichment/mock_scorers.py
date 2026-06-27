"""Scenario-driven mock scorers + a tiny enricher zoo for resilience tests.

These satisfy the protocols in :mod:`podcast_scraper.enrichment.scorers.protocol`
without loading real models. They are deterministic, scriptable, and never
hit the network — safe for CI ([[feedback_no_llm_in_ci]]).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from podcast_scraper.enrichment.protocol import (
    EnricherManifest,
    EnricherResult,
    EnricherScope,
    EnricherTier,
    EpisodeArtifactBundle,
    RunContext,
    STATUS_OK,
)
from podcast_scraper.enrichment.resilience import (
    BadInputError,
    DependencyAccessError,
    ModelLoadError,
    ScorerTimeoutError,
)
from podcast_scraper.enrichment.scorers.protocol import (
    EmbeddingProvider,
    LLMScorer,
    NliScore,
    NliScorer,
)

# ---------------------------------------------------------------------------
# Scenario script type — drives "fail N times then succeed" patterns
# ---------------------------------------------------------------------------


@dataclass
class Script:
    """A callable script of return values / exceptions.

    Indexed by attempt number (1-based). When the script runs out, the
    last entry is repeated. Use this to drive deterministic
    "fail-twice-then-succeed" retry scenarios in integration tests.
    """

    steps: list[Any]
    call_count: int = 0

    def next(self) -> Any:
        i = min(self.call_count, len(self.steps) - 1)
        self.call_count += 1
        step = self.steps[i]
        if isinstance(step, BaseException):
            raise step
        if callable(step):
            return step()
        return step


# ---------------------------------------------------------------------------
# Mock scorers
# ---------------------------------------------------------------------------


@dataclass
class MockNliScorer:
    """NLI scorer driven by a script per (premise, hypothesis) call.

    Default script returns a neutral (uncalibrated) score with zero
    cost — drop a ``ScorerTimeoutError`` / ``DependencyAccessError`` in
    the script to exercise the retry path.
    """

    script: Script = field(default_factory=lambda: Script(steps=[NliScore(0.1, 0.7, 0.2)]))

    async def score(self, premise: str, hypothesis: str) -> NliScore:
        await asyncio.sleep(0)  # yield control so the executor's cancel can fire
        result = self.script.next()
        assert isinstance(result, NliScore)
        return result


@dataclass
class MockEmbeddingProvider:
    """Embedding provider keyed by topic id; missing ids return ``None``.

    Pass ``vectors={"topic:a": [0.1, 0.2, 0.3]}`` to seed; absent keys
    drive the ``BadInputError`` path in enrichers that require a
    vector.
    """

    vectors: dict[str, list[float]] = field(default_factory=dict)
    script: Script | None = None

    async def topic_vector(self, topic_id: str) -> list[float] | None:
        await asyncio.sleep(0)
        if self.script is not None:
            result = self.script.next()
            assert result is None or isinstance(result, list)
            return result
        return self.vectors.get(topic_id)


@dataclass
class MockLLMScorer:
    """LLM scorer driven by a script. ``cost_usd`` per call is fixed.

    Real LLM-tier enrichers will accumulate cost via the executor's
    cost-cap state; this mock just returns the script value and any
    cost-cap behaviour is tested by enrichers that record cost
    explicitly.
    """

    script: Script = field(default_factory=lambda: Script(steps=["ok"]))

    async def score(self, prompt: str) -> str:
        await asyncio.sleep(0)
        result = self.script.next()
        return str(result)


# Runtime-checkable protocol smoke (caught by mypy; asserted at module import).
assert isinstance(MockNliScorer(), NliScorer)
assert isinstance(MockEmbeddingProvider(), EmbeddingProvider)
assert isinstance(MockLLMScorer(), LLMScorer)


# ---------------------------------------------------------------------------
# A tiny enricher zoo for integration tests
# ---------------------------------------------------------------------------


@dataclass
class ScriptedEnricher:
    """Enricher whose ``enrich()`` body is a script.

    Use to exercise the executor's resilience loop:
    - ``[EnricherResult(status="ok", data={"x": 1})]`` → succeeds first try.
    - ``[ScorerTimeoutError("boom"), EnricherResult(status="ok", data={"x": 1})]``
      → fails once (retryable), succeeds on retry.
    - ``[BadInputError("missing")]`` → fails non-retryably.
    - ``[ModelLoadError("oom"), ModelLoadError("oom again")]`` → RETRYABLE_ONCE
      then non-retryable.
    """

    manifest: EnricherManifest
    script: Script
    config_assert: Callable[[dict[str, Any]], None] | None = None

    async def enrich(
        self,
        *,
        bundle: EpisodeArtifactBundle | None,
        corpus_root: Path,
        all_bundles: list[EpisodeArtifactBundle] | None,
        config: dict[str, Any],
        ctx: RunContext,
    ) -> EnricherResult:
        await asyncio.sleep(0)
        if self.config_assert is not None:
            self.config_assert(config)
        if ctx.cancel_event.is_set():
            from podcast_scraper.enrichment.protocol import STATUS_CANCELLED

            return EnricherResult(status=STATUS_CANCELLED, error="cancel_requested")
        step = self.script.next()
        if isinstance(step, EnricherResult):
            return step
        if isinstance(step, dict):
            return EnricherResult(status=STATUS_OK, data=step)
        raise TypeError(
            f"ScriptedEnricher script step must be EnricherResult or dict, got {step!r}"
        )


@dataclass
class SlowEnricher:
    """Enricher whose body sleeps for *delay_s* — used to exercise the hard
    timeout path. ``delay_s`` > ``expected_duration_s`` triggers timeout."""

    manifest: EnricherManifest
    delay_s: float

    async def enrich(
        self,
        *,
        bundle: EpisodeArtifactBundle | None,
        corpus_root: Path,
        all_bundles: list[EpisodeArtifactBundle] | None,
        config: dict[str, Any],
        ctx: RunContext,
    ) -> EnricherResult:
        await asyncio.sleep(self.delay_s)
        return EnricherResult(status=STATUS_OK, data={"slow": True})


# ---------------------------------------------------------------------------
# Manifest factories — keep tests terse
# ---------------------------------------------------------------------------


def manifest(
    id: str = "scripted",
    *,
    tier: EnricherTier = EnricherTier.DETERMINISTIC,
    scope: EnricherScope = EnricherScope.CORPUS,
    writes: str = "scripted.json",
    max_cost_usd_per_run: float | None = None,
    expected_duration_s: int | None = None,
) -> EnricherManifest:
    return EnricherManifest(
        id=id,
        version="1.0.0",
        scope=scope,
        tier=tier,
        reads=[],
        writes=writes,
        description=f"mock enricher {id!r}",
        max_cost_usd_per_run=max_cost_usd_per_run,
        expected_duration_s=expected_duration_s,
    )


__all__ = [
    "MockEmbeddingProvider",
    "MockLLMScorer",
    "MockNliScorer",
    "Script",
    "ScriptedEnricher",
    "SlowEnricher",
    "manifest",
    # Re-export common failures so test files don't need to import them
    # from resilience just to script a scenario.
    "BadInputError",
    "DependencyAccessError",
    "ModelLoadError",
    "ScorerTimeoutError",
]
