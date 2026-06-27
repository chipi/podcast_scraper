"""Concrete ``NliScorer`` implementations.

Two shipped:

* :class:`FixedNliScorer` — deterministic, dependency-free. Returns
  scripted ``NliScore`` for known ``(premise, hypothesis)`` pairs;
  defaults to a configurable neutral score for unknown pairs. Used by
  tests + CI smoke ([[feedback_no_llm_in_ci]]).

* :class:`DeBERTaNliScorer` — production-shape lazy wrapper around the
  ``cross-encoder/nli-deberta-v3-small`` model via
  ``sentence-transformers``. The model is loaded on first ``score``
  call (no import-time penalty). Loading failures raise
  :class:`podcast_scraper.enrichment.resilience.ModelLoadError` so the
  ML tier's RETRYABLE_ONCE policy applies; per-call exceptions raise
  :class:`podcast_scraper.enrichment.resilience.ScorerTimeoutError` so
  the executor retries per the ML-tier policy (2 retries, 60s max
  backoff, circuit at 3).

Operators integrate the real scorer via the optional ``[ml]`` extra;
omitting it leaves the production scorer permanently in a non-usable
state, which is fine for airgapped_thin and other no-ML profiles.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from podcast_scraper.enrichment.scorers.protocol import NliScore


@dataclass
class FixedNliScorer:
    """Deterministic ``NliScorer`` for tests + CI smoke.

    Pre-populate ``scores`` keyed by ``(premise, hypothesis)`` tuples;
    pairs that aren't in the map return ``default``.
    """

    scores: dict[tuple[str, str], NliScore] = field(default_factory=dict)
    default: NliScore = field(default_factory=lambda: NliScore(0.05, 0.85, 0.10))

    async def score(self, premise: str, hypothesis: str) -> NliScore:
        await asyncio.sleep(0)
        return self.scores.get((premise, hypothesis), self.default)


@dataclass
class DeBERTaNliScorer:
    """Lazy wrapper around ``cross-encoder/nli-deberta-v3-small``.

    Loads on first ``score`` call via ``sentence-transformers``'
    ``CrossEncoder``. Cost is recorded as ``0.0`` (CPU-only).

    Failures during model load raise ``ModelLoadError`` (RETRYABLE_ONCE
    per the ML-tier policy). Failures during a single ``predict()``
    call raise ``ScorerTimeoutError`` (RETRYABLE).
    """

    model_id: str = "cross-encoder/nli-deberta-v3-small"
    model_version: str = "v1"
    _model: object | None = field(default=None, init=False, repr=False)

    def _load(self) -> object:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover — ML extra not installed in CI
            from podcast_scraper.enrichment.resilience import ModelLoadError

            raise ModelLoadError(
                "sentence-transformers not installed; install the [ml] extra"
            ) from exc
        try:
            return CrossEncoder(self.model_id)
        except Exception as exc:  # pragma: no cover — model unreachable
            from podcast_scraper.enrichment.resilience import ModelLoadError

            raise ModelLoadError(f"failed to load NLI model {self.model_id!r}: {exc}") from exc

    async def score(self, premise: str, hypothesis: str) -> NliScore:
        if self._model is None:
            self._model = await asyncio.to_thread(self._load)
        try:
            scores = await asyncio.to_thread(
                self._model.predict, [(premise, hypothesis)]  # type: ignore[attr-defined]
            )
        except Exception as exc:  # pragma: no cover — runtime path
            from podcast_scraper.enrichment.resilience import ScorerTimeoutError

            raise ScorerTimeoutError(f"NLI predict failed: {exc}") from exc
        # CrossEncoder for nli-deberta-v3-small returns logits in
        # [contradiction, entailment, neutral] order per HF model card.
        try:
            row = scores[0]
        except Exception:  # pragma: no cover
            row = scores
        try:
            contradiction = float(row[0])
            entailment = float(row[1])
            neutral = float(row[2])
        except Exception:  # pragma: no cover — unexpected shape
            contradiction, neutral, entailment = 0.0, 1.0, 0.0
        return NliScore(
            contradiction=contradiction, neutral=neutral, entailment=entailment, cost_usd=0.0
        )


__all__ = [
    "DeBERTaNliScorer",
    "FixedNliScorer",
]
