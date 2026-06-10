"""Anthropic Claude Sonnet 4.6 judge client (G-Eval finale primary).

Per #932, Sonnet 4.6 is the **primary** judge — every (candidate summary,
G-Eval dimension) pair is scored by this client.

Sizing note: Sonnet 4.6 is the *standard* (non-thinking) variant; the
``temperature`` parameter is still respected (unlike Opus 4.7's reasoning
mode), so we send ``temperature=0.0`` for deterministic-ish scoring.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

from podcast_scraper.evaluation.judges.base import JudgeResult, JudgeUnavailableError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-6"
# Anthropic public pricing — Sonnet 4.6 input/output per million tokens.
# Authoritative copy lives in pricing_assumptions.yaml; the values here are
# duplicated only so the judge can self-report cost without loading the YAML.
PRICE_PER_MTOK_INPUT_USD = 3.00
PRICE_PER_MTOK_OUTPUT_USD = 15.00


class Sonnet46Judge:
    """Thin wrapper over ``anthropic.Anthropic.messages.create``."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        client: object | None = None,
    ) -> None:
        """Instantiate the judge.

        Args:
            api_key: Anthropic API key. If not given, reads from env in this
                order (operator's autoresearch-vs-prod account separation —
                NEVER falls through to the plain prod key):
                  1. ``AUTORESEARCH_JUDGE_ANTHROPIC_API_KEY`` (preferred — dedicated judge slot)
                  2. ``AUTORESEARCH_EXPERIMENT_ANTHROPIC_API_KEY`` (autoresearch namespace fallback)
                Errors if neither is set. The plain ``ANTHROPIC_API_KEY`` is
                deliberately NOT consulted — it's reserved for prod / personal
                inference and must not be charged for autoresearch judging.
            model: Sonnet model id to address (default ``claude-sonnet-4-6``).
            client: Optional pre-constructed ``anthropic.Anthropic`` (tests).
        """
        self._model = model
        self._client = client
        self._api_key = (
            api_key
            or os.environ.get("AUTORESEARCH_JUDGE_ANTHROPIC_API_KEY", "").strip()
            or os.environ.get("AUTORESEARCH_EXPERIMENT_ANTHROPIC_API_KEY", "").strip()
        )

    @property
    def model(self) -> str:
        return self._model

    def _ensure_client(self) -> object:
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise JudgeUnavailableError(
                "Sonnet46Judge: neither AUTORESEARCH_JUDGE_ANTHROPIC_API_KEY nor "
                "AUTORESEARCH_EXPERIMENT_ANTHROPIC_API_KEY is set. The plain "
                "ANTHROPIC_API_KEY is intentionally ignored (prod/personal). "
                "Set one of the autoresearch-namespaced keys or inject a client."
            )
        try:
            import anthropic  # local import to avoid hard dep at module load
        except ImportError as exc:  # pragma: no cover
            raise JudgeUnavailableError(
                f"Sonnet46Judge: anthropic package not installed ({exc})"
            ) from exc
        self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    @staticmethod
    def _estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
        return (prompt_tokens / 1_000_000.0) * PRICE_PER_MTOK_INPUT_USD + (
            completion_tokens / 1_000_000.0
        ) * PRICE_PER_MTOK_OUTPUT_USD

    def score(self, prompt: str, *, max_tokens: int = 1024) -> JudgeResult:
        """Send ``prompt`` as a single user message, return the parsed reply."""
        client = self._ensure_client()
        start = time.monotonic()
        try:
            msg = client.messages.create(  # type: ignore[attr-defined]
                model=self._model,
                max_tokens=max_tokens,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
                # 120s per-request hard cap. Without this the Anthropic SDK
                # defaults to 600s + 2 retries = up to 30 min on a single hung
                # connection — we observed this hanging the finale at the
                # 2nd-finalist mark on 2026-06-09, idle for 17min with one
                # ESTABLISHED-but-dead TCP socket and zero CPU. 120s is well
                # above the ~3s typical Sonnet judge call latency; we'd rather
                # surface a timeout error and retry one finalist than block the
                # whole sweep for hours.
                timeout=120.0,
            )
        except Exception as exc:  # noqa: BLE001
            raise JudgeUnavailableError(f"Sonnet46Judge: Anthropic API call failed: {exc}") from exc
        latency = time.monotonic() - start

        # Extract assistant text blocks (Anthropic returns a list of content blocks).
        if isinstance(msg, str):
            text = msg.strip()
        else:
            parts = []
            for block in getattr(msg, "content", []) or []:
                t = getattr(block, "text", None)
                if t:
                    parts.append(t)
            text = "".join(parts).strip()

        usage = getattr(msg, "usage", None)
        prompt_tokens = int(getattr(usage, "input_tokens", 0) or 0) if usage else 0
        completion_tokens = int(getattr(usage, "output_tokens", 0) or 0) if usage else 0
        cost = self._estimate_cost(prompt_tokens, completion_tokens)
        logger.debug(
            "Sonnet46 judge: ptok=%d ctok=%d cost=$%.4f lat=%.2fs",
            prompt_tokens,
            completion_tokens,
            cost,
            latency,
        )
        return JudgeResult(
            text=text,
            model=self._model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            latency_seconds=latency,
        )
