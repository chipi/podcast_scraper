"""OpenAI Chat-completions judge client (G-Eval finale cross-check).

Per RFC-057 and #932/#940, the cross-check judge in the finale tier is the
OpenAI flagship chat model — paired with Sonnet 4.6 (Anthropic) for the
"two-judge bias reduction" pattern set in the autoresearch RFC. Default
model is ``gpt-5.4`` (analog tier to Sonnet 4.6); ``gpt-4o`` is supported
for backward-compatibility with EVAL_CLEANING_AUTORESEARCH_2026_06_08.

Why this judge (vs Gemini 2.5 Pro, the original #932 cross-check):
  - Gemini 2.5 Pro is a reasoning model whose dynamic thinking budget
    can consume the entire ``max_output_tokens`` budget on internal
    reasoning, returning empty ``response.text`` (observed 2026-06-09:
    20/20 fluency-and-coverage calls returned text=''). OpenAI chat
    models don't have this pathology.
  - RFC-057 explicitly specifies "OpenAI + Anthropic flagship chat
    models" as the dual-judge pair.
  - The operator's autoresearch ``.env`` already provisions a dedicated
    ``AUTORESEARCH_JUDGE_OPENAI_API_KEY`` — no operator setup required.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

from podcast_scraper.evaluation.judges.base import JudgeResult, JudgeUnavailableError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-5.4"
# Public pricing per million tokens, gpt-5.4 standard tier (≤128K context).
# Authoritative copy lives in pricing_assumptions.yaml; duplicated here only so
# the judge can self-report cost without loading the YAML.
PRICE_PER_MTOK_INPUT_USD = 3.00
PRICE_PER_MTOK_OUTPUT_USD = 15.00


class OpenAIChatJudge:
    """Thin wrapper over ``openai.OpenAI.chat.completions.create``."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        client: object | None = None,
    ) -> None:
        """Instantiate the judge.

        Args:
            api_key: OpenAI API key. If not given, reads from env in this
                order (operator's autoresearch-vs-prod account separation —
                NEVER falls through to the plain prod key):
                  1. ``AUTORESEARCH_JUDGE_OPENAI_API_KEY`` (preferred — dedicated judge slot)
                  2. ``AUTORESEARCH_EXPERIMENT_OPENAI_API_KEY`` (autoresearch namespace fallback)
                Errors if neither is set. The plain ``OPENAI_API_KEY`` is
                deliberately NOT consulted — it's reserved for prod / personal
                inference and must not be charged for autoresearch judging.
            model: OpenAI chat model id (default ``gpt-5.4``).
            client: Optional pre-constructed ``openai.OpenAI`` (tests).
        """
        self._model = model
        self._client = client
        self._api_key = (
            api_key
            or os.environ.get("AUTORESEARCH_JUDGE_OPENAI_API_KEY", "").strip()
            or os.environ.get("AUTORESEARCH_EXPERIMENT_OPENAI_API_KEY", "").strip()
        )

    @property
    def model(self) -> str:
        return self._model

    def _ensure_client(self) -> object:
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise JudgeUnavailableError(
                "OpenAIChatJudge: neither AUTORESEARCH_JUDGE_OPENAI_API_KEY nor "
                "AUTORESEARCH_EXPERIMENT_OPENAI_API_KEY is set. The plain "
                "OPENAI_API_KEY is intentionally ignored (prod/personal). "
                "Set one of the autoresearch-namespaced keys or inject a client."
            )
        try:
            from openai import OpenAI  # local import to avoid hard dep at module load
        except ImportError as exc:  # pragma: no cover
            raise JudgeUnavailableError(
                f"OpenAIChatJudge: openai package not installed ({exc})"
            ) from exc
        self._client = OpenAI(api_key=self._api_key)
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
            # GPT-5.x and o-series chat completions reject ``max_tokens`` and
            # require ``max_completion_tokens``; older 4o-tier models accept
            # both. Send ``max_completion_tokens`` for forward compatibility.
            resp = client.chat.completions.create(  # type: ignore[attr-defined]
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_completion_tokens=max_tokens,
                # 120s per-request hard cap. Matches Sonnet/Gemini judges'
                # cap. Without this, a hung TCP socket can block the whole
                # sweep for tens of minutes (see sonnet46.py rationale).
                timeout=120.0,
            )
        except Exception as exc:  # noqa: BLE001
            raise JudgeUnavailableError(f"OpenAIChatJudge: OpenAI API call failed: {exc}") from exc
        latency = time.monotonic() - start

        # Extract assistant text. OpenAI returns choices[0].message.content.
        text = ""
        choices = getattr(resp, "choices", None) or []
        if choices:
            msg = getattr(choices[0], "message", None)
            text = (getattr(msg, "content", "") or "").strip()

        usage = getattr(resp, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
        cost = self._estimate_cost(prompt_tokens, completion_tokens)
        logger.debug(
            "OpenAIChat judge (%s): ptok=%d ctok=%d cost=$%.4f lat=%.2fs",
            self._model,
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
