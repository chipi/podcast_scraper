"""Google Gemini 2.5 Pro judge client (G-Eval finale cross-check).

Per #932, Gemini 2.5 Pro acts as a **cross-check** judge — invoked only on the
top-2 finalists from the Sonnet primary pass, to cap cost. Disagreement (per
the divergence rule in ``g_eval.py``) escalates a pair for manual review in
the finale report.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

from podcast_scraper.evaluation.judges.base import JudgeResult, JudgeUnavailableError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-2.5-pro"
# Gemini 2.5 Pro public pricing per million tokens (≤128K context tier).
PRICE_PER_MTOK_INPUT_USD = 1.25
PRICE_PER_MTOK_OUTPUT_USD = 10.00


class Gemini25ProJudge:
    """Thin wrapper over ``google.genai`` content generation."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        client: object | None = None,
    ) -> None:
        # Operator's autoresearch-vs-prod account separation — NEVER fall through
        # to the plain prod key (reserved for prod inference). Order:
        #   1. AUTORESEARCH_JUDGE_GEMINI_API_KEY (preferred — dedicated judge slot
        #      if the operator adds one)
        #   2. AUTORESEARCH_EXPERIMENT_GEMINI_API_KEY (autoresearch namespace
        #      fallback; the only Gemini key currently in the operator's .env
        #      under that prefix)
        self._model = model
        self._client = client
        self._api_key = (
            api_key
            or os.environ.get("AUTORESEARCH_JUDGE_GEMINI_API_KEY", "").strip()
            or os.environ.get("AUTORESEARCH_EXPERIMENT_GEMINI_API_KEY", "").strip()
        )

    @property
    def model(self) -> str:
        return self._model

    def _ensure_client(self) -> object:
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise JudgeUnavailableError(
                "Gemini25ProJudge: neither AUTORESEARCH_JUDGE_GEMINI_API_KEY "
                "nor AUTORESEARCH_EXPERIMENT_GEMINI_API_KEY is set. The plain "
                "GEMINI_API_KEY is intentionally ignored (prod/personal). "
                "Set one of the autoresearch-namespaced keys or inject a client."
            )
        try:
            from google import genai  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise JudgeUnavailableError(
                f"Gemini25ProJudge: google-genai package not installed ({exc})"
            ) from exc
        self._client = genai.Client(api_key=self._api_key)
        return self._client

    @staticmethod
    def _estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
        return (prompt_tokens / 1_000_000.0) * PRICE_PER_MTOK_INPUT_USD + (
            completion_tokens / 1_000_000.0
        ) * PRICE_PER_MTOK_OUTPUT_USD

    def score(self, prompt: str, *, max_tokens: int = 1024) -> JudgeResult:
        client = self._ensure_client()
        start = time.monotonic()
        try:
            # genai 1.x API
            response = client.models.generate_content(  # type: ignore[attr-defined]
                model=self._model,
                contents=prompt,
                config={
                    "temperature": 0.0,
                    "max_output_tokens": max_tokens,
                },
            )
        except Exception as exc:  # noqa: BLE001
            raise JudgeUnavailableError(
                f"Gemini25ProJudge: google-genai API call failed: {exc}"
            ) from exc
        latency = time.monotonic() - start

        # Gemini returns ``response.text`` for the canonical join.
        text = (getattr(response, "text", "") or "").strip()
        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0) if usage else 0
        completion_tokens = int(getattr(usage, "candidates_token_count", 0) or 0) if usage else 0
        cost = self._estimate_cost(prompt_tokens, completion_tokens)
        logger.debug(
            "Gemini25Pro judge: ptok=%d ctok=%d cost=$%.4f lat=%.2fs",
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
