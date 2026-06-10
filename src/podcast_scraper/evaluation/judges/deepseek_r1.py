"""DeepSeek-R1 32B judge client over DGX Ollama (#940 Track 1).

R1 distill is reasoning-tuned — well suited to "score X on dimension Y and
justify why" prompts. We address ``deepseek-r1:32b`` via the local DGX Ollama
host (no marginal $ cost). R1's chain-of-thought is wrapped in ``<think>``
tags by default; we strip those before returning the final text, leaving the
G-Eval parser unconcerned with reasoning markup.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Optional

from podcast_scraper.evaluation.judges.base import JudgeResult, JudgeUnavailableError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "deepseek-r1:32b"
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _resolve_dgx_ollama_base() -> str:
    """Compose the DGX Ollama OpenAI-compatible endpoint.

    Honors a precomputed ``OLLAMA_API_BASE`` first; otherwise composes from
    ``DGX_TAILNET_FQDN``. Falls back to localhost:11434 when neither is set —
    useful in test/dev environments that proxy DGX through a local tunnel.
    """
    direct = os.environ.get("OLLAMA_API_BASE", "").strip()
    if direct:
        return direct.rstrip("/")
    fqdn = os.environ.get("DGX_TAILNET_FQDN", "").strip()
    if fqdn:
        return f"http://{fqdn}:11434/v1"
    return "http://localhost:11434/v1"


def strip_reasoning_tags(text: str) -> str:
    """Strip ``<think>...</think>`` blocks from R1's reply.

    R1 emits its chain-of-thought between ``<think>`` tags; we keep only the
    final answer so the G-Eval JSON parser doesn't trip on prose.
    """
    return THINK_BLOCK_RE.sub("", text).strip()


class DeepSeekR1Judge:
    """Thin wrapper over Ollama's OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        *,
        api_base: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        client: object | None = None,
        request_timeout_s: float = 300.0,
    ) -> None:
        """Instantiate the judge.

        Args:
            api_base: Override the DGX Ollama OpenAI-compatible base URL.
            model: Ollama model tag (``deepseek-r1:32b`` is the sweet spot per #940).
            client: Optional pre-constructed ``openai.OpenAI`` (tests).
            request_timeout_s: Per-call timeout — R1 32B can take ~60-90s.
        """
        self._model = model
        self._client = client
        self._api_base = (api_base or _resolve_dgx_ollama_base()).rstrip("/")
        self._timeout = request_timeout_s

    @property
    def model(self) -> str:
        return self._model

    @property
    def api_base(self) -> str:
        return self._api_base

    def _ensure_client(self) -> object:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise JudgeUnavailableError(
                f"DeepSeekR1Judge: openai package not installed ({exc})"
            ) from exc
        # Ollama doesn't require a real key but openai SDK insists on one being set.
        self._client = OpenAI(
            base_url=self._api_base,
            api_key="ollama",
            timeout=self._timeout,
        )
        return self._client

    def score(self, prompt: str, *, max_tokens: int = 1024) -> JudgeResult:
        """Send ``prompt`` to DGX Ollama; return the R1 reply with ``<think>`` stripped."""
        client = self._ensure_client()
        start = time.monotonic()
        try:
            resp = client.chat.completions.create(  # type: ignore[attr-defined]
                model=self._model,
                temperature=0.0,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:  # noqa: BLE001
            raise JudgeUnavailableError(
                f"DeepSeekR1Judge: Ollama API call failed at {self._api_base}: {exc}"
            ) from exc
        latency = time.monotonic() - start

        raw_text = ""
        try:
            raw_text = (resp.choices[0].message.content or "").strip()
        except (AttributeError, IndexError):
            raw_text = ""

        text = strip_reasoning_tags(raw_text)
        usage = getattr(resp, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
        # R1 on DGX = local inference = $0 marginal cost.
        cost = 0.0
        logger.debug(
            "DeepSeekR1 judge: ptok=%d ctok=%d (local, $0) lat=%.2fs base=%s",
            prompt_tokens,
            completion_tokens,
            latency,
            self._api_base,
        )
        return JudgeResult(
            text=text,
            model=self._model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            latency_seconds=latency,
        )
