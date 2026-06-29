"""Generic Ollama chat judge — non-reasoning model variant.

Sibling of ``deepseek_r1.py`` (which strips ``<think>...</think>`` blocks
emitted by reasoning models). This judge talks to Ollama's OpenAI-compatible
chat-completions endpoint and assumes the model returns the score JSON
directly, without reasoning preamble. Suitable for ``llama3.1``, ``gemma3``,
``mistral-*``, ``qwen2.5/3.5`` (non-deepseek-r1) families.

Wire it into Track A by setting a judge entry's ``provider: ollama`` in
``judge_config*.yaml`` — see ``autoresearch_track_a._score_one`` for the
dispatcher.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, cast, Optional

import httpx

from podcast_scraper.evaluation.judges.base import JudgeUnavailableError
from podcast_scraper.utils.retry import retry_with_exponential_backoff

logger = logging.getLogger(__name__)


def _resolve_ollama_base() -> str:
    """Pick the Ollama base URL from env (same precedence as DeepSeekR1Judge).

    1. ``OLLAMA_API_BASE`` (explicit, OpenAI-compatible /v1 endpoint)
    2. ``DGX_TAILNET_FQDN`` (compose ``http://<fqdn>:11434/v1``)
    3. ``http://localhost:11434/v1`` (tests / local dev)
    """
    direct = os.environ.get("OLLAMA_API_BASE", "").strip()
    if direct:
        return direct.rstrip("/")
    fqdn = os.environ.get("DGX_TAILNET_FQDN", "").strip()
    if fqdn:
        return f"http://{fqdn}:11434/v1"
    return "http://localhost:11434/v1"


_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _parse_score(text: str) -> float:
    """Extract a float in [0, 10] from the model's reply.

    Accepts JSON like ``{"score": 7.5}`` or bare floats like ``"7.5"``.
    Clips to [0, 10]. Raises on unparseable replies.
    """
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "score" in obj:
            val = float(obj["score"])
            return max(0.0, min(10.0, val))
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    m = _NUMBER_RE.search(text)
    if m is None:
        raise ValueError(f"could not parse score from judge reply: {text!r}")
    return max(0.0, min(10.0, float(m.group(0))))


class OllamaChatJudge:
    """Generic Ollama judge for non-reasoning chat models."""

    def __init__(
        self,
        *,
        api_base: Optional[str] = None,
        model: str,
        request_timeout_s: float = 300.0,
        client: object | None = None,
    ) -> None:
        self.api_base = (api_base or _resolve_ollama_base()).rstrip("/")
        self.model = model
        self.request_timeout_s = request_timeout_s
        self._client = client

    def _do_request(self, *, user_content: str) -> dict[str, Any]:
        url = f"{self.api_base}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": user_content}],
            "temperature": 0.0,
        }
        if self._client is not None:
            return cast(
                dict, self._client.post(url, json=payload).json()  # type: ignore[attr-defined]
            )
        with httpx.Client(timeout=self.request_timeout_s) as client:
            resp = client.post(url, json=payload)
            # 5xx + 408/429 = transient (server load, brief Ollama model
            # swap, rate limit). 4xx = client bug, don't retry.
            resp.raise_for_status()
            return cast(dict, resp.json())

    def score(self, *, user_content: str) -> float:
        """POST to /chat/completions; return the parsed float.

        Retries on transient HTTP errors (5xx, 408, 429) and network/IO
        errors with exponential backoff. A single brittle Ollama hiccup
        shouldn't kill a 10-judge-call sweep; this matters more as cohort
        size grows.
        """
        # Only retry on transient errors. httpx.HTTPStatusError for 4xx
        # is NOT retried (we let it pop out of the retry helper untouched
        # by re-raising as a non-retryable type).
        retryable = (
            httpx.RequestError,
            httpx.ConnectError,
            httpx.ReadTimeout,
            httpx.RemoteProtocolError,
            httpx.HTTPStatusError,
            OSError,
        )
        try:
            reply = retry_with_exponential_backoff(
                lambda: self._do_request(user_content=user_content),
                max_retries=3,
                initial_delay=2.0,
                max_delay=15.0,
                retryable_exceptions=retryable,
            )
        except (httpx.HTTPError, OSError) as exc:
            raise JudgeUnavailableError(
                f"Ollama judge ({self.model} at {self.api_base}) unreachable "
                f"after retries: {exc}"
            ) from exc

        choices = reply.get("choices") or []
        if not choices:
            raise JudgeUnavailableError(f"Ollama judge ({self.model}) returned no choices")
        text = (choices[0].get("message") or {}).get("content") or ""
        return _parse_score(text)
