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

    Handles reasoning-model outputs that emit CoT before the JSON answer
    (nemotron_h, gpt-oss, qwen3.x thinking). Strategy:

    1. Try ``_extract_json_object`` to strip ``</think>`` prefix + code
       fences and extract the first balanced ``{...}``. If the object
       has a ``score`` key, return it.
    2. If that fails (no JSON object), fall back to a regex scan for
       the first number AFTER ``</think>`` (if present) — never grab
       numbers from the reasoning trace itself.
    3. Raise if nothing parseable found.

    Clips to [0, 10]. Note: the current rubric emits 0.0-1.0 values;
    the wider clip range preserves compatibility with older 0-10 rubric
    judges (autoresearch_track_a.py::parse_judge_score_json enforces the
    stricter [0, 1] check on top for the rubric flow).
    """
    from podcast_scraper.evaluation.pairwise import _extract_json_object, _THINK_TAG_RE

    # Try structured JSON first — handles both raw JSON and <think>-wrapped
    try:
        blob = _extract_json_object(text)
        obj = json.loads(blob)
        if isinstance(obj, dict) and "score" in obj:
            return max(0.0, min(10.0, float(obj["score"])))
    except (ValueError, json.JSONDecodeError, TypeError):
        pass

    # No JSON found — regex fallback, scoped to POST-</think> if present
    m_think = _THINK_TAG_RE.search(text)
    scan_from = text[m_think.end() :] if m_think else text
    m = _NUMBER_RE.search(scan_from)
    if m is None:
        raise ValueError(f"could not parse score from judge reply: {text!r}")
    return max(0.0, min(10.0, float(m.group(0))))


class OllamaChatJudge:
    """Generic Ollama judge — non-reasoning models by default; pass
    ``reasoning_effort`` and ``max_tokens`` to support reasoning-tuned
    models (gpt-oss:120b, qwen3.6, etc.)."""

    def __init__(
        self,
        *,
        api_base: Optional[str] = None,
        model: str,
        request_timeout_s: float = 300.0,
        reasoning_effort: Optional[str] = None,
        max_tokens: Optional[int] = None,
        client: object | None = None,
    ) -> None:
        self.api_base = (api_base or _resolve_ollama_base()).rstrip("/")
        self.model = model
        self.request_timeout_s = request_timeout_s
        # For reasoning-tuned Ollama models (gpt-oss:120b, qwen3.6, etc.)
        # Ollama routes CoT to a ``reasoning`` field and leaves ``content``
        # empty by default. Pass reasoning_effort="low" (or "none") to
        # keep CoT short + fit within max_tokens.
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self._client = client

    def _do_request(self, *, user_content: str) -> dict[str, Any]:
        url = f"{self.api_base}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": user_content}],
            "temperature": 0.0,
        }
        # Explicit overrides win, else auto-detect reasoning models by name.
        # Reasoning-tuned Ollama models leave content="" and route CoT to
        # a ``reasoning`` field by default — reasoning_effort="low" keeps
        # the CoT short so both fit within max_tokens.
        effort = self.reasoning_effort
        max_tok = self.max_tokens
        if effort is None and ("gpt-oss" in self.model or "qwen3.6" in self.model):
            effort = "low"
        if max_tok is None and effort is not None:
            max_tok = 1024
        if effort is not None:
            payload["reasoning_effort"] = effort
        if max_tok is not None:
            payload["max_tokens"] = max_tok
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

    def raw(self, *, user_content: str) -> str:
        """POST to /chat/completions; return the raw assistant text.

        Same retry policy as ``score`` — pairwise judging (which parses
        its own JSON schema) needs the raw text, not a float. Both
        ``score`` and ``raw`` share this transport so scalar and pairwise
        modes use identical retry, error-wrapping, and timeouts.
        """
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
        return (choices[0].get("message") or {}).get("content") or ""

    def score(self, *, user_content: str) -> float:
        """POST to /chat/completions; return the parsed float.

        Thin wrapper over :meth:`raw` that runs :func:`_parse_score` on
        the reply. Scalar-mode callers use this; pairwise-mode callers
        use ``raw`` directly.
        """
        return _parse_score(self.raw(user_content=user_content))
