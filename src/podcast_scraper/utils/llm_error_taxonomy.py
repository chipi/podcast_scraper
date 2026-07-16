"""What KIND of failure did an LLM API just return? Retry it, back off, or abandon the run?

The retry layer was binary: retryable or not. That is not enough for LLM APIs, where a
same-looking error can mean three very different things:

* **the endpoint is BUSY** — 500/502/503/504, Anthropic's 529 ``overloaded_error``, Gemini's 503
  ``UNAVAILABLE``. gemini-2.5-flash-lite throws these under load because it is cheap and everyone
  hammers it. → back off and try again; this is what resilience is FOR.
* **we are being RATE LIMITED** — 429, "rate limit exceeded", a transient ``RESOURCE_EXHAUSTED``.
  → honour Retry-After, back off, try again.
* **we are OUT OF MONEY or ACCESS** — Anthropic "you have reached your specified API usage limits",
  OpenAI ``insufficient_quota`` / billing hard cap, DeepSeek 402 "Insufficient Balance", a revoked
  or invalid key (401/403). No amount of backoff fixes this. → **TERMINAL: stop the whole run,
  loudly, and say why.** Retrying it just wastes time and muddies the logs; the old code did exactly
  that, because it lumped every "quota" string into "retryable".

TERMINAL is a fuse, the same shape as the call-count fuse: it aborts the run rather than limp on.
Everything else stays on the retry/backoff/circuit-breaker path.

Kept provider-agnostic on purpose: the signals below come from what these seven APIs actually
return (and, for the usage-limit case, from the exact 400 that stopped our Anthropic account).
"""

from __future__ import annotations

import enum
import re
from typing import Optional


class LLMErrorClass(enum.Enum):
    """How the retry layer should treat an LLM API error."""

    RETRYABLE_OVERLOAD = "retryable_overload"  # endpoint busy (5xx / 529 / UNAVAILABLE) — back off
    RETRYABLE_RATE_LIMIT = "retryable_rate_limit"  # 429 / rate limit — honour Retry-After, back off
    TERMINAL = "terminal"  # out of money/access — abort the run, do NOT retry
    NON_RETRYABLE = "non_retryable"  # bad request / our fault — re-raise, do NOT retry


class LLMTerminalError(RuntimeError):
    """A terminal LLM condition: no budget, no credit, or no access on this key.

    Raised as a hard stop — the fuse for money/access, mirroring the call-count fuse. The message is
    written for a human staring at a stopped run: which provider, and why it is over.
    """


# OUT OF MONEY / ACCESS. Checked FIRST — they win over any rate-limit look-alike, because the
# Anthropic spend-cap 400 and OpenAI insufficient_quota 429 both contain "limit"/"quota" yet must
# NOT retry.
_TERMINAL_SIGNALS = (
    "insufficient_quota",
    "insufficient balance",  # DeepSeek 402
    "billing_hard_limit",
    "billing hard limit",
    "exceeded your current quota",
    "usage limit",  # Anthropic: "you have reached your specified API usage limits"
    "usage limits",
    "spending limit",
    "credit balance is too low",
    "payment required",
    "account is not active",
    "access denied",
    "permission denied",
    "invalid api key",
    "invalid x-api-key",
    "incorrect api key",
    "authentication",
    "unauthorized",
)

# BUSY / OVERLOADED — the endpoint is up but can't serve us right now. Back off and retry.
_OVERLOAD_SIGNALS = (
    "overloaded",  # Anthropic overloaded_error (HTTP 529)
    "service unavailable",
    "unavailable",  # Gemini 503 UNAVAILABLE
    "bad gateway",
    "gateway timeout",
    "internal server error",
    "server error",
    "try again",
    "temporarily",
)
_OVERLOAD_STATUS = (500, 502, 503, 504, 529)

# RATE LIMITED — transient, honour Retry-After. Distinct from a terminal quota above.
_RATE_LIMIT_SIGNALS = (
    "rate limit",
    "rate_limit",
    "too many requests",
    "resource exhausted",  # Gemini RESOURCE_EXHAUSTED, transient form
    "resource_exhausted",
    "quota exceeded",  # transient per-minute quota, NOT the terminal account quota above
)

# Our fault / not worth retrying — malformed request, unsupported param, etc.
_NON_RETRYABLE_STATUS = (400, 404, 405, 409, 413, 422)


def _status_of(error: Exception) -> Optional[int]:
    for attr in ("status_code", "status", "http_status", "code"):
        v = getattr(error, attr, None)
        if isinstance(v, int) and 100 <= v <= 599:
            return v
    m = re.search(r"\b(4\d\d|5\d\d)\b", str(error))
    return int(m.group(1)) if m else None


def _auth_status(status: Optional[int]) -> bool:
    return status in (401, 403)


def classify_llm_error(error: Exception) -> LLMErrorClass:
    """Classify an LLM API exception. TERMINAL is checked first so an out-of-money error is never
    mistaken for a retryable rate limit."""
    text = str(error).lower()
    status = _status_of(error)

    # 1. TERMINAL — out of money / access. Wins over everything.
    if any(sig in text for sig in _TERMINAL_SIGNALS):
        return LLMErrorClass.TERMINAL
    if _auth_status(status):
        return LLMErrorClass.TERMINAL
    if status == 402:  # Payment Required — DeepSeek insufficient balance
        return LLMErrorClass.TERMINAL

    # 2. OVERLOAD — busy endpoint. Back off.
    if status in _OVERLOAD_STATUS:
        return LLMErrorClass.RETRYABLE_OVERLOAD
    if any(sig in text for sig in _OVERLOAD_SIGNALS):
        return LLMErrorClass.RETRYABLE_OVERLOAD

    # 3. RATE LIMIT — transient. Honour Retry-After.
    if status == 429 or any(sig in text for sig in _RATE_LIMIT_SIGNALS):
        return LLMErrorClass.RETRYABLE_RATE_LIMIT

    # 4. Transport/connection issues read as overload-retryable.
    for sig in (
        "timeout",
        "timed out",
        "connection",
        "connection reset",
        "broken pipe",
        "network",
        "econnreset",
    ):
        if sig in text:
            return LLMErrorClass.RETRYABLE_OVERLOAD

    # 5. Explicit client errors — our fault, do not retry.
    if status in _NON_RETRYABLE_STATUS:
        return LLMErrorClass.NON_RETRYABLE

    # Unknown: treat as non-retryable so we surface it rather than loop on it.
    return LLMErrorClass.NON_RETRYABLE


def terminal_message(provider: str, error: Exception) -> str:
    """A human-facing one-liner for a terminal stop — what's over, and the raw reason."""
    text = str(error)
    lowered = text.lower()
    if any(
        s in lowered
        for s in ("usage limit", "quota", "billing", "balance", "credit", "payment", "spending")
    ):
        why = "no budget/credit left on this key"
    elif any(s in lowered for s in ("api key", "auth", "unauthorized", "permission", "denied")):
        why = "the key is invalid or lacks access"
    else:
        why = "a terminal account condition"
    return (
        f"{provider}: {why} — this is NOT retryable, so the run is hard-stopping (the resilience "
        f"fuse for money/access, same as the call-count fuse). Raw provider error: {text[:200]}"
    )
