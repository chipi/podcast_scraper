"""Shared types for finale-tier judge clients."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


class JudgeUnavailableError(RuntimeError):
    """Raised when a judge cannot serve a request (missing key, host down, etc.).

    The finale runner catches this per-(judge, summary) pair so a single
    transient failure does not collapse a 1000-call sweep — the failed pair is
    recorded with ``error`` set and excluded from agreement / aggregation.
    """


@dataclass(frozen=True)
class JudgeResult:
    """Outcome of a single judge invocation.

    Attributes:
        text: Raw assistant text (G-Eval parsing happens at the caller).
        model: Concrete model identifier the judge reported (post-routing).
        prompt_tokens: Best-effort prompt token count when the API reports it.
        completion_tokens: Best-effort completion token count.
        cost_usd: Estimated USD cost — 0.0 for local-Ollama judges.
        latency_seconds: Wall-clock for the API call (for reporting).
        raw: Optional provider-specific payload for debugging; not persisted.
    """

    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_seconds: float = 0.0
    raw: Optional[Dict[str, Any]] = field(default=None, repr=False)
