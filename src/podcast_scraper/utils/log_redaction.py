"""Redact likely secrets from strings before logging or returning to callers.

Used for pipeline errors, API exception messages, and any user/server-provided text
that might echo tokens (Bearer, OpenAI-style keys, etc.).
"""

from __future__ import annotations

import re
from typing import Optional, Pattern

# Order matters: more specific patterns first.
_REDACT_PATTERNS: tuple[tuple[Pattern[str], str], ...] = (
    (re.compile(r"(?i)(Bearer\s+)[A-Za-z0-9._\-~+/=]+"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(Basic\s+)[A-Za-z0-9+/=]+"), r"\1[REDACTED]"),
    (re.compile(r"\bsk-ant-api[A-Za-z0-9\-_]{8,}\b"), "sk-ant-[REDACTED]"),
    (re.compile(r"\bsk-proj-[A-Za-z0-9\-_]{8,}\b"), "sk-proj-[REDACTED]"),
    (re.compile(r"\bsk-[A-Za-z0-9\-_]{12,}\b"), "sk-[REDACTED]"),
    (
        re.compile(r"(?i)(api[_-]?key)([\"']?\s*[:=]\s*)([\"']?)[A-Za-z0-9\-_.]{8,}\3?"),
        r"\1\2\3[REDACTED]\3",
    ),
    (re.compile(r"(?i)(password)([\"']?\s*[:=]\s*)([\"']?)[^\s\"']{4,}\3?"), r"\1\2\3[REDACTED]\3"),
    (re.compile(r"(?i)(secret)([\"']?\s*[:=]\s*)([\"']?)[^\s\"']{4,}\3?"), r"\1\2\3[REDACTED]\3"),
)

_DEFAULT_MAX_LEN = 2000


def redact_for_log(text: Optional[str], *, max_len: int = _DEFAULT_MAX_LEN) -> str:
    """Return *text* with common token patterns replaced and length capped.

    Args:
        text: Raw message (may be None).
        max_len: Maximum length after redaction; longer strings are truncated with a suffix.

    Returns:
        Safe string for logs or external error fields.
    """
    if text is None:
        return ""
    s = str(text)
    if len(s) > max_len * 4:
        s = s[: max_len * 4]
    for pattern, repl in _REDACT_PATTERNS:
        s = pattern.sub(repl, s)
    if len(s) > max_len:
        s = s[:max_len] + "…[truncated]"
    return s


def format_exception_for_log(exc: BaseException, *, max_len: int = _DEFAULT_MAX_LEN) -> str:
    """String form of *exc* after redaction (for ``logger...(..., %s, exc)`` replacements)."""
    return redact_for_log(str(exc), max_len=max_len)
