"""Uniform result envelope shared by every probe (core, CLI, and later MCP).

Every source function returns one of these dicts so callers get a consistent shape and a
half-configured control plane degrades gracefully: a source whose credentials are absent
returns ``ok=False, configured=False`` with a clear "set X" message instead of raising.
"""

from __future__ import annotations

from typing import Any


def ok(source: str, data: Any, **extra: Any) -> dict:
    """A successful probe result."""
    return {"ok": True, "source": source, "data": data, **extra}


def err(source: str, message: str, *, configured: bool = True, **extra: Any) -> dict:
    """A failed probe result. ``configured=False`` means "not wired in this target"."""
    return {"ok": False, "source": source, "error": message, "configured": configured, **extra}
