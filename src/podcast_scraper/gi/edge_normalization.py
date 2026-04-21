"""Normalise GIL graph edge ``type`` strings for comparisons."""

from __future__ import annotations

from typing import Any


def normalize_gil_edge_type(raw: Any) -> str:
    """Return uppercased stripped edge type (empty if missing)."""
    return str(raw or "").strip().upper()
