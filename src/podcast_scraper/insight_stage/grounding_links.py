"""Each insight's grounding quote, text and time, read from ``gi.json``: what the two-tier index
links insights to transcript segments by (moved verbatim from ``search.two_tier_indexer``, E04
S4).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple


def _insight_grounding_quotes(gi_path: Path) -> Dict[str, Tuple[float, Optional[float]]]:
    """Map each insight node id → its first grounding quote's (start_s, end_s).

    Reads the episode's ``*.gi.json`` (Insight ``SUPPORTED_BY`` Quote; quotes carry
    ``timestamp_*_ms``). This is what lets ``link_insights_to_segments`` connect an
    insight to the transcript segment it was spoken in — the input the compound-result
    path (``dedup``) needs but the retired migration couldn't supply.
    """
    if not gi_path.is_file():
        return {}
    try:
        art = json.loads(gi_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    quote_ts: Dict[str, Tuple[float, Optional[float]]] = {}
    for node in art.get("nodes") or []:
        if node.get("type") == "Quote":
            props = node.get("properties") or {}
            start = props.get("timestamp_start_ms")
            if start is not None:
                end = props.get("timestamp_end_ms")
                quote_ts[node.get("id")] = (
                    float(start) / 1000.0,
                    float(end) / 1000.0 if end is not None else None,
                )
    out: Dict[str, Tuple[float, Optional[float]]] = {}
    for edge in art.get("edges") or []:
        if edge.get("type") == "SUPPORTED_BY":
            insight_id, quote_id = edge.get("from"), edge.get("to")
            if insight_id not in out and quote_id in quote_ts:
                out[insight_id] = quote_ts[quote_id]
    return out


def _insight_grounding_quote_texts(gi_path: Path) -> Dict[str, str]:
    """Map each insight node id → its first grounding quote's verbatim text.

    The text-based counterpart to :func:`_insight_grounding_quotes`. Used to link
    insights to the transcript segment that contains the quote when segments carry
    no usable timestamps (``summary.timestamps`` unpopulated → segment spans at 0.0).
    """
    if not gi_path.is_file():
        return {}
    try:
        art = json.loads(gi_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    quote_text: Dict[str, str] = {}
    for node in art.get("nodes") or []:
        if node.get("type") == "Quote":
            txt = (node.get("properties") or {}).get("text")
            if isinstance(txt, str) and txt.strip():
                quote_text[node.get("id")] = txt.strip()
    out: Dict[str, str] = {}
    for edge in art.get("edges") or []:
        if edge.get("type") == "SUPPORTED_BY":
            insight_id, quote_id = edge.get("from"), edge.get("to")
            if insight_id not in out and quote_id in quote_text:
                out[insight_id] = quote_text[quote_id]
    return out
