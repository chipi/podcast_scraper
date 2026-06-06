"""Positional heuristics for commercial segment detection."""

from __future__ import annotations

from typing import Dict, Tuple

POSITION_WINDOWS: Dict[str, Tuple[float, float]] = {
    "pre_roll": (0.00, 0.15),
    "mid_roll": (0.35, 0.65),
    "post_roll": (0.80, 1.00),
}

POSITION_BOOST = 0.15


def position_score(char_index: int, text_length: int) -> float:
    """Return positional confidence boost when match falls in an ad-break window.

    Decision (B6): RFC-060 §3 sketched a -0.10 penalty for matches in the
    conversation zones, but we deliberately use **boosts only** (no penalty). A
    blanket off-window penalty suppresses legitimate mid-roll host-read ads, and
    precision for the risky low-confidence inline CTAs is already enforced by the
    corroboration gate in the detector (a brand/promo/intro must be nearby). So
    out-of-window matches get a neutral 0.0 rather than a penalty.
    """
    if text_length <= 0:
        return 0.0
    ratio = char_index / text_length
    for _name, (start, end) in POSITION_WINDOWS.items():
        if start <= ratio <= end:
            return POSITION_BOOST
    return 0.0
