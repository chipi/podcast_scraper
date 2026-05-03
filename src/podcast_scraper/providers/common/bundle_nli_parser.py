"""Robust JSON parser for the bundled ``score_entailment`` response (#698 Layer B).

The bundled call asks the LLM to return ``{pair_id_str: score_float}`` covering all
pairs in one chunk. This parser tolerates:

- Code fences (`````json ... `````) wrapping the JSON.
- Top-level envelope keys (``scores`` / ``results`` / ``entailment``).
- Missing keys for some pair indices (caller treats those as "no score" and
  falls back to a per-pair staged call OR drops the candidate).
- Non-numeric values (skipped — caller decides on fallback).
- Scores outside [0, 1] (clamped to that range).

The same fallback policy as Layer A applies upstream: any pair without a usable
score from the bundled call is re-scored individually by the dispatcher (or
discarded if even that fails).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class BundleNliParseError(Exception):
    """Raised when the bundled NLI response is unparseable as JSON."""


def _strip_code_fences(content: str) -> str:
    text = (content or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0]
    return text.strip()


def _coerce_score(raw: Any) -> float | None:
    """Cast a raw value to a clamped [0, 1] float; return None when not numeric."""
    if isinstance(raw, bool):
        # bool is subclass of int — explicit reject so True/False don't masquerade.
        return None
    if isinstance(raw, (int, float)):
        v = float(raw)
    elif isinstance(raw, str):
        try:
            v = float(raw.strip())
        except (TypeError, ValueError):
            return None
    else:
        return None
    if v != v:  # NaN
        return None
    return max(0.0, min(1.0, v))


def parse_bundled_nli_response(
    content: str,
    expected_count: int,
) -> Dict[int, float]:
    """Parse the bundled NLI response into ``{pair_idx: score}``.

    Args:
        content: Raw model response text (may include code fences).
        expected_count: Number of pairs the bundled call covered. Pair indices
            outside ``range(expected_count)`` are dropped.

    Returns:
        Dict mapping pair index to its NLI score (clamped to [0, 1]). Indices
        without a usable score are absent — callers should treat absence as
        "fall back to a per-pair staged call or drop the candidate".

    Raises:
        BundleNliParseError: When the content is not valid JSON or the
            top-level shape isn't a mapping. Callers should fall back to the
            staged per-pair NLI path for the whole batch.
    """
    if expected_count <= 0:
        return {}

    text = _strip_code_fences(content)
    if not text:
        raise BundleNliParseError("empty content")

    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BundleNliParseError(f"invalid JSON: {exc}") from exc

    if not isinstance(obj, dict):
        raise BundleNliParseError(f"top-level must be an object, got {type(obj).__name__}")

    inner = obj
    for envelope_key in ("scores", "results", "entailment", "by_pair"):
        v = obj.get(envelope_key)
        if isinstance(v, dict):
            inner = v
            break

    out: Dict[int, float] = {}
    for raw_key, raw_val in inner.items():
        try:
            idx = int(str(raw_key).strip())
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= expected_count:
            continue
        score = _coerce_score(raw_val)
        if score is None:
            continue
        out[idx] = score

    return out
