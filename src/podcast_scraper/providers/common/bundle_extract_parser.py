"""Robust JSON parser for the bundled ``extract_quotes`` response (#698 Layer A).

The bundled call asks the LLM to return ``{insight_id_str: [quote_text, ...]}``
covering all N insights in one response. This parser tolerates:

- Code fences (`````json ... `````) wrapping the JSON.
- Top-level ``{"insights": {...}}`` envelope (some models add an outer key).
- Missing keys for some insight indices (returned as empty lists).
- Non-string ids (cast to str).
- Non-list values (treated as empty for that index).
- Per-quote dict shape ``{"text": "..."}`` instead of bare strings.

The same fallback policy as ``mega_bundled`` applies upstream: if this parser
returns nothing useful for an insight, the dispatcher falls back to the per-
insight staged path for that insight (or the whole batch on hard parse failure).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class BundleExtractParseError(Exception):
    """Raised when the bundled extract response is unparseable as JSON."""


def _strip_code_fences(content: str) -> str:
    text = (content or "").strip()
    if text.startswith("```"):
        # `````json\n...\n````` or `````\n...\n`````
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0]
    return text.strip()


def _coerce_quote_strings(raw: Any) -> List[str]:
    """Normalise per-insight quote payload to a list of non-empty strings."""
    if raw is None:
        return []
    if isinstance(raw, str):
        s = raw.strip()
        return [s] if s else []
    if not isinstance(raw, list):
        return []
    out: List[str] = []
    for item in raw:
        if isinstance(item, str):
            s = item.strip()
            if s:
                out.append(s)
            continue
        if isinstance(item, dict):
            text_val = item.get("text") or item.get("quote") or item.get("quote_text")
            if isinstance(text_val, str):
                s = text_val.strip()
                if s:
                    out.append(s)
    return out


def parse_bundled_extract_response(
    content: str,
    expected_count: int,
) -> Dict[int, List[str]]:
    """Parse the bundled extract response into ``{insight_idx: [quote_text, ...]}``.

    Args:
        content: Raw model response text (may include code fences).
        expected_count: Number of insights the bundled call covered. Used to
            seed the result with empty lists for missing indices so the caller
            can iterate uniformly.

    Returns:
        Dict mapping each insight index in ``range(expected_count)`` to its
        list of quote strings (possibly empty).

    Raises:
        BundleExtractParseError: When the content is not valid JSON or the
            top-level shape isn't a mapping. Callers should fall back to the
            staged extract path for the whole batch.
    """
    if expected_count <= 0:
        return {}

    text = _strip_code_fences(content)
    if not text:
        raise BundleExtractParseError("empty content")

    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BundleExtractParseError(f"invalid JSON: {exc}") from exc

    if not isinstance(obj, dict):
        raise BundleExtractParseError(f"top-level must be an object, got {type(obj).__name__}")

    # Tolerate envelope: ``{"insights": {...}}`` or ``{"quotes": {...}}``.
    inner = obj
    for envelope_key in ("insights", "quotes", "by_insight", "results"):
        v = obj.get(envelope_key)
        if isinstance(v, dict):
            inner = v
            break

    out: Dict[int, List[str]] = {idx: [] for idx in range(expected_count)}
    for raw_key, raw_val in inner.items():
        try:
            idx = int(str(raw_key).strip())
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= expected_count:
            continue
        out[idx] = _coerce_quote_strings(raw_val)

    return out
