"""Defensive JSON parsing helpers for LLM provider responses.

LLM APIs (especially Gemini) sometimes return malformed JSON:
- Markdown code fences wrapping valid JSON
- Duplicate JSON objects on separate lines
- Raw control characters inside string values

This module provides a single ``parse_llm_json()`` entry point that
handles all known edge cases.  Use it for every ``json.loads()`` call
on LLM output instead of raw ``json.loads()``.

See GitHub #583 for the full inventory of issues.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict


def strip_code_fences(text: str) -> str:
    """Strip markdown code fences (``\\`\\`\\`json ... \\`\\`\\```) from LLM responses."""
    stripped = re.sub(r"^```(?:json)?\s*\n?", "", text.strip(), flags=re.MULTILINE)
    stripped = re.sub(r"\n?```\s*$", "", stripped.strip(), flags=re.MULTILINE)
    return stripped.strip()


def parse_llm_json(raw: str) -> Dict[str, Any]:
    """Parse JSON from an LLM response, handling known provider quirks.

    Applies defenses in order:

    1. Strip markdown code fences (Anthropic, Mistral, sometimes Gemini)
    2. ``json.loads(strict=False)`` to tolerate control characters (Gemini)
    3. First-line fallback for duplicate JSON objects (Gemini)
    4. Empty-object fallback if all parsing fails

    Args:
        raw: Raw text from LLM response (may include fences, duplicates, etc.)

    Returns:
        Parsed dict.  Returns ``{}`` if all parsing attempts fail.
    """
    if not raw or not raw.strip():
        return {}

    cleaned = strip_code_fences(raw)

    # Attempt 1: full text, strict=False
    try:
        result = json.loads(cleaned, strict=False)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    # Attempt 2: first line only (handles duplicate JSON objects)
    first_line = cleaned.split("\n")[0].strip()
    if first_line:
        try:
            result = json.loads(first_line, strict=False)
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    # Attempt 3: find first { ... } block via regex
    m = re.search(r"\{[\s\S]*?\}", cleaned)
    if m:
        try:
            result = json.loads(m.group(0), strict=False)
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    return {}
