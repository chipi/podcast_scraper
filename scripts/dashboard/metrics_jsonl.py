#!/usr/bin/env python3
"""
Parse metrics history files (JSONL).

History must be one JSON object per line. Legacy CI used pretty-printed JSON with
`echo "$(cat latest.json)" >> history.jsonl`, which broke line-based parsing.
This module recovers full objects by scanning with json.JSONDecoder.raw_decode.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def recover_json_objects_from_text(text: str) -> List[Dict[str, Any]]:
    """Return all JSON objects found in *text* (legacy pretty-printed blobs or mixed garbage)."""
    decoder = json.JSONDecoder()
    pos = 0
    n = len(text)
    out: List[Dict[str, Any]] = []

    while pos < n:
        while pos < n and text[pos].isspace():
            pos += 1
        if pos >= n:
            break
        if text[pos] != "{":
            nxt = text.find("{", pos)
            if nxt == -1:
                break
            pos = nxt
        try:
            obj, end = decoder.raw_decode(text, pos)
            if isinstance(obj, dict):
                out.append(obj)
            pos = end
        except json.JSONDecodeError:
            nxt = text.find("{", pos + 1)
            if nxt == -1:
                break
            pos = nxt

    return out


def load_metrics_history(path: Path) -> List[Dict[str, Any]]:
    """Load metrics history from *path*; empty list if missing or unreadable."""
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        logger.warning("Could not read history %s: %s", path, e)
        return []

    if not text.strip():
        return []

    non_empty = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not non_empty:
        return []

    line_records: List[Dict[str, Any]] = []
    for line in non_empty:
        try:
            rec = json.loads(line)
            if isinstance(rec, dict):
                line_records.append(rec)
            else:
                line_records = []
                break
        except json.JSONDecodeError:
            line_records = []
            break

    if line_records:
        return line_records

    recovered = recover_json_objects_from_text(text)
    if recovered:
        logger.warning(
            "Recovered %d metrics record(s) from non-JSONL history %s (legacy multi-line append). "
            "Run repair_metrics_jsonl.py --in-place to rewrite as proper JSONL.",
            len(recovered),
            path,
        )
    return recovered


def dump_compact_line(record: Dict[str, Any]) -> str:
    """Serialize *record* as one JSON line (no trailing newline)."""
    return json.dumps(record, separators=(",", ":"), ensure_ascii=False)
