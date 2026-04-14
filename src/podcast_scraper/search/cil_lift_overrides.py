"""Optional corpus-level overrides for RFC-072 transcript chunk lift (#528).

``cil_lift_overrides.json`` at the corpus root (pipeline output directory) adjusts
char alignment and resolves split canonical ids without rebuilding the FAISS index.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping

from podcast_scraper.utils.path_validation import normpath_if_under_root, safe_resolve_directory

logger = logging.getLogger(__name__)

_OVERRIDES_FILENAME = "cil_lift_overrides.json"
_MAX_ALIAS_HOPS = 16


@dataclass(frozen=True)
class CilLiftOverrides:
    """Parsed ``cil_lift_overrides.json`` (all fields optional with safe defaults)."""

    transcript_char_shift: int = 0
    entity_id_aliases: Dict[str, str] = field(default_factory=dict)
    topic_id_aliases: Dict[str, str] = field(default_factory=dict)


def _aliases_from_raw(raw: Mapping[str, Any], key: str) -> Dict[str, str]:
    block = raw.get(key)
    if not isinstance(block, dict):
        return {}
    out: Dict[str, str] = {}
    for k, v in block.items():
        if isinstance(k, str) and isinstance(v, str) and k.strip() and v.strip():
            out[k.strip()] = v.strip()
    return out


def load_cil_lift_overrides(corpus_root: Path) -> CilLiftOverrides:
    """Load ``<corpus_root>/cil_lift_overrides.json`` if present; else empty defaults."""
    root_resolved = safe_resolve_directory(corpus_root)
    if root_resolved is None:
        return CilLiftOverrides()
    root_s = os.path.normpath(str(root_resolved))
    safe_prefix = root_s + os.sep
    joined = os.path.normpath(os.path.join(root_s, _OVERRIDES_FILENAME))
    safe = normpath_if_under_root(joined, root_s)
    if not safe or not safe.startswith(safe_prefix) or not os.path.isfile(safe):
        return CilLiftOverrides()
    try:
        with open(safe, encoding="utf-8") as fh:
            raw = json.loads(fh.read())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("cil_lift_overrides: failed to read %s: %s", safe, exc)
        return CilLiftOverrides()
    if not isinstance(raw, dict):
        logger.warning("cil_lift_overrides: expected object at top level: %s", safe)
        return CilLiftOverrides()
    shift_raw = raw.get("transcript_char_shift", 0)
    try:
        shift = int(shift_raw)
    except (TypeError, ValueError):
        shift = 0
    return CilLiftOverrides(
        transcript_char_shift=shift,
        entity_id_aliases=_aliases_from_raw(raw, "entity_id_aliases"),
        topic_id_aliases=_aliases_from_raw(raw, "topic_id_aliases"),
    )


def resolve_id_alias(canonical_id: str, aliases: Mapping[str, str]) -> str:
    """Follow ``aliases`` from ``canonical_id`` until fixed point (cycle-safe)."""
    if not canonical_id or not aliases:
        return canonical_id
    cur = canonical_id
    seen: set[str] = set()
    for _ in range(_MAX_ALIAS_HOPS):
        if cur in seen:
            return cur
        seen.add(cur)
        nxt = aliases.get(cur)
        if not nxt or nxt == cur:
            return cur
        cur = nxt
    return cur
