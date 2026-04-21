"""Optional corpus-level overrides for transcript chunk lift to CIL (#528).

``cil_lift_overrides.json`` at the corpus root (pipeline output directory) adjusts
char alignment and resolves split canonical ids without rebuilding the FAISS index.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from podcast_scraper.utils.path_validation import normpath_if_under_root, safe_resolve_directory

logger = logging.getLogger(__name__)

_OVERRIDES_FILENAME = "cil_lift_overrides.json"
_MAX_ALIAS_HOPS = 16


def _cil_overrides_json_path(corpus_root: Path) -> Optional[str]:
    """Return safe absolute path to ``<corpus_root>/cil_lift_overrides.json``."""
    root_resolved = safe_resolve_directory(corpus_root)
    if root_resolved is None:
        return None
    root_s = os.path.normpath(str(root_resolved))
    safe_prefix = root_s + os.sep
    joined = os.path.normpath(os.path.join(root_s, _OVERRIDES_FILENAME))
    safe = normpath_if_under_root(joined, root_s)
    if not safe or not safe.startswith(safe_prefix):
        return None
    return safe


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
    safe = _cil_overrides_json_path(corpus_root)
    if not safe or not os.path.isfile(safe):
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


def write_cil_lift_overrides_merged_topic_id_aliases(
    corpus_root: Path,
    auto_topic_aliases: Mapping[str, str],
) -> Dict[str, str]:
    """Merge auto-generated topic aliases into ``cil_lift_overrides.json``.

    Keys already present in the file's ``topic_id_aliases`` **win** over *auto_topic_aliases*
    (hand edits and prior runs preserved). Other top-level keys are kept. Creates the file
    if it was missing.

    Args:
        corpus_root: Pipeline output directory (corpus root).
        auto_topic_aliases: Typically from :func:`topic_id_aliases_from_clusters_payload`.

    Returns:
        Merged ``topic_id_aliases`` dict written to JSON.

    Raises:
        ValueError: If *corpus_root* cannot be resolved safely.
    """
    path = _cil_overrides_json_path(corpus_root)
    if path is None:
        raise ValueError(f"invalid corpus root for cil_lift_overrides: {corpus_root!r}")

    raw: Dict[str, Any] = {}
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as fh:
                loaded = json.loads(fh.read())
            if isinstance(loaded, dict):
                raw = loaded
            else:
                logger.warning(
                    "cil_lift_overrides: top-level JSON was %s; resetting to object for merge",
                    type(loaded).__name__,
                )
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "cil_lift_overrides: unreadable %s (%s); starting merge from empty",
                path,
                exc,
            )

    existing = _aliases_from_raw(raw, "topic_id_aliases")
    merged: Dict[str, str] = dict(auto_topic_aliases)
    merged.update(existing)
    sorted_merged = dict(sorted(merged.items()))
    raw["topic_id_aliases"] = sorted_merged

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(raw, indent=2, sort_keys=True) + "\n")
    return sorted_merged


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
