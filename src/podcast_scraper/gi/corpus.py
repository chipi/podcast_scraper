"""GIL corpus I/O: load artifacts for export (NDJSON / merged bundle), mirroring ``kg.corpus``."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from podcast_scraper.utils.log_redaction import format_exception_for_log

from .io import read_artifact

logger = logging.getLogger(__name__)


def load_gi_artifacts(
    paths: List[Path],
    *,
    validate: bool = False,
    strict: bool = False,
) -> List[Tuple[Path, Dict[str, Any]]]:
    """Load ``.gi.json`` from paths; skip invalid with warning when not strict."""
    out: List[Tuple[Path, Dict[str, Any]]] = []
    for path in paths:
        try:
            data = read_artifact(path, validate=validate, strict=strict)
            out.append((path, data))
        except Exception as e:
            if strict:
                raise
            logger.warning(
                "Skip invalid GIL artifact %s: %s",
                path,
                format_exception_for_log(e),
            )
    return out


#: An artifact carrying ONLY this insight is a placeholder, not a result. Matched on the text
#: because that is the one field stable across every version of the stub: the properties changed
#: (#1657 item 9 made it ungrounded / FILLER / ``drop``), so a detector keyed on ``grounded`` or
#: ``routing_tag`` would miss precisely the older artifacts that need finding.
STUB_INSIGHT_TEXT = "Summary insight (stub)."


def is_stub_artifact(artifact: Dict[str, Any]) -> bool:
    """True when this ``.gi.json`` holds a placeholder instead of real insights.

    The shape is exact: one Insight node whose text is the stub string. An episode that
    genuinely produced a single real insight is NOT a stub and must not be swept up — the point
    of re-deriving these is to replace failures, not to redo work that succeeded.
    """
    insights = [
        n
        for n in (artifact.get("nodes") or [])
        if isinstance(n, dict) and n.get("type") == "Insight"
    ]
    if len(insights) != 1:
        return False
    return str((insights[0].get("properties") or {}).get("text", "")).strip() == STUB_INSIGHT_TEXT


def find_stub_artifacts(
    corpus_root: Path,
    *,
    validate: bool = False,
) -> List[Tuple[Path, str]]:
    """Every stub ``.gi.json`` under *corpus_root*, as ``(path, episode_id)``.

    The work-list for corpus repair. 112 of 678 production episodes (16.5 %) reached the stub
    path while insight generation failed silently; they are indistinguishable from real episodes
    to anything that only asks "does this episode have GI?", which is what
    ``episode_complete_for_append_resume`` asks. An append-mode re-run will therefore SKIP them
    forever. Re-deriving them needs an explicit list, and this produces it.

    Ordered by path so two runs of the same corpus yield the same work-list.
    """
    paths = sorted(corpus_root.rglob("*.gi.json"))
    out: List[Tuple[Path, str]] = []
    for path, doc in load_gi_artifacts(paths, validate=validate, strict=False):
        if is_stub_artifact(doc):
            out.append((path, str(doc.get("episode_id") or "")))
    return out


def summarize_stub_artifacts(corpus_root: Path) -> Dict[str, Any]:
    """Counts for an operator deciding whether a repair run is worth starting."""
    total = len(sorted(corpus_root.rglob("*.gi.json")))
    stubs = find_stub_artifacts(corpus_root)
    return {
        "artifacts_total": total,
        "stub_artifacts": len(stubs),
        "stub_share": round(len(stubs) / total, 4) if total else 0.0,
        "episode_ids": [eid for _, eid in stubs if eid],
        "paths": [str(p) for p, _ in stubs],
    }


def export_ndjson(
    loaded: List[Tuple[Path, Dict[str, Any]]],
    *,
    output_dir: Optional[Path],
    stream_write: Callable[[str], None],
) -> None:
    """Write one JSON object per line; each includes ``_artifact_path``."""
    for path, art in loaded:
        row = dict(art)
        rel = str(path)
        if output_dir:
            try:
                rel = str(path.relative_to(output_dir))
            except ValueError:
                pass
        row["_artifact_path"] = rel
        stream_write(json.dumps(row, ensure_ascii=False) + "\n")


def export_merged_json(
    loaded: List[Tuple[Path, Dict[str, Any]]],
    *,
    output_dir: Optional[Path],
) -> Dict[str, Any]:
    """Single JSON document with all GIL artifacts (corpus bundle)."""
    artifacts: List[Dict[str, Any]] = []
    total_insights = 0
    total_quotes = 0
    for path, art in loaded:
        copy = dict(art)
        rel = str(path)
        if output_dir:
            try:
                rel = str(path.relative_to(output_dir))
            except ValueError:
                pass
        copy["_artifact_path"] = rel
        artifacts.append(copy)
        nodes = art.get("nodes") or []
        for n in nodes:
            t = n.get("type")
            if t == "Insight":
                total_insights += 1
            elif t == "Quote":
                total_quotes += 1
    return {
        "export_kind": "gi_corpus_bundle",
        "schema_version": "1.0",
        "artifact_count": len(artifacts),
        "insight_count_total": total_insights,
        "quote_count_total": total_quotes,
        "artifacts": artifacts,
    }
