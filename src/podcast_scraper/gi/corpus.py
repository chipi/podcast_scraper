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
