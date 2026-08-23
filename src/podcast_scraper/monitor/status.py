"""Atomic read/write for ``.pipeline_status.json``."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from podcast_scraper.utils.atomic_io import write_json_atomic

PIPELINE_STATUS_FILENAME = ".pipeline_status.json"


def pipeline_status_path(output_dir: str | os.PathLike[str]) -> Path:
    """Path to the pipeline status file under ``output_dir``."""
    return Path(output_dir) / PIPELINE_STATUS_FILENAME


def read_pipeline_status(output_dir: str | os.PathLike[str]) -> Optional[Dict[str, Any]]:
    """Return parsed status JSON, or ``None`` if missing or invalid."""
    path = pipeline_status_path(output_dir)
    if not path.is_file():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError):
        return None


def write_pipeline_status_atomic(path: Path, payload: Dict[str, Any]) -> None:
    """Write JSON atomically (temp + ``os.replace``).

    The implementation this function used to carry inline is now ``utils.atomic_io`` — it was one
    of six independent copies in this repo, and the GI/KG artifact writers that needed it most had
    none. Kept as a named function because callers read better for it.
    """
    write_json_atomic(path, payload, indent=2, sort_keys=True)


def maybe_update_pipeline_status(
    cfg: Any,
    output_dir: str,
    *,
    stage: str,
    episode_idx: Optional[int] = None,
    episode_total: Optional[int] = None,
) -> None:
    """Update status file when ``cfg.monitor`` is true; no-op otherwise."""
    if not bool(getattr(cfg, "monitor", False)):
        return
    path = pipeline_status_path(output_dir)
    prev = read_pipeline_status(output_dir) or {}
    now = time.time()
    pid = os.getpid()
    if not prev:
        started_at = now
    else:
        started_at = float(prev.get("started_at", now))
    if stage != prev.get("stage"):
        stage_started_at = now
    else:
        stage_started_at = float(prev.get("stage_started_at", now))
    payload: Dict[str, Any] = {
        "pid": pid,
        "stage": stage,
        "started_at": started_at,
        "stage_started_at": stage_started_at,
    }
    if episode_idx is not None:
        payload["episode_idx"] = int(episode_idx)
    if episode_total is not None:
        payload["episode_total"] = int(episode_total)
    write_pipeline_status_atomic(path, payload)
