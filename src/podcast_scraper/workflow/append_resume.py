"""Append / resume helpers (GitHub #444).

Skip logic prefers **on-disk metadata + episode_id** validation. ``index.json`` may be
missing or stale after crashes; it is not used as the sole skip signal in v1.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

import yaml

from .. import config
from .helpers import get_episode_id_from_episode
from .run_index import find_episode_metadata_relative_path

logger = logging.getLogger(__name__)


def _load_metadata_doc(path: str, metadata_format: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as handle:
            if metadata_format == "yaml":
                raw = yaml.safe_load(handle)
            else:
                raw = json.load(handle)
        return raw if isinstance(raw, dict) else None
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        logger.debug("Append resume: could not read metadata %s: %s", path, exc)
        return None


def _summary_looks_present(summary_block: Any) -> bool:
    if not isinstance(summary_block, dict):
        return False
    bullets = summary_block.get("bullets")
    if isinstance(bullets, list) and bullets:
        return True
    raw = summary_block.get("raw_text")
    return isinstance(raw, str) and bool(raw.strip())


def episode_complete_for_append_resume(
    cfg: config.Config,
    episode: Any,
    feed_url: str,
    effective_output_dir: str,
    run_suffix: Optional[str],
) -> bool:
    """Return True if the episode should be skipped when ``cfg.append`` is enabled.

    Validates:

    - Resolved metadata file on disk
    - ``episode.episode_id`` matches the ID derived from the RSS episode
    - ``content.transcript_file_path`` exists under *effective_output_dir*
    - When ``backfill_transcript_segments`` and ``generate_gi`` are true, a sibling
      ``.segments.json`` next to a ``.txt`` transcript is also required (GitHub #542).
    - Optional stages enabled in *cfg* (summary / GI / KG) have corresponding artifacts

    Args:
        cfg: Run configuration
        episode: Episode model instance
        feed_url: RSS URL for this run (episode id derivation)
        effective_output_dir: Run root directory
        run_suffix: Filename suffix for this run (Whisper/metadata), if any

    Returns:
        True if the pipeline considers this episode complete for append mode.
    """
    if not cfg.generate_metadata:
        return False

    expected_id, _ = get_episode_id_from_episode(episode, feed_url)
    meta_rel = find_episode_metadata_relative_path(episode, effective_output_dir, run_suffix)
    if not meta_rel:
        return False
    meta_path = os.path.join(effective_output_dir, meta_rel)
    doc = _load_metadata_doc(meta_path, cfg.metadata_format)
    if not doc:
        return False

    ep_block = doc.get("episode")
    if not isinstance(ep_block, dict) or ep_block.get("episode_id") != expected_id:
        return False

    content = doc.get("content")
    if not isinstance(content, dict):
        return False
    trel = content.get("transcript_file_path")
    if not isinstance(trel, str) or not trel.strip():
        return False
    tpath = os.path.join(effective_output_dir, trel)
    if not os.path.isfile(tpath):
        return False

    if (
        getattr(cfg, "backfill_transcript_segments", False)
        and getattr(cfg, "generate_gi", False)
        and trel.strip().lower().endswith(".txt")
    ):
        seg_path = os.path.splitext(tpath)[0] + ".segments.json"
        if not os.path.isfile(seg_path):
            return False

    if cfg.generate_summaries and not _summary_looks_present(doc.get("summary")):
        return False

    if getattr(cfg, "generate_gi", False):
        gi = doc.get("grounded_insights")
        if not isinstance(gi, dict):
            return False
        gi_path = gi.get("artifact_path")
        if not isinstance(gi_path, str) or not gi_path.strip():
            return False
        if not os.path.isfile(os.path.join(effective_output_dir, gi_path)):
            return False

    if getattr(cfg, "generate_kg", False):
        kg = doc.get("knowledge_graph")
        if not isinstance(kg, dict):
            return False
        kg_path = kg.get("artifact_path")
        if not isinstance(kg_path, str) or not kg_path.strip():
            return False
        if not os.path.isfile(os.path.join(effective_output_dir, kg_path)):
            return False

    return True


__all__ = ["episode_complete_for_append_resume"]
