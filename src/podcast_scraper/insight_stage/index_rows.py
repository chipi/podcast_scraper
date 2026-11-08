"""The insight and quote rows the search index takes from an episode's ``gi.json`` (the block moved
verbatim from ``search.indexer._collect_docs_for_episode``, E04 S4).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast, Dict, List, Optional, Tuple

from podcast_scraper.search.indexer import _gi_path

logger = logging.getLogger(__name__)


def gi_index_rows(
    episode_root: Path,
    metadata_path: Path,
    doc: Dict[str, Any],
    *,
    scope_tag: str,
    episode_id: str,
    raw_feed_id: Any,
    published: Any,
) -> List[Tuple[str, str, Dict[str, Any]]]:
    """``(doc_id, text, metadata)`` rows for the episode's Insight and Quote nodes."""
    rows: List[Tuple[str, str, Dict[str, Any]]] = []
    gi_path = _gi_path(episode_root, metadata_path, doc)
    artifact: Optional[Dict[str, Any]] = None
    if gi_path.is_file():
        try:
            artifact = cast(Dict[str, Any], json.loads(gi_path.read_text(encoding="utf-8")))
        except Exception as exc:
            logger.warning("Skip gi.json %s: %s", gi_path, exc)

    if artifact:
        for n in artifact.get("nodes") or []:
            nt = n.get("type")
            props = n.get("properties") or {}
            nid = n.get("id")
            if not isinstance(nid, str):
                continue
            if nt == "Insight":
                text = props.get("text")
                if isinstance(text, str) and text.strip():
                    # RFC-072 GIL v1.1 insight_type is carried on the GI node
                    # properties (claim / recommendation / observation /
                    # question / unknown). Emit it into the index payload so
                    # downstream scope filters (Search v3 §S8 compare
                    # ``insight_types``) can narrow by type. Legacy nodes with
                    # no type default to ``unknown`` — matching the migration
                    # in ``migrations/gil_kg_identity_migrations.py``.
                    raw_itype = props.get("insight_type")
                    insight_type = (
                        str(raw_itype).strip().lower()
                        if isinstance(raw_itype, str) and raw_itype.strip()
                        else "unknown"
                    )
                    rows.append(
                        (
                            f"insight:{scope_tag}:{nid}",
                            text.strip(),
                            {
                                "doc_type": "insight",
                                "episode_id": episode_id,
                                "feed_id": raw_feed_id,
                                "publish_date": published,
                                "source_id": nid,
                                "grounded": bool(props.get("grounded")),
                                "insight_type": insight_type,
                            },
                        )
                    )
            elif nt == "Quote":
                text = props.get("text")
                if isinstance(text, str) and text.strip():
                    rows.append(
                        (
                            f"quote:{scope_tag}:{nid}",
                            text.strip(),
                            {
                                "doc_type": "quote",
                                "episode_id": episode_id,
                                "feed_id": raw_feed_id,
                                "publish_date": published,
                                "source_id": nid,
                                "speaker_id": props.get("speaker_id"),
                                "char_start": props.get("char_start"),
                                "char_end": props.get("char_end"),
                                "timestamp_start_ms": props.get("timestamp_start_ms"),
                                "timestamp_end_ms": props.get("timestamp_end_ms"),
                            },
                        )
                    )

    return rows
