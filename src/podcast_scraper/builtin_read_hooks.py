"""Today's in-tree read-side code on the read hooks (E04 S3).

Each entry imports its code inside its body, so this module loads without it. An entry goes when
its code moves to an extension, which then contributes it through ``Extension.read_hooks``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .read_hooks import ReadHooks


def _segment_speaker(raw: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    from podcast_scraper.speaker_stage import display

    return display._segment_speaker(raw)


def _unidentified_label() -> str:
    from podcast_scraper.speaker_stage import display

    return display._UNIDENTIFIED_TEXT


def _ad_reads(raw: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    from podcast_scraper.insight_stage import read_side

    return read_side._detected_ad_reads(raw)


def _looks_like_publisher(name: str, language: Optional[str]) -> bool:
    from podcast_scraper.speaker_detectors.hosts import looks_like_publisher

    return looks_like_publisher(name, language)


def _entity_id_map(root: Path) -> Dict[str, str]:
    from podcast_scraper.kg.entity_clusters import cached_entity_id_map

    return cached_entity_id_map(root)


def _gi_integrity(root: Path) -> Dict[str, Any]:
    from podcast_scraper.gi.integrity import assess_gi_integrity

    return assess_gi_integrity(root)


BUILTIN = ReadHooks(
    name="builtin",
    segment_speaker=_segment_speaker,
    unidentified_label=_unidentified_label,
    ad_reads=_ad_reads,
    looks_like_publisher=_looks_like_publisher,
    entity_id_map=_entity_id_map,
    gi_integrity=_gi_integrity,
)
