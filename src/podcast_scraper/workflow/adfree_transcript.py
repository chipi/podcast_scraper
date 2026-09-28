"""Produce the ad-free processing-base transcript (#974).

The two-artifact transcript model keeps the raw screenplay ``.txt`` (with ads, full
timeline) as the canonical source-of-truth (future subtitle player) and derives an
**ad-free** sibling that becomes the base for all NLP — GI quote offsets, enrich-edges
SPOKEN_BY, search chunking, and the viewer reader. Producing it here (at transcript
save time) means a single coordinate space: the ad-free text is *saved*, and is the
space GI's ``char_start`` lives in, so the consumers that read it never drift.

Artifacts written next to the raw ``<base>.txt``:

- ``<base>.adfree.txt``          — ad-free screenplay (the processing base)
- ``<base>.adfree.segments.json``— segments, each carrying its ``char_start`` /
  ``char_end`` range in the ad-free text (so a quote maps to a segment exactly, with
  no cumulative-length guard — #974 Fault B)
- ``<base>.adfree.admap.json``   — the ad-map: excised ranges in raw-screenplay space,
  to reconcile an ad-free offset back to the raw transcript for the future player

This module PRODUCES those artifacts. Deciding which variant a reader should consume is a
separate job and lives in :mod:`podcast_scraper.workflow.transcript_resolution` (#2170) —
the resolver names are re-exported here so existing importers keep working.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..cleaning.commercial.crosspromo import crosspromo_char_end
from ..gi.ad_regions import (
    _overlaps_any,
    excise_ad_regions,
    excise_ad_regions_with_offsets,
    merge_preroll_range,
)
from ..providers.ml.diarization.formatting import format_diarized_screenplay_with_offsets
from .transcript_resolution import (  # noqa: F401 - re-exported for existing importers
    ADFREE_SUFFIX as _ADFREE_SUFFIX,
    adfree_transcript_relpath,
    load_processing_transcript,
    load_transcript,
    ProcessingTranscript,
    TranscriptPurpose,
)

logger = logging.getLogger(__name__)

#: Re-exported for importers that predate ``transcript_resolution`` (#2170).
ADFREE_SUFFIX = _ADFREE_SUFFIX


@dataclass
class AdfreeArtifacts:
    """The ad-free text + offset-carrying segments + ad-map for one episode."""

    text: str
    segments: List[Dict[str, Any]]
    ad_map: Dict[str, Any]
    chars_removed: int


def _derive_offsets_by_find(text: str, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Locate each segment's stripped text in ``text`` (non-diarized / provider format).

    Used when the transcript is not the diarized screenplay (so we cannot re-derive
    exact offsets by reformatting). Progressive search keeps segments in order.
    """
    out: List[Dict[str, Any]] = []
    cursor = 0
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        t = (seg.get("text") or "").strip()
        if not t:
            continue
        idx = text.find(t, cursor)
        if idx < 0:
            idx = text.find(t)  # fall back to a global search
        if idx < 0:
            continue  # cannot locate (e.g. provider reflowed text) — skip this segment
        emitted = {
            "start": float(seg.get("start") or 0.0),
            "end": float(seg.get("end") or 0.0),
            "speaker_label": seg.get("speaker_label") or seg.get("speaker"),
            "text": t,
            "char_start": idx,
            "char_end": idx + len(t),
        }
        # Preserve the per-segment role/type truth (the metadata reader prefers the ad-free
        # sidecar; dropping speaker_role here resurrects the guest-as-host bug).
        for key in ("speaker", "speaker_role", "voice_type"):
            val = seg.get(key)
            if val is not None:
                emitted[key] = val
        out.append(emitted)
        cursor = idx + len(t)
    return out


def build_adfree_artifacts(
    text: str,
    segments: Optional[List[Dict[str, Any]]],
    *,
    extra_cue_patterns: Optional[List[str]] = None,
) -> Optional[AdfreeArtifacts]:
    """Build the ad-free text + offset segments + ad-map from the saved transcript.

    Returns ``None`` when there is nothing to process (no text / no segments). When no
    ad regions are detected the ad-free text equals the input and all segments survive
    — still a valid (identity) processing base, so consumers can always read it.

    ``extra_cue_patterns`` extends the built-in opening cross-promo cue set (#1188)
    with feed-onboarding patterns; ``None`` uses the built-ins.
    """
    if not text or not segments:
        return None

    rebuilt, offset_segs = format_diarized_screenplay_with_offsets(segments)
    # Opening host-read cross-promo (#1188): a diarization-detected leading ad the
    # pattern/density passes miss. Both branches locate it the same way — the char
    # offset where it ends — then fold it into the pre-roll so the roster (dropped
    # segments), text, offsets, and ad-map all stay in one coordinate space.
    cp_end = crosspromo_char_end(offset_segs, extra_cue_patterns=extra_cue_patterns)
    if rebuilt == text:
        # Diarized screenplay: detect ad ranges, DROP the segments inside them, then
        # RE-RENDER the survivors. Re-rendering (vs a raw char-cut) guarantees every
        # surviving turn keeps its ``Name:`` marker — a char-cut would sever the marker
        # of an ad that coalesced into the same-speaker turn as the following content —
        # and the offsets come straight from the formatter, so they stay exact.
        _, _, meta = excise_ad_regions(text)
        merge_preroll_range(meta, cp_end, text=text)
        ranges = meta.excised_ranges
        survivors = (
            [s for s in offset_segs if not _overlaps_any(s["char_start"], s["char_end"], ranges)]
            if ranges
            else offset_segs
        )
        adfree_text, adfree_segs = format_diarized_screenplay_with_offsets(survivors)
        return AdfreeArtifacts(
            text=adfree_text,
            segments=adfree_segs,
            ad_map=meta.to_dict(),
            chars_removed=meta.chars_removed,
        )

    # Plain / provider transcript (no speaker markers to preserve): a char-level cut is
    # exact. Derive each segment's offset by progressive search, then excise. The
    # cross-promo end is recomputed on THESE offsets (their own char space).
    offset_segs = _derive_offsets_by_find(text, segments)
    if not offset_segs:
        return None
    cp_end = crosspromo_char_end(offset_segs, extra_cue_patterns=extra_cue_patterns)
    adfree_text, adfree_segs, meta = excise_ad_regions_with_offsets(
        text, offset_segs, extra_preroll_end=cp_end
    )
    return AdfreeArtifacts(
        text=adfree_text,
        segments=adfree_segs,
        ad_map=meta.to_dict(),
        chars_removed=meta.chars_removed,
    )


def save_adfree_artifacts(
    rel_transcript_path: str,
    effective_output_dir: str,
    artifacts: AdfreeArtifacts,
) -> Optional[str]:
    """Write the three ad-free sidecars next to the raw transcript.

    Returns the relative path to ``<base>.adfree.txt`` (or ``None`` on write failure).
    """
    if not rel_transcript_path:
        return None
    full_path = os.path.join(effective_output_dir, rel_transcript_path)
    base, _ = os.path.splitext(full_path)
    adfree_txt = base + ADFREE_SUFFIX + ".txt"
    adfree_segs = base + ADFREE_SUFFIX + ".segments.json"
    adfree_admap = base + ADFREE_SUFFIX + ".admap.json"
    try:
        with open(adfree_txt, "w", encoding="utf-8") as f:
            f.write(artifacts.text)
        with open(adfree_segs, "w", encoding="utf-8") as f:
            json.dump(artifacts.segments, f, indent=0, allow_nan=False)
        with open(adfree_admap, "w", encoding="utf-8") as f:
            json.dump(artifacts.ad_map, f, indent=2)
    except OSError as exc:
        # A write failure here silently degrades every downstream NLP consumer to the RAW,
        # ad-laden transcript (is_adfree=False) with no other signal — so it is a WARNING, not a
        # swallowed DEBUG (C3).
        logger.warning("Could not save ad-free artifacts for %s: %s", rel_transcript_path, exc)
        return None
    logger.debug(
        "Saved ad-free transcript base: %s (%d ad chars removed)",
        adfree_txt,
        artifacts.chars_removed,
    )
    return os.path.relpath(adfree_txt, effective_output_dir)


def produce_adfree_transcript(
    text: str,
    segments: Optional[List[Dict[str, Any]]],
    rel_transcript_path: str,
    effective_output_dir: str,
    *,
    extra_cue_patterns: Optional[List[str]] = None,
) -> Optional[str]:
    """Convenience: build + save the ad-free artifacts. Returns the ``.adfree.txt`` relpath."""
    artifacts = build_adfree_artifacts(text, segments, extra_cue_patterns=extra_cue_patterns)
    if artifacts is None:
        return None
    return save_adfree_artifacts(rel_transcript_path, effective_output_dir, artifacts)
