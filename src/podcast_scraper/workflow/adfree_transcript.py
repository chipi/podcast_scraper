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
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..gi.ad_regions import _overlaps_any, excise_ad_regions, excise_ad_regions_with_offsets
from ..providers.ml.diarization.formatting import format_diarized_screenplay_with_offsets

logger = logging.getLogger(__name__)

ADFREE_SUFFIX = ".adfree"


@dataclass
class AdfreeArtifacts:
    """The ad-free text + offset-carrying segments + ad-map for one episode."""

    text: str
    segments: List[Dict[str, Any]]
    ad_map: Dict[str, Any]
    chars_removed: int


def adfree_transcript_relpath(transcript_relpath: str) -> str:
    """``transcripts/01 - ep.txt`` -> ``transcripts/01 - ep.adfree.txt``."""
    base, ext = os.path.splitext(transcript_relpath)
    return f"{base}{ADFREE_SUFFIX}{ext or '.txt'}"


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
        out.append(
            {
                "start": float(seg.get("start") or 0.0),
                "end": float(seg.get("end") or 0.0),
                "speaker_label": seg.get("speaker_label") or seg.get("speaker"),
                "text": t,
                "char_start": idx,
                "char_end": idx + len(t),
            }
        )
        cursor = idx + len(t)
    return out


def build_adfree_artifacts(
    text: str, segments: Optional[List[Dict[str, Any]]]
) -> Optional[AdfreeArtifacts]:
    """Build the ad-free text + offset segments + ad-map from the saved transcript.

    Returns ``None`` when there is nothing to process (no text / no segments). When no
    ad regions are detected the ad-free text equals the input and all segments survive
    — still a valid (identity) processing base, so consumers can always read it.
    """
    if not text or not segments:
        return None

    rebuilt, offset_segs = format_diarized_screenplay_with_offsets(segments)
    if rebuilt == text:
        # Diarized screenplay: detect ad ranges, DROP the segments inside them, then
        # RE-RENDER the survivors. Re-rendering (vs a raw char-cut) guarantees every
        # surviving turn keeps its ``Name:`` marker — a char-cut would sever the marker
        # of an ad that coalesced into the same-speaker turn as the following content —
        # and the offsets come straight from the formatter, so they stay exact.
        _, _, meta = excise_ad_regions(text)
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
    # exact. Derive each segment's offset by progressive search, then excise.
    offset_segs = _derive_offsets_by_find(text, segments)
    if not offset_segs:
        return None
    adfree_text, adfree_segs, meta = excise_ad_regions_with_offsets(text, offset_segs)
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
        logger.debug("Could not save ad-free artifacts for %s: %s", rel_transcript_path, exc)
        return None
    logger.debug(
        "Saved ad-free transcript base: %s (%d ad chars removed)",
        adfree_txt,
        artifacts.chars_removed,
    )
    return os.path.relpath(adfree_txt, effective_output_dir)


def _read_text(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def _read_json(path: str) -> Optional[Any]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


@dataclass
class ProcessingTranscript:
    """The transcript a consumer should reason over (#974).

    ``is_adfree`` is True when the ad-free base was found and loaded; in that case the
    text + segments already exclude ads and ``segments`` carry exact ``char_start`` /
    ``char_end`` ranges, and ``ad_map`` describes what was cut. When False the caller
    got the raw ``.txt`` (old corpus / ad-free disabled) and must excise itself.
    """

    text: str
    segments: Optional[List[Dict[str, Any]]]
    transcript_ref: str
    ad_map: Optional[Dict[str, Any]]
    is_adfree: bool


def load_processing_transcript(output_dir: str, transcript_file_path: str) -> ProcessingTranscript:
    """Load the ad-free base if present, else the raw transcript.

    This is the single resolver all NLP consumers (GI, enrich-edges, search) use so
    they read one coordinate space. ``transcript_ref`` is the relpath that was actually
    loaded — point quote/viewer references at it so highlights align.
    """
    adfree_rel = adfree_transcript_relpath(transcript_file_path)
    adfree_full = os.path.join(output_dir, adfree_rel)
    if os.path.isfile(adfree_full):
        adfree_base = os.path.splitext(adfree_full)[0]  # <…>.adfree
        text = _read_text(adfree_full)
        segs = _read_json(adfree_base + ".segments.json")
        ad_map = _read_json(adfree_base + ".admap.json")
        return ProcessingTranscript(
            text=text,
            segments=segs if isinstance(segs, list) else None,
            transcript_ref=adfree_rel,
            ad_map=ad_map if isinstance(ad_map, dict) else None,
            is_adfree=True,
        )

    raw_full = os.path.join(output_dir, transcript_file_path)
    text = _read_text(raw_full) if os.path.isfile(raw_full) else ""
    segs = _read_json(os.path.splitext(raw_full)[0] + ".segments.json")
    return ProcessingTranscript(
        text=text,
        segments=segs if isinstance(segs, list) else None,
        transcript_ref=transcript_file_path,
        ad_map=None,
        is_adfree=False,
    )


def produce_adfree_transcript(
    text: str,
    segments: Optional[List[Dict[str, Any]]],
    rel_transcript_path: str,
    effective_output_dir: str,
) -> Optional[str]:
    """Convenience: build + save the ad-free artifacts. Returns the ``.adfree.txt`` relpath."""
    artifacts = build_adfree_artifacts(text, segments)
    if artifacts is None:
        return None
    return save_adfree_artifacts(rel_transcript_path, effective_output_dir, artifacts)
