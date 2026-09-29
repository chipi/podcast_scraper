"""Write the ``turns.json`` sidecar next to each transcript variant (RFC-123 / S1.2).

WHAT THIS MODULE IS FOR. :func:`...diarization.turns.build_turns` is pure: offset segments in,
turns out. This module is the part that touches disk — it decides *whether* an episode's text is
the kind of text turns can honestly describe, attaches provenance so a consumer can tell whether
its turns are stale, and reports what happened so the manifest can say so.

**Nothing reads the artifact yet.** Its three consumers (GI attribution, search chunking, the
player's sentence granularity) are v2, and that is deliberate: the artifact has to exist before
translation units can be defined as sentence groups inside a turn (MULTILINGUAL_ARC §4, S2.3).

TURNS ARE ONLY MEANINGFUL FOR A DIARIZED SCREENPLAY. A plain provider transcript has no speaker
lines, so "consecutive segments with the same speaker" is not a turn — it is one turn covering the
episode, which is a fact-shaped lie. The test for which kind of text we hold is the one
:func:`...adfree_transcript.build_adfree_artifacts` already uses: re-render the segments and
compare. Equal means the text WAS rendered by this formatter, so the offsets it computes index the
text on disk exactly. Unequal means it was not, and we write nothing and say why (RFC-123 §2,
``turns: unavailable``).

ONE ARTIFACT PER VARIANT, NEVER ONE PER EPISODE (RFC-123 §Key Decisions 5). Char offsets are only
meaningful in one text, and the ad-free variant drops whole turns and renumbers from ``t0000`` — so
a single artifact would have to pick a coordinate space and silently mis-anchor the other, and a
cross-variant join by ``turn_id`` would point at the wrong turn.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from ..providers.ml.diarization.formatting import format_diarized_screenplay_with_offsets
from ..providers.ml.diarization.turns import build_turns, TurnInvariantError

logger = logging.getLogger(__name__)

#: RFC-123 §1. Additive fields (word anchors, V2-F) do not break this.
TURNS_SCHEMA_VERSION = "1.0"

TURNS_SUFFIX = ".turns.json"


@dataclass
class TurnsOutcome:
    """What happened for ONE variant — written or not, and the manifest's numbers either way.

    ``relpath is None`` with ``unavailable_reason`` set is a first-class result, not an error:
    a legacy or plain-text transcript genuinely has no turns, and RFC-123 §2 says consumers fall
    back to current behaviour. ``invariant_failures`` is separate precisely because it is NOT that
    — it means the builder produced a structure that disagreed with its own text.
    """

    relpath: Optional[str] = None
    count: int = 0
    backchannels: int = 0
    median_turn_s: Optional[float] = None
    invariant_failures: int = 0
    unavailable_reason: Optional[str] = None

    def to_metrics(self) -> Dict[str, Any]:
        """The four RFC-123 monitoring keys, plus the reason when there is one."""
        out: Dict[str, Any] = {
            "count": self.count,
            "backchannels": self.backchannels,
            "median_turn_s": self.median_turn_s,
            "invariant_failures": self.invariant_failures,
        }
        if self.unavailable_reason:
            out["unavailable_reason"] = self.unavailable_reason
        return out


def turns_path(rel_transcript_path: str, effective_output_dir: str) -> str:
    """``transcripts/ep1.txt`` -> ``<out>/transcripts/ep1.turns.json``.

    The ad-free variant needs no special case: its relpath already carries ``.adfree``, so
    ``ep1.adfree.txt`` yields ``ep1.adfree.turns.json``.
    """
    full = os.path.join(effective_output_dir, rel_transcript_path)
    base, _ = os.path.splitext(full)
    return base + TURNS_SUFFIX


def _segments_sha256(segments_path: str) -> Optional[str]:
    """Hash of the segments sidecar AS WRITTEN, for provenance-keyed invalidation.

    The file, not the in-memory list: a consumer checking whether its turns are stale can only
    hash what is on disk, so that has to be what the artifact claims. ``None`` when the sidecar
    is absent — honest, and better than a hash of something the consumer cannot reproduce.
    """
    try:
        with open(segments_path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return None


def _median_seconds(durations: Sequence[float]) -> Optional[float]:
    if not durations:
        return None
    ordered = sorted(durations)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return round(ordered[mid], 3)
    return round((ordered[mid - 1] + ordered[mid]) / 2.0, 3)


def build_turns_document(
    text: str,
    segments: Sequence[Dict[str, Any]],
    *,
    rel_transcript_path: str,
    episode_slug: Optional[str],
    language: Optional[str],
    segments_sha256: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """The full sidecar document, or ``None`` when this text is not a diarized screenplay.

    Raises :class:`TurnInvariantError` from the builder — deliberately not caught here, so a
    caller that wants the artifact to be trustworthy can choose to fail, and the one caller that
    must not lose an episode (:func:`write_turns_artifact`) makes that choice explicitly and in
    one place.
    """
    if not text or not segments:
        return None
    rebuilt, offset_segments = format_diarized_screenplay_with_offsets(list(segments))
    if rebuilt != text:
        return None

    turns = build_turns(offset_segments, screenplay_text=rebuilt)
    base, _ = os.path.splitext(rel_transcript_path)
    doc: Dict[str, Any] = {
        "version": TURNS_SCHEMA_VERSION,
        "episode_slug": episode_slug,
        "language": language,
        "source": {
            "segments_ref": base + ".segments.json",
            "segments_sha256": segments_sha256,
            "transcript_ref": rel_transcript_path,
        },
    }
    doc.update(turns.to_dict())
    # `turns.to_dict()` also carries `version`; the builder's and the artifact's are the same
    # schema, so let the envelope's win rather than shipping two version keys that could drift.
    doc["version"] = TURNS_SCHEMA_VERSION
    return doc


def write_turns_artifact(
    text: str,
    segments: Optional[Sequence[Dict[str, Any]]],
    rel_transcript_path: str,
    effective_output_dir: str,
    *,
    language: Optional[str] = None,
) -> TurnsOutcome:
    """Build and write one variant's ``turns.json``. Never raises into the pipeline.

    AN INVARIANT FAILURE DOES NOT LOSE THE EPISODE — a deviation from RFC-123 §Monitoring ("any
    invariant failure fails the build for that episode"), taken on purpose and only while the
    artifact has no consumers. Nothing reads these turns in v1, so a builder bug can corrupt
    nothing downstream; aborting the episode would convert a cosmetic defect in a brand-new
    sidecar into lost ASR and a lost GPU hour. The failure is recorded (``invariant_failures``
    in the manifest, plus an ERROR log with the episode), which is the opposite of suppressing
    it. **When S1.4 switches GI attribution to turn lookup this has to harden**: from that point
    a silent invariant failure mis-attributes quotes, and the RFC's rule becomes the right one.
    """
    if not rel_transcript_path or not text or not segments:
        return TurnsOutcome(unavailable_reason="no_segments")

    base, _ = os.path.splitext(rel_transcript_path)
    episode_slug = os.path.basename(base)
    segments_file = os.path.join(effective_output_dir, base + ".segments.json")

    try:
        doc = build_turns_document(
            text,
            segments,
            rel_transcript_path=rel_transcript_path,
            episode_slug=episode_slug,
            language=language,
            segments_sha256=_segments_sha256(segments_file),
        )
    except TurnInvariantError as exc:
        logger.error(
            "turns: invariant failure for %s — no turns.json written: %s",
            rel_transcript_path,
            exc,
        )
        return TurnsOutcome(invariant_failures=1, unavailable_reason="invariant_failure")

    if doc is None:
        # Not a diarized screenplay (or nothing to build from). RFC-123 §2: `turns: unavailable`
        # and consumers keep their current behaviour.
        logger.debug(
            "turns: %s is not a rendered diarized screenplay; no turns.json", rel_transcript_path
        )
        return TurnsOutcome(unavailable_reason="not_a_diarized_screenplay")

    out_path = turns_path(rel_transcript_path, effective_output_dir)
    try:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=0, allow_nan=False)
    except OSError as exc:
        logger.warning("turns: could not write %s: %s", out_path, exc)
        return TurnsOutcome(unavailable_reason="write_failed")

    # Measured from the rows actually written, not from the builder's objects: the manifest should
    # describe the file on disk.
    turn_rows: List[Dict[str, Any]] = doc.get("turns") or []
    count = len(turn_rows)
    backchannels = sum(1 for t in turn_rows if t.get("backchannel"))
    median = _median_seconds([(int(t["end_ms"]) - int(t["start_ms"])) / 1000.0 for t in turn_rows])
    logger.debug("turns: wrote %s (%d turns, %d backchannels)", out_path, count, backchannels)
    return TurnsOutcome(
        relpath=os.path.relpath(out_path, effective_output_dir),
        count=count,
        backchannels=backchannels,
        median_turn_s=median,
        invariant_failures=0,
    )


def turns_manifest_metrics(
    raw: TurnsOutcome, adfree: Optional[TurnsOutcome] = None
) -> Dict[str, Any]:
    """The manifest ``turns`` block's metrics: the source variant's numbers, plus the ad-free's.

    The four RFC-123 keys stay at the top level and describe the SOURCE variant, because that is
    the one every episode has; the ad-free variant is nested because it is conditional on
    ``save_adfree_transcript`` and its ids live in a different coordinate space.
    """
    metrics = raw.to_metrics()
    if adfree is not None:
        metrics["adfree"] = adfree.to_metrics()
    return metrics
