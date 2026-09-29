"""``translation.json`` and the English render (RFC-124 §5.1/§5.3 / S2.4).

TWO ARTIFACTS, TWO MEANINGS, and keeping them apart is the whole design:

- ``<base>.translation.json`` is written whenever translation is **attempted**. It is the
  per-unit ledger AND the resume state (D-33): every successfully translated unit is stored
  with its content key, so a repair re-requests only what failed and nothing already paid for
  is thrown away.
- ``<base>.en.txt`` / ``<base>.en.segments.json`` are written **only when every unit
  succeeded**, atomically. Their EXISTENCE is the completeness signal.

WHY EXISTENCE HAS TO BE THE SIGNAL. The resolver's contract is "file present → read it first",
with no status check (`transcript_resolution.py`), and D-38 makes `.en.txt` the player's default
with no marker (D-36). So a partial English render would be consumed as if it were whole, by
every stage and by the listener. That is why the threshold for writing it is zero failed units
rather than a percentage: the harm of a missing unit is not proportional to how many are
missing — one dropped unit can be the pivot the whole episode turns on — and "is 2% acceptable"
is the quality question v1 explicitly deferred.

A failed unit's text is therefore NOTHING, because the render is not written at all. Leaving
source text in `.en.txt` would feed Spanish to English NER; an empty string would silently
shorten the episode. With `.en.txt` absent the resolver falls back to the canonical source and
the right thing happens by construction.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..providers.ml.diarization.formatting import format_diarized_screenplay_with_offsets
from .units import pack_stats, TranslationUnit

logger = logging.getLogger(__name__)

TRANSLATION_SCHEMA_VERSION = "1.0"

#: Unit outcomes. A closed vocabulary so the corpus ledger can GROUP BY them.
UNIT_OK = "ok"
UNIT_FAILED = "failed"

#: Episode-level translation states (mirrors ``translation_stage``'s vocabulary).
STATUS_TRANSLATED = "translated"
STATUS_PENDING = "pending"
STATUS_FAILED = "failed"


@dataclass
class UnitRecord:
    """One unit's row in the ledger — and its resume state."""

    unit_id: str
    turn_id: str
    content_key: str
    status: str
    #: ``sentence`` when numbered output aligned, ``unit`` when it fell back to one block.
    alignment: str = "sentence"
    #: ``[{sent_id, en_text}]`` — empty when the unit failed.
    sentences: List[Dict[str, str]] = field(default_factory=list)
    error: Optional[str] = None
    attempts: int = 0

    @property
    def ok(self) -> bool:
        return self.status == UNIT_OK

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "unit_id": self.unit_id,
            "turn_id": self.turn_id,
            "content_key": self.content_key,
            "status": self.status,
            "alignment": self.alignment,
            "sentences": list(self.sentences),
            "attempts": self.attempts,
        }
        if self.error:
            out["error"] = self.error
        return out

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "UnitRecord":
        return cls(
            unit_id=str(raw.get("unit_id") or ""),
            turn_id=str(raw.get("turn_id") or ""),
            content_key=str(raw.get("content_key") or ""),
            status=str(raw.get("status") or UNIT_FAILED),
            alignment=str(raw.get("alignment") or "sentence"),
            sentences=list(raw.get("sentences") or []),
            error=raw.get("error"),
            attempts=int(raw.get("attempts") or 0),
        )


@dataclass
class TranslationDocument:
    """The ``translation.json`` document."""

    version: str = TRANSLATION_SCHEMA_VERSION
    episode_slug: Optional[str] = None
    source_language: Optional[str] = None
    target_language: str = "en"
    model: Optional[str] = None
    prompt: Optional[Dict[str, Any]] = None
    source: Dict[str, Any] = field(default_factory=dict)
    units: List[UnitRecord] = field(default_factory=list)

    @property
    def failed_units(self) -> List[UnitRecord]:
        return [u for u in self.units if not u.ok]

    @property
    def complete(self) -> bool:
        """Every unit translated. The ONLY condition under which `.en.*` may be written."""
        return bool(self.units) and not self.failed_units

    @property
    def status(self) -> str:
        if not self.units:
            return STATUS_PENDING
        return STATUS_TRANSLATED if self.complete else STATUS_FAILED

    def by_content_key(self) -> Dict[str, UnitRecord]:
        """The resume index: a unit whose text is unchanged need not be translated again."""
        return {u.content_key: u for u in self.units if u.ok and u.content_key}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "episode_slug": self.episode_slug,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "model": self.model,
            "prompt": self.prompt,
            "source": dict(self.source),
            "status": self.status,
            "units_total": len(self.units),
            "units_failed": len(self.failed_units),
            "units": [u.to_dict() for u in self.units],
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "TranslationDocument":
        return cls(
            version=str(raw.get("version") or TRANSLATION_SCHEMA_VERSION),
            episode_slug=raw.get("episode_slug"),
            source_language=raw.get("source_language"),
            target_language=str(raw.get("target_language") or "en"),
            model=raw.get("model"),
            prompt=raw.get("prompt"),
            source=dict(raw.get("source") or {}),
            units=[UnitRecord.from_dict(u) for u in (raw.get("units") or [])],
        )


# --- paths ---------------------------------------------------------------------------------
def translation_json_path(rel_transcript_path: str, effective_output_dir: str) -> str:
    base, _ = os.path.splitext(os.path.join(effective_output_dir, rel_transcript_path))
    return base + ".translation.json"


def english_text_relpath(rel_transcript_path: str) -> str:
    base, ext = os.path.splitext(rel_transcript_path)
    return f"{base}.en{ext or '.txt'}"


def english_segments_relpath(rel_transcript_path: str) -> str:
    base, _ = os.path.splitext(rel_transcript_path)
    return f"{base}.en.segments.json"


# --- the ledger ----------------------------------------------------------------------------
def write_translation_json(
    doc: TranslationDocument, rel_transcript_path: str, effective_output_dir: str
) -> Optional[str]:
    """Write the ledger. Written on EVERY attempt, complete or not — it is the resume state."""
    path = translation_json_path(rel_transcript_path, effective_output_dir)
    try:
        _atomic_write_json(path, doc.to_dict())
    except OSError as exc:
        logger.warning("translation: could not write %s: %s", path, exc)
        return None
    return os.path.relpath(path, effective_output_dir)


def load_translation_json(
    rel_transcript_path: str, effective_output_dir: str
) -> Optional[TranslationDocument]:
    path = translation_json_path(rel_transcript_path, effective_output_dir)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, ValueError):
        return None
    return TranslationDocument.from_dict(raw) if isinstance(raw, dict) else None


# --- the English render --------------------------------------------------------------------
def render_english(
    turns: Sequence[Dict[str, Any]],
    units: Sequence[TranslationUnit],
    doc: TranslationDocument,
) -> Tuple[str, List[Dict[str, Any]]]:
    """``(en_text, en_offset_segments)`` — ONE PSEUDO-SEGMENT PER SENTENCE (RFC-124 §5.1).

    Rendered through the SAME formatter the source screenplay uses, so the two bodies cannot
    diverge in shape and a reader of either works identically.

    Per sentence, not per unit, because the ad-free builder drops any segment overlapping an
    excised range: with one segment per ~120-word unit, an ad boundary would discard up to ~45
    seconds of real speech instead of one fragment. And these segments are served as subtitle
    cues, where a unit-sized cue is a paragraph.

    THE SPEAKER LABEL IS CARRIED VERBATIM (D-24 / S2.6) from the source turn. It never went
    through the translator, which would have renamed the same person inconsistently between
    units.
    """
    by_unit = {u.unit_id: u for u in units}
    records = {r.unit_id: r for r in doc.units}
    turn_label = {str(t.get("turn_id")): str(t.get("speaker_label") or "") for t in turns}
    # Source sentence times, so the English cues line up with the original audio.
    sent_time: Dict[str, Tuple[float, float]] = {}
    for t in turns:
        for s in t.get("sentences") or []:
            sent_time[str(s.get("sent_id"))] = (
                float(s.get("start_ms") or 0) / 1000.0,
                float(s.get("end_ms") or 0) / 1000.0,
            )

    pseudo: List[Dict[str, Any]] = []
    for unit in units:
        rec = records.get(unit.unit_id)
        if rec is None or not rec.ok:
            continue
        label = turn_label.get(unit.turn_id, unit.speaker_label)
        if rec.alignment == "unit":
            # The whole-unit fallback: one cue spanning the unit, timed from its first to its
            # last source sentence. Coarser, and the artifact says so via `alignment`.
            first, last = unit.sentences[0], unit.sentences[-1]
            start = sent_time.get(first.sent_id, (0.0, 0.0))[0]
            end = sent_time.get(last.sent_id, (0.0, 0.0))[1]
            text = (rec.sentences[0]["en_text"] if rec.sentences else "").strip()
            if text:
                pseudo.append(
                    {
                        "start": start,
                        "end": end,
                        "speaker_label": label,
                        "text": text,
                        "unit_id": unit.unit_id,
                        "sent_id": first.sent_id,
                    }
                )
            continue
        for entry in rec.sentences:
            sid = str(entry.get("sent_id") or "")
            text = str(entry.get("en_text") or "").strip()
            if not text:
                continue
            start, end = sent_time.get(sid, (0.0, 0.0))
            pseudo.append(
                {
                    "start": start,
                    "end": end,
                    "speaker_label": label,
                    "text": text,
                    "unit_id": unit.unit_id,
                    "sent_id": sid,
                }
            )

    # The formatter sorts by start time. Source sentence times are monotonic within and across
    # turns, so document order is preserved — but an interpolated tie would otherwise reorder
    # two cues silently, so nudge equal starts by their document position.
    for i, seg in enumerate(pseudo):
        seg["start"] = float(seg["start"]) + i * 1e-6
    _ = by_unit  # (kept for symmetry with resolve_units_for_span's index)
    return format_diarized_screenplay_with_offsets(pseudo)


def write_english_artifacts(
    doc: TranslationDocument,
    turns: Sequence[Dict[str, Any]],
    units: Sequence[TranslationUnit],
    rel_transcript_path: str,
    effective_output_dir: str,
) -> Optional[str]:
    """Write `.en.txt` + `.en.segments.json` — ONLY when the translation is complete.

    Returns the `.en.txt` relpath, or ``None`` when nothing was written (and logs why).

    ATOMIC AS A GROUP. A reader that found `.en.txt` without its sidecar would resolve English
    text against source-language segments — the displacement bug in its newest shape — so both
    are staged and moved, and a failure removes whatever landed.
    """
    if not doc.complete:
        logger.warning(
            "translation: %d of %d units failed for %s — NOT writing the English artifacts, so "
            "summary/GI/KG cannot consume an incomplete translation (RFC-124 §5.3)",
            len(doc.failed_units),
            len(doc.units),
            rel_transcript_path,
        )
        return None

    en_text, en_segments = render_english(turns, units, doc)
    if not en_text.strip() or not en_segments:
        logger.warning("translation: the English render is empty for %s", rel_transcript_path)
        return None

    text_rel = english_text_relpath(rel_transcript_path)
    seg_rel = english_segments_relpath(rel_transcript_path)
    text_path = os.path.join(effective_output_dir, text_rel)
    seg_path = os.path.join(effective_output_dir, seg_rel)
    written: List[str] = []
    try:
        _atomic_write_text(text_path, en_text)
        written.append(text_path)
        _atomic_write_json(seg_path, en_segments, indent=0)
        written.append(seg_path)
    except OSError as exc:
        logger.warning("translation: could not write the English artifacts: %s", exc)
        for path in written:
            try:
                os.remove(path)
            except OSError:
                pass
        return None
    logger.info(
        "    saved English transcript: %s (%d cues, %d units)",
        text_rel,
        len(en_segments),
        len(doc.units),
    )
    return text_rel


def write_english_adfree(
    rel_transcript_path: str,
    effective_output_dir: str,
    extra_cue_patterns: Optional[List[str]] = None,
) -> Optional[str]:
    """Build ``<base>.en.adfree.*`` from the English render, with the EXISTING machinery (S2.5).

    AD EXCISION HAS TO RUN ON THE ENGLISH, and this is the slice where that becomes true rather
    than asserted. ``_AD_PATTERNS`` are English regexes: measured on the V.6a fixture, the
    Spanish source matched **zero** patterns while its English render matched **two**
    ("sponsored by", "visit strava.com"). A source-language ad-free artifact for a non-English
    episode is therefore an IDENTITY artifact — a file claiming ads were removed when the
    patterns simply could not see them — which is what S2.7 removes from the save sites.

    Called only after the English render exists, so it cannot produce an ad-free base for a
    translation that was withheld.
    """
    from ..workflow.adfree_transcript import produce_adfree_artifacts

    en_rel = english_text_relpath(rel_transcript_path)
    en_path = os.path.join(effective_output_dir, en_rel)
    seg_path = os.path.join(effective_output_dir, english_segments_relpath(rel_transcript_path))
    try:
        with open(en_path, "r", encoding="utf-8") as fh:
            en_text = fh.read()
        with open(seg_path, "r", encoding="utf-8") as fh:
            en_segments = json.load(fh)
    except (OSError, ValueError) as exc:
        logger.warning("translation: cannot read the English render for ad-free: %s", exc)
        return None
    if not isinstance(en_segments, list) or not en_segments:
        return None

    produced = produce_adfree_artifacts(
        en_text, en_segments, en_rel, effective_output_dir, extra_cue_patterns=extra_cue_patterns
    )
    if produced is None:
        return None
    rel, artifacts = produced
    logger.info(
        "    saved English ad-free base: %s (%d ad chars removed)", rel, artifacts.chars_removed
    )
    return rel


def resolve_units_for_span(
    segments: Sequence[Dict[str, Any]], char_start: int, char_end: int
) -> List[str]:
    """The ``unit_id``s an English char span touches — provenance for every claim (S2.11).

    RESOLVED THROUGH THE SEGMENTS AND ``unit_id``, NOT THROUGH THE AD-MAP. The ad-map records
    which ranges were excised in the raw coordinate space; it cannot INVERT the ad-free
    transform, so it cannot answer "which unit produced this ad-free offset" (§5.4 C-5,
    measured). The English pseudo-segments carry ``unit_id`` through the render precisely so
    this is a lookup rather than a reconstruction.

    OVERLAP, NOT CONTAINMENT. A quote span routinely touches a ``Label: `` prefix or the
    whitespace between turns, neither of which belongs to any segment — under containment those
    spans would resolve to nothing and the claim would silently carry no provenance. Overlap
    returns every unit the span reaches, which is the honest answer for a span that crosses a
    turn boundary too.
    """
    if char_end <= char_start:
        return []
    out: List[str] = []
    for seg in segments:
        unit_id = seg.get("unit_id")
        if not unit_id:
            continue
        s = int(seg.get("char_start") or 0)
        e = int(seg.get("char_end") or 0)
        if s < char_end and char_start < e:  # half-open overlap
            if unit_id not in out:
                out.append(str(unit_id))
    return out


def verify_span_excerpt(text: str, char_start: int, char_end: int, excerpt: str) -> bool:
    """Whether ``text[char_start:char_end]`` really is ``excerpt`` (S2.5's provenance check).

    Catches a RE-TRANSLATION, which a file hash alone would not: the offsets still look
    plausible and the file still exists, but the text at them has changed. Compared on stripped
    text because the renderer strips each cue.
    """
    if char_end <= char_start or char_end > len(text):
        return False
    return text[char_start:char_end].strip() == (excerpt or "").strip()


def english_artifacts_present(rel_transcript_path: str, effective_output_dir: str) -> bool:
    """Both halves on disk. The completeness signal every consumer keys on."""
    return os.path.isfile(
        os.path.join(effective_output_dir, english_text_relpath(rel_transcript_path))
    ) and os.path.isfile(
        os.path.join(effective_output_dir, english_segments_relpath(rel_transcript_path))
    )


def translation_metrics(
    doc: TranslationDocument, units: Sequence[TranslationUnit]
) -> Dict[str, Any]:
    """What the manifest records — the numbers S2.10 sizes capacity from."""
    metrics: Dict[str, Any] = {
        "status": doc.status,
        "units": len(doc.units),
        "units_failed": len(doc.failed_units),
        "alignment_unit_fallbacks": sum(1 for u in doc.units if u.alignment == "unit"),
        "attempts_total": sum(u.attempts for u in doc.units),
    }
    metrics.update({f"packed_{k}": v for k, v in pack_stats(units).items()})
    failed_ids = [u.unit_id for u in doc.failed_units]
    if failed_ids:
        # Bounded: enough to find the pattern, not an unbounded list in every manifest.
        metrics["failed_unit_ids"] = failed_ids[:20]
    return metrics


# --- io ------------------------------------------------------------------------------------
def _atomic_write_text(path: str, text: str) -> None:
    _atomic_write(path, text.encode("utf-8"))


def _atomic_write_json(path: str, payload: Any, indent: int = 2) -> None:
    _atomic_write(
        path,
        json.dumps(payload, indent=indent, allow_nan=False, ensure_ascii=False).encode("utf-8"),
    )


def _atomic_write(path: str, payload: bytes) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def unresolved_units(
    units: Iterable[TranslationUnit], doc: Optional[TranslationDocument]
) -> List[TranslationUnit]:
    """Units still needing the translator — the resume path (D-33).

    Keyed by CONTENT, not by unit id: a rename changes every offset and every turn-ordinal id
    but no unit's text, so a naming repair on a translated show re-renders at zero GPU cost.
    """
    if doc is None:
        return list(units)
    have = doc.by_content_key()
    return [u for u in units if u.content_key not in have]
