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
    #: The model and prompt that produced THIS unit's text (S2.11).
    #:
    #: Per-unit, not per-document, because a partial resume mixes them: run 1 under model A,
    #: one turn edited, run 2 under model B leaves N-1 units of A's output in a ledger whose
    #: document-level model says B. Every claim resolving to an A-unit would then carry
    #: `model: B` and a hash computed under B — the misattribution the document-level carry was
    #: meant to stop, narrowed to the partial case rather than removed.
    model: Optional[str] = None
    prompt_sha256: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status == UNIT_OK

    def to_dict(self) -> Dict[str, Any]:
        """One unit's ledger row. `error` appears only when there is one.

        `content_key` is the field that makes the ledger a resume index rather than a log: the
        memory is keyed on the unit's TEXT and deliberately not on the model (D-33), so a model
        change does not invalidate work already done. `model` is recorded for provenance, which
        is a different job from keying.
        """
        out: Dict[str, Any] = {
            "unit_id": self.unit_id,
            "turn_id": self.turn_id,
            "content_key": self.content_key,
            "status": self.status,
            "alignment": self.alignment,
            "sentences": list(self.sentences),
            "attempts": self.attempts,
            "model": self.model,
            "prompt_sha256": self.prompt_sha256,
        }
        if self.error:
            out["error"] = self.error
        return out

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "UnitRecord":
        """Read a ledger row back, defaulting an unreadable status to FAILED.

        Every field is coerced and nothing raises, because this parses an artifact from a
        previous run: a ledger that cannot be read must cost us a re-translation, never the run.
        A missing status becoming FAILED is the safe direction — it re-does work rather than
        skipping it.
        """
        return cls(
            unit_id=str(raw.get("unit_id") or ""),
            turn_id=str(raw.get("turn_id") or ""),
            content_key=str(raw.get("content_key") or ""),
            status=str(raw.get("status") or UNIT_FAILED),
            alignment=str(raw.get("alignment") or "sentence"),
            sentences=list(raw.get("sentences") or []),
            error=raw.get("error"),
            attempts=int(raw.get("attempts") or 0),
            model=raw.get("model"),
            prompt_sha256=raw.get("prompt_sha256"),
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
    #: True when every unit translated but the set on disk is NOT CONSUMABLE — the ANALYSIS body
    #: could not be written. Without it ``status`` reads ``translated`` (it is derived from unit
    #: outcomes alone) and the API repeats that for an episode nothing can analyse.
    #:
    #: THE NAME IS A LEGACY SPELLING. It dates from before D-44, when the response was to delete
    #: the `.en.*` files — to "withdraw" them. The atomic swap leaves no partial set to withdraw,
    #: so nothing is deleted any more and the flag now means only "not consumable". It keeps its
    #: name because it is a PERSISTED ledger key: every `translation.json` already on disk spells
    #: it this way, and renaming it would split the field across two spellings for no gain.
    english_withdrawn: bool = False
    #: The EPISODE title in English, when it was translated (S2.4's title decision).
    #:
    #: THE EPISODE TITLE IS TRANSLATED; THE SHOW NAME IS NOT. The deployed model turns
    #: `Sesiones de Sendero` into `Trail Sessions` (ADR-157's evidence), and a show's name is its
    #: identity — renaming it would change what the feed IS on every surface, in search, and in
    #: every person's saved library. An EPISODE title is a description of that episode's content,
    #: which is exactly the kind of thing translation is for, and §5.4 C-6 needs it in English:
    #: the roster reads the title and description for host/guest context and for NER candidate
    #: discovery, so a Spanish title feeding an English NER is the §5.2 hazard (recall 2/2,
    #: precision 67% -> 18%) pointed at the one input naming trusts most.
    #:
    #: ``None`` means it was not translated — an English episode, or a title that failed. The
    #: naming stage falls back to the source title in that case rather than passing nothing,
    #: because a missing title costs the roster its role context entirely.
    title_en: Optional[str] = None

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
        if self.english_withdrawn:
            # Every unit translated, but the set on disk is not consumable. Reporting
            # `translated` here is what let the API claim success for an episode with no
            # English artifacts at all.
            return STATUS_FAILED
        return STATUS_TRANSLATED if self.complete else STATUS_FAILED

    def by_content_key(self) -> Dict[str, UnitRecord]:
        """The resume index: a unit whose text is unchanged need not be translated again."""
        return {u.content_key: u for u in self.units if u.ok and u.content_key}

    def to_dict(self) -> Dict[str, Any]:
        """The whole ledger, with the unit tallies computed rather than stored.

        `units_total` and `units_failed` are derived here on every write so they cannot drift
        from the `units` list beside them — the completeness gate (RFC-124 §5.3) reads them, and
        a stored counter that disagreed with its own list would let an incomplete translation
        pass as complete.
        """
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
            "english_withdrawn": self.english_withdrawn,
            "title_en": self.title_en,
            "units": [u.to_dict() for u in self.units],
        }

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "TranslationDocument":
        """Read a ledger back for resume. The derived tallies are recomputed, not trusted.

        `units_total`/`units_failed` are absent from the constructor for that reason: they are
        written by `to_dict` and re-derived from `units` on load, so a hand-edited or truncated
        file cannot assert a completeness it does not have.
        """
        return cls(
            version=str(raw.get("version") or TRANSLATION_SCHEMA_VERSION),
            episode_slug=raw.get("episode_slug"),
            source_language=raw.get("source_language"),
            target_language=str(raw.get("target_language") or "en"),
            model=raw.get("model"),
            prompt=raw.get("prompt"),
            source=dict(raw.get("source") or {}),
            units=[UnitRecord.from_dict(u) for u in (raw.get("units") or [])],
            english_withdrawn=bool(raw.get("english_withdrawn")),
            title_en=raw.get("title_en"),
        )


# --- paths ---------------------------------------------------------------------------------
def translation_json_path(rel_transcript_path: str, effective_output_dir: str) -> str:
    """Absolute path of the ledger beside the transcript it describes.

    Derived from the CANONICAL transcript path — the suffix STACK (`.en`, `.adfree`, `.cleaned`,
    `.anon`) is only valid when built from the canonical name, so handing this an
    already-suffixed path yields a ledger for an episode that does not exist.
    """
    base, _ = os.path.splitext(os.path.join(effective_output_dir, rel_transcript_path))
    return base + ".translation.json"


def english_text_relpath(rel_transcript_path: str) -> str:
    """The English render's relpath — which is the CANONICAL path itself (D-44).

    ENGLISH IS THE FILE WITHOUT A LANGUAGE SUFFIX. That is the whole naming rule, and it is what
    lets every generic reader stay ignorant of language: `<base>.txt` is English on an English
    episode because ASR wrote it there, and English on a translated episode because translation
    swapped it there. One name, one meaning, no variant for anyone to choose between.

    This used to return `<base>.en.txt`, which forced ~23 references to the `.en.` suffix into
    modules that have nothing to do with language — the resolver, the indexer, the API route, the
    metadata stage — each of them trying a candidate that never matches on an English episode.
    Inverting the rule deletes all of that rather than reverting it file by file.
    """
    return rel_transcript_path


def canonical_segments_relpath(rel_transcript_path: str) -> str:
    """Relpath of the English segments sidecar — the canonical sidecar name (D-44)."""
    base, _ = os.path.splitext(rel_transcript_path)
    return f"{base}.segments.json"


def source_text_relpath(rel_transcript_path: str, language: str) -> str:
    """``<base>.<lang>.<ext>`` — where the SOURCE body lives once translation has swapped it.

    The record of what was actually said (D-2) keeps its own name rather than being overwritten.
    `language` is the normalized primary subtag, so a feed declaring `es-ES` produces `.es.txt`
    and not `.es-ES.txt`: the suffix has to be predictable for the toggle to ask for it.

    Refuses ``en``. An English episode has no separate source body — the canonical file IS the
    English one — and generating `<base>.en.txt` here would reintroduce exactly the suffix this
    scheme exists to remove.
    """
    normalized = (language or "").strip().lower().split("-")[0]
    if not normalized or normalized == "en":
        raise ValueError(
            f"source_text_relpath is for a NON-English source language, got {language!r}: "
            "English is the unsuffixed canonical file (D-44), so it has no source variant"
        )
    base, ext = os.path.splitext(rel_transcript_path)
    return f"{base}.{normalized}{ext or '.txt'}"


def source_segments_relpath(rel_transcript_path: str, language: str) -> str:
    """Relpath of the SOURCE-language segments sidecar (D-44)."""
    text = source_text_relpath(rel_transcript_path, language)
    base, _ = os.path.splitext(text)
    return f"{base}.segments.json"


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
    """The previous run's ledger, or None if there is not a readable one.

    None for both "no file" and "unreadable file", on purpose: the caller's next move is the
    same either way — translate the units — and distinguishing them would only offer a chance to
    treat a corrupt ledger as authoritative.
    """
    path = translation_json_path(rel_transcript_path, effective_output_dir)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, ValueError):
        return None
    return TranslationDocument.from_dict(raw) if isinstance(raw, dict) else None


# --- the English render --------------------------------------------------------------------
def render_target_text(
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

    # No tie-breaking nudge. `sorted()` is stable, so cues with equal start times already keep
    # document order — an earlier version added `i * 1e-6` to every start "to be safe", which
    # persisted into the artifact and gave a zero-duration sentence `start > end`.
    return format_diarized_screenplay_with_offsets(pseudo)


def write_translated_artifacts(
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

    en_text, en_segments = render_target_text(turns, units, doc)
    if not en_text.strip() or not en_segments:
        logger.warning("translation: the English render is empty for %s", rel_transcript_path)
        return None

    return _swap_in_translation(
        rel_transcript_path,
        effective_output_dir,
        doc.source_language,
        target_text=en_text,
        target_segments=en_segments,
        unit_count=len(doc.units),
    )


def _swap_in_translation(
    rel_transcript_path: str,
    effective_output_dir: str,
    source_language: Optional[str],
    *,
    target_text: str,
    target_segments: Any,
    unit_count: int,
) -> Optional[str]:
    """Move the SOURCE body aside and put the translation at the canonical path. Both, or neither.

    D-44 makes ``<base>.txt`` the analysis-language body, so finishing a translation means two
    renames: the source moves to ``<base>.<lang>.txt`` and the rendered translation takes its place.
    The pair has to be indivisible. Every generic reader opens the canonical path without asking
    what language it holds, so a half-applied swap is not a degraded episode — it is an episode that
    LIES, and summary/GI/KG/search would run English prompts over source-language text and produce
    confident nonsense (§5.2: precision 67% -> 18%, inventing people rather than finding none).

    HOW ATOMICITY IS ACHIEVED without a transactional filesystem: everything that can fail is done
    BEFORE anything is visible. The translation is staged to temp files beside their targets, the
    source is renamed to its language-tagged name, and only then do the staged files take the
    canonical names. ``os.replace`` is atomic within a filesystem, so each individual step either
    happens or does not. If a later step fails, the earlier ones are rolled back in reverse — and
    the rollback only has to move files that are already written, which is the cheap direction.

    THE WINDOW THAT REMAINS, stated rather than hidden: between renaming the source away and
    renaming the staged body in, the canonical path does not exist. A reader in that instant sees a
    missing transcript, not a wrong one — which is the failure we can tolerate, because every reader
    already handles an absent body and none of them can handle a mislabelled one. The window is two
    rename syscalls wide.

    Returns the canonical relpath on success, ``None`` on failure with the episode untouched.
    """
    normalized = (source_language or "").strip().lower().split("-")[0]
    if not normalized or normalized == "en":
        logger.warning(
            "translation: refusing to swap for source_language=%r — English needs no swap, and a "
            "`.en` suffix is exactly what D-44 removed",
            source_language,
        )
        return None

    canon_text = rel_transcript_path
    canon_seg = canonical_segments_relpath(rel_transcript_path)
    src_text = source_text_relpath(rel_transcript_path, normalized)
    src_seg = source_segments_relpath(rel_transcript_path, normalized)

    abs_canon_text = os.path.join(effective_output_dir, canon_text)
    abs_canon_seg = os.path.join(effective_output_dir, canon_seg)
    abs_src_text = os.path.join(effective_output_dir, src_text)
    abs_src_seg = os.path.join(effective_output_dir, src_seg)

    staged_text = f"{abs_canon_text}.swap.tmp"
    staged_seg = f"{abs_canon_seg}.swap.tmp"

    # Already swapped: the source body is at its tagged name. Re-running must not move the
    # TRANSLATION aside as though it were the source, which would hide the English behind a
    # language tag and leave the canonical path holding nothing.
    if os.path.exists(abs_src_text):
        logger.info("translation: the swap already happened for %s; nothing to do", canon_text)
        return canon_text

    done: List[tuple] = []  # (kind, args) for rollback, newest last
    try:
        _atomic_write_text(staged_text, target_text)
        done.append(("unlink", staged_text))
        _atomic_write_json(staged_seg, target_segments, indent=0)
        done.append(("unlink", staged_seg))

        os.replace(abs_canon_text, abs_src_text)
        done.append(("move_back", abs_src_text, abs_canon_text))
        if os.path.exists(abs_canon_seg):
            os.replace(abs_canon_seg, abs_src_seg)
            done.append(("move_back", abs_src_seg, abs_canon_seg))

        os.replace(staged_text, abs_canon_text)
        os.replace(staged_seg, abs_canon_seg)
    except OSError as exc:
        logger.warning(
            "translation: the swap failed for %s (%s) — rolling back, the episode is unchanged "
            "and its canonical body is still the source language",
            canon_text,
            exc,
        )
        for entry in reversed(done):
            try:
                if entry[0] == "unlink":
                    if os.path.exists(entry[1]):
                        os.remove(entry[1])
                else:
                    if os.path.exists(entry[1]):
                        os.replace(entry[1], entry[2])
            except OSError:
                logger.warning("translation: rollback step %s failed", entry, exc_info=True)
        return None

    logger.info(
        "    swapped in the translation: %s now holds the analysis language, source kept at %s "
        "(%d cues, %d units)",
        canon_text,
        src_text,
        len(target_segments) if hasattr(target_segments, "__len__") else -1,
        unit_count,
    )
    return canon_text


def write_analysis_base(
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
    seg_path = os.path.join(effective_output_dir, canonical_segments_relpath(rel_transcript_path))
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


def analysis_text_relpath(rel_transcript_path: str) -> str:
    """``<base>.adfree.txt`` — the English body in the ANALYSIS coordinate space (D-44).

    The same name an English episode's ad-free body already has, which is the point: the ad-free
    reader does not know whether it is reading a translated episode.
    """
    base, ext = os.path.splitext(rel_transcript_path)
    return f"{base}.adfree{ext or '.txt'}"


def translation_swap_happened(
    rel_transcript_path: str, effective_output_dir: str, language: str
) -> bool:
    """Whether translation COMPLETED and swapped the bodies over (D-44).

    The signal is the presence of ``<base>.<lang>.txt`` — the source body at its own name. Only the
    atomic swap creates that file, and the swap only runs once a complete English render exists, so
    its presence proves the canonical ``<base>.txt`` now holds English. Absent means the swap never
    happened and ``<base>.txt`` is still the source language.

    ABSENCE IS STILL THE GATE, which is what the previous scheme got right and worth keeping. There
    it was the absence of `.en.txt`; here it is the absence of the suffixed SOURCE. Either way no
    flag has to be remembered and no partial state exists to misread — under an atomic swap there
    is no "half translated" on disk to interpret.

    This replaced a predicate that listed three English files and checked all of them, which was
    needed because a partial write could leave some present and some not. The swap removes that
    class of bug rather than checking for it.
    """
    if not language or language.strip().lower().split("-")[0] == "en":
        # An English episode needs no swap; its canonical body is already English.
        return True
    rel = source_text_relpath(rel_transcript_path, language)
    return os.path.isfile(os.path.join(effective_output_dir, rel))


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
