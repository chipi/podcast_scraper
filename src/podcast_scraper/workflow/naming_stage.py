"""Naming, run AFTER translation, on the English render (D-34 / S2.6).

WHY THE ORDER HAD TO CHANGE. Every cue the roster reads is English — the self-intro patterns
(``I'm X``, ``Welcome to … I'm X``), the interview cues ``corroborate_guests`` requires, and
``en_core_web_sm`` behind the NER. On a non-English transcript they do not find nothing; §5.2
measured them finding the WRONG things, recall holding at 2/2 while precision fell 67% to 18%.

And the consequence of leaving the order alone is not "slightly worse names". ``corroborate_guests``
needs an English interview cue in the title or description, so on a Spanish feed every guest is
rejected and only bare metadata names survive. Guests carry the positions, so a translated episode
yields **no position-bearing insights at all** — which is D-34's own stated reasoning.

THE SEQUENCE, and what each step must not do:

1. diarize → anonymous ``SPEAKER_NN`` labels. Nothing reads words yet.
2. write ``.txt`` (anonymous), ``.segments.json``, ``.anon.txt`` (D-40), turns.
3. translate → ``.en.txt`` + ``.en.segments.json``, each English cue carrying the SOURCE turn's
   label verbatim (D-24) — which at this point is still the anonymous voice id, and that is what
   makes this whole stage possible.
4. **here**: resolve the voices by reading the ENGLISH text, then re-render both bodies named.

WHAT IS DELIBERATELY NOT RE-DERIVED. The diarization is reconstructed from the SOURCE
``.segments.json``'s ``speaker`` field and its real audio times, never from the English cues: the
roster weighs talk time, and English cue durations are the source sentences' durations only
because the renderer copies them. Taking timing from the audio side keeps "who talked longest" a
fact about the recording.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..languages import transcription_language
from ..languages_guard import is_english_text_language

logger = logging.getLogger(__name__)

#: Naming ran and renamed at least one voice.
STATUS_NAMED = "named"
#: Naming ran and resolved nobody. A result, not a failure — the episode keeps anonymous labels.
STATUS_UNRESOLVED = "unresolved"
#: Naming was not deferred for this episode, so it already happened inside diarization.
STATUS_NOT_DEFERRED = "not_deferred"
#: Deferred, but the English render this stage needs is not there.
STATUS_NO_ENGLISH = "no_english_render"
#: Deferred, but the source segments carry no diarized voice to name.
STATUS_NO_VOICES = "no_diarized_voices"


@dataclass
class NamingOutcome:
    """What the stage did, in the shape the manifest and the caller both need."""

    status: str
    renamed: Dict[str, str] = field(default_factory=dict)
    voices: int = 0
    duration_s: Optional[float] = None
    reason: str = ""

    @property
    def ran(self) -> bool:
        return self.status in (STATUS_NAMED, STATUS_UNRESOLVED)

    def to_metrics(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "voices": self.voices,
            "named": len(self.renamed),
            "reason": self.reason,
        }


def naming_is_deferred(cfg: Any) -> bool:
    """Whether THIS episode's naming waits for a translation (D-34).

    True only when the episode is non-English **and** a translator is deployed. Both halves
    matter, and for different reasons:

    * English episode — there is nothing to wait for, and deferring would move the 678-episode
      corpus onto a new path to buy nothing.
    * non-English with no translator — the English render will never arrive, so deferring would
      leave the episode permanently anonymous. Naming in place is worse (§5.2) but recoverable;
      never naming at all is not, and the §5.3 gate already stops the analysis stages from
      trusting that transcript.

    This is the same predicate `decide_translation` uses to reach `pending`, and it must stay
    that way: if the two disagree, an episode either gets named twice or never.
    """
    from ..translation.factory import is_translation_configured

    language = transcription_language(cfg)
    if is_english_text_language(language):
        return False
    return bool(is_translation_configured(cfg))


def _load_json(path: str) -> Optional[Any]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def diarization_from_segments(segments: List[Dict[str, Any]]) -> Optional[Any]:
    """Rebuild a ``DiarizationResult`` from a saved ``.segments.json``.

    From the SOURCE segments, whose ``start``/``end`` are real audio times. The roster weighs
    talk time, and the English cues' durations are the source sentences' durations only because
    the renderer copies them across — so deriving timing from the English side would make "who
    talked longest" a fact about the translation.

    ``None`` when no segment carries a ``speaker``: nothing was diarized, so there is no voice to
    name.
    """
    from ..providers.ml.diarization.base import DiarizationResult, DiarizationSegment

    turns: List[Any] = []
    voices: set = set()
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        voice = seg.get("speaker")
        if not voice:
            continue
        voices.add(str(voice))
        turns.append(
            DiarizationSegment(
                start=float(seg.get("start") or 0.0),
                end=float(seg.get("end") or 0.0),
                speaker=str(voice),
            )
        )
    if not turns:
        return None
    return DiarizationResult(segments=turns, num_speakers=len(voices), model_name="from-segments")


def align_english_to_voices(
    english_segments: List[Dict[str, Any]],
) -> List[Tuple[Dict[str, Any], str]]:
    """``[(english_segment, voice_id)]`` — the English text, attributed to the diarized voices.

    The English cue's ``speaker_label`` IS the voice id, because the render carries the source
    turn's label verbatim (D-24) and at translation time naming has not run yet. That is the
    whole hinge of D-34: without it there would be no way back from an English sentence to the
    voice that spoke it.

    A cue whose label is not an anonymous voice id is skipped rather than trusted. That happens
    when this stage runs twice — the second pass would see NAMES in the label position — and
    silently aligning those would attribute an already-named voice to a voice called "Dana
    Reyes".
    """
    out: List[Tuple[Dict[str, Any], str]] = []
    for seg in english_segments:
        if not isinstance(seg, dict):
            continue
        label = str(seg.get("speaker_label") or "")
        if not label.startswith("SPEAKER_"):
            continue
        out.append((seg, label))
    return out


def relabel_segments(
    segments: List[Dict[str, Any]], names: Dict[str, str]
) -> Tuple[List[Dict[str, Any]], int]:
    """Apply ``{voice_id: name}`` to a segment list. Returns ``(segments, changed_count)``.

    Keyed on the segment's own ``speaker`` where it has one (the source body) and on
    ``speaker_label`` otherwise (the English cues, which carry only the label). ``speaker`` is
    never rewritten — it is the frozen voice id every later stage and the ``.anon.txt`` render
    depend on.
    """
    changed = 0
    out: List[Dict[str, Any]] = []
    for seg in segments:
        if not isinstance(seg, dict):
            out.append(seg)
            continue
        voice = str(seg.get("speaker") or seg.get("speaker_label") or "")
        name = names.get(voice)
        row = dict(seg)
        if name and name != seg.get("speaker_label"):
            row["speaker_label"] = name
            changed += 1
        out.append(row)
    return out, changed


def _write_text(path: str, body: str) -> bool:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(body)
        return True
    except OSError as exc:
        logger.warning("naming: could not write %s: %s", path, exc)
        return False


def _write_json(path: str, payload: Any) -> bool:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        return True
    except OSError as exc:
        logger.warning("naming: could not write %s: %s", path, exc)
        return False


def run_naming_stage(
    cfg: Any,
    *,
    transcript_relpath: str,
    effective_output_dir: str,
    detected_speaker_names: Optional[List[str]] = None,
    metadata_named: Optional[List[str]] = None,
    feed_hosts: Optional[List[str]] = None,
    episode_title: Optional[str] = None,
    episode_description: Optional[str] = None,
    episode_id: Optional[str] = None,
    feed_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> NamingOutcome:
    """Resolve the voices from the ENGLISH render and re-render both transcripts named (D-34).

    Called at the metadata seam, immediately after the translation stage. A no-op for every
    episode whose naming was not deferred, which is every English one — so the cost on the
    corpus that works today is one language comparison.

    NEVER RAISES. A naming failure costs an episode anonymous labels, which a relabel recovers;
    raising here would land in `processing.py`'s generic handler, count the episode as an error
    and write no metadata at all. That asymmetry is the same one #876 records for naming a voice
    wrongly versus not naming it.
    """
    import time

    from ..providers.ml.diarization.formatting import format_diarized_screenplay_with_offsets
    from ..providers.ml.diarization.pipeline import resolve_names_on_result
    from .transcript_resolution import (
        _segments_relpath,
        english_transcript_relpath,
    )

    started = time.monotonic()

    if not naming_is_deferred(cfg):
        return NamingOutcome(
            status=STATUS_NOT_DEFERRED,
            reason="naming already ran inside diarization for this episode",
        )
    if not transcript_relpath:
        return NamingOutcome(status=STATUS_NO_ENGLISH, reason="no transcript path")

    root = effective_output_dir
    src_segments_rel = _segments_relpath(transcript_relpath)
    en_rel = english_transcript_relpath(transcript_relpath)
    en_segments_rel = _segments_relpath(en_rel)

    src_segments = _load_json(os.path.join(root, src_segments_rel))
    en_segments = _load_json(os.path.join(root, en_segments_rel))
    if not isinstance(src_segments, list) or not isinstance(en_segments, list):
        # The §5.3 completeness gate already stops the analysis stages here; naming simply has
        # nothing to read, and the episode keeps its anonymous labels.
        return NamingOutcome(
            status=STATUS_NO_ENGLISH,
            reason="the English render or its segments sidecar is absent",
            duration_s=time.monotonic() - started,
        )

    diarization = diarization_from_segments(src_segments)
    if diarization is None:
        return NamingOutcome(
            status=STATUS_NO_VOICES,
            reason="the source segments carry no diarized `speaker`, so there is no voice to name",
            duration_s=time.monotonic() - started,
        )

    aligned = align_english_to_voices(en_segments)
    if not aligned:
        return NamingOutcome(
            status=STATUS_NO_VOICES,
            voices=diarization.num_speakers,
            reason=(
                "no English cue carries an anonymous voice id — naming has already run on this "
                "render, or the labels were rewritten by something else"
            ),
            duration_s=time.monotonic() - started,
        )

    english_body = "\n".join(str(seg.get("text") or "") for seg, _v in aligned).strip()

    # S2.4's title decision: read the TRANSLATED episode title when the translation stage
    # produced one. The roster reads the title for host/guest context and for NER candidate
    # discovery, so handing it the SOURCE title while every per-voice sample is English is the
    # same silent mismatch `naming_text` exists to prevent one level down — both are strings, and
    # the roster would simply resolve fewer voices.
    #
    # Falls back to the source title rather than to nothing: a missing title costs the roster its
    # role context entirely, which is worse than a title in the wrong language.
    #
    # The SHOW name is never translated and is not read here (ADR-157: the model renames it).
    ledger = _load_json(
        os.path.join(root, os.path.splitext(transcript_relpath)[0] + ".translation.json")
    )
    if isinstance(ledger, dict):
        translated_title = str(ledger.get("title_en") or "").strip()
        if translated_title:
            episode_title = translated_title
    try:
        resolved = resolve_names_on_result(
            {"segments": [dict(seg) for seg, _v in aligned], "text": english_body},
            cfg,
            diarization,
            aligned,
            detected_speaker_names,
            metadata_named=metadata_named,
            feed_hosts=feed_hosts,
            episode_title=episode_title,
            episode_description=episode_description,
            detection_ran=True,
            naming_text=english_body,
        )
    except Exception:  # noqa: BLE001 - see the docstring: this must never cost the episode
        logger.warning("naming: resolution failed; labels stay anonymous", exc_info=True)
        return NamingOutcome(
            status=STATUS_UNRESOLVED,
            voices=diarization.num_speakers,
            reason="stage_error",
            duration_s=time.monotonic() - started,
        )

    # {voice_id: resolved name}, keeping only the voices that actually got a NAME. A voice the
    # roster left as `SPEAKER_NN` must not be "renamed" to itself, or every episode would report
    # a rename it did not make.
    names: Dict[str, str] = {}
    for row in resolved.get("segments") or []:
        if not isinstance(row, dict):
            continue
        voice = str(row.get("speaker") or "")
        label = str(row.get("speaker_label") or "")
        if voice and label and label != voice and not label.startswith("SPEAKER_"):
            names[voice] = label

    if not names:
        return NamingOutcome(
            status=STATUS_UNRESOLVED,
            voices=diarization.num_speakers,
            reason="the roster named no voice from the English text",
            duration_s=time.monotonic() - started,
        )

    # RE-RENDER BOTH BODIES. The source transcript keeps its own text and gains the names; the
    # English render keeps its text and gains the same names. Neither is re-translated and
    # neither is re-diarized — only the label column moves, which is why D-33's content-keyed
    # memory makes this cost zero GPU.
    new_src, _ = relabel_segments(src_segments, names)
    new_en, _ = relabel_segments(en_segments, names)

    src_text, src_offsets = format_diarized_screenplay_with_offsets(new_src)
    en_text, en_offsets = format_diarized_screenplay_with_offsets(new_en)

    wrote = (
        _write_text(os.path.join(root, transcript_relpath), src_text)
        and _write_json(os.path.join(root, src_segments_rel), src_offsets)
        and _write_text(os.path.join(root, en_rel), en_text)
        and _write_json(os.path.join(root, en_segments_rel), en_offsets)
    )
    if not wrote:
        # Partially written is the dangerous state, so say so loudly. The offsets in a body and
        # its sidecar must describe the same text or every GI quote in this episode is wrong.
        logger.error(
            "naming: re-render was incomplete for %s — the transcript bodies and their segment "
            "sidecars may disagree on char offsets. Re-run a relabel for this episode.",
            transcript_relpath,
        )

    _record(
        NamingOutcome(
            status=STATUS_NAMED,
            renamed=names,
            voices=diarization.num_speakers,
            duration_s=time.monotonic() - started,
            reason="resolved from the English render",
        ),
        transcript_relpath=transcript_relpath,
        effective_output_dir=root,
        episode_id=episode_id,
        feed_id=feed_id,
        run_id=run_id,
    )
    logger.info(
        "    naming (after translation): named %d of %d voice(s) — %s",
        len(names),
        diarization.num_speakers,
        ", ".join(f"{v}->{n}" for v, n in sorted(names.items())),
    )
    return NamingOutcome(
        status=STATUS_NAMED,
        renamed=names,
        voices=diarization.num_speakers,
        duration_s=time.monotonic() - started,
        reason="resolved from the English render",
    )


def _record(
    outcome: NamingOutcome,
    *,
    transcript_relpath: str,
    effective_output_dir: str,
    episode_id: Optional[str],
    feed_id: Optional[str],
    run_id: Optional[str],
) -> None:
    from . import processing_manifest as pm

    try:
        pm.update_stage(
            effective_output_dir,
            transcript_relpath,
            "naming",
            pm.stage_block(
                ran=outcome.ran,
                method_version=pm.METHOD_VERSIONS["naming"],
                duration_s=outcome.duration_s,
                cost_usd=0.0,
                metrics=outcome.to_metrics(),
            ),
            episode_id=episode_id,
            feed_id=feed_id,
            run_id=run_id,
        )
    except Exception:  # noqa: BLE001 - the manifest never fails the episode
        logger.debug("naming: manifest write failed", exc_info=True)
