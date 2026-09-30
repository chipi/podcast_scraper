"""The translation stage's one seam in the pipeline (RFC-124 / S2.2).

WHY ONE INSERTION POINT. Translation has to happen after the source transcript and its naming
exist and before anything reads English — which is every path that ends in
``generate_episode_metadata``: ASR, a transcript-cache hit, a direct download, a
publisher-supplied transcript, and every ``relabel_only`` / ``rediarize_only`` /
``retranscript`` cascade. They all converge there, so there is exactly one place to put this. Two
insertion points would mean two places for the decision to drift, and the drift would be silent:
an episode translated on one path and not the other looks identical on disk until an English
stage reads Spanish.

WHAT THIS SLICE DOES AND DOES NOT DO. S2.2 is the STAGE SLOT: the decision, the manifest block,
and the deadline accounting. It performs no translation — S2.3 brings the vLLM client and S2.4
the artifacts. So a non-English episode with the flag on records ``pending`` here, which is the
honest state: the pipeline knows it owes a translation and has not produced one.

EVERY EPISODE GETS A LEDGER ENTRY. THERE IS ONE PIPELINE, NOT A PER-LANGUAGE ONE.

ASR -> diarization -> naming -> translation -> summary -> GI -> KG runs for every episode in
every language. Translation is a stage in that graph; on an English episode it runs and finds
nothing to do, which is a RESULT, not an absence. So the block is written for every outcome,
``already_english`` included.

THIS REVERSES WHAT THIS MODULE DID FIRST, AND THE REVERSAL IS THE CORRECTION. The first version
withheld the block for English episodes to keep ``pipeline_composition_version`` from moving on
the 678 English episodes already on disk. Two things were wrong with that:

1. The hash exists to say WHICH PIPELINE SHAPE produced an episode. The pipeline gained a
   stage, so the hash moving is the hash working. Suppressing the record to hold a provenance
   value still is falsifying the description to avoid an operational inconvenience — and the
   inconvenience is real but one-time: a "reprocess below version X" query gets reissued once.

2. Worse, withholding it made the hash a function of the EPISODE'S LANGUAGE rather than of the
   code. Two episodes off the same commit produced different stage-graph hashes because one was
   Spanish. That is not a stable hash, it is a broken one, and it would split exactly the
   queries the hash exists to serve.

``ran`` distinguishes the two facts that remain: ``ran=False`` means the stage ran and produced
nothing (nothing to translate, flag off), ``ran=True`` means it attempted a translation. The
stage's PRESENCE says it was part of the pipeline; ``ran`` says what it did. Absence would have
said neither — which is the same measured-vs-defaulted confusion ``language_source`` exists to
prevent, one slice after Phase 0 built the machinery to prevent it.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..languages import resolve_config_language
from ..utils.timeout import deadline_credit

logger = logging.getLogger(__name__)

#: Nothing to translate, or the pipeline was not asked to.
STATUS_SKIPPED = "skipped"
#: A translation is owed and has not been produced. Not a failure — an unpaid debt.
STATUS_PENDING = "pending"
#: The English artifacts exist and are current. Written by S2.4, never by this slice.
STATUS_TRANSLATED = "translated"
#: A translation was attempted and did not produce a usable English artifact set.
STATUS_FAILED = "failed"

#: The decision said "translate this" and the work has not been attempted yet in THIS call.
#: `decide_translation` is pure, so it reports readiness; `run_translation_stage` does the work.
REASON_TRANSLATOR_READY = "ready"

#: Why a translation was skipped. A closed vocabulary so the corpus ledger can GROUP BY it.
REASON_ALREADY_ENGLISH = "already_english"
REASON_FLAG_OFF = "flag_off"
REASON_NO_LANGUAGE = "no_language"
REASON_NO_TRANSCRIPT = "no_transcript"
REASON_NOT_IMPLEMENTED = "translator_not_wired"


@dataclass
class TranslationOutcome:
    """What the stage decided, in the shape the manifest and the caller both need."""

    status: str
    source_language: Optional[str] = None
    language_source: Optional[str] = None
    reason: Optional[str] = None
    duration_s: float = 0.0
    #: Units the episode packed to. ``0`` for every skip — translation did not run.
    units: int = 0
    units_failed: int = 0
    #: True only when the complete English artifact set is on disk. THE gate condition.
    english_ready: bool = False
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def ran(self) -> bool:
        """Whether the stage did work, as opposed to deciding it had none.

        ``pending`` counts as NOT having run: it means the stage looked, found work it cannot
        yet do, and said so. Reporting ``ran=True`` for it would put a stage that produced
        nothing in the same bucket as one that produced a translation.
        """
        return self.status in (STATUS_TRANSLATED, STATUS_FAILED)

    def to_metrics(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "status": self.status,
            "source_language": self.source_language,
            "language_source": self.language_source,
            "reason": self.reason,
        }
        out.update(self.metrics)
        return out


def decide_translation(
    cfg: Any,
    *,
    feed_language: Optional[str] = None,
    transcript_relpath: Optional[str] = None,
) -> TranslationOutcome:
    """The decision, with no side effects — so it can be reasoned about and tested alone.

    Order matters here. The language is resolved BEFORE the flag is consulted, so that an
    episode's recorded ``source_language`` is the same value whether the flag is on or off. If
    the flag short-circuited first, every episode processed with translation disabled would
    record a null language, and the corpus could not later be queried for "which episodes would
    need translating" — which is the question the flag's rollout depends on.
    """
    _raw, language, language_source = resolve_config_language(cfg, feed_language=feed_language)

    if not language:
        # An honest unknown. S0.7's guard refuses to transcribe in this state, so reaching here
        # means the transcript came from somewhere else (a publisher file, a cache hit).
        return TranslationOutcome(
            status=STATUS_SKIPPED,
            source_language=None,
            language_source=language_source,
            reason=REASON_NO_LANGUAGE,
        )

    if language == "en":
        return TranslationOutcome(
            status=STATUS_SKIPPED,
            source_language=language,
            language_source=language_source,
            reason=REASON_ALREADY_ENGLISH,
        )

    if not getattr(cfg, "multilingual_ingest", False):
        # A non-English episode with the flag off. Recorded rather than ignored: this is the
        # population the flag's rollout is sized against.
        return TranslationOutcome(
            status=STATUS_SKIPPED,
            source_language=language,
            language_source=language_source,
            reason=REASON_FLAG_OFF,
        )

    if not transcript_relpath:
        return TranslationOutcome(
            status=STATUS_SKIPPED,
            source_language=language,
            language_source=language_source,
            reason=REASON_NO_TRANSCRIPT,
        )

    return TranslationOutcome(
        status=STATUS_PENDING,
        source_language=language,
        language_source=language_source,
        reason=REASON_TRANSLATOR_READY,
    )


def run_translation_stage(
    cfg: Any,
    *,
    feed_language: Optional[str] = None,
    transcript_relpath: Optional[str] = None,
    effective_output_dir: Optional[str] = None,
    episode_id: Optional[str] = None,
    feed_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> TranslationOutcome:
    """Decide, record, and credit the deadline. Never raises into metadata generation.

    THE DEADLINE CREDIT IS THE POINT OF DOING THIS HERE RATHER THAN OUTSIDE. The seam sits
    inside the block ``processing.py`` observes under the ``summarization_timeout`` key — the
    block that already reports GI's overruns under the summariser's name. Translation's wall
    time is credited back, so an overrun alert keeps meaning "summary + GI + KG were slow"
    rather than quietly becoming "this episode was translated".
    """
    started = time.monotonic()
    try:
        outcome = decide_translation(
            cfg,
            feed_language=feed_language,
            transcript_relpath=transcript_relpath,
        )
    except Exception:  # noqa: BLE001 - a stage that produces nothing must not lose the episode
        logger.warning("translation: decision failed for %s", transcript_relpath, exc_info=True)
        outcome = TranslationOutcome(status=STATUS_SKIPPED, reason=REASON_NO_LANGUAGE)

    # THE WORK HAPPENS BEFORE THE CLOCK IS READ. An earlier version measured `duration_s` and
    # credited the deadline HERE, above the translation call — so it credited the decision's
    # microseconds, the manifest recorded ~0 for an episode that took minutes, and the whole
    # translation was charged to the `summarization_timeout` block it was supposed to be
    # excluded from. Every long translated episode would have logged "METADATA GENERATION
    # OVERRAN", bumped `summarization_deadline_overruns` and filed an incident — precisely the
    # misattribution this seam exists to prevent, reintroduced by measuring in the wrong order.
    if outcome.status == STATUS_PENDING and effective_output_dir and transcript_relpath:
        try:
            outcome = _translate_episode(
                cfg,
                outcome,
                transcript_relpath=transcript_relpath,
                effective_output_dir=effective_output_dir,
            )
        except Exception as exc:  # noqa: BLE001 - see the docstring: never raise into metadata
            # "Never raises into metadata generation" was false: `create_translation_provider`
            # raises on an unknown provider id and `initialize()` raises on a served-model
            # mismatch, both of which escaped into processing.py's generic handler — episode
            # counted as an error, NO `metadata.json` written, backlog invisible. A translation
            # that cannot run is a recorded outcome, not a lost episode.
            # exc_info: without it a code bug (AttributeError, TypeError) becomes a one-line
            # `stage_error:AttributeError` in the manifest with no stack to debug from.
            logger.error(
                "translation: the stage failed for %s (%s): %s",
                transcript_relpath,
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            outcome.status = STATUS_FAILED
            outcome.reason = f"stage_error:{type(exc).__name__}"

    outcome.duration_s = time.monotonic() - started
    if outcome.duration_s > 0:
        deadline_credit(outcome.duration_s, reason="translation stage")

    if effective_output_dir and transcript_relpath:
        _record(
            outcome,
            effective_output_dir=effective_output_dir,
            transcript_relpath=transcript_relpath,
            episode_id=episode_id,
            feed_id=feed_id,
            run_id=run_id,
        )
    return outcome


def analysis_blocked_reason(
    cfg: Any,
    *,
    transcript_relpath: Optional[str],
    effective_output_dir: Optional[str],
    feed_language: Optional[str] = None,
) -> Optional[str]:
    """Why summary/GI/KG must NOT run for this episode, or ``None`` to proceed (RFC-124 §5.3).

    THE GATE, AS THE CONSUMERS SEE IT. The producer side withholds `.en.*` when a translation is
    incomplete; this is the half that stops the English stages reading the source anyway. Both
    halves are needed: without the check, a pending or failed translation falls straight through
    the resolver's precedence to `.adfree.txt` or `.txt` and runs English prompts over Spanish
    text — which §5.2 measured as confidently wrong rather than blind.

    ENGLISH EPISODES ARE NEVER BLOCKED, and the ordering here says so first: the check resolves
    the language before anything else, so the 678 English episodes in the corpus take one
    comparison and return. A gate that could stall the English pipeline in exchange for
    protecting the Spanish one would not be worth having.

    Returns a human-readable sentence, so the log line, the manifest and the metric all carry
    the same words — the same rule ``_unsupported_language_skip_reason`` follows.
    """
    from ..translation.artifacts import english_artifacts_present, load_translation_json

    _raw, language, _source = resolve_config_language(cfg, feed_language=feed_language)
    if not language or language == "en":
        return None
    if not transcript_relpath or not effective_output_dir:
        return None
    if english_artifacts_present(transcript_relpath, effective_output_dir):
        return None

    doc = load_translation_json(transcript_relpath, effective_output_dir)
    if doc is None:
        detail = "no translation was attempted"
    elif doc.failed_units:
        detail = f"{len(doc.failed_units)} of {len(doc.units)} units failed to translate"
    else:
        detail = "the English artifacts are missing"
    return (
        f"episode language is {language!r} and there is no complete English artifact set "
        f"({detail}), so summary, GI and KG were SKIPPED rather than run over "
        f"{language!r} text with English prompts (RFC-124 §5.3). Repair the translation and "
        "reprocess; the source transcript and every successful unit are on disk."
    )


def _withdraw_english_render(transcript_relpath: str, effective_output_dir: str) -> None:
    """Remove `.en.txt` + `.en.segments.json` when the analysis base could not be built.

    The gate keys on the English set being present, and a set without its ANALYSIS body is not
    complete — it would pass the gate and then serve the source language to GI and KG. So the
    render is withdrawn and the episode records `failed`, which is recoverable; the alternative
    is a corpus entry nobody can tell from a correct one.
    """
    from ..translation.artifacts import english_artifact_relpaths

    # EVERY `.en.*` file, not just the two the gate reads. A partial ad-free save leaves
    # `.en.adfree.txt` behind, and ANALYSIS readers outside the gate — the indexer, `gi/load`,
    # the processing stage — resolve that file FIRST.
    for rel in english_artifact_relpaths(transcript_relpath):
        try:
            os.remove(os.path.join(effective_output_dir, rel))
        except OSError:
            continue
    logger.warning(
        "translation: the English ad-free base could not be built for %s, so the English render "
        "was withdrawn — a set without its ANALYSIS body would pass the gate and then serve the "
        "source language to GI and KG",
        transcript_relpath,
    )


def _load_source_transcript(effective_output_dir: str, transcript_relpath: str) -> tuple[str, list]:
    """The SOURCE body and its own sidecar, by exact path. No precedence, by design.

    Paired deliberately: the sidecar is derived from the body's own base rather than resolved
    separately, which is the same rule :func:`transcript_resolution.load_transcript` follows —
    mixing a body with another variant's segments is the displacement bug this arc keeps meeting.
    """
    base, _ = os.path.splitext(transcript_relpath)
    text_path = os.path.join(effective_output_dir, transcript_relpath)
    seg_path = os.path.join(effective_output_dir, base + ".segments.json")
    try:
        with open(text_path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        return "", []
    try:
        with open(seg_path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, ValueError):
        return text, []
    return text, raw if isinstance(raw, list) else []


def _translate_episode(
    cfg: Any,
    outcome: TranslationOutcome,
    *,
    transcript_relpath: str,
    effective_output_dir: str,
) -> TranslationOutcome:
    """Pack, translate, write the ledger, and write the English render only when complete.

    THE GATE IS THE ABSENCE OF `.en.txt`, not a flag anyone has to remember to check. RFC-124
    §5.3: an episode without a complete English set skips summary, GI and KG. Because the
    resolver keys on file presence, not writing the render IS the gate — a partial render would
    be consumed as though it were whole by every stage and by the listener (D-36/D-38).
    """
    from ..providers.ml.diarization.turns import build_turns
    from ..translation.artifacts import (
        load_translation_json,
        translation_metrics,
        TranslationDocument,
        UNIT_FAILED,
        UNIT_OK,
        UnitRecord,
        unresolved_units,
        write_english_adfree,
        write_english_artifacts,
        write_translation_json,
    )
    from ..translation.factory import create_translation_provider, is_translation_configured
    from ..translation.units import pack_units

    language = outcome.source_language or ""
    if not is_translation_configured(cfg):
        outcome.status = STATUS_SKIPPED
        outcome.reason = "translator_not_configured"
        return outcome

    # TRANSLATION'S INPUT IS DEFINED, NOT RESOLVED. It reads the SOURCE body at the exact path
    # it was given, never through a purpose precedence.
    #
    # This was a bug first, and a dangerous one. `TranscriptPurpose.TIMELINE` prefers `.en.txt`
    # by D-38, so once an episode had been translated a second run resolved the ENGLISH body,
    # packed units from it, found no matching content keys (English text hashes differently
    # from Spanish), and re-translated English into English — overwriting the ledger with
    # garbage and calling the model for every unit. The resolver is for CONSUMERS choosing which
    # rendering to read; the producer of a rendering must never ask it what to read.
    source_text, source_segments = _load_source_transcript(effective_output_dir, transcript_relpath)
    if not source_text or not source_segments:
        outcome.status = STATUS_SKIPPED
        outcome.reason = REASON_NO_TRANSCRIPT
        return outcome

    # Turns are rebuilt rather than read from `turns.json`: the artifact stores no sentence text,
    # and rebuilding from the segments we just resolved guarantees the offsets index the body
    # we hold rather than a body written at some other time.
    from ..providers.ml.diarization.formatting import format_diarized_screenplay_with_offsets

    rebuilt, offset_segments = format_diarized_screenplay_with_offsets(list(source_segments))
    if rebuilt != source_text:
        outcome.status = STATUS_SKIPPED
        outcome.reason = "not_a_diarized_screenplay"
        return outcome
    turns = build_turns(offset_segments, screenplay_text=rebuilt)
    turn_dicts = [t.to_dict() for t in turns.turns]

    provider = create_translation_provider(cfg)
    units = pack_units(
        turn_dicts,
        screenplay_text=rebuilt,
        source_language=language,
        max_input_tokens=(
            getattr(provider, "MODEL_INPUT_TOKEN_LIMIT", 2048)
            if hasattr(provider, "MODEL_INPUT_TOKEN_LIMIT")
            else 2048
        ),
        count_tokens=getattr(provider, "count_tokens", None),
    )
    if not units:
        outcome.status = STATUS_SKIPPED
        outcome.reason = "nothing_to_translate"
        return outcome

    previous = load_translation_json(transcript_relpath, effective_output_dir)
    todo = unresolved_units(units, previous)
    reused = {r.content_key: r for r in (previous.units if previous else []) if r.ok}
    if previous and len(todo) < len(units):
        logger.info(
            "    translation: reusing %d of %d units from the existing ledger (content-keyed)",
            len(units) - len(todo),
            len(units),
        )

    base, _ = os.path.splitext(transcript_relpath)
    doc = TranslationDocument(
        episode_slug=os.path.basename(base),
        source_language=language,
        model=getattr(cfg, "translate_model", None),
        source={"transcript_ref": transcript_relpath, "turns": len(turn_dicts)},
    )
    # CARRY THE PREVIOUS LEDGER'S PROVENANCE FORWARD. On the D-33 path this design advertises —
    # a relabel where every unit hits the content-keyed memory — NO unit is fresh, so nothing
    # would set `doc.prompt` and the ledger was rewritten with `prompt: null`. Worse, the model
    # was taken from the CURRENT config, re-attributing cached units to a model that never saw
    # them; every claim decorated afterwards then carried a null prompt hash and a wrong model.
    # The units did not change, so neither may their provenance.
    if previous is not None and todo:
        doc.prompt = previous.prompt or doc.prompt
    elif previous is not None:
        # Nothing was re-translated at all: the ledger describes exactly the previous run's work.
        doc.prompt = previous.prompt
        doc.model = previous.model or doc.model

    todo_ids = {u.unit_id for u in todo}
    for unit in units:
        if unit.unit_id not in todo_ids:
            cached = reused.get(unit.content_key)
            if cached is not None:
                doc.units.append(
                    UnitRecord(
                        unit_id=unit.unit_id,
                        turn_id=unit.turn_id,
                        content_key=unit.content_key,
                        status=UNIT_OK,
                        alignment=cached.alignment,
                        # Re-key the cached sentences onto THIS packing's sent_ids: the text is
                        # identical by content key, but the ids are turn-ordinal and may have
                        # been renumbered by a relabel.
                        sentences=[
                            {"sent_id": s.sent_id, "en_text": e.get("en_text", "")}
                            for s, e in zip(unit.sentences, cached.sentences)
                        ],
                        attempts=0,
                        # The CACHED unit keeps the model and prompt that produced its text.
                        # Falling back to the previous document's values covers a ledger written
                        # before these fields existed.
                        model=cached.model or (previous.model if previous else None),
                        prompt_sha256=cached.prompt_sha256
                        or ((previous.prompt or {}).get("sha256") if previous else None),
                    )
                )
                continue
        result = provider.translate_unit(unit, source_language=language)
        meta = result.get("metadata") or {}
        # A freshly translated unit's prompt wins: it is the one that actually produced text in
        # THIS run, and it is what a mixed ledger should be attributed to.
        if meta.get("prompt"):
            doc.prompt = meta["prompt"]
        ok = result.get("alignment") in ("sentence", "unit") and result.get("sentences")
        doc.units.append(
            UnitRecord(
                unit_id=unit.unit_id,
                turn_id=unit.turn_id,
                content_key=unit.content_key,
                status=UNIT_OK if ok else UNIT_FAILED,
                alignment=str(result.get("alignment") or "failed"),
                sentences=list(result.get("sentences") or []),
                error=meta.get("error"),
                attempts=int(meta.get("attempts") or 0),
                # A FRESH unit is attributed to this run's model and prompt.
                model=meta.get("model") or getattr(cfg, "translate_model", None),
                prompt_sha256=(meta.get("prompt") or {}).get("sha256"),
            )
        )

    write_translation_json(doc, transcript_relpath, effective_output_dir)
    en_rel = write_english_artifacts(
        doc, turn_dicts, units, transcript_relpath, effective_output_dir
    )
    adfree_rel = None
    if en_rel:
        # THE ENGLISH AD-FREE BASE IS NOT GOVERNED BY `save_adfree_transcript`, and a review
        # found why that matters. `.en.adfree.txt` is the ONLY English body in the ANALYSIS
        # coordinate space — the space GI's and KG's `char_start` live in. With the flag off,
        # `.en.txt` + `.en.segments.json` existed, the gate passed, `translation_status` said
        # `translated`, and then ANALYSIS fell through the absent `.en.adfree.txt` and the
        # deliberately-absent source `.adfree.txt` (S2.7) to the SPANISH `.txt` — English
        # prompts over Spanish, with provenance resolved against English segments. The flag
        # governs whether the SOURCE gets an ad-free derivative; it has no business deciding
        # whether the analysis base for a translated episode exists at all.
        adfree_rel = write_english_adfree(
            transcript_relpath,
            effective_output_dir,
            extra_cue_patterns=getattr(cfg, "crosspromo_cue_patterns", None),
        )
        if adfree_rel is None:
            # No analysis base means the English set is NOT complete, whatever the render says.
            # Withdraw the render rather than let the gate pass on a partial set.
            _withdraw_english_render(transcript_relpath, effective_output_dir)
            en_rel = None
            # And say so IN the ledger. `TranslationDocument.status` is derived from unit
            # outcomes, so with zero failed units it reads `translated` — and the API's
            # `translation_status` reads exactly that field, so it would report `translated` for
            # an episode with no English at all. The flag is what stops the ledger contradicting
            # the disk.
            doc.english_withdrawn = True
            write_translation_json(doc, transcript_relpath, effective_output_dir)

    outcome.units = len(doc.units)
    outcome.units_failed = len(doc.failed_units)
    outcome.english_ready = en_rel is not None
    outcome.metrics = translation_metrics(doc, units)
    outcome.status = STATUS_TRANSLATED if en_rel else STATUS_FAILED
    if en_rel:
        outcome.reason = None
    elif adfree_rel is None and not doc.failed_units:
        # Distinct from `incomplete_translation`: every unit translated, but the ANALYSIS body
        # could not be written, so the render was withdrawn. Calling that "incomplete
        # translation" would send an operator looking for failed units that do not exist.
        outcome.reason = "adfree_base_failed"
    else:
        outcome.reason = "incomplete_translation"
    return outcome


def _record(
    outcome: TranslationOutcome,  # noqa: D401
    *,
    effective_output_dir: str,
    transcript_relpath: str,
    episode_id: Optional[str],
    feed_id: Optional[str],
    run_id: Optional[str],
) -> None:
    from . import processing_manifest as pm

    try:
        pm.update_stage(
            effective_output_dir,
            transcript_relpath,
            "translation",
            pm.stage_block(
                ran=outcome.ran,
                method_version=pm.METHOD_VERSIONS["translation"],
                duration_s=outcome.duration_s,
                # Local GPU, so a measured zero rather than an unmeasured None — and it stays
                # honest for now because no translation is performed. S2.10 measures the real
                # GPU cost and this becomes a real number.
                cost_usd=0.0,
                metrics=outcome.to_metrics(),
            ),
            episode_id=episode_id,
            feed_id=feed_id,
            run_id=run_id,
        )
    except Exception:  # noqa: BLE001 - the manifest never fails the episode
        logger.debug("translation: manifest write failed", exc_info=True)
