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
from ..translation.factory import is_translation_configured

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
REASON_ALREADY_TARGET_LANGUAGE = "already_english"
#: No translator endpoint is deployed, so there is nothing to call. A DEFECT when a non-English
#: language is enabled, not a decision — which is why it is distinct from every other reason
#: here. It replaced `flag_off`, and the difference matters: `flag_off` said "we chose not to
#: translate this", which turned out to be a choice nobody could coherently make (see the module
#: docstring).
REASON_NO_TRANSLATOR = "translator_not_configured"
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
        """The translation stage's run-summary row.

        `language_source` travels beside `source_language` because the pair is the claim: `es`
        from the feed's own tag and `es` from a profile default are the same value with very
        different standing, and only the first makes the corpus a measured one.
        """
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
            reason=REASON_ALREADY_TARGET_LANGUAGE,
        )

    if not is_translation_configured(cfg):
        # Reaching here means the episode is non-English AND its language is `enabled: true`, so
        # an operator has already approved ingesting it. Having no translator deployed at that
        # point is a misconfiguration, not a policy: the episode will get a transcript and then
        # be blocked out of summary/GI/KG by the §5.3 completeness gate. Recorded so the ledger
        # can GROUP BY it and an operator sees a cause rather than a silent gap.
        return TranslationOutcome(
            status=STATUS_SKIPPED,
            source_language=language,
            language_source=language_source,
            reason=REASON_NO_TRANSLATOR,
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
    episode_title: Optional[str] = None,
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
                episode_title=episode_title,
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
    from ..translation.artifacts import load_translation_json, translation_swap_happened

    _raw, language, _source = resolve_config_language(cfg, feed_language=feed_language)
    if not language or language == "en":
        return None
    if not transcript_relpath or not effective_output_dir:
        return None
    doc = load_translation_json(transcript_relpath, effective_output_dir)

    # TWO CONDITIONS: the swap happened AND the ledger does not say the set is unusable.
    #
    # The swap alone is not enough, and the gap was reachable. `write_analysis_base` returns None
    # only when the English body or its segments are unreadable, empty, or not a list — NOT when
    # there are simply no ads to cut, which yields an identity base. So a None there means the
    # render is broken, the stage records `failed`, and the gate — asking only "did the swap
    # happen?" — said PROCEED anyway. Summary, GI and KG then ran over the English body with its
    # ads still in it while the ledger and `translation_status` both reported failure.
    #
    # THE LEDGER, NOT THE AD-FREE FILE. An earlier attempt made this require `<base>.adfree.txt`
    # to exist and was reverted: a NATIVE ENGLISH episode with `save_adfree_transcript` off has no
    # such file either and is perfectly healthy, so requiring it would refuse a translated episode
    # in a state an English one runs in — the language branch D-39 forbids. Reading the ledger adds
    # no branch: this function already returns for English before reaching here, and an English
    # episode has no `translation.json` at all, so it can never be caught by this.
    #
    # (`english_withdrawn` is a legacy key name from when the response was to delete the `.en.*`
    # files. Nothing is deleted now; it means "every unit translated but the set is not
    # consumable". It keeps its name because it is already persisted in every ledger on disk.)
    if translation_swap_happened(transcript_relpath, effective_output_dir, language):
        if doc is None or not doc.english_withdrawn:
            return None
        return (
            f"episode language is {language!r} and the translation completed, but the ledger "
            "records the English set as NOT CONSUMABLE (the ANALYSIS body could not be built from "
            "it, which means the render or its segments are unreadable). Summary, GI and KG were "
            "SKIPPED rather than run over a body whose ad-free coordinate space does not exist "
            "(RFC-124 §5.3). Repair the translation and reprocess; the source transcript and every "
            "successful unit are on disk."
        )

    if doc is None:
        detail = "no translation was attempted"
    elif doc.failed_units:
        detail = f"{len(doc.failed_units)} of {len(doc.units)} units failed to translate"
    else:
        detail = "the swap did not happen, so the canonical body is still the source language"
    return (
        f"episode language is {language!r} and there is no complete English artifact set "
        f"({detail}), so summary, GI and KG were SKIPPED rather than run over "
        f"{language!r} text with English prompts (RFC-124 §5.3). Repair the translation and "
        "reprocess; the source transcript and every successful unit are on disk."
    )


def _load_source_transcript(
    effective_output_dir: str, transcript_relpath: str, language: Optional[str] = None
) -> tuple[str, list]:
    """The SOURCE body and its own sidecar, by exact path. No precedence, by design.

    Paired deliberately: the sidecar is derived from the body's own base rather than resolved
    separately, which is the same rule :func:`transcript_resolution.load_transcript` follows —
    mixing a body with another variant's segments is the displacement bug this arc keeps meeting.

    IT FOLLOWS THE SWAP (D-44). Once translation has completed, the canonical ``<base>.txt`` holds
    the TRANSLATION and the source lives at ``<base>.<lang>.txt`` — so reading the canonical path on
    a second run would feed the model its own output. The integration test named
    ``TestTranslationReadsTheSourceNotItsOwnOutput`` caught exactly that: every unit missed the
    ledger's content-keyed cache (the text had changed from Spanish to English), so a resume
    re-translated all four units and the ledger's provenance would have recorded English as the
    source of its own translation.

    So: prefer the tagged source when it exists, else the canonical path. Both are exact lookups —
    this never searches a precedence list, because "which body was I spoken in" has one answer.
    """
    base, _ = os.path.splitext(transcript_relpath)
    rel = transcript_relpath
    normalized = (language or "").strip().lower().split("-")[0]
    if normalized and normalized != "en":
        from ..translation.artifacts import source_text_relpath

        tagged = source_text_relpath(transcript_relpath, normalized)
        if os.path.isfile(os.path.join(effective_output_dir, tagged)):
            rel = tagged
            base, _ = os.path.splitext(tagged)
    text_path = os.path.join(effective_output_dir, rel)
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


#: How many of the episode's opening sentences ride along as context for the title.
#: Two is enough to establish the subject and keeps the unit small — a title unit is sent per
#: episode and must stay cheap.
_TITLE_CONTEXT_SENTENCES = 2


def _translate_title(
    cfg: Any,
    provider: Any,
    episode_title: Optional[str],
    source_language: str,
    context_sentences: Optional[list] = None,
) -> Optional[str]:
    """The EPISODE title in English, or ``None`` (S2.4's title decision).

    THE SHOW NAME IS NOT TRANSLATED and is not passed here. ADR-157 measured the model turning
    `Sesiones de Sendero` into `Trail Sessions`; a show's name is its identity, so renaming it
    would change what the feed IS on every surface, in search, and in every listener's saved
    library. An episode title describes that episode's content, which is what translation is
    for, and §5.4 C-6 needs it in English because the roster reads it for host/guest context and
    NER candidate discovery.

    BEST EFFORT, ALWAYS. A title is one short unit and it is not part of the completeness gate:
    a failure costs the roster some context and the naming stage falls back to the source title.
    It must never cost the episode its translation, which is why every exception is swallowed
    here rather than propagated into the unit loop's accounting.

    THE TITLE TRAVELS WITH CONTEXT (2026-10-01). It used to be sent as a unit holding ONE
    sentence, which is this module's own contract read backwards: "the unit is the translation
    CONTEXT; the sentence is the alignment atom". A one-sentence unit has no context by
    construction, and a title is the shortest, most ambiguous string in the episode.

    Measured: the German fixture's `Wege Bauen, Die Bleiben` came back as "Building bridges,
    creating connections that last" — both nouns invented — while the same conversation in es,
    it, fr and pt all produced the correct "Building Trails That Last." `Wege bauen` in isolation
    is genuinely ambiguous between literal path-building and the English idiom; the body never
    had the problem because its units carry neighbouring sentences.

    So the episode's opening sentences ride along in the same unit and only the FIRST translated
    sentence is taken. No extra request, no new provider API — the mechanism was already there
    and the title simply was not using it.
    """
    title = (episode_title or "").strip()
    if not title:
        return None
    try:
        from ..translation.units import TranslationUnit, UnitSentence

        # The title FIRST, so `sentences[0]` is the answer; the context after it, purely to tell
        # the model what the episode is about. Only the first result is read — the rest are
        # translated and discarded, which is the cost of the fix and is one short unit's worth.
        sents = [UnitSentence(sent_id="title.s01", text=title, char_start=0, char_end=0)]
        for i, ctx in enumerate(context_sentences or [], start=2):
            ctx_text = str(ctx or "").strip()
            if ctx_text:
                sents.append(
                    UnitSentence(sent_id=f"title.s{i:02d}", text=ctx_text, char_start=0, char_end=0)
                )
        unit = TranslationUnit(
            unit_id="title",
            turn_id="title",
            speaker_label="",
            # char_start/char_end are 0 because a title has no span in the screenplay — it is
            # not part of the transcript body at all. Nothing resolves a span against this unit:
            # it never enters `doc.units`, so `resolve_units_for_span` cannot see it and no
            # claim can be provenanced to it.
            sentences=sents,
        )
        result = provider.translate_unit(unit, source_language=source_language)
        sentences = result.get("sentences") or []
        if result.get("alignment") in ("sentence", "unit") and sentences:
            out = str(sentences[0].get("en_text") or "").strip()
            if out:
                logger.info("    translated the episode title: %r -> %r", title, out)
                return out
        logger.info("    the episode title did not translate; naming will read the source title")
    except Exception:  # noqa: BLE001 - see the docstring
        logger.warning(
            "    translating the episode title failed; naming will read the source title",
            exc_info=True,
        )
    return None


def _translate_episode(
    cfg: Any,
    outcome: TranslationOutcome,
    *,
    transcript_relpath: str,
    effective_output_dir: str,
    episode_title: Optional[str] = None,
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
        write_analysis_base,
        write_translated_artifacts,
        write_translation_json,
    )
    from ..translation.factory import create_translation_provider
    from ..translation.units import pack_units

    language = outcome.source_language or ""
    if not is_translation_configured(cfg):
        # Belt and braces: `decide_translation` already returns SKIPPED for this, so reaching
        # here means a caller drove the work directly. The CONSTANT, not the bare literal that
        # used to sit here — the reason vocabulary is documented as closed, and a literal
        # outside it is how a ledger ends up with two spellings of one state.
        outcome.status = STATUS_SKIPPED
        outcome.reason = REASON_NO_TRANSLATOR
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
    source_text, source_segments = _load_source_transcript(
        effective_output_dir, transcript_relpath, language
    )
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

    # S2.4's TITLE DECISION: the EPISODE title is translated, the SHOW name is not.
    #
    # The show's name is its identity. ADR-157 measured the model turning `Sesiones de Sendero`
    # into `Trail Sessions`, and renaming a show would change what the feed IS on every surface,
    # in search, and in every listener's saved library. An episode title is a description of that
    # episode's content, which is what translation is for — and §5.4 C-6 needs it in English,
    # because the roster reads the title for host/guest context and for NER candidate discovery.
    # Passing a Spanish title to an English NER points the §5.2 hazard (recall 2/2, precision
    # 67% -> 18%) at the one input naming trusts most.
    #
    # Best-effort by design: a failed title costs the roster some context, and the naming stage
    # falls back to the source title. It must never cost the episode its translation.
    # The episode's opening sentences go with the title — see `_translate_title`. Taken from the
    # SOURCE units (the model translates source -> English in one pass), and from the first unit
    # that carries any, so a leading backchannel ("Sí.") does not become the whole context.
    _title_context: list[str] = []
    for _u in units:
        for _s in getattr(_u, "sentences", []) or []:
            _text = str(getattr(_s, "text", "") or "").strip()
            if len(_text) >= 20:
                _title_context.append(_text)
            if len(_title_context) >= _TITLE_CONTEXT_SENTENCES:
                break
        if len(_title_context) >= _TITLE_CONTEXT_SENTENCES:
            break
    doc.title_en = _translate_title(cfg, provider, episode_title, language, _title_context)

    write_translation_json(doc, transcript_relpath, effective_output_dir)
    en_rel = write_translated_artifacts(
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
        adfree_rel = write_analysis_base(
            transcript_relpath,
            effective_output_dir,
            extra_cue_patterns=getattr(cfg, "crosspromo_cue_patterns", None),
        )
        if adfree_rel is None:
            # NOTHING IS DELETED HERE, and that is the D-44 change. This used to call
            # `_withdraw_english_render` to remove the `.en.*` files so ANALYSIS readers would
            # fall back to the source. Under the atomic swap there is no partial set to withdraw:
            # the swap either happened (canonical body holds the translation, source kept at its
            # tagged name) or it did not. Deleting the canonical body now would leave the episode
            # with NO body at all, which is worse than the state it is in.
            #
            # THE RECORD IS WHAT CHANGES, AND THE GATE READS THE RECORD. The flag set below is
            # what `analysis_blocked_reason` consults, so summary/GI/KG are skipped for this
            # episode.
            #
            # THIS COMMENT HAS BEEN WRONG TWICE, in opposite directions, which is why the
            # mechanism is spelled out rather than summarised. It first claimed the gate "checks
            # for this exact file" — it never did. It was then corrected to "NOTHING GATES ON IT",
            # and that became false on 2026-10-03 when the gate was taught to read
            # `english_withdrawn`; it sat here asserting that summary/GI/KG still ran over the
            # ad-laden body while they no longer did.
            #
            # WHAT THE GATE DOES NOT DO, because the distinction is the whole reason this took two
            # tries: it does not look for `<base>.adfree.txt`. Requiring that file was tried and
            # reverted — a NATIVE ENGLISH episode with `save_adfree_transcript` off has no such
            # file either and is perfectly healthy, so requiring it would refuse a translated
            # episode in a state an English one runs in, which is the language branch D-39 forbids.
            # The gate reads the LEDGER, which records a decision this stage made, and an English
            # episode has no `translation.json` to be caught by.
            logger.warning(
                "translation: %s translated and swapped, but the ANALYSIS base could not be "
                "built — recording `failed` so the gate blocks summary/GI/KG rather than letting "
                "them index a body with no ad-free coordinate space",
                transcript_relpath,
            )
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
