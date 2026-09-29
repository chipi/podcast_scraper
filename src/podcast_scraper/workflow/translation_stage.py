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

import logging
import time
from dataclasses import dataclass
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

    @property
    def ran(self) -> bool:
        """Whether the stage did work, as opposed to deciding it had none.

        ``pending`` counts as NOT having run: it means the stage looked, found work it cannot
        yet do, and said so. Reporting ``ran=True`` for it would put a stage that produced
        nothing in the same bucket as one that produced a translation.
        """
        return self.status in (STATUS_TRANSLATED, STATUS_FAILED)

    def to_metrics(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "source_language": self.source_language,
            "language_source": self.language_source,
            "reason": self.reason,
        }


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

    # S2.3/S2.4 replace this branch with the real thing.
    return TranslationOutcome(
        status=STATUS_PENDING,
        source_language=language,
        language_source=language_source,
        reason=REASON_NOT_IMPLEMENTED,
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

    outcome.duration_s = time.monotonic() - started
    if outcome.duration_s > 0:
        deadline_credit(outcome.duration_s, reason="translation stage")

    if outcome.status == STATUS_PENDING:
        logger.info(
            "    translation PENDING for a %s episode: the translator is not wired yet (%s)",
            outcome.source_language,
            outcome.reason,
        )

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
