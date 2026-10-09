"""Refuse English-only NLP on non-English text (S2.14 / MULTILINGUAL_ARC §5.2).

DEFENCE IN DEPTH, AND THE FAILURE IT CATCHES IS SILENT. If translation ran correctly nothing
non-English reaches an English NLP stage: under D-44 the canonical ``<base>.txt`` holds the
ANALYSIS language once the atomic swap completes, and the completeness gate (RFC-124 §5.3)
withholds the translation and stops summary/GI/KG when it has not. But a stage run out of order,
a reprocess with the wrong flag, or a future caller that resolves its own transcript would feed
source-language text to an English model — and §5.2 measured what that produces.

(This paragraph used to cite D-34, "naming comes after translation", as the second belt. D-34 was
REVERTED on 2026-10-02 — it was never asked for — so the guarantee now comes from the swap plus
the completeness gate, which is where it belonged anyway: an ordering rule inside one stage was
always a weaker claim than "the canonical path holds the analysis language or the gate holds the
episode".)

WHAT IT PRODUCES IS NOT AN ABSENCE. Measured on the V.6a Spanish fixture against an English
control:

- ad excision found **0** patterns where the English render found **6** — so a source-language
  ad-free artifact asserts ads were removed when none could be seen;
- the sniff gate **over**-counted, 98 against 65 — a wrong number, not a missing one;
- English NER over Spanish prose produced candidate names with recall 2/2 but precision
  collapsing 67% → 18%.

A missing name is visible. A wrong one is not. That asymmetry is the whole argument for a guard
that refuses rather than a stage that copes.

WHICH OF THOSE THREE THE PER-LANGUAGE VOCABULARIES NOW ADDRESS — and which this guard is still the
only answer for (2026-10-02):

- Ad excision (0 of 6) is a PATTERN problem, and patterns are data.
  :data:`podcast_scraper.gi.filters.AD_PATTERNS_BY_LANGUAGE` now carries a row per tier-1
  language, so the vocabulary exists. It is UNMEASURED — the rows are translations, not
  observations — so this guard still refuses rather than trusting them.
- The sniff gate's over-count (98 vs 65) is likewise threshold-and-pattern shaped, and
  ``sniff_gate`` consults :func:`is_target_language` for exactly that reason.
- NER precision (67% -> 18%) is MODEL-bound and nothing here changes it. The entity model is
  English; a Spanish one is a different model, not a different constant. This is the finding that
  keeps the guard necessary, and it is why
  :data:`podcast_scraper.speaker_detectors.constants.SPEAKER_CUE_LANGUAGES` being non-empty for a
  language must not be read as "NER works for that language" — the cue vocabulary feeds the
  pattern-based detectors that sit AROUND the model, not the model.

THIS MODULE REFUSES; IT DOES NOT RAISE. A guard that raised would turn a stage-ordering mistake
into a lost episode, and the episode is recoverable while a corpus of confidently wrong claims
is not — so the caller gets an empty result and a recorded reason instead.
"""

from __future__ import annotations

import logging
from typing import Optional

from .languages import primary_language

logger = logging.getLogger(__name__)

#: Recorded on the stage that declined, so an audit can GROUP BY it.
#:
#: The NAME carries no language, per the platform rule that a function or constant must not name
#: one. The VALUE deliberately still reads ``input_not_english``: it is an audit key already
#: written into manifests, and renaming it would split every existing GROUP BY across two
#: spellings for no gain. The language it names is :data:`podcast_scraper.languages.TARGET_LANGUAGE`
#: — if that ever stops being English, this value becomes a legacy spelling and the comment is
#: what says so.
REASON_INPUT_NOT_TARGET_LANGUAGE = "input_not_english"


def is_target_language(language: Optional[str]) -> bool:
    """Whether English-only NLP may run on text in *language*.

    ``None`` PASSES. Most of the corpus predates language resolution and resolves to nothing;
    refusing those would stop the English pipeline that works today in order to protect a
    Spanish one that does not exist yet — the same reasoning
    ``_unsupported_language_skip_reason`` records for the transcription guard.
    """
    # Strip FIRST. A whitespace-only tag is an unknown language, not a non-English one, and
    # treating it as non-English would have refused a stage over a formatting artefact.
    normalized = str(language or "").strip().lower()
    if not normalized:
        return True
    return primary_language(normalized) == "en"


def refuse_unsupported_language(stage: str, language: Optional[str]) -> Optional[str]:
    """``None`` to proceed, or the reason this stage must not run on this text.

    The reason names the STAGE and the LANGUAGE, so the log line, the manifest entry and the
    metric all carry the same sentence and an operator reading any one of them learns which
    model was pointed at which language.
    """
    if is_target_language(language):
        return None
    reason = (
        f"{stage} uses English-only models, and this text resolved to {language!r}. Declining: "
        "on non-English text these models do not fail, they return confidently wrong results "
        "(§5.2 — ad patterns found 0 of 6, the entity count over-read 98 vs 65, naming "
        "precision fell 67% to 18%). A missing result is visible; a wrong one is not."
    )
    logger.warning("[#2169] %s", reason)
    return reason


#: Below this many words a transcript is too short to judge (a trailer, a stinger).
TRANSCRIPT_LANGUAGE_MIN_WORDS = 200
#: The declared language's own function words must be at least this share of the transcript.
#:
#: Measured 2026-10-07 on every transcript at hand (54 v3 fixtures in six languages, three real
#: 80,000 Hours episodes and their DGX transcripts): a CORRECTLY declared language never fell below
#: 0.051 (Portuguese, whose unique-word list is the thinnest), and a wrongly declared one never rose
#: above 0.032 (Spanish words in Italian prose). The floor sits well under the first; the second is
#: caught by the confident-detection half of the check regardless.
DECLARED_LANGUAGE_MIN_SHARE = 0.02


def transcript_language_contradiction(text: str, declared: Optional[str]) -> Optional[str]:
    """``None`` when the transcript reads as *declared*, else why it does not.

    THE FAILURE THIS CATCHES IS SILENT. The pipeline sends the declared language to ASR, and
    Whisper honours it: Spanish audio requested as ``en`` came back as correct SPANISH text with
    ``language: en`` reported, because the service echoes the request (measured on p10_e01,
    2026-10-07). So ``asr_language_mismatch`` cannot fire, and a feed whose ``<language>`` tag is
    wrong would put source-language text through every English stage, labelled English.

    The text is the only witness left, so this reads it with the function-word counts
    :mod:`ad_signatures` already uses on the corpus. It fails on either signal:

    * the text confidently reads as ANOTHER language (``detect_language``), or
    * the declared language's own function words are under :data:`DECLARED_LANGUAGE_MIN_SHARE`
      — which also covers a language with no word list, read under a tag that has one.

    It does not judge what it cannot: a declared language with no word list, or a transcript
    under :data:`TRANSCRIPT_LANGUAGE_MIN_WORDS`, passes.
    """
    from .providers.ml.diarization.ad_signatures import detect_language, FUNCTION_WORDS, words

    lang = primary_language(declared)
    if lang not in FUNCTION_WORDS:
        return None
    ws = words(text)
    if len(ws) < TRANSCRIPT_LANGUAGE_MIN_WORDS:
        return None
    detected = detect_language(ws)
    share = sum(w in FUNCTION_WORDS[lang] for w in ws) / len(ws)
    if detected in (None, lang) and share >= DECLARED_LANGUAGE_MIN_SHARE:
        return None
    reads_as = f"reads as {detected!r}" if detected else "does not read as any known language"
    return (
        f"the transcript {reads_as}, but the episode's language is {lang!r} "
        f"({share:.1%} of its {len(ws)} words are {lang!r} function words; a correctly declared "
        f"episode measures at least 5%). The feed's <language> tag is wrong, or the audio is not "
        "in the language it claims. Nothing was saved: set an operator override with the real "
        "language (#2283) and re-run."
    )
