"""Refuse English-only NLP on non-English text (S2.14 / MULTILINGUAL_ARC §5.2).

DEFENCE IN DEPTH, AND THE FAILURE IT CATCHES IS SILENT. If translation ran correctly nothing
non-English reaches an English NLP stage: the completeness gate (RFC-124 §5.3) withholds
the translation and stops summary/GI/KG, and D-34 puts naming after translation. But a stage
run out of
order, a reprocess with the wrong flag, or a future caller that resolves its own transcript
would feed source-language text to an English model — and §5.2 measured what that produces.

WHAT IT PRODUCES IS NOT AN ABSENCE. Measured on the V.6a Spanish fixture against an English
control:

- ad excision found **0** patterns where the English render found **6** — so a source-language
  ad-free artifact asserts ads were removed when none could be seen;
- the sniff gate **over**-counted, 98 against 65 — a wrong number, not a missing one;
- English NER over Spanish prose produced candidate names with recall 2/2 but precision
  collapsing 67% → 18%.

A missing name is visible. A wrong one is not. That asymmetry is the whole argument for a guard
that refuses rather than a stage that copes.

THIS MODULE REFUSES; IT DOES NOT RAISE. A guard that raised would turn a stage-ordering mistake
into a lost episode, and the episode is recoverable while a corpus of confidently wrong claims
is not — so the caller gets an empty result and a recorded reason instead.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

#: Recorded on the stage that declined, so an audit can GROUP BY it.
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
    return normalized.split("-")[0] == "en"


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
