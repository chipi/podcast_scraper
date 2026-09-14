"""Fit a transcript to a token budget by COUNTING, not estimating (#2050).

Every context-overflow bug in this repo has the same shape: a budget expressed in tokens, a
transcript measured in characters, and a guessed conversion rate between them. The guess is always
wrong somewhere, because the rate depends on the content:

    tests/fixtures/transcripts/v2/*   4.44 chars/token   (prod's format, simple vocabulary)
    real corpus transcripts           3.48 chars/token   (proper nouns, names, domain jargon)

Sizing at the fixture rate "proves" a budget fits when production proves it does not — measured
2026-09-13, vLLM rejected the same 106,905-char prompt 508 times in 30 days, each time at 30,721
input tokens against a 32,768 limit. Over by ONE token, deterministically, because the conversion
rate was fractionally optimistic.

A server that can tokenize ends the argument. vLLM exposes ``POST /tokenize``; ask it, clip to
what actually fits, and no constant can be wrong. The estimate survives only as the fallback for
backends with no tokenizer, and is deliberately pessimistic there.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

#: Fallback conversion rate when the backend cannot tokenize.
#:
#: 3.5 alone is NOT pessimistic enough: production exhibits 3.48 (106,905 chars / 30,721 tokens,
#: 508 rejections in 30 days), so sizing at 3.5 lands fractionally over — which is precisely the
#: one-token overflow this module exists to delete. The margin below is what makes the fallback
#: safe, so the two are always applied together via :data:`FALLBACK_EFFECTIVE_CHARS_PER_TOKEN`.
FALLBACK_CHARS_PER_TOKEN = 3.5

#: Headroom on the fallback, matching ``CONTEXT_BUDGET_SAFETY_MARGIN``. Over-estimating costs a
#: little transcript; under-estimating costs the entire stage for that episode.
FALLBACK_SAFETY_MARGIN = 0.9

#: What the fallback actually sizes at. Strictly below the rate production exhibits, by design.
FALLBACK_EFFECTIVE_CHARS_PER_TOKEN = FALLBACK_CHARS_PER_TOKEN * FALLBACK_SAFETY_MARGIN

#: How many measure-and-shrink rounds before giving up and taking the last safe clip. Convergence
#: is fast because chars->tokens is very nearly linear within one document; 4 is generous.
_MAX_ROUNDS = 4

#: Shrink slightly past the target each round so we converge from below rather than oscillating.
_UNDERSHOOT = 0.98


def fit_text_to_token_budget(
    text: str,
    max_tokens: int,
    count_tokens: Optional[Callable[[str], Optional[int]]] = None,
    *,
    label: str = "transcript",
) -> str:
    """Return the longest prefix of *text* that fits *max_tokens*, measured rather than assumed.

    ``count_tokens`` is the backend's tokenizer. When it is ``None`` — or returns ``None``, which
    is how a transient tokenizer failure is reported — this falls back to the pessimistic character
    estimate rather than sending an unmeasured prompt.

    ``max_tokens`` of 0 or less returns ``""``: the caller's reserves already exhaust the window,
    and an empty transcript is the honest answer, not a negative slice.
    """
    if max_tokens <= 0:
        return ""
    text = text or ""
    if not text:
        return text

    if count_tokens is None:
        return _estimate_clip(text, max_tokens)

    actual = count_tokens(text)
    if actual is None:
        logger.warning(
            "tokenizer unavailable while budgeting %s; falling back to the pessimistic "
            "%.2f chars/token estimate (#2050)",
            label,
            FALLBACK_EFFECTIVE_CHARS_PER_TOKEN,
        )
        return _estimate_clip(text, max_tokens)

    if actual <= max_tokens:
        return text

    clipped = text
    for _ in range(_MAX_ROUNDS):
        # chars->tokens is near-linear within a document, so scaling by the measured ratio lands
        # close on the first round; the undershoot keeps us approaching from the safe side.
        ratio = max_tokens / actual
        target = max(1, int(len(clipped) * ratio * _UNDERSHOOT))
        if target >= len(clipped):
            target = len(clipped) - 1
        clipped = clipped[:target]
        measured = count_tokens(clipped)
        if measured is None:
            return _estimate_clip(text, max_tokens)
        if measured <= max_tokens:
            logger.info(
                "%s clipped to %d chars = %d tokens (budget %d), measured not estimated (#2050)",
                label,
                len(clipped),
                measured,
                max_tokens,
            )
            return clipped
        actual = measured

    # Did not converge — take the pessimistic estimate, which is strictly shorter than anything
    # we measured as still-too-long.
    logger.warning(
        "%s did not converge on a token budget in %d rounds; using the pessimistic estimate",
        label,
        _MAX_ROUNDS,
    )
    return _estimate_clip(text, max_tokens)


def _estimate_clip(text: str, max_tokens: int) -> str:
    return text[: max(0, int(max_tokens * FALLBACK_EFFECTIVE_CHARS_PER_TOKEN))]
