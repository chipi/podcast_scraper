"""Guest-intent filtering from episode title and description."""

from __future__ import annotations

import logging
import re

from .constants import (
    INTERVIEW_INDICATOR_PATTERNS,
    INTERVIEW_TRAILING_PATTERNS,
    MENTIONED_ONLY_PATTERNS,
)

logger = logging.getLogger(__name__)

# "Our guest Dr. Sarah Chen, a professor at MIT, joins us..." — cue and name ARE adjacent, but an
# honorific sits between them, and a period is not in the gap's character class. That exclusion is
# deliberate (it stops a cue reaching across a sentence boundary into the next sentence), so the cue
# could not see the name and a real guest was dropped.
#
# Stripping the honorific before matching keeps the sentence-boundary guard AND lets the cue reach
# the name. The period is required to strip — bare "dr"/"ms" as words would otherwise be mangled.
_HONORIFIC_RE = re.compile(
    r"\b(?:dr|mr|mrs|ms|prof|sen|rep|gov|lt|sgt|capt|rev)\.\s+", re.IGNORECASE
)


def _strip_honorifics(text: str) -> str:
    return _HONORIFIC_RE.sub("", text)


def _has_interview_indicator(name: str, text: str) -> bool:
    """Is *name* introduced as a guest — cue and name TOGETHER, not merely in the same paragraph?

    The cue must sit next to the name. The old matcher was ``cue + ".*?" + name``, an unbounded gap,
    so ANY cue anywhere earlier in the text "introduced" ANY name later in it. On Hard Fork's
    "OpenAI's Big Reset" that let the cue belonging to Dr. Adam Rodman ("...returns to discuss...")
    reach backwards 200 characters and introduce **Elon Musk**, who appears only as the man suing
    OpenAI. His name was then painted onto the doctor's diarized voice.

    ``is_introduced_guest`` in this module already bounded its gap and explains why in a comment
    directly below the broken code. The bound was never applied to the description path — the one
    that actually runs.

    Also accepts a cue AFTER the name ("Dr. Adam Rodman, of Harvard Medical School, returns"), which
    is how guests are usually introduced in a feed description and which the leading-cue list could
    never see.
    """
    text_lower = _strip_honorifics(text.lower())
    name_lower = re.escape(_strip_honorifics(name.lower()))
    gap = r"[\s,'\-\w]{0," + str(_CUE_MAX_GAP) + r"}?"

    for pattern in INTERVIEW_INDICATOR_PATTERNS:
        if re.search(pattern + gap + name_lower, text_lower):
            return True
    for pattern in INTERVIEW_TRAILING_PATTERNS:
        if re.search(name_lower + pattern, text_lower):
            return True
    return False


# Max chars allowed between an interview cue and the name for a *transcript intro* guest. On a
# clean feed description the cue and name are always adjacent; on a long, noisy ASR intro a loose
# ``.*?`` match lets a cue "introduce" a merely-mentioned person 1000s of chars away.
# A cue only introduces the name it SITS NEXT TO. Unbounded, one cue introduces every name in
# the paragraph — which is how a lawsuit defendant became a podcast guest.
_CUE_MAX_GAP = 40
_INTRO_CUE_MAX_GAP = _CUE_MAX_GAP


def is_introduced_guest(name: str, intro_text: str) -> bool:
    """True when the transcript intro *introduces* ``name`` as a guest (not merely mentions them).

    ASR-grade precision, stricter than the feed-description filter: requires a First-Last name
    (drops mononym fragments — "Ezra", "Kevin" — and single-token noise — "Trump", "RN") AND an
    interview cue *immediately* before the name (drops people the episode is merely about). A wrong
    name painted on a diarized voice is worse than leaving it ``SPEAKER_NN`` (#876 discipline).
    """
    if len([t for t in name.split() if t]) < 2:
        return False
    text_lower = _strip_honorifics(intro_text.lower())
    name_lower = re.escape(_strip_honorifics(name.lower()))
    for pattern in INTERVIEW_INDICATOR_PATTERNS:
        if re.search(
            pattern + r"[\s,'\-\w]{0," + str(_INTRO_CUE_MAX_GAP) + r"}?" + name_lower, text_lower
        ):
            return True
    return False


def _has_mentioned_only_indicator(name: str, text: str) -> bool:
    """Is *name* talked ABOUT — the marker sitting next to the name, not merely in the same text?

    Same unbounded-gap bug as the interview matcher, and the same fix: a "discusses"/"about" marker
    only marks the name it is adjacent to. Left unbounded it fires on every name in the paragraph,
    so it could not discriminate either.
    """
    text_lower = _strip_honorifics(text.lower())
    name_lower = re.escape(_strip_honorifics(name.lower()))
    gap = r"[\s,'\-\w]{0," + str(_CUE_MAX_GAP) + r"}?"

    for pattern in MENTIONED_ONLY_PATTERNS:
        if re.search(pattern + gap + name_lower, text_lower):
            return True
    return False


def _is_likely_actual_guest(name: str, title: str, description: str | None) -> bool:
    """Determine if a detected person is likely an actual guest vs merely mentioned.

    A guest must LOOK like a person before any cue is even considered. ``is_introduced_guest``
    (the transcript path) has always required a First-Last name; the description path — the one
    production actually runs — never got the check, so a single-token entity sitting near a cue
    was accepted as a speaker. "Microsoft", "OpenAI", "Anthropic" and "Taiwan" all reach this
    function, and NER hands them over as PERSON often enough to matter.
    """
    if len([t for t in name.split() if t]) < 2:
        return False

    combined_text = title
    if description:
        combined_text += " " + description

    has_interview = _has_interview_indicator(name, combined_text)
    has_mentioned_only = _has_mentioned_only_indicator(name, combined_text)

    if has_interview:
        logger.debug("Name '%s' has interview indicator - likely actual guest", name)
        return True
    if has_mentioned_only:
        logger.debug("Name '%s' has mentioned-only indicator - NOT a guest", name)
        return False

    return False
