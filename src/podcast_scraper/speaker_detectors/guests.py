"""Guest-intent filtering from episode title and description."""

from __future__ import annotations

import logging
import re

from .constants import INTERVIEW_INDICATOR_PATTERNS, MENTIONED_ONLY_PATTERNS

logger = logging.getLogger(__name__)


def _has_interview_indicator(name: str, text: str) -> bool:
    """Check if name appears in an interview context."""
    text_lower = text.lower()
    name_lower = name.lower()

    for pattern in INTERVIEW_INDICATOR_PATTERNS:
        full_pattern = pattern + r".*?" + re.escape(name_lower)
        if re.search(full_pattern, text_lower, re.IGNORECASE):
            return True
    return False


# Max chars allowed between an interview cue and the name for a *transcript intro* guest. On a
# clean feed description the cue and name are always adjacent; on a long, noisy ASR intro a loose
# ``.*?`` match lets a cue "introduce" a merely-mentioned person 1000s of chars away.
_INTRO_CUE_MAX_GAP = 40


def is_introduced_guest(name: str, intro_text: str) -> bool:
    """True when the transcript intro *introduces* ``name`` as a guest (not merely mentions them).

    ASR-grade precision, stricter than the feed-description filter: requires a First-Last name
    (drops mononym fragments — "Ezra", "Kevin" — and single-token noise — "Trump", "RN") AND an
    interview cue *immediately* before the name (drops people the episode is merely about). A wrong
    name painted on a diarized voice is worse than leaving it ``SPEAKER_NN`` (#876 discipline).
    """
    if len([t for t in name.split() if t]) < 2:
        return False
    text_lower = intro_text.lower()
    name_lower = re.escape(name.lower())
    for pattern in INTERVIEW_INDICATOR_PATTERNS:
        if re.search(
            pattern + r"[\s,'\-\w]{0," + str(_INTRO_CUE_MAX_GAP) + r"}?" + name_lower, text_lower
        ):
            return True
    return False


def _has_mentioned_only_indicator(name: str, text: str) -> bool:
    """Check if name appears in a mentioned-only context (not an actual guest)."""
    text_lower = text.lower()
    name_lower = name.lower()

    for pattern in MENTIONED_ONLY_PATTERNS:
        full_pattern = pattern + r".*?" + re.escape(name_lower)
        if re.search(full_pattern, text_lower, re.IGNORECASE):
            return True
    return False


def _is_likely_actual_guest(name: str, title: str, description: str | None) -> bool:
    """Determine if a detected person is likely an actual guest vs merely mentioned."""
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
