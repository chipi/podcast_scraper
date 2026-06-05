"""Thresholds, defaults, and pattern lists for NER-based speaker detection."""

from __future__ import annotations

import re

# Default speaker names when detection fails (Issue #428: use typed placeholder, not "Guest")
DEFAULT_SPEAKER_NAMES = ["Host", "unknown_guest_1"]

_VALID_MODEL_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]+$")

MAX_MODEL_NAME_LENGTH = 100
MIN_NAME_LENGTH = 2
MIN_RAW_NAME_LENGTH = 2
MIN_SEGMENT_LENGTH = 2

DEFAULT_CONFIDENCE_SCORE = 1.0
PATTERN_BASED_CONFIDENCE_SCORE = 0.7

DESCRIPTION_SNIPPET_LENGTH = 500
DEFAULT_SAMPLE_SIZE = 5
MIN_SPEAKERS_REQUIRED = 2

INTERVIEW_INDICATOR_PATTERNS = [
    r"interview(?:ed|ing|s)?\s+(?:with\s+)?",
    r"(?:we(?:'re|'ve)?|i(?:'m|'ve)?)\s+(?:been\s+)?joined\s+by\s+",
    r"speaking\s+(?:with|to)\s+",
    r"talking\s+(?:with|to)\s+",
    r"talks?\s+(?:with|to)\s+",
    r"conversation\s+with\s+",
    r"guest(?:s)?(?:\s*:|\s+is|\s+are)?\s*",
    r"featuring\s+",
    r"(?:special\s+)?guest\s+",
    r"welcomes?\s+",
    r"sits?\s+down\s+with\s+",
    r"chats?\s+with\s+",
    r"joining\s+us\s+",
]

MENTIONED_ONLY_PATTERNS = [
    r"about\s+",
    r"on\s+\w+(?:'s)?\s+",
    r"discuss(?:es|ing|ed)?\s+",
    r"analysis\s+of\s+",
    r"according\s+to\s+",
    r"(?:he|she|they)\s+says?\s+",
    r"'s\s+(?:\w+\s+)*(?:policy|plan|speech|decision|statement)",
    r"(?:the\s+)?(?:president|ceo|senator|governor)\s+",
    r"covers?\s+",
    r"examines?\s+",
    r"looks?\s+at\s+",
    r"(?:news|story|report)\s+(?:about|on)\s+",
]
