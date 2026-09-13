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
# Transcript-intro window scanned for guests with the SAME NER + interview-indicator logic used on
# the feed description — the opening few minutes name the guests the feed metadata often omits.
INTRO_SNIPPET_LENGTH = 3000
DEFAULT_SAMPLE_SIZE = 5
MIN_SPEAKERS_REQUIRED = 2

INTERVIEW_INDICATOR_PATTERNS = [
    r"interview(?:ed|ing|s)?\s+(?:with\s+)?",
    # "we ARE joined by" / "I AM joined by" — the plain forms were missing, and they are the most
    # common way a description introduces a guest.
    r"(?:we(?:'re|'ve|\s+are|\s+have)?|i(?:'m|'ve|\s+am|\s+have)?)\s+(?:been\s+)?joined\s+by\s+",
    r"speaks?\s+(?:with|to)\s+",
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
    # A panel: "speaks with Chris Miller, author of Chip War, AND WITH analyst Stacy Rasgon".
    # The leading cue only reaches the first name; the second guest is coordinated onto it.
    r"(?:and|along)\s+with\s+",
]

#: Cues whose SUBJECT — the name immediately BEFORE them — is the one conducting the interview,
#: i.e. the HOST of this episode (#2061).
#:
#: ``INTERVIEW_INDICATOR_PATTERNS`` above reads the same sentence from the other side: it confirms
#: the name AFTER the cue is a guest. Nothing read the left-hand side, so the interviewer could only
#: ever be classified as a guest unless the FEED description already listed them as a show host.
#: That breaks for exactly the case that exposed it — a strand inside a main feed with its own
#: interviewer: WSJ's "The Journal." is "Hosted by Ryan Knutson and Jessica Mendoza", while its
#: "My Monday Morning" episodes say "Lane Florsheim sits down with Twiggy".
#:
#: Only TRANSITIVE forms belong here. "we are joined by X" has no left-hand person (the subject is
#: "we"), so it says nothing about who is asking the questions.
INTERVIEWER_LEAD_PATTERNS = [
    r"sits?\s+down\s+with\s+",
    r"sat\s+down\s+with\s+",
    r"interview(?:s|ed|ing)?\s+(?:with\s+)?",
    r"speaks?\s+(?:with|to)\s+",
    r"spoke\s+(?:with|to)\s+",
    r"talks?\s+(?:with|to)\s+",
    r"talked\s+(?:with|to)\s+",
    r"chats?\s+with\s+",
    r"chatted\s+with\s+",
    r"welcomes?\s+",
    r"welcomed\s+",
    r"is\s+joined\s+by\s+",
    r"was\s+joined\s+by\s+",
    r"in\s+conversation\s+with\s+",
]

# Cues that come AFTER the name ("Dr. Adam Rodman ... returns"). The list above only matches a
# cue BEFORE the name, which is why the real guest of "OpenAI's Big Reset" was invisible to the
# safe path — the description introduces him as "the A.I. researcher Dr. Adam Rodman, of Harvard
# Medical School, returns to discuss...". A guest the detector cannot see leaves a voice cluster
# free for a mentioned celebrity to claim.
INTERVIEW_TRAILING_PATTERNS = [
    r"\s*,?\s*(?:of|from|at)\s+[\w .'-]{2,40},?\s+returns?\b",
    r"\s*,?\s*returns?\s+to\s+(?:discuss|talk|explain|join)",
    r"\s*,?\s*(?:is\s+back|rejoins?|comes?\s+back)\b",
    r"\s*,?\s*joins?\s+(?:us|the\s+show|me)\b",
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
