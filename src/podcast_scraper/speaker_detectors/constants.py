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
    # Third-person passive — "Elena Burger is joined by a16z's Andy McCall". The list only had
    # the first-person form ("we're joined by"), so every show that writes its blurb in the
    # third person was invisible. 48 fires, 77.1% a real speaker on the ground-truth set.
    r"(?:is|are|was|were)\s+joined\s+by\s+",
    r"deep\s+dive\s+with\s+",
    # A panel: "speaks with Chris Miller, author of Chip War, AND WITH analyst Stacy Rasgon".
    # The leading cue only reaches the first name; the second guest is coordinated onto it.
    r"(?:and|along)\s+with\s+",
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

#: Trailing cues matched with a BOUNDED GAP after the name — ``NAME <role clause> CUE``.
#:
#: WHY A SECOND LIST. Every pattern above is glued to the name (``name + pattern``), which only
#: works when the cue is immediately adjacent. Real episode descriptions put the guest's job title
#: in between: "Sarah Laszlo, senior director of Visa's machine learning platform, joins the AI
#: Podcast", "Mike Pritchard, Director of Climate Simulation Research at NVIDIA, discusses". These
#: are matched as ``name + gap + cue`` instead.
#:
#: DIRECTION IS WHAT MAKES ``discusses`` SAFE HERE. The same verb appears in
#: :data:`MENTIONED_ONLY_PATTERNS`, and that is not a contradiction: mentioned-only is matched
#: cue-BEFORE-name ("discusses Mike Pritchard" — he is the topic), this is matched
#: name-BEFORE-cue ("Mike Pritchard ... discusses" — he is speaking). The two can never fire on the
#: same text in the same direction.
#:
#: MEASURED against 1,400 episodes whose roster already names a real guest, so a wrong pick is a
#: genuine error rather than a roster gap:
#:     NAME joins ...................... 46 fires, 82.6% a real speaker
#:     NAME, <role>, discusses ......... 49 fires, 75.5%
#: The looser "NAME ... discusses anywhere within 60 chars" scored 58.1% and is NOT included.
INTERVIEW_TRAILING_GAPPED_PATTERNS = [
    r",?\s*joins?\b",
    r",?\s*(?:discusses|explains|shares|unpacks|breaks\s+down)\b",
    r",?\s*(?:tells|speaks?\s+(?:with|to)|sits?\s+down\s+with)\b",
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
