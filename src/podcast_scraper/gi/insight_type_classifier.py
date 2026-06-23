"""Rule-based ``insight_type`` classifier (RFC-072 §2a, RFC-097 v3.0).

Classifies a free-text insight into the schema enum
``claim | recommendation | observation | question | unknown``. Used by
:func:`gi.pipeline._resolve_insight_specs` when a provider returns bare
strings (the common case — all 9 ``generate_insights`` implementations
return ``List[str]`` today). Provider-supplied dict items with an
``insight_type`` already set pass through unchanged.

Design choices:

- **Rule-based, deterministic, fast.** No extra LLM call per insight —
  classification cost is sub-millisecond per call. Critical for the 10k+
  episode operational target where a second LLM round-trip per insight
  would dominate latency and cost.
- **Conservative bucketing.** Order: ``question`` → ``recommendation`` →
  ``claim`` → ``observation`` (default for declarative content with no
  stronger signal) → ``unknown`` (empty / whitespace only). ``observation``
  as the descriptive default is the right floor for podcast insights —
  most insights *describe* what was said in the episode.
- **No NLP dependency.** Pure regex; no spaCy load, no model file. Keeps
  the classifier shippable in cloud-thin profiles that omit ML weights.

The vocabulary is owned by the schema; this module is the rule layer
on top. An LLM-based classifier can replace this later by exposing the
same ``classify_insight_type(text) -> str`` interface — call sites are
the only thing that has to change to swap.
"""

from __future__ import annotations

import re
from typing import Literal

InsightTypeStr = Literal["claim", "recommendation", "observation", "question", "unknown"]

# First word patterns that signal an interrogative even without a "?".
# Bounded to short sentences (≤25 words) — a 100-word "What if we..." that
# pivots into a claim is a claim, not a question.
_INTERROGATIVE_STARTERS = frozenset(
    {
        "why",
        "how",
        "what",
        "when",
        "where",
        "who",
        "which",
        "should",
        "shouldnt",
        "shouldn",  # "shouldn't" first word after .lower().rstrip("?,.!'")
        "can",
        "cant",
        "could",
        "couldnt",
        "would",
        "wouldnt",
        "will",
        "wont",
        "is",
        "isnt",
        "are",
        "arent",
        "was",
        "wasnt",
        "were",
        "werent",
        "do",
        "dont",
        "does",
        "doesnt",
        "did",
        "didnt",
        "has",
        "hasnt",
        "have",
        "havent",
    }
)
_MAX_QUESTION_WORDS = 25

_RECOMMENDATION = re.compile(
    r"\b(?:"
    r"should|shouldn['’]?t|must|mustn['’]?t|ought\s+to|"
    r"recommend(?:s|ed)?|recommendation|advise(?:s|d)?|"
    r"consider(?:ing)?|suggest(?:s|ed|ion)?|"
    r"need(?:s)?\s+to|need(?:ed)?\s+to|"
    r"better\s+(?:to|off)|prefer(?:s|red)?|"
    r"avoid(?:s|ed)?|prioriti[sz]e|"
    r"try\s+to|don['’]?t\s+(?:try|use|do|forget)"
    r")\b",
    re.IGNORECASE,
)

_CLAIM = re.compile(
    r"\b(?:"
    r"argue(?:s|d)?|claim(?:s|ed)?|assert(?:s|ed)?|"
    r"believe(?:s|d)?|maintain(?:s|ed)?|insist(?:s|ed)?|"
    r"prove(?:s|d|n)?|demonstrate(?:s|d)?|establish(?:es|ed)?|"
    r"contend(?:s|ed)?|posit(?:s|ed)?|"
    r"data\s+show(?:s|ed)?|evidence\s+show(?:s|ed)?|"
    r"caus(?:es|ed)|leads?\s+to|results?\s+in"
    r")\b"
    # OR numeric-evidence signal: "+13pp", "50%", "3x", "2.5x", "$100M"
    r"|(?:[+-]?\d+(?:\.\d+)?\s*(?:%|pp|x|×|million|billion|usd|\$))",
    re.IGNORECASE,
)

_OBSERVATION = re.compile(
    r"\b(?:"
    r"note(?:s|d)?|observe(?:s|d)?|"
    r"appear(?:s|ed)?|seem(?:s|ed)?|"
    r"mention(?:s|ed)?|discuss(?:es|ed)?|"
    r"reflect(?:s|ed)?|describe(?:s|d)?|"
    r"acknowledge(?:s|d)?|recall(?:s|ed)?|share(?:s|d)?|"
    r"explain(?:s|ed)?|relate(?:s|d)?"
    r")\b",
    re.IGNORECASE,
)


def classify_insight_type(text: str) -> InsightTypeStr:
    """Classify an insight string into the RFC-072 §2a / RFC-097 v3.0 enum.

    Order of precedence:

    1. ``question`` — explicit ``?`` suffix OR short sentence starting
       with an interrogative word.
    2. ``recommendation`` — modal verbs / explicit recommendation language
       (``should``, ``recommend``, ``advise``, ``consider``, etc.).
    3. ``claim`` — strong assertion verbs (``argue``, ``claim``, ``prove``)
       or numeric-evidence patterns (``+13pp``, ``50%``, ``3x``).
    4. ``observation`` — descriptive verbs (``note``, ``observe``,
       ``mentioned``, ``described``) OR the conservative default for
       any non-empty declarative content with no stronger signal.
    5. ``unknown`` — only for empty / whitespace-only input.

    Args:
        text: Insight text (typically 1-3 sentence noun phrase summarizing
            something said in the episode).

    Returns:
        One of ``"claim"``, ``"recommendation"``, ``"observation"``,
        ``"question"``, ``"unknown"``. Guaranteed to be in
        :data:`gi.pipeline._INSIGHT_TYPE_ALLOWED`.
    """
    s = (text or "").strip()
    if not s:
        return "unknown"

    # 1. Question — most specific signal first.
    if s.endswith("?"):
        return "question"
    words = s.split()
    if words and len(words) <= _MAX_QUESTION_WORDS:
        first = words[0].lower().rstrip("?,.!'’")
        if first in _INTERROGATIVE_STARTERS:
            return "question"

    # 2. Recommendation — modal / advisory verbs.
    if _RECOMMENDATION.search(s):
        return "recommendation"

    # 3. Claim — strong assertion verbs / numeric-evidence signal.
    if _CLAIM.search(s):
        return "claim"

    # 4. Observation — descriptive verbs OR conservative default.
    if _OBSERVATION.search(s):
        return "observation"

    # 5. Default to ``observation`` rather than ``unknown`` — most podcast
    #    insights describe what was said in the episode. ``unknown`` is
    #    reserved for empty / garbage inputs above.
    return "observation"
