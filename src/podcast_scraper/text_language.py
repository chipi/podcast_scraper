"""Which language a text reads as, from function words alone.

No model and no dependency: each language's commonest short words, kept only where unique to
that language. The language checks on transcripts (``languages_guard``,
``transcription.language_identification``) and the cross-show ad signatures all read text with
these.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, FrozenSet, List, Optional

#: Below this many words :func:`detect_language` does not guess.
DETECT_MIN_WORDS = 15

_WORD = re.compile(r"[^\W\d_]+")

# Function words per language, kept only where UNIQUE to that language: shared short words ("a",
# "de", "no", "as") read English and German as Portuguese and taught the corpus a false ad language.
_FUNCTION_WORDS_RAW: Dict[str, List[str]] = {
    "en": "the and is that you this with was are have they what but not".split(),
    "de": (
        "der die das und ist nicht für mit sie auf dem ein eine wir sich auch noch ich du".split()
    ),
    "es": "el los las por para con del una pero muy está porque nosotros también y".split(),
    "pt": "os não uma você também do da na isso muito ele com".split(),
    "fr": "le les des et une pour avec pas qui est du au je nous vous ce".split(),
    "it": "il che di della gli sono anche per è non".split(),
    "nl": "het een van dat op te voor niet zijn ook ik je".split(),
}
_ALL_FUNCTION_WORDS = Counter(w for ws in _FUNCTION_WORDS_RAW.values() for w in ws)
FUNCTION_WORDS: Dict[str, FrozenSet[str]] = {
    lang: frozenset(w for w in ws if _ALL_FUNCTION_WORDS[w] == 1)
    for lang, ws in _FUNCTION_WORDS_RAW.items()
}


def words(text: str) -> List[str]:
    """Lowercased word tokens of ``text``."""
    return _WORD.findall((text or "").lower())


def detect_language(ws: List[str]) -> Optional[str]:
    """Best-guess language from function words, or ``None`` when the text does not say clearly."""
    if len(ws) < DETECT_MIN_WORDS:
        return None
    scored = sorted(
        ((sum(w in fw for w in ws) / len(ws), lang) for lang, fw in FUNCTION_WORDS.items()),
        reverse=True,
    )
    (best, lang), (second, _) = scored[0], scored[1]
    return lang if best >= 0.06 and best >= 2 * second else None
