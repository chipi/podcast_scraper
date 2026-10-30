"""Person identity: the one home for cross-episode identity decisions (RFC-131).

This module grows in steps. First it holds the name-similarity helpers that were copied across
the code base. The roster, speaker resolution, the corpus graph, name canonicalisation and the
name report each had their own edit distance or Soundex. Later steps of RFC-131 add the
same-person rule modes, the guards (ADR-167), the spelling rule and the display-name function.

Standard library only, so any layer can import it without a cycle.
"""

from __future__ import annotations

__all__ = ["edit_distance", "one_edit_apart", "soundex"]

_SOUNDEX_CODES = {
    **dict.fromkeys("BFPV", "1"),
    **dict.fromkeys("CGJKQSXZ", "2"),
    **dict.fromkeys("DT", "3"),
    **dict.fromkeys("L", "4"),
    **dict.fromkeys("MN", "5"),
    **dict.fromkeys("R", "6"),
}


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance between ``a`` and ``b``, case-insensitive."""
    a, b = a.lower(), b.lower()
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def one_edit_apart(a: str, b: str) -> bool:
    """True for strings at Levenshtein distance exactly 1 (case-insensitive), in linear time."""
    a, b = a.lower(), b.lower()
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) > len(b):
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i + (len(a) == len(b)) :] == b[i + 1 :]


def soundex(word: str) -> str:
    """Classic American Soundex of one token.

    Catches ASR substitutions that swap vowels but keep the consonant skeleton ("Roose" ->
    "Russo"). It is blind to "Newton" -> "Noon", which edit distance catches instead; the two are
    complementary, and neither alone is enough. Letters are kept by ``str.isalpha``, so a
    non-ASCII letter can be the first letter of the code; an empty string for a token without
    letters.
    """
    w = "".join(c for c in word.upper() if c.isalpha())
    if not w:
        return ""
    out, prev = w[0], _SOUNDEX_CODES.get(w[0], "")
    for ch in w[1:]:
        code = _SOUNDEX_CODES.get(ch, "")
        if code and code != prev:
            out += code
        if ch not in "HW":
            prev = code
    return (out + "000")[:4]
