"""Surface ASR-garbled person names in transcript BODY text for review (#1285).

The ASR mishears a name the diarization roster resolved correctly: "Kevin Roose" is the host, but
the body says "Kevin Russo" / "Kevin Roos"; "Ryan Knutson" appears as "Ryan Knudsen". The roster
holds the spelling it settled on for every voice that SPEAKS. This proposes mapping a garbled body
mention back to that name.

CAUTION — this is a REVIEW-GATED CANDIDATE GENERATOR, not a safe automatic corpus transform. Do NOT
wire ``canonicalize_text`` into a path that writes the corpus body unattended. Two failure modes are
confirmed on the real corpus and the gate below does NOT catch them (#876 — a wrong name is worse
than a garble left alone):
  - **Distinct-person collision.** The gate only abstains when TWO *speaking* voices share a first
    name; it does nothing for "speaker A + a *mention* of a different real person B with the same
    first name and a near surname". "Kevin Rose" (Digg founder) → "Kevin Roose" (NYT), or
    "Eric Schmidt" (Google) → "Eric Schmitt" (reporter) — both edit-1/same-Soundex — would be
    silently rewritten, corrupting a correct mention of A into speaker B's name.
  - **Reverse garble.** The roster name can itself be the garble ("Duncan Macmillans", trailing-s
    ASR error) while the body is correct ("Duncan Macmillan") — this then rewrites correct → wrong.

Safe use = generate candidates + their surrounding sentence, have a human confirm each is a garble
of the SPEAKER before any body write. Automatic corpus application needs a stronger gate (abstain
when the garbled form is itself a plausible distinct real name, and pick the more-canonical of
roster-vs-body) — not yet built.

Gate applied by :func:`canonicalize_text` (candidate criteria, all required):
  1. First name matches EXACTLY (case-insensitive).
  2. The canonical person is a voice that SPEAKS in this episode (a roster person-named voice).
  3. The surname is a phonetic near-variant: same Soundex AND edit distance ≤ 3, but not identical.
  4. EXACTLY ONE speaking person has that first name (two speaking "Eric"s → abstain).
"""

from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

_MAX_SURNAME_EDIT = 3


def _soundex(s: str) -> str:
    """American Soundex of a token (surname-matching key). Empty for a non-alpha token."""
    s = re.sub(r"[^a-z]", "", s.lower())
    if not s:
        return ""
    codes = {
        "b": "1",
        "f": "1",
        "p": "1",
        "v": "1",
        "c": "2",
        "g": "2",
        "j": "2",
        "k": "2",
        "q": "2",
        "s": "2",
        "x": "2",
        "z": "2",
        "d": "3",
        "t": "3",
        "l": "4",
        "m": "5",
        "n": "5",
        "r": "6",
    }
    first = s[0].upper()
    prev = codes.get(s[0], "")
    tail = ""
    for ch in s[1:]:
        c = codes.get(ch, "")
        if c and c != prev:
            tail += c
        if ch not in "hw":
            prev = c
    return (first + tail + "000")[:4]


def _edit_distance(a: str, b: str) -> int:
    a, b = a.lower(), b.lower()
    n = len(b)
    d = list(range(n + 1))
    for i in range(1, len(a) + 1):
        prev = d[0]
        d[0] = i
        for j in range(1, n + 1):
            t = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = t
    return d[n]


def build_canonical_map(present_names: Sequence[str]) -> Dict[str, str]:
    """Lookup of ``first-name (lower) -> the single canonical name`` for the speaking people.

    ``present_names`` are the canonical names of voices that actually speak in the episode (roster
    person-named). A first name shared by two DIFFERENT speakers is dropped (ambiguous → abstain).
    """
    by_first: Dict[str, List[str]] = {}
    for n in present_names:
        parts = (n or "").split()
        if len(parts) < 2:
            continue
        by_first.setdefault(parts[0].lower(), []).append(n)
    return {f: names[0] for f, names in by_first.items() if len(set(names)) == 1}


def _canonical_for(first: str, surname: str, first_to_canonical: Dict[str, str]) -> str | None:
    canonical = first_to_canonical.get(first.lower())
    if not canonical:
        return None
    canon_surname = canonical.split()[-1]
    if canon_surname.lower() == surname.lower():
        return None  # already correct
    same_sound = _soundex(canon_surname) == _soundex(surname)
    if same_sound and _edit_distance(canon_surname, surname) <= _MAX_SURNAME_EDIT:
        return canonical
    return None


def canonicalize_text(text: str, present_names: Sequence[str]) -> Tuple[str, List[Tuple[str, str]]]:
    """Return ``(rewritten_text, [(garbled, canonical)])`` — body text with garbled speaking-person
    names snapped to their canonical spelling under the gate. ``present_names`` = the episode's
    roster person-named voices. Idempotent; leaves everything not gated untouched.
    """
    if not text or not present_names:
        return text, []
    first_to_canonical = build_canonical_map(present_names)
    if not first_to_canonical:
        return text, []
    fixes: List[Tuple[str, str]] = []
    # Anchor on the known first names so a sentence-initial capital ("So Kevin Russo") can't swallow
    # the first name into a spurious "So Kevin" pair. Longest-first avoids a short-name prefix win.
    firsts = sorted(
        {n.split()[0] for n in present_names if len(n.split()) >= 2}, key=len, reverse=True
    )
    pattern = re.compile(r"\b(" + "|".join(re.escape(f) for f in firsts) + r")\s+([A-Z][a-z]+)\b")

    def _sub(m: "re.Match[str]") -> str:
        first, surname = m.group(1), m.group(2)
        canonical = _canonical_for(first, surname, first_to_canonical)
        if canonical is None:
            return m.group(0)
        fixes.append((f"{first} {surname}", canonical))
        return canonical

    return pattern.sub(_sub, text), fixes
