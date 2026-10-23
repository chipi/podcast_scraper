"""Widen the candidate list with names the episode's own title and description present as on it.

``widen_candidates_from_metadata``: speaker detection reads the show notes before the audio exists
and sometimes misses a name it was given; the candidate list is closed, so a voice whose name is
missing from it can never be named (podcast-scraper-eval-data NAMING-RETHINK, M2: of the unnamed
nameable voices, 5 were a name in the episode's own title / description that detection missed).

Only names the text presents as PEOPLE ON THE EPISODE are added: after "with", "guest", "joined
by", "featuring", "interview with", "conversation with", or followed by "joins". A description
that is ABOUT a person ("the lawsuit against Elon Musk") adds nothing (#876: a name in the
candidate list is a name the resolver may bind). Each name must also pass the publish gate in the
feed's language, and is recorded in the trace. Off by default.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence

_NAME = r"((?:[A-Z][a-zà-öø-ÿ'’\-]+|[A-Z]\.)(?:\s+(?:[A-Z][a-zà-öø-ÿ'’\-]+|[A-Z]\.)){1,3})"
_BEFORE = re.compile(
    # the cue words in any case ("Featuring", "With"); the name itself stays capitalised
    r"\b(?i:with|guest|guests|joined by|featuring|feat\.|ft\.|interview with|conversation with|"
    r"talks? (?:to|with)|sits? down with|speaks? (?:to|with))\s+"
    r"(?:(?:Dr|Prof|Professor|Sir|Dame)\.?\s+)?" + _NAME,
)
_AFTER = re.compile(_NAME + r"\s+(?i:joins|returns|is back)\b")
#: A capitalised phrase starting with one of these is a title, a show or an organisation.
_NOT_A_FIRST_NAME = {
    "the",
    "a",
    "an",
    "our",
    "my",
    "your",
    "this",
    "that",
    "episode",
    "part",
    "season",
    "special",
    "live",
    "new",
    "us",
    "uk",
    "eu",
    "un",
}


def cue_names(*texts: Optional[str]) -> List[str]:
    """Names the texts present as people on the episode, in order, without duplicates."""
    out: List[str] = []
    seen = set()
    for text in texts:
        if not text:
            continue
        for pat in (_BEFORE, _AFTER):
            for m in pat.finditer(text):
                name = m.group(1).strip()
                key = name.lower()
                if key.split()[0] in _NOT_A_FIRST_NAME:
                    continue
                if key not in seen:
                    seen.add(key)
                    out.append(name)
    return out


def _known(name: str, existing: Sequence[str]) -> bool:
    n = name.lower().split()
    for e in existing:
        x = e.lower().split()
        if not x:
            continue
        if x == n or (x[-1] == n[-1] and x[0][:3] == n[0][:3]):
            return True
    return False


def widen(
    candidates: Sequence[str],
    title: Optional[str],
    description: Optional[str],
    *,
    language: Optional[str],
) -> List[str]:
    """The names to add: cue names of the title and description not already among ``candidates``."""
    from ....speaker_detectors.hosts import is_publishable_speaker_name
    from .roster import plausible_person_names

    added: List[str] = []
    for name in plausible_person_names(cue_names(title, description)):
        if _known(name, list(candidates) + added):
            continue
        if not is_publishable_speaker_name(name, language=language):
            continue
        added.append(name)
    return added
