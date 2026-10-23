"""Widen the candidate list with names the episode's own title and description present as on it.

``widen_candidates_from_metadata``: speaker detection reads the show notes before the audio exists
and sometimes misses a name it was given; the candidate list is closed, so a voice whose name is
missing from it can never be named (podcast-scraper-eval-data NAMING-RETHINK, M2: of the unnamed
nameable voices, 5 were a name in the episode's own title / description that detection missed).

Only names the text presents as PEOPLE ON THE EPISODE are added:

- after "guest", "joined by", or a conversation word followed by "with" / "to" ("a conversation
  with", "talks to", "sits down with", "catches up with");
- after a bare "with" that opens the text or a clause ("With Ana Ruiz", "Episode 12 — with Ana
  Ruiz");
- after "featuring" / "feat." / "ft." when the name closes the clause or a list ("Featuring Ana
  Ruiz.", "featuring Ana Ruiz and Bo Lee");
- before "joins" / "returns" / "is back" when nothing else follows in the clause or the object is
  the show ("Ana Ruiz joins us", "Ana Ruiz is back!", "Ana Ruiz joins the show to talk water").

Everything else is text ABOUT a person and adds nothing: "his feud with Elon Musk", "Taylor
Swift returns to Nashville", "featuring Taylor Swift songs", "the lawsuit against Elon Musk"
(#876: a name in the candidate list is a name the resolver may bind). A phrase that starts with a
question or sentence opener ("Why Elon Musk ...", "How Ana Ruiz ...") is no name. Each name must
also pass the publish gate in the feed's language, and is recorded in the trace. Off by default.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence

#: One word of a name: an inner capital ("McCormick", "O'Brien", "DeShawn") and hyphenated parts
#: ("Maclean-Jones") belong to it; a lowercase-only tail used to cut those to "Mc" / "Maclean-".
_WORD = r"[A-Z][a-zà-öø-ÿ'’]+(?:[A-Z][a-zà-öø-ÿ'’]+)?(?:-[A-Z][a-zà-öø-ÿ'’]+)*"
_NAME = r"((?:" + _WORD + r"|[A-Z]\.)(?:\s+(?:" + _WORD + r"|[A-Z]\.)){1,3})"
_HONORIFIC = r"(?:(?:Dr|Prof|Professor|Sir|Dame)\.?\s+)?"
#: Where a clause or a list ends right after a name.
_CLOSES = r"(?=\s*(?:$|[.,;:!?|)(\]—–]|\s-\s|\s(?i:and|&)\s))"
_BEFORE = re.compile(
    # the cue words in any case ("Guest", "With"); the name itself stays capitalised
    r"(?:\b(?i:guest|guests|joined by)"
    r"|\b(?i:(?:conversations?|chats?|interviews?|talks?|speaks?|sits? down|"
    r"catch(?:es|ing)? up)\s+(?:with|to))"
    # a bare "with" only where it opens the text or a clause
    r"|(?:^|(?<=[\-—–:|(,;]))\s*(?i:with))\s+" + _HONORIFIC + _NAME,
    re.MULTILINE,
)
#: "featuring X songs" is about X; "featuring X." / "featuring X and Y" presents X.
_FEATURING = re.compile(
    r"\b(?i:featuring|feat\.|ft\.)\s+" + _HONORIFIC + _NAME + _CLOSES, re.MULTILINE
)
#: "X joins us" / "X is back!" presents X; "X returns to Nashville" is about X.
_AFTER = re.compile(
    _NAME
    + r"\s+(?i:joins|returns|is back)"
    + r"(?:\s+(?i:us|me|the show|the podcast|the pod|the programme|the program))?"
    + r"(?=\s*(?:$|[.,;:!?|)(\]—–]|\s-\s|\s(?i:to talk|to discuss|for|on|this week)\b))",
    re.MULTILINE,
)
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
    # question and sentence openers: "Why Elon Musk joins ..." is no name
    "why",
    "how",
    "what",
    "when",
    "where",
    "who",
    "whose",
    "which",
    "is",
    "are",
    "was",
    "were",
    "can",
    "could",
    "does",
    "do",
    "did",
    "will",
    "would",
    "should",
    "has",
    "have",
    "inside",
    "meet",
}


def cue_names(*texts: Optional[str]) -> List[str]:
    """Names the texts present as people on the episode, in order, without duplicates."""
    out: List[str] = []
    seen = set()
    for text in texts:
        if not text:
            continue
        for pat in (_BEFORE, _FEATURING, _AFTER):
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
