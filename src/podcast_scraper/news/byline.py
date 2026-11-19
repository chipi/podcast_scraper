"""A byline string as a list of author names (E05 "Bylines"): split, order kept, organisations
removed. A name joins a person only through the joining rules (ADR-180), never here.

A comma means two things in real bylines: a list of names ("Ana Example, Bo Example and Cy
Example") or a name followed by its role and institution ("Lahra Smith, Associate Professor of
Political Science, Georgetown University", The Conversation's form). So authors are split on ";",
and within an author the comma parts are read in order up to the first one that looks like an
affiliation; only name-like parts are split further on "and" / "&", since an affiliation can
contain "and" ("Professor and Chair of …").
"""

from __future__ import annotations

import re
from typing import Collection, List, Optional

_BY = re.compile(r"^\s*by\s+", re.IGNORECASE)
_AUTHORS = re.compile(r"\s*[;|]\s*")
_COMMA = re.compile(r"\s*,\s*")
_AND = re.compile(r"\s+(?:and|&)\s+|^(?:and|&)\s+", re.IGNORECASE)
_EMAIL = re.compile(r"\S+@\S+")
#: A desk credit, not a person ("Al Jazeera Staff", "Rest of World Staff", "Reuters staff").
_DESK = re.compile(r"\b(staff|contributors|newsroom|editors|desk)\b", re.IGNORECASE)
#: Words that make a byline part a role or an institution, not a name.
_AFFILIATION = re.compile(
    r"\b(professor|lecturer|researcher|research|fellow|director|manager|candidate|associate|"
    r"assistant|reader|senior|deputy|chair|dean|scientist|editor|correspondent|reporter|"
    r"postdoctoral|doctoral|ph\.?d|catedr[aá]tico|university|universidade|universit[eé]|"
    r"college|institute|institution|school|centre|center|department|faculty|laboratory|lab|"
    r"news|policy|programmes?|programs?|studies|foundation|council|ministry|agency)\b",
    re.IGNORECASE,
)
#: A part this long is not a personal name.
_MAX_NAME_WORDS = 6
#: News agencies credited as the author ("AFP; Reuters"): an organisation, never a person.
_AGENCIES = frozenset(
    {
        "afp",
        "ap",
        "associated press",
        "reuters",
        "bloomberg",
        "pa",
        "pa media",
        "dpa",
        "efe",
        "ansa",
        "xinhua",
        "tass",
        "kyodo",
        "upi",
        "anadolu",
    }
)


def _is_affiliation(part: str) -> bool:
    words = part.split()
    if not words or len(words) > _MAX_NAME_WORDS or _AFFILIATION.search(part):
        return True
    # An acronym on its own is an institution (UNSW), never a person.
    return len(words) == 1 and len(part) >= 3 and part.isupper()


def _names_of_author(author: str) -> List[str]:
    """The names in one author block: its comma parts up to the first affiliation."""
    names: List[str] = []
    for part in _COMMA.split(author):
        part = part.strip()
        if not part:
            continue
        if _is_affiliation(part):
            break
        names.extend(p for p in _AND.split(part) if p and p.strip())
    return names


def parse(raw: Optional[str], *, organisations: Collection[str] = ()) -> List[str]:
    """The names in ``raw``, in order, without duplicates, e-mail addresses, roles, institutions,
    or a name that is one of ``organisations`` (the publisher or the source writing as itself)."""
    if not raw or not raw.strip():
        return []
    orgs = {o.casefold().strip() for o in organisations if o and o.strip()}
    names: List[str] = []
    seen = set()
    for author in _AUTHORS.split(_BY.sub("", raw.strip())):
        for part in _names_of_author(_EMAIL.sub("", author)):
            name = " ".join(part.split())
            key = name.casefold()
            if not name or key in orgs or key in _AGENCIES or key in seen or _DESK.search(name):
                continue
            seen.add(key)
            names.append(name)
    return names
