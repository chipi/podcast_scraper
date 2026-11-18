"""A byline string as a list of author names (E05 "Bylines"): split, order kept, organisations
removed. A name joins a person only through the joining rules (ADR-180), never here.
"""

from __future__ import annotations

import re
from typing import Collection, List, Optional

_BY = re.compile(r"^\s*by\s+", re.IGNORECASE)
_SPLIT = re.compile(r"\s*(?:,|;|\||&|\band\b)\s*", re.IGNORECASE)
_EMAIL = re.compile(r"\S+@\S+")
#: A desk credit, not a person ("Al Jazeera Staff", "Rest of World Staff", "Reuters staff").
_DESK = re.compile(r"\b(staff|contributors|newsroom|editors|desk)\b", re.IGNORECASE)


def parse(raw: Optional[str], *, organisations: Collection[str] = ()) -> List[str]:
    """The names in ``raw``, in order, without duplicates, e-mail addresses, or a name that is
    one of ``organisations`` (the publisher or the source writing as itself)."""
    if not raw or not raw.strip():
        return []
    orgs = {o.casefold().strip() for o in organisations if o and o.strip()}
    names: List[str] = []
    seen = set()
    for part in _SPLIT.split(_BY.sub("", raw.strip())):
        name = " ".join(_EMAIL.sub("", part).split())
        key = name.casefold()
        if not name or key in orgs or key in seen or _DESK.search(name):
            continue
        seen.add(key)
        names.append(name)
    return names
