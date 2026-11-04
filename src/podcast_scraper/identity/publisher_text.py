"""The publisher's own text as identity evidence (record ``l_publisher_verified_mode``).

The ledger's "publisher" names are the pipeline's extracted names: the speakers diagnostics'
``known_hosts``, ``metadata_named`` and ``detected_guests``. On snapshot-prod-20261103, 1,063 of
4,307 ``known_hosts`` name-episodes are not in the episode's own publisher text; many are ASR
spellings the recurrence scan wrote there (record s6h). The evidence mode ``publisher_verified``
reads the text itself:

- **V1** :func:`verify_and_find` keeps an extracted name as the publisher's only when the text
  contains it (folded);
- **V2** it adds the text's own spelling of a person the episode names, found by the first name and
  a resembling surname, when the text holds exactly one such spelling;
- **V3** :func:`host_route` decides whether a host-seat spelling in a show is one of that show's
  verified names, by three guarded routes.

The publisher text is the feed and episode title and description. Pure functions; no I/O.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple

#: The evidence modes of :func:`podcast_scraper.identity.guards.collect_episode_evidence`.
EXTRACTED = "extracted"
PUBLISHER_VERIFIED = "publisher_verified"
EVIDENCE_MODES: Tuple[str, ...] = (EXTRACTED, PUBLISHER_VERIFIED)

_WORD = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ'’\-]+")
_POSSESSIVE = re.compile(r"['’]s$")
_SOUNDEX = {
    **{c: "1" for c in "bfpv"},
    **{c: "2" for c in "cgjkqsxz"},
    **{c: "3" for c in "dt"},
    "l": "4",
    **{c: "5" for c in "mn"},
    "r": "6",
}


def fold(text: Any) -> str:
    """Accents, case and punctuation folded to single spaces: how text and names are compared."""
    ascii_text = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", ascii_text.lower()).strip()


def soundex(word: str) -> str:
    """American Soundex of one word (letters only); ``""`` for no letters."""
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return ""
    out, last = w[0].upper(), _SOUNDEX.get(w[0], "")
    for c in w[1:]:
        code = _SOUNDEX.get(c, "")
        if code and code != last:
            out += code
        if c not in "hw":
            last = code
    return (out + "000")[:4]


def _ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.replace(" ", ""), b.replace(" ", "")).ratio()


def _joined(name: str) -> str:
    return fold(name).replace(" ", "")


def surname_resembles(a: str, b: str) -> bool:
    """V2's surname test: the same Soundex, or a folded ratio of at least 0.5."""
    sa, sb = _joined(a), _joined(b)
    return bool(sa and sb) and (soundex(sa) == soundex(sb) or _ratio(sa, sb) >= 0.5)


def strong_surname(a: str, b: str) -> bool:
    """V3's surname test: the same Soundex, or a folded ratio of at least 0.6."""
    sa, sb = _joined(a), _joined(b)
    return bool(sa and sb) and (soundex(sa) == soundex(sb) or _ratio(sa, sb) >= 0.6)


def split_name(name: str) -> Tuple[str, str]:
    """``(first name, the rest)`` folded; ``("", "")`` for fewer than two words."""
    toks = fold(name).split()
    return (toks[0], " ".join(toks[1:])) if len(toks) >= 2 else ("", "")


def publisher_text(meta: Mapping[str, Any]) -> str:
    """The episode's publisher-written text: feed and episode title and description."""
    feed, ep = meta.get("feed") or {}, meta.get("episode") or {}
    parts = (feed.get("title"), feed.get("description"), ep.get("title"), ep.get("description"))
    return " ".join(str(x or "") for x in parts)


def text_spans(text: str, first: str) -> List[str]:
    """Spans of ``text`` that start with the (folded) first name ``first`` and go on with one or two
    capitalised words, as written; a trailing possessive is dropped."""
    words = [m.group(0) for m in _WORD.finditer(text)]
    out: List[str] = []
    for i, w in enumerate(words):
        if fold(w) != first:
            continue
        for n in (1, 2):
            tail = words[i + 1 : i + 1 + n]
            if len(tail) == n and all(t[:1].isupper() for t in tail):
                out.append(_POSSESSIVE.sub("", " ".join([w, *tail])))
    return out


def verify_and_find(
    meta: Mapping[str, Any], by_key: Mapping[str, Sequence[str]], roster: Iterable[str]
) -> Tuple[Dict[str, List[str]], FrozenSet[Tuple[str, str]]]:
    """V1 and V2 over one episode. ``by_key``: the extracted names per diagnostics key;
    ``roster``: the names the episode's speaker record carries.

    Returns the names the publisher's text supports per key (V1), with the text's own spellings
    added under ``metadata_named`` (V2), and the ``(spelling, name)`` pairs (folded) V2 found, which
    the display rule takes as the same person."""
    text = publisher_text(meta)
    padded = f" {fold(text)} "
    raw = " ".join(text.casefold().split())

    def in_text(name: str) -> bool:
        folded = fold(name)
        if folded:
            return f" {folded} " in padded
        # A name with no Latin letters (Cyrillic, CJK) folds to nothing: compare it as written.
        written = " ".join(str(name).casefold().split())
        return bool(written) and written in raw

    out: Dict[str, List[str]] = {k: [n for n in names if in_text(n)] for k, names in by_key.items()}
    found: List[str] = []
    pairs = set()
    for name in [*roster, *(n for names in by_key.values() for n in names)]:
        first, rest = split_name(name)
        if not first or f" {fold(name)} " in padded:
            continue
        spans = [
            s
            for s in dict.fromkeys(text_spans(text, first))
            if surname_resembles(split_name(s)[1], rest)
        ]
        if len({fold(s) for s in spans}) == 1:
            found.append(spans[0])
            pairs.add((fold(spans[0]), fold(name)))
    out["metadata_named"] = list(dict.fromkeys([*out.get("metadata_named", []), *found]))
    return out, frozenset(pairs)


def host_route(host: str, other: str, by_first: Mapping[str, Sequence[str]]) -> Optional[str]:
    """V3: which route makes ``other`` (a host-seat spelling in a show) the show's verified name
    ``host``, or None. ``by_first``: the show's verified names by folded first name.

    - ``A``: the same first name, the only verified name with it, surnames strongly resembling;
    - ``B``: first names at ratio >= 0.75, surnames strongly resembling;
    - ``C``: the same last word, the same first letter, the rest at ratio >= 0.5.

    Measured on snapshot-prod-20261103 (prototype runs 1-4): a whole-name ratio route joined two
    pairs of different people, a loose first-name route a dozen, route C without the first letter
    two; these three routes join only ASR spellings of a stated host there."""
    host_toks, other_toks = fold(host).split(), fold(other).split()
    if len(host_toks) < 2 or len(other_toks) < 2 or host_toks == other_toks:
        return None
    host_first, host_rest, other_first, other_rest = (
        host_toks[0],
        " ".join(host_toks[1:]),
        other_toks[0],
        " ".join(other_toks[1:]),
    )
    if (
        host_first == other_first
        and len(by_first.get(host_first, ())) == 1
        and strong_surname(host_rest, other_rest)
    ):
        return "A"
    if (
        host_first != other_first
        and _ratio(host_first, other_first) >= 0.75
        and strong_surname(host_rest, other_rest)
    ):
        return "B"
    if (
        host_toks[-1] == other_toks[-1]
        and host_first[:1] == other_first[:1]
        and _ratio(" ".join(host_toks[:-1]), " ".join(other_toks[:-1])) >= 0.5
    ):
        return "C"
    return None
