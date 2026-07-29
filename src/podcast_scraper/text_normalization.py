"""ADR-137 — the text-normalization contract's single source of truth.

Three canonical forms exist in the pipeline: **raw** (verbatim producer output),
**display** (human-readable case + punctuation), and **match** (the comparable form every
recognition / matching / dedup / indexing path folds its INPUT to at its own boundary). This
module owns the **match** form and the name-matching primitives built on it, so casing (and unicode
/ whitespace / punctuation drift) can never again change a downstream result depending on which
producer — turbo ASR (lowercase), openai-whisper (truecased), a feed (Title Case) — emitted the
text.

The rule (ADR-137): a matching stage casts to match-form **at entry, unconditionally**; it never
trusts the surface the previous stage emitted. Recognition is the first adopter; GI/KG/identity/
search migrate onto the same SSOT incrementally.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, FrozenSet, List

# Curly quotes / dashes the ASR and feeds emit interchangeably with their ASCII forms. Folded so a
# name or cue matches whether it was written with a typographic or a straight quote/dash.
_QUOTES = str.maketrans({"‘": "'", "’": "'", "`": "'", "“": '"', "”": '"'})
_DASHES = str.maketrans({"–": "-", "—": "-", "‑": "-", "‐": "-"})
_WS_RE = re.compile(r"\s+")


def normalize_for_match(text: str) -> str:
    """The match-form of ``text``: NFKD → strip diacritics → casefold → normalize quotes/dashes →
    collapse whitespace.

    Diacritics are folded ("Gómez" → "gomez") because ASR routinely drops or mangles accents, so a
    match-form that kept them would miss "gomez" vs "Gómez" — the same reason `identity/slugify.py`
    strips them for IDs. Idempotent and lossless of word content (it does not drop punctuation or
    tokens — callers that need token-level comparison split afterwards). This is the ONE function
    every matching boundary calls; no matching path keeps its own ``.lower()`` / ``casefold``.
    """
    if not text:
        return ""
    t = unicodedata.normalize("NFKD", text)
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = t.translate(_QUOTES).translate(_DASHES)
    t = t.casefold()
    return _WS_RE.sub(" ", t).strip()


def normalize_name_for_match(name: str) -> str:
    """Match-form of a NAME: :func:`normalize_for_match` then drop apostrophes/periods so
    "O'Shaughnessy" == "oshaughnessy" and "Jr." == "jr". Hyphens are kept (they separate tokens in
    "Gómez-Bombarelli")."""
    t = normalize_for_match(name)
    # normalize_for_match already folded typographic apostrophes to the straight form.
    return re.sub(r"[.']", "", t)


# --- Nickname equivalence (ADR-137 / ADR-128) -----------------------------------------------------
# Given-name variants an ASR renders as spoken ("I'm Rich Gelfond") that a feed states formally
# ("Richard Gelfond"). Edit distance alone misses these — "rich"→"richard" is 3 edits — so a table
# is required. Bidirectional: every member of a group matches every other. Kept deliberately small
# and high-confidence; a wrong nickname bind is worse than an unnamed voice (#876), so speculative
# entries are excluded.
_NICKNAME_GROUPS: List[FrozenSet[str]] = [
    frozenset(g)
    for g in (
        {"richard", "rich", "rick", "ricky", "dick"},
        {"robert", "rob", "bob", "bobby", "robbie"},
        {"william", "will", "bill", "billy", "willie"},
        {"james", "jim", "jimmy", "jamie"},
        {"michael", "mike", "mikey", "mick"},
        {"david", "dave", "davey"},
        {"thomas", "tom", "tommy"},
        {"christopher", "chris"},
        {"daniel", "dan", "danny"},
        {"joseph", "joe", "joey"},
        {"stephen", "steven", "steve", "stevie"},
        {"edward", "ed", "eddie", "ted", "teddy"},
        {"benjamin", "ben", "benji"},
        {"andrew", "andy", "drew"},
        {"anthony", "tony"},
        {"nicholas", "nick", "nicky"},
        # Alexander and Alexandra are DIFFERENT people — split into two groups sharing only the
        # ambiguous short form "alex" (the Patrick/Patricia pattern), so alexander↔alexandra is
        # False while each still matches "alex". "sasha" is cross-gender ambiguous → excluded
        # (speculative, per this table's own rule). (review: nickname false-friend)
        {"alexander", "alex"},
        {"alexandra", "alex"},
        {"matthew", "matt", "matty"},
        {"samuel", "sam", "sammy"},
        {"gregory", "greg"},
        {"jeffrey", "geoffrey", "jeff", "geoff"},
        {"kenneth", "ken", "kenny"},
        {"ronald", "ron", "ronnie"},
        {"donald", "don", "donnie"},
        {"patrick", "pat", "paddy"},
        {"charles", "charlie", "chuck", "chas"},
        {"peter", "pete"},
        {"raymond", "ray"},
        {"timothy", "tim", "timmy"},
        # Jonathan and John are DIFFERENT people (distinct formal names) — split, sharing only the
        # ambiguous short form "jon", so jonathan↔john is False while each keeps its own nicknames.
        {"jonathan", "jon", "jonny"},
        {"john", "jon", "johnny"},
        {"katherine", "kathryn", "catherine", "kate", "katie", "kathy", "cathy", "kat"},
        {"elizabeth", "liz", "lizzie", "beth", "betsy", "eliza"},
        {"margaret", "maggie", "meg", "peggy", "greta"},
        {"jennifer", "jen", "jenny"},
        {"rebecca", "becca", "becky"},
        {"deborah", "deb", "debbie"},
        {"susan", "sue", "susie"},
        {"patricia", "pat", "patty", "trish"},
        {"victoria", "vicky", "tori"},
        {"nathaniel", "nathan", "nate"},
        {"zachary", "zach", "zack"},
        {"joshua", "josh"},
        {"gabriel", "gabe"},
        {"theodore", "theo", "ted"},
    )
]
_NICKNAME_LOOKUP: Dict[str, FrozenSet[str]] = {}
for _g in _NICKNAME_GROUPS:
    for _m in _g:
        _NICKNAME_LOOKUP[_m] = _NICKNAME_LOOKUP.get(_m, frozenset()) | _g


def first_names_match(a: str, b: str) -> bool:
    """Whether two given names denote the same person: exact (match-form), an initial of the other
    ("R." ↔ "Robert"), or a known nickname pair ("Rich" ↔ "Richard"). Case/quote/unicode-blind."""
    na, nb = normalize_name_for_match(a), normalize_name_for_match(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    if len(na) == 1 and nb.startswith(na):
        return True
    if len(nb) == 1 and na.startswith(nb):
        return True
    return nb in _NICKNAME_LOOKUP.get(na, frozenset())
