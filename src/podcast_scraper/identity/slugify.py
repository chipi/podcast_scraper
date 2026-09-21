"""Canonical slugifier for CIL identifiers.

See ``docs/guides/GIL_KG_CIL_CROSS_LAYER.md`` for canonical identity rules (shared
namespace).
"""

from __future__ import annotations

import re
import unicodedata


def slugify(text: str) -> str:
    """Canonical slugifier for CIL identifiers.

    Behaviour:
    - Unicode normalise (NFKD), strip non-ASCII (diacritics dropped for stable IDs)
    - Lowercase
    - Replace whitespace and punctuation with hyphens
    - Collapse consecutive hyphens
    - Strip leading/trailing hyphens
    - Raise ValueError if result is empty (preserves original input in error message)

    Args:
        text: Raw human-readable label or name.

    Returns:
        Non-empty ASCII slug.

    Raises:
        ValueError: If the slug is empty after normalisation.
    """
    original = text
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text)
    text = text.strip("-")
    if not text:
        raise ValueError(f"Slug is empty after normalisation of: {original!r}")
    return text


#: Post-nominal CREDENTIALS. Not part of a name: `Peter Attia, MD` is the same man as
#: `Peter Attia`, and minting `person:peter-attia-md` beside `person:peter-attia` is two KG nodes
#: for one human. Measured on 400 sampled prod episodes: The Peter Attia Drive carries the pair on
#: four of them.
CREDENTIAL_SUFFIXES = frozenset(
    {
        "md",
        "phd",
        "dphil",
        "dds",
        "dvm",
        "do",
        "rn",
        "jd",
        "esq",
        "mba",
        "msc",
        "ma",
        "bsc",
        "ba",
        "cfa",
        "cpa",
        "pe",
        "pharmd",
        "psyd",
        "edd",
        "llm",
    }
)

#: Generational suffixes — the OPPOSITE of a credential. These exist to tell a father from a son,
#: so they are never stripped and two names disagreeing about one are two people.
GENERATIONAL_SUFFIXES = frozenset({"jr", "jnr", "junior", "sr", "snr", "senior", "ii", "iii", "iv"})


#: Closing brackets, and the opener each one pairs with. Used only to decide whether a trailing
#: bracket is a cut mark or part of the name — see `letters_or_digits_at_the_edges`.
_CLOSERS = {")": "(", "]": "[", "}": "{"}


def letters_or_digits_at_the_edges(name: str) -> str:
    """A name starts and ends with a LETTER OR A DIGIT; nothing else is part of it.

    Stated as the rule rather than a list of characters to strip, because the list was always
    going to be incomplete — `Aaron Levie)` (show-notes name cut at a bracket) and `Peter Attia,`
    (credential removed, punctuation left behind) are one defect reached two ways, and each
    leftover mark mints a separate person id from the clean spelling elsewhere.

    Digits are admitted so a regnal or generational number survives (`Louis 14`). Internal
    punctuation is untouched: `John F. Kennedy`, `Anne-Marie Slaughter`, `O'Neill`.

    A CLOSING BRACKET IS JUNK ONLY WHEN IT IS UNMATCHED. `Aaron Levie)` is a name cut at a
    bracket; `Empress Elisabeth (Sisi)` is a name that CONTAINS one, and stripping its final `)`
    left `Empress Elisabeth (Sisi` — a corrupt spelling, worse than the input, and one this
    function would then have published everywhere. Measured on the 2,257-episode production
    snapshot: 3 names carry a balanced bracket pair (`Empress Elisabeth (Sisi)`,
    `Valderis (Deco)`, `Aramar Castro (Mara)`) against 8 carrying an unmatched trailing one.
    Both are real; only the unmatched one is damage.
    """
    cleaned = (name or "").strip()
    while cleaned and not cleaned[0].isalnum():
        cleaned = cleaned[1:]
    while cleaned and not cleaned[-1].isalnum():
        if cleaned[-1] in _CLOSERS and _CLOSERS[cleaned[-1]] in cleaned[:-1]:
            break  # it has a partner — part of the name, not a cut mark
        cleaned = cleaned[:-1]
    return cleaned


def canonical_person_name(name: str) -> str:
    """The ONE spelling of a person's name that every surface must write.

    Applied both where a name is published (the roster) and where its node id is minted, because
    normalising in only one of those places is what produced the duplicate it is meant to remove:
    the roster could tidy `Peter Attia, MD` to `Peter Attia` while any other writer minting an id
    from the raw string still created `person:peter-attia-md` beside it.
    """
    cleaned = letters_or_digits_at_the_edges(" ".join((name or "").split()))
    while "," in cleaned:
        head, _, tail = cleaned.rpartition(",")
        if tail.strip().lower().replace(".", "") not in CREDENTIAL_SUFFIXES:
            break
        cleaned = letters_or_digits_at_the_edges(head)
    return cleaned


def person_id(name: str) -> str:
    """Return canonical ``person:{slug}`` (Phase 2+); slugifier is shared from Phase 1.

    The name is normalised FIRST, here at the lowest layer, so every minting path inherits it.
    `graph_id_utils.entity_node_id` is the other one; normalising in only one of them is what
    `TestEveryLayerMintsTheSameId` exists to catch, and did.
    """
    return f"person:{slugify(canonical_person_name(name) or name)}"


def org_id(name: str) -> str:
    """Return canonical ``org:{slug}`` (Phase 2+)."""
    return f"org:{slugify(name)}"


def topic_id(label: str) -> str:
    """Return canonical ``topic:{slug}`` (aligned with existing ``topic:`` nodes)."""
    return f"topic:{slugify(label)}"
