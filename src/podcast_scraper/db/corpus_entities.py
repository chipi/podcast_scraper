"""The column values of a corpus database ``entities`` row (CRFC-005 §3.3).

The one place a written id becomes a row: the KG importer, the GI importer and the identity
importer all call :func:`entity_row`, so a written id has the same row whichever importer reaches
it first (each inserts on ``natural_id`` and leaves an existing row as it is). Reads no file and no
database. ``canonical_id`` is not here: the ``entities_canonical_default`` trigger makes a new row
its own canonical, and only the identity import repoints it.

What each column comes from:

- ``natural_id``: the written id as written (``person:jane-doe``), the id every artifact and
  route carries; it is never rewritten.
- ``type``: the written id's prefix as written (``person``, ``org``, ``topic``, ``object``), the
  key the ``(type, normalized_label)`` lookup filters by.
- ``scope``: the identity scope of that type, as the identity tables spell it
  (``org`` is ``organisation``).
- ``label``: the name the caller read for the id, unchanged.
- ``normalized_label``: :func:`normalized_label`, the exact-lookup key of entity resolution today
  (Common's ``stages.identity.resolver._norm``, which ``resolve_entity`` matches names with;
  ``search.corpus_search._normalize_kg_surface_text`` gives the same result). Common's tests check
  that the two agree.
"""

from __future__ import annotations

from typing import Any

#: The identity scope of each entity type (corpus_0005's ``_SCOPES``).
SCOPES = {"person": "person", "org": "organisation", "topic": "topic", "object": "object"}


class NotAnEntity(ValueError):
    """The id is not a written entity id, or there is no label for it."""


def entity_type(written_id: str) -> str:
    """The written id's prefix; :class:`NotAnEntity` for an id of no entity type."""
    prefix, sep, rest = str(written_id).partition(":")
    if not sep or not rest or prefix not in SCOPES:
        raise NotAnEntity(f"{written_id!r} is not a written entity id")
    return prefix


def normalized_label(label: str) -> str:
    """Lowercase, whitespace collapsed: the key an exact name lookup matches on."""
    return " ".join(str(label).lower().split())


def entity_row(written_id: str, label: str) -> dict[str, Any]:
    """The ``entities`` row of a written id named ``label``."""
    kind = entity_type(written_id)
    if not str(label or "").strip():
        raise NotAnEntity(f"{written_id!r} has no label")
    return {
        "natural_id": written_id,
        "type": kind,
        "scope": SCOPES[kind],
        "label": label,
        "normalized_label": normalized_label(label),
    }
