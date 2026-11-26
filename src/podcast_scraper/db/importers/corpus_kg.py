"""An item's KG file as rows: its entities, ``item_entities`` and ``person_roles`` (CRFC-005 §3.4).

:func:`import_kg` reads one ``*.kg.json`` the way the readers do (``kg.as_v3.kg_as_v3``) and
writes, in the caller's transaction:

- an ``entities`` row for each Person, Organization, Topic and Object node
  (``corpus_entities.entity_row``), inserted only when its written id has none: the identity
  import, which runs after every entity-writing importer, merges them;
- ``item_entities``: a ``mentions`` row for each ``MENTIONS`` edge from an entity to the item, a
  ``contributes_to`` row for each ``CONTRIBUTES_TO`` edge from an entity (a host or guest of the
  show), with the item's ``published_at`` and ``content_type`` copied on (§3.4, run 2);
- ``person_roles``: each Person node's ``role`` when it is one ``kg.speaker_coherence`` knows
  (host, guest, mentioned), the value ``app_momentum``'s people list reads today.

A node whose id is no entity id, or that has no label, is refused with the whole item: nothing of
it is guessed. An edge naming a node the file does not hold is refused too.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Connection

from podcast_scraper.db.corpus_entities import entity_row, NotAnEntity
from podcast_scraper.db.importers.corpus_documents import DocumentRefused
from podcast_scraper.db.schema import entities, item_entities, person_roles
from podcast_scraper.kg.as_v3 import kg_as_v3

#: The KG 3.0 node types that are entities.
ENTITY_NODES = frozenset({"Person", "Organization", "Topic", "Object"})
#: ``item_entities.role`` of each KG edge type it keeps.
EDGE_ROLES = {"MENTIONS": "mentions", "CONTRIBUTES_TO": "contributes_to"}
#: ``kg/speaker_coherence.py`` KNOWN_ROLES (corpus_0009's ``person_roles_role_check``).
PERSON_ROLES = frozenset({"host", "guest", "mentioned"})


@dataclass(frozen=True)
class KgWritten:
    """What one :func:`import_kg` wrote."""

    entities: int
    edges: int
    roles: int


def _label(node: Mapping[str, Any]) -> str:
    props = node.get("properties")
    props = props if isinstance(props, Mapping) else {}
    for key in ("label", "name"):
        value = props.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def read_kg(path: Path, language: Optional[str] = None) -> dict[str, Any]:
    """The KG file at ``path`` as KG 3.0; :class:`DocumentRefused` when it cannot be read."""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DocumentRefused(f"unreadable KG {path.name}: {exc}") from exc
    if not isinstance(doc, dict):
        raise DocumentRefused(f"KG {path.name} is not an object")
    return kg_as_v3(doc, language=language)


def import_kg(
    conn: Connection,
    item_id: int,
    kg: Mapping[str, Any],
    *,
    published_at: Optional[datetime],
    content_type: str,
) -> KgWritten:
    """Write the entities, ``item_entities`` and ``person_roles`` of ``kg`` (KG 3.0) for the item
    ``item_id``; returns what was written."""
    nodes = {n["id"]: n for n in kg.get("nodes") or [] if isinstance(n, Mapping) and n.get("id")}
    rows: dict[str, dict[str, Any]] = {}
    for written_id, node in nodes.items():
        if node.get("type") not in ENTITY_NODES:
            continue
        try:
            rows[written_id] = entity_row(written_id, _label(node))
        except NotAnEntity as exc:
            raise DocumentRefused(f"KG node {exc}") from exc

    edges: set[tuple[str, str]] = set()
    # The item is the file's ContentItem node: its id is the root ``item_id`` with the type's
    # prefix (``episode:...``), so the root value is not compared.
    items = {i for i, n in nodes.items() if n.get("type") == "ContentItem"}
    for edge in kg.get("edges") or []:
        if not isinstance(edge, Mapping):
            continue
        role = EDGE_ROLES.get(str(edge.get("type")))
        if role is None:
            continue
        source, target = edge.get("from"), edge.get("to")
        if source not in nodes or target not in nodes:
            raise DocumentRefused(f"KG edge {source} -> {target} names a node the file lacks")
        if source not in rows:
            continue
        if role == "mentions" and target not in items:
            continue
        edges.add((source, role))

    roles: set[tuple[str, str]] = set()
    for written_id, row in rows.items():
        props = nodes[written_id].get("properties")
        role = str((props or {}).get("role") or "").strip().lower()
        if row["type"] == "person" and role in PERSON_ROLES:
            roles.add((written_id, role))

    if not rows:
        return KgWritten(0, 0, 0)
    inserted = conn.execute(
        insert(entities)
        .values(list(rows.values()))
        .on_conflict_do_nothing(index_elements=["natural_id"])
        .returning(entities.c.entity_id)
    ).all()
    ids = dict(
        conn.execute(
            select(entities.c.natural_id, entities.c.entity_id).where(
                entities.c.natural_id.in_(sorted(rows))
            )
        ).all()
    )
    if edges:
        conn.execute(
            insert(item_entities)
            .values(
                [
                    {
                        "item_id": item_id,
                        "entity_id": ids[written_id],
                        "role": role,
                        "published_at": published_at,
                        "content_type": content_type,
                    }
                    for written_id, role in sorted(edges)
                ]
            )
            .on_conflict_do_nothing()
        )
    if roles:
        conn.execute(
            insert(person_roles)
            .values(
                [
                    {"item_id": item_id, "person_id": ids[written_id], "role": role}
                    for written_id, role in sorted(roles)
                ]
            )
            .on_conflict_do_nothing()
        )
    return KgWritten(len(inserted), len(edges), len(roles))
