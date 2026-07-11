"""Project a ``*.kg.json`` artifact to the consumer entities shape (#1068).

Pure functions over the parsed KG artifact dict (RFC-055/097) — no HTTP, no disk.
Returns the people, organisations, and topics for one episode. Defensive: malformed
nodes are skipped rather than raising.
"""

from __future__ import annotations

from typing import Any

from podcast_scraper.enrichment.enrichers._loaders import is_unresolved_speaker_placeholder
from podcast_scraper.server.schemas import AppEntity, AppTopic


def _role_of(props: dict) -> str | None:
    """Normalised speaker role (``host``/``guest``/``mentioned``) from a person node, else None."""
    role = props.get("role")
    if isinstance(role, str) and role.strip():
        return role.strip().lower()
    return None


def _name(props: dict, fallback_id: Any) -> str:
    for key in ("name", "label", "display_name"):
        val = props.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    if isinstance(fallback_id, str) and ":" in fallback_id:
        return fallback_id.split(":", 1)[1]
    return str(fallback_id) if fallback_id is not None else ""


def entities_from_kg(artifact: Any) -> tuple[list[AppEntity], list[AppEntity], list[AppTopic]]:
    """Return ``(persons, orgs, topics)`` from a KG artifact dict, de-duplicated by id."""
    persons: dict[str, AppEntity] = {}
    orgs: dict[str, AppEntity] = {}
    topics: dict[str, AppTopic] = {}

    if not isinstance(artifact, dict):
        return [], [], []
    nodes = artifact.get("nodes")
    if not isinstance(nodes, list):
        return [], [], []

    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id:
            continue
        ntype = node.get("type")
        props = node.get("properties")
        props = props if isinstance(props, dict) else {}

        if ntype == "Topic" or node_id.startswith("topic:"):
            topics.setdefault(node_id, AppTopic(id=node_id, label=_name(props, node_id)))
            continue

        # Person / Org entities — typed nodes (v2) or legacy ``Entity`` + ``kind``.
        kind = props.get("kind")
        if ntype == "Person" or node_id.startswith("person:") or kind == "person":
            person_name = _name(props, node_id)
            # Unresolved diarization voices (``person:speaker-NN``) are anonymous
            # labels, not real people — never surface them as Person entities (#1167).
            if is_unresolved_speaker_placeholder(node_id, person_name):
                continue
            persons.setdefault(
                node_id,
                AppEntity(id=node_id, name=person_name, kind="person", role=_role_of(props)),
            )
        elif ntype in ("Organization", "Org") or node_id.startswith("org:") or kind == "org":
            orgs.setdefault(node_id, AppEntity(id=node_id, name=_name(props, node_id), kind="org"))

    return list(persons.values()), list(orgs.values()), list(topics.values())
