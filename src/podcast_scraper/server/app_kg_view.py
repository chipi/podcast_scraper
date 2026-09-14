"""Project a ``*.kg.json`` artifact to the consumer entities shape (#1068).

Pure functions over the parsed KG artifact dict (RFC-055/097) — no HTTP, no disk.
Returns the people, organisations, and topics for one episode. Defensive: malformed
nodes are skipped rather than raising.
"""

from __future__ import annotations

from typing import Any

from podcast_scraper.graph_id_utils import (
    is_bare_speaker_label,
    is_scoped_placeholder_person_id,
)
from podcast_scraper.identity.bare_name_scope import is_scoped_person_id
from podcast_scraper.kg.filters import is_filler_topic
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


def objects_from_kg(artifact: Any) -> list[AppEntity]:
    """``Object`` entities from a KG artifact, de-duplicated by id (#2057, KG schema 2.1).

    A separate function rather than a fourth element of :func:`entities_from_kg`, deliberately:
    that tuple is unpacked at ten call sites which want people and topics and would all have to
    change to ignore a value they never use. This keeps the blast radius at the one route that
    wants Objects.

    It exists because the alternative was silence. `entities_from_kg` matches Person or
    Organization and falls through everything else, so an Object — extracted, typed, migrated and
    indexed — was discarded at the last projection before the client. Deciding not to SHOW
    something is a client choice; dropping it in the projection is an accident.
    """
    out: dict[str, AppEntity] = {}
    if not isinstance(artifact, dict):
        return []
    nodes = artifact.get("nodes")
    if not isinstance(nodes, list):
        return []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_id = node.get("id")
        if not isinstance(node_id, str) or not node_id:
            continue
        props = node.get("properties")
        props = props if isinstance(props, dict) else {}
        if (
            node.get("type") == "Object"
            or node_id.startswith("object:")
            or props.get("kind") == "object"
        ):
            out.setdefault(
                node_id, AppEntity(id=node_id, name=_name(props, node_id), kind="object")
            )
    return list(out.values())


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
            topic_label = _name(props, node_id)
            # Conversational boilerplate is not a subject and must not be rendered as an episode
            # topic chip, a followable interest, or a discover signal. Same predicate the corpus
            # enrichers apply (``_loaders.topic_nodes``), applied HERE because this is the other
            # chokepoint: filtering one and not the other produced a chip a listener could tap
            # whose entity card was guaranteed empty, since the card's signals come from the
            # already-filtered artifacts.
            if is_filler_topic(topic_label, node_id):
                continue
            topics.setdefault(node_id, AppTopic(id=node_id, label=topic_label))
            continue

        # Person / Org entities — typed nodes (v2) or legacy ``Entity`` + ``kind``.
        kind = props.get("kind")
        if ntype == "Person" or node_id.startswith("person:") or kind == "person":
            person_name = _name(props, node_id)
            # ANONYMOUS voices — ``person:speaker-{ep}-03``, ``person:speaker-{ep}-host`` and the
            # legacy global ``person:host`` — are labels, not people. Rendering one puts
            # "SPEAKER_03" on the card as a human (#1167/#2059). Still dropped.
            if is_scoped_placeholder_person_id(node_id) or is_bare_speaker_label(person_name):
                continue
            # NAMED but episode-scoped (``person:unresolved-twiggy-{ep}``, #1685) is a DIFFERENT
            # thing, and filtering it here was wrong. A single-token name identifies one person
            # within an episode and nobody globally, which is why the id is scoped and why
            # corpus-scope surfaces exclude it — that reasoning is untouched. But THIS surface is
            # one episode's own card, where "Twiggy" is not under-specified at all: she is the
            # guest. The rule was defending a corpus-wide invariant on an episode-scoped surface,
            # and the cost was hiding the guest on the page where the guest matters most.
            #
            # `episode_scoped` lets the client render the chip without offering a tap into an
            # entity card that would be empty — #1685's own stated worry, now expressed as data
            # instead of as an omission.
            episode_scoped = is_scoped_person_id(node_id)
            persons.setdefault(
                node_id,
                AppEntity(
                    id=node_id,
                    name=person_name,
                    kind="person",
                    role=_role_of(props),
                    episode_scoped=episode_scoped,
                ),
            )
        elif ntype in ("Organization", "Org") or node_id.startswith("org:") or kind == "org":
            orgs.setdefault(node_id, AppEntity(id=node_id, name=_name(props, node_id), kind="org"))

    return list(persons.values()), list(orgs.values()), list(topics.values())
