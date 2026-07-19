"""CIL (cross-layer intelligence) MCP tools (RFC-095 slice 3).

Wrap :mod:`podcast_scraper.server.cil_queries` — the position-arc / person-profile /
topic-timeline traversals over the RFC-072 bridge. The CIL functions take a corpus
``root`` plus an ``anchor`` (a path-injection sanitisation seam in the HTTP route); for
the MCP server both are simply the corpus directory. All take canonical ids — call
``resolve_entity`` first.
"""

from __future__ import annotations

from typing import Any, Dict

from ...enrichment.enrichers._loaders import is_unresolved_speaker_placeholder
from ..context import CorpusContext


def person_profile(ctx: CorpusContext, person_id: str) -> Dict[str, Any]:
    """A person's CIL profile (``person:`` id) — their grounded insights across episodes."""
    from ...server import cil_queries

    if is_unresolved_speaker_placeholder(person_id):
        return {"subject": person_id, "profile": {}}  # #1193: no profile for a placeholder voice
    root = str(ctx.corpus_dir)
    return {"subject": person_id, "profile": cil_queries.person_profile(root, root, person_id)}


def topic_timeline(ctx: CorpusContext, topic_id: str) -> Dict[str, Any]:
    """A topic's CIL timeline (``topic:`` id) — insights about it across episodes, over time."""
    from ...server import cil_queries

    root = str(ctx.corpus_dir)
    return {"subject": topic_id, "timeline": cil_queries.topic_timeline(root, root, topic_id)}


def position_arc(ctx: CorpusContext, person_id: str, topic_id: str) -> Dict[str, Any]:
    """How a person's position on a topic evolves over time (``person:`` + ``topic:`` ids)."""
    from ...server import cil_queries

    if is_unresolved_speaker_placeholder(person_id):
        return {"subject_person": person_id, "subject_topic": topic_id, "arc": {}}  # #1193
    root = str(ctx.corpus_dir)
    arc = cil_queries.position_arc(root, root, person_id, topic_id)
    return {"subject_person": person_id, "subject_topic": topic_id, "arc": arc}
