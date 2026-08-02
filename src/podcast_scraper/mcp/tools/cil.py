"""CIL (cross-layer intelligence) MCP tools (RFC-095 slice 3).

Wrap :mod:`podcast_scraper.server.cil_queries` — the position-arc / person-profile /
topic-timeline traversals over the RFC-072 bridge. The CIL functions take a corpus
``root`` plus an ``anchor`` (a path-injection sanitisation seam in the HTTP route); for
the MCP server both are simply the corpus directory. All take canonical ids — call
``resolve_entity`` first.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

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


def topic_conversation_arc(
    ctx: CorpusContext, topic_id: str, insight_types: Optional[List[str]] = None
) -> Dict[str, Any]:
    """A topic's conversation arc — weekly insight volume + sentiment mix over time.

    Unlike ``topic_timeline`` (per-week insight blocks), this is the aggregated *arc*: each
    week's insight count and neg/neu/pos sentiment split + mean compound, so an agent can
    read "how did the conversation on this topic evolve / heat up / sour". ``insight_types``
    (e.g. ``["claim"]``) narrows the arc to those types. ``topic_id`` is a ``topic:`` id.
    """
    from ...server import cil_queries

    root = str(ctx.corpus_dir)
    types = tuple(insight_types) if insight_types else None
    arc = cil_queries.topic_conversation_arc(root, root, topic_id, insight_types=types)
    return {"subject": topic_id, "arc": arc}


def topic_perspective_leaders(ctx: CorpusContext, limit: int = 12) -> Dict[str, Any]:
    """Topics with the widest cross-speaker engagement — the corpus's most-debated nodes.

    Ranks topics by distinct-speaker count (≥2 speakers), most-contested first — the
    closest thing to graph centrality the corpus surfaces, and a strong corpus-level
    entrypoint ("what is everyone weighing in on"). Each leader carries a ``topic:`` id to
    pivot into ``who_said_about_topic`` / ``topic_conversation_arc``.
    """
    from ...server import cil_queries

    root = str(ctx.corpus_dir)
    return {"leaders": cil_queries.topic_perspective_leaders(root, root, limit=max(1, int(limit)))}
