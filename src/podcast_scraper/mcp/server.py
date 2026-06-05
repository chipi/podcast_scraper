"""FastMCP server construction (RFC-095 slice 1).

Registers the plain ``tools/`` functions as MCP tools and runs over stdio. The MCP SDK is
imported inside :func:`build_server` so the rest of the package (and its tests) import
without the ``[mcp]`` extra installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .context import CorpusContext
from .tools import relational as _relational
from .tools.resolve import resolve_entity as _resolve_entity
from .tools.search import search_corpus as _search_corpus


def build_server(corpus_dir: Path | str) -> Any:
    """Build a FastMCP server bound to *corpus_dir* with the slice-1 tools registered."""
    from mcp.server.fastmcp import FastMCP

    ctx = CorpusContext.from_path(corpus_dir)
    server = FastMCP("podcast-scraper")

    @server.tool()
    def resolve_entity(name: str, kind: Optional[str] = None) -> dict:
        """Resolve a freeform name to a canonical corpus entity id.

        Use this FIRST when a user names a person, organization, or topic
        ("Sam Altman", "OpenAI", "inflation") — the relational and intelligence tools take
        canonical ids (``person:…`` / ``org:…`` / ``topic:…``), not names. Returns the best
        match with its kind, display name, score, and method (or no candidates if unknown).
        """
        return _resolve_entity(ctx, name, kind)

    @server.tool()
    def search_corpus(
        query: str,
        tier: str = "both",
        grounded_only: bool = False,
        feed: Optional[str] = None,
        since: Optional[str] = None,
        top_k: int = 10,
    ) -> dict:
        """Search the corpus with hybrid two-tier retrieval and get grounded evidence.

        ``tier``: "insight" (synthesized claims), "segment" (raw transcript quotes), or
        "both". ``grounded_only`` keeps only insights backed by a supporting quote. Each
        result carries ``source_tier``, a relevance ``score``, and provenance
        (``metadata`` with episode/feed ids); the response carries the detected
        ``query_type``. For exact quotes use ``tier="segment"``; for positions/claims use
        ``tier="insight"``.
        """
        return _search_corpus(
            ctx,
            query,
            tier=tier,
            grounded_only=grounded_only,
            feed=feed,
            since=since,
            top_k=top_k,
        )

    # --- relational tools (RFC-095 slice 2): all take canonical ids (resolve first) ---

    @server.tool()
    def person_positions(person_id: str, k: int = 20) -> dict:
        """Insights a person has stated — their positions (the Person→STATES→Insight edge).

        ``person_id`` is a canonical ``person:`` id (use ``resolve_entity`` on a name first).
        Results are hybrid-re-ranked by relevance to the person.
        """
        return _relational.person_positions(ctx, person_id, k=k)

    @server.tool()
    def who_said_about_topic(topic_id: str, k: int = 20) -> dict:
        """Who said what about a topic — insights grouped by the person who stated them.

        ``topic_id`` is a canonical ``topic:`` id. Returns ``{groups: {person_id: [insights]}}``;
        people with no attributed speaker are omitted (attribution is diarization-gated).
        """
        return _relational.who_said_about_topic(ctx, topic_id, k=k)

    @server.tool()
    def cross_show_synthesis(topic_id: str, per_show: int = 1) -> dict:
        """Cross-show synthesis — the top insight from each distinct show covering a topic.

        ``topic_id`` is a canonical ``topic:`` id. The corpus differentiator: returns
        ``{groups: {show_id: [insights]}}``, one (or ``per_show``) insight per show.
        """
        return _relational.cross_show_synthesis(ctx, topic_id, per_show=per_show)

    @server.tool()
    def insights_about_entity(entity_id: str, k: int = 20) -> dict:
        """Insights that mention a person or organization (``person:`` / ``org:`` id).

        Hybrid-re-ranked by relevance to the entity. Distinct from ``person_positions``
        (what they *stated*) — this is what insights *say about* them.
        """
        return _relational.insights_about_entity(ctx, entity_id, k=k)

    @server.tool()
    def topic_entities(topic_id: str, k: int = 20) -> dict:
        """The people and organizations a topic's insights mention, ranked by mention frequency.

        ``topic_id`` is a canonical ``topic:`` id.
        """
        return _relational.topic_entities(ctx, topic_id, k=k)

    @server.tool()
    def related_insights(insight_id: str, k: int = 20) -> dict:
        """Insights related to a given insight — siblings sharing a topic or mentioned entity.

        ``insight_id`` is a canonical ``insight:`` id (e.g. from a ``search_corpus`` hit's
        ``metadata.source_id``). Hybrid-re-ranked.
        """
        return _relational.related_insights(ctx, insight_id, k=k)

    @server.tool()
    def show_episodes(podcast_id: str, k: int = 20) -> dict:
        """A show's episodes (``podcast:`` id; the HAS_EPISODE relationship)."""
        return _relational.show_episodes(ctx, podcast_id, k=k)

    return server


def run_stdio(corpus_dir: Path | str) -> None:
    """Build and run the MCP server over stdio (the default agent-client transport)."""
    build_server(corpus_dir).run()
