"""FastMCP server construction (RFC-095).

Registers the plain ``tools/`` functions as MCP tools and runs over stdio. The MCP SDK is
imported inside :func:`build_server` so the rest of the package (and its tests) import
without it installed (the SDK rides in the ``[dev]`` extra).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .context import CorpusContext
from .tools import (
    catalog as _catalog,
    cil as _cil,
    connectivity as _connectivity,
    relational as _relational,
)
from .tools.resolve import resolve_entity as _resolve_entity
from .tools.search import search_corpus as _search_corpus


def build_server(corpus_dir: Path | str) -> Any:
    """Build a FastMCP server bound to *corpus_dir* with the read tools registered."""
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

    # --- CIL intelligence tools (RFC-095 slice 3): canonical ids (resolve first) ---

    @server.tool()
    def person_profile(person_id: str) -> dict:
        """A person's CIL profile — their grounded insights across episodes (``person:`` id)."""
        return _cil.person_profile(ctx, person_id)

    @server.tool()
    def topic_timeline(topic_id: str) -> dict:
        """A topic's timeline — insights about it across episodes, over time (``topic:`` id)."""
        return _cil.topic_timeline(ctx, topic_id)

    @server.tool()
    def position_arc(person_id: str, topic_id: str) -> dict:
        """How a person's position on a topic evolves over time (``person:`` + ``topic:`` ids)."""
        return _cil.position_arc(ctx, person_id, topic_id)

    # --- connectivity / neighborhood tools (#1054): one-call multi-faceted exploration ---

    @server.tool()
    def entity_neighborhood(entity_id: str, k: int = 8) -> dict:
        """Everything connected to an entity, in ONE call — the exploration keystone.

        Pass a canonical id (``resolve_entity`` first). person → what they stated, what's
        said about them, their topics, co-speakers, shows; topic → entities, speakers,
        cross-show synthesis; org → mentioned-in; podcast → episodes. Uniform envelope
        ``{ok, kind, subject, data, note}`` — ``note`` explains empty/sparse results. Use
        this to understand an entity before drilling in with the focused tools.
        """
        return _connectivity.entity_neighborhood(ctx, entity_id, k=k)

    @server.tool()
    def person_topics(person_id: str, k: int = 20) -> dict:
        """The topics a person engages, ranked by how much of their output touches each.

        ``person_id`` is a canonical ``person:`` id. Closes the person→topic traversal
        (person → stated insights → their topics) — pair with ``cross_show_synthesis`` /
        ``who_said_about_topic`` to jump from a person to the wider conversation.
        """
        return _connectivity.person_topics(ctx, person_id, k=k)

    @server.tool()
    def co_occurring_entities(entity_id: str, k: int = 20) -> dict:
        """Who is discussed *alongside* an entity — the social graph (the connectivity link).

        For a ``person:`` id: people who speak on the same topics, ranked by shared-topic
        count. Use to fan out from one voice to the others in the same conversation.
        """
        return _connectivity.co_occurring_entities(ctx, entity_id, k=k)

    # --- catalog / navigation tools (RFC-095 slice 3) ---

    @server.tool()
    def list_feeds() -> dict:
        """List the shows (feeds) in the corpus, with display titles and episode counts."""
        return _catalog.list_feeds(ctx)

    @server.tool()
    def list_episodes(
        feed: Optional[str] = None, since: Optional[str] = None, limit: int = 50
    ) -> dict:
        """List episodes newest-first, optionally filtered by ``feed`` substring and ``since`` date.

        ``since`` is a ``YYYY-MM-DD`` lower bound. Returns compact rows; use ``episode_detail``
        (with a row's ``metadata_path``) for one episode's full summary.
        """
        return _catalog.list_episodes(ctx, feed=feed, since=since, limit=limit)

    @server.tool()
    def episode_detail(metadata_path: str) -> dict:
        """Full detail for one episode by its ``metadata_path`` (from a list or search result)."""
        return _catalog.episode_detail(ctx, metadata_path)

    @server.tool()
    def top_people(limit: int = 10) -> dict:
        """The corpus's top voices — people ranked by grounded (quote-backed) insight count."""
        return _catalog.top_people(ctx, limit=limit)

    return server


def run_stdio(corpus_dir: Path | str) -> None:
    """Build and run the MCP server over stdio (the default agent-client transport)."""
    build_server(corpus_dir).run()
