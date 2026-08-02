"""FastMCP server construction (RFC-095).

Registers the plain ``tools/`` functions as MCP tools and runs over stdio. The MCP SDK is
imported inside :func:`build_server` so the rest of the package (and its tests) import
without it installed (the SDK rides in the ``[dev]`` extra).
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def _safe(call: Callable[[], Any]) -> dict:
    """Normalise any tool result to a uniform ``{ok, data, note}`` envelope (#1054).

    A result that already carries ``ok`` (the connectivity tools) passes through; any other
    payload is wrapped under ``data``; an exception becomes ``ok=False`` with the reason. So
    an agent can ALWAYS check ``ok`` and read ``data`` — no per-tool special-casing, no
    confusing "no data" with a crash.
    """
    try:
        result = call()
    except Exception as exc:  # noqa: BLE001 — a tool error must reach the agent as ok=False
        # Only the exception CLASS goes to the MCP client — the full str(exc) can
        # carry absolute host paths / corpus layout that fingerprints the box
        # (review 2026-07-17 low/mcp-leak). Full detail is logged server-side.
        logger.debug("MCP tool error", exc_info=True)
        return {"ok": False, "data": {}, "note": type(exc).__name__}
    if isinstance(result, dict) and "ok" in result:
        return result
    return {"ok": True, "data": result, "note": ""}


def _enveloped(fn: Callable[..., Any]) -> Callable[..., dict]:
    """Wrap a tool fn so it returns the uniform envelope; keeps its signature for FastMCP."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> dict:
        return _safe(lambda: fn(*args, **kwargs))

    return wrapper


from .context import CorpusContext
from .tools import (
    catalog as _catalog,
    cil as _cil,
    connectivity as _connectivity,
    enrichment as _enrichment,
    gi as _gi,
    relational as _relational,
)
from .tools.briefing_pack import corpus_briefing_pack as _corpus_briefing_pack
from .tools.resolve import resolve_entity as _resolve_entity
from .tools.search import search_corpus as _search_corpus
from .tools.trending import corpus_trending as _corpus_trending


def build_server(corpus_dir: Path | str) -> Any:
    """Build a FastMCP server bound to *corpus_dir* with the read tools registered.

    Tool registration is split into per-family ``_register_*`` helpers (search / relational /
    CIL / connectivity / catalog) so this stays flat as the tool count grows.
    """
    from mcp.server.fastmcp import FastMCP

    ctx = CorpusContext.from_path(corpus_dir)
    server = FastMCP("podcast-scraper")
    _register_core(server, ctx)
    _register_relational(server, ctx)
    _register_cil(server, ctx)
    _register_gi(server, ctx)
    _register_enrichment(server, ctx)
    _register_connectivity(server, ctx)
    _register_catalog(server, ctx)
    return server


def _register_core(server: Any, ctx: CorpusContext) -> None:
    """Entry tools: resolve, hybrid search, briefing pack, trending."""

    @server.tool()
    @_enveloped
    def resolve_entity(name: str, kind: Optional[str] = None) -> dict:
        """Resolve a freeform name to a canonical corpus entity id.

        Use this FIRST when a user names a person, organization, or topic
        ("Sam Altman", "OpenAI", "inflation") — the relational and intelligence tools take
        canonical ids (``person:…`` / ``org:…`` / ``topic:…``), not names. Returns the best
        match with its kind, display name, score, and method (or no candidates if unknown).
        """
        return _resolve_entity(ctx, name, kind)

    @server.tool()
    @_enveloped
    def search_corpus(
        query: str,
        tier: str = "both",
        grounded_only: bool = False,
        feed: Optional[str] = None,
        since: Optional[str] = None,
        speaker: Optional[str] = None,
        topic: Optional[str] = None,
        episode_id: Optional[str] = None,
        top_k: int = 10,
    ) -> dict:
        """Search the corpus with hybrid two-tier retrieval and get grounded evidence.

        ``tier``: "insight" (synthesized claims), "segment" (raw transcript quotes), or
        "both". ``grounded_only`` keeps only insights backed by a supporting quote.
        ``speaker``/``topic``/``episode_id`` scope the search (parity with the web
        ``/api/search``): pass a resolved ``person:``/``topic:`` id from ``resolve_entity``,
        or an episode id, to pivot a text search onto one entity/episode. Each result
        carries ``source_tier``, a relevance ``score``, and provenance (``metadata`` with
        episode/feed/entity ids); the response carries the detected ``query_type``. For
        exact quotes use ``tier="segment"``; for positions/claims use ``tier="insight"``.
        """
        return _search_corpus(
            ctx,
            query,
            tier=tier,
            grounded_only=grounded_only,
            feed=feed,
            since=since,
            speaker=speaker,
            topic=topic,
            episode_id=episode_id,
            top_k=top_k,
        )

    @server.tool()
    @_enveloped
    def corpus_briefing_pack(
        query: str,
        tier: str = "both",
        grounded_only: bool = False,
        feed: Optional[str] = None,
        since: Optional[str] = None,
        top_k: int = 10,
        max_tokens: int = 8000,
    ) -> dict:
        """LITM-positioned briefing pack over the corpus (RFC-093).

        Wraps ``search_corpus`` + the existing pack builder: returns a
        ready-to-paste-into-context block ordered for LLM attention
        — critical grounding at the top, supporting evidence in the
        middle, caveats / low-confidence at the bottom (LITM
        positioning, Liu et al. 2023). Use INSTEAD of ``search_corpus``
        when you want one assembled brief; use ``search_corpus`` when
        you want raw hits to assemble yourself.

        ``max_tokens`` is a soft budget (default 8000; ~4 chars / token
        approximation). The builder trims supporting evidence to fit.
        """
        return _corpus_briefing_pack(
            ctx,
            query,
            tier=tier,
            grounded_only=grounded_only,
            feed=feed,
            since=since,
            top_k=top_k,
            max_tokens=max_tokens,
        )

    @server.tool()
    @_enveloped
    def corpus_trending(kind: Optional[str] = None, limit: int = 8) -> dict:
        """What's rising corpus-wide right now (RFC-103 momentum).

        Time-weighted "hot" ranking (EWMA velocity), the same signal the discover ranker
        and the web trending views use. ``kind``: topic|cluster|storyline|person|episode|
        show|insight, or omit for all kinds. ``limit`` is per-kind. Each entity carries a
        namespaced ``entity_id`` you can pivot straight into the graph tools
        (``entity_neighborhood``, ``insights_about_entity``, ``topic_entities`` …) — use it
        to go "what's hot → expand it". ``velocity`` >1 = rising; ``series`` is the weekly
        sparkline.
        """
        return _corpus_trending(ctx, kind=kind, limit=limit)


def _register_relational(server: Any, ctx: CorpusContext) -> None:
    """Relational traversals (RFC-095 slice 2): all take canonical ids (resolve first)."""

    @server.tool()
    @_enveloped
    def person_positions(person_id: str, k: int = 20) -> dict:
        """Insights a person has stated — their positions (the Person→STATES→Insight edge).

        ``person_id`` is a canonical ``person:`` id (use ``resolve_entity`` on a name first).
        Results are hybrid-re-ranked by relevance to the person.
        """
        return _relational.person_positions(ctx, person_id, k=k)

    @server.tool()
    @_enveloped
    def who_said_about_topic(topic_id: str, k: int = 20) -> dict:
        """Who said what about a topic — insights grouped by the person who stated them.

        ``topic_id`` is a canonical ``topic:`` id. Returns ``{groups: {person_id: [insights]}}``;
        people with no attributed speaker are omitted (attribution is diarization-gated).
        """
        return _relational.who_said_about_topic(ctx, topic_id, k=k)

    @server.tool()
    @_enveloped
    def cross_show_synthesis(topic_id: str, per_show: int = 1) -> dict:
        """Cross-show synthesis — the top insight from each distinct show covering a topic.

        ``topic_id`` is a canonical ``topic:`` id. The corpus differentiator: returns
        ``{groups: {show_id: [insights]}}``, one (or ``per_show``) insight per show.
        """
        return _relational.cross_show_synthesis(ctx, topic_id, per_show=per_show)

    @server.tool()
    @_enveloped
    def insights_about_entity(entity_id: str, k: int = 20) -> dict:
        """Insights that mention a person or organization (``person:`` / ``org:`` id).

        Hybrid-re-ranked by relevance to the entity. Distinct from ``person_positions``
        (what they *stated*) — this is what insights *say about* them.
        """
        return _relational.insights_about_entity(ctx, entity_id, k=k)

    @server.tool()
    @_enveloped
    def topic_entities(topic_id: str, k: int = 20) -> dict:
        """The people and organizations a topic's insights mention, ranked by mention frequency.

        ``topic_id`` is a canonical ``topic:`` id.
        """
        return _relational.topic_entities(ctx, topic_id, k=k)

    @server.tool()
    @_enveloped
    def related_insights(insight_id: str, k: int = 20) -> dict:
        """Insights related to a given insight — siblings sharing a topic or mentioned entity.

        ``insight_id`` is a canonical ``insight:`` id (e.g. from a ``search_corpus`` hit's
        ``metadata.source_id``). Hybrid-re-ranked.
        """
        return _relational.related_insights(ctx, insight_id, k=k)

    @server.tool()
    @_enveloped
    def insight_detail(insight_id: str) -> dict:
        """Resolve one insight's own content — the pivot bridge from search into the graph.

        The master hop: take a ``search_corpus`` insight hit's ``pivot.id`` (or any
        ``insight:`` id) and get the insight's text, type, grounded flag, supporting
        ``quotes``, the ``topics`` it is about, and the ``entities`` it mentions — each with
        an id you can hand to ``topic_entities`` / ``entity_neighborhood`` / etc. to keep
        chaining. Unlike ``related_insights`` (structural neighbours), this is the insight
        itself. Returns ``detail: None`` if the id is not an insight.
        """
        return _relational.insight_detail(ctx, insight_id)

    @server.tool()
    @_enveloped
    def show_episodes(podcast_id: str, k: int = 20) -> dict:
        """A show's episodes (``podcast:`` id; the HAS_EPISODE relationship)."""
        return _relational.show_episodes(ctx, podcast_id, k=k)


def _register_cil(server: Any, ctx: CorpusContext) -> None:
    """CIL intelligence tools (RFC-095 slice 3): canonical ids (resolve first)."""

    @server.tool()
    @_enveloped
    def person_profile(person_id: str) -> dict:
        """A person's CIL profile — their grounded insights across episodes (``person:`` id)."""
        return _cil.person_profile(ctx, person_id)

    @server.tool()
    @_enveloped
    def topic_timeline(topic_id: str) -> dict:
        """A topic's timeline — insights about it across episodes, over time (``topic:`` id)."""
        return _cil.topic_timeline(ctx, topic_id)

    @server.tool()
    @_enveloped
    def position_arc(person_id: str, topic_id: str) -> dict:
        """How a person's position on a topic evolves over time (``person:`` + ``topic:`` ids)."""
        return _cil.position_arc(ctx, person_id, topic_id)

    @server.tool()
    @_enveloped
    def topic_conversation_arc(topic_id: str, insight_types: Optional[list] = None) -> dict:
        """A topic's conversation arc — weekly insight volume + sentiment mix over time.

        The aggregated arc (vs ``topic_timeline``'s per-week blocks): each week's count and
        neg/neu/pos split + mean compound sentiment, so an agent can read how the
        conversation evolved / heated up / soured. ``insight_types`` (e.g. ``["claim"]``)
        narrows it. ``topic_id`` is a ``topic:`` id.
        """
        return _cil.topic_conversation_arc(ctx, topic_id, insight_types=insight_types)

    @server.tool()
    @_enveloped
    def topic_perspective_leaders(limit: int = 12) -> dict:
        """Topics with the widest cross-speaker engagement — the corpus's most-debated nodes.

        Ranks topics by distinct-speaker count (≥2), most-contested first — the corpus's
        centrality proxy and a strong "what is everyone weighing in on" entrypoint. Each
        leader carries a ``topic:`` id to pivot into ``who_said_about_topic`` /
        ``topic_conversation_arc``.
        """
        return _cil.topic_perspective_leaders(ctx, limit=limit)


def _register_gi(server: Any, ctx: CorpusContext) -> None:
    """GI / grounded-insight tools: faceted discovery, per-episode insights, compare."""

    @server.tool()
    @_enveloped
    def explore_insights(
        topic: Optional[str] = None,
        speaker: Optional[str] = None,
        grounded_only: bool = False,
        min_confidence: Optional[float] = None,
        sort_by: str = "confidence",
        limit: int = 50,
    ) -> dict:
        """Faceted cross-episode insight discovery (UC5) — insights matching these facets.

        Filter the corpus's grounded insights by ``topic`` / ``speaker`` (canonical ids),
        ``grounded_only``, ``min_confidence``; ``sort_by`` = ``confidence`` | ``time``. The
        discovery complement to the entity/topic-scoped relational tools; each insight's id
        pivots on via ``insight_detail``.
        """
        return _gi.explore_insights(
            ctx,
            topic=topic,
            speaker=speaker,
            grounded_only=grounded_only,
            min_confidence=min_confidence,
            sort_by=sort_by,
            limit=limit,
        )

    @server.tool()
    @_enveloped
    def episode_insights(metadata_path: str, limit: Optional[int] = None) -> dict:
        """Salience-ranked grounded insights for one episode (ADR-135), with quotes.

        ``metadata_path`` from a ``list_episodes`` / ``search_corpus`` hit. ``limit`` caps to
        the top-N by salience. Fills the per-episode insight gap (the relational tools are
        entity/topic-scoped).
        """
        return _gi.episode_insights(ctx, metadata_path, limit=limit)

    @server.tool()
    @_enveloped
    def compare_subjects(
        subject_a: str,
        subject_b: str,
        q: str = "",
        top_k: int = 10,
        max_tokens: int = 2000,
        insight_types: Optional[list] = None,
    ) -> dict:
        """Compare two subjects — a briefing pack per side + a deterministic judge summary.

        ``subject_a``/``subject_b`` are canonical ids (resolve names first). ``q`` focuses the
        comparison; ``insight_types`` (e.g. ``["claim"]``) narrows both sides symmetrically.
        The Search-v3 compare, so an agent doesn't run two packs and diff them badly.
        """
        return _gi.compare_subjects(
            ctx,
            subject_a,
            subject_b,
            q=q,
            top_k=top_k,
            max_tokens=max_tokens,
            insight_types=insight_types,
        )


def _register_enrichment(server: Any, ctx: CorpusContext) -> None:
    """Enrichment envelopes (RFC-088) + diarized speaker roster / talk-share."""

    @server.tool()
    @_enveloped
    def corpus_enrichment_signals() -> dict:
        """Corpus-scope enrichment signals (RFC-088) — the ``enrichments/`` aggregates.

        One call for topic_similarity / topic_consensus / temporal_velocity / grounding_rate
        / guest_coappearance / topic_cooccurrence etc. Same data as ``/api/corpus/enrichment``
        — a capability probe + corpus aggregates.
        """
        return _enrichment.corpus_enrichment_signals(ctx)

    @server.tool()
    @_enveloped
    def episode_enrichment_signals(metadata_path: str) -> dict:
        """Per-episode enrichment signals (RFC-088) — sentiment, density, co-occurrence.

        ``metadata_path`` from a ``list_episodes`` / ``search_corpus`` hit. Supplements
        ``episode_detail`` with pacing/sentiment context (same data as the app episode
        enrichment route).
        """
        return _enrichment.episode_enrichment_signals(ctx, metadata_path)

    @server.tool()
    @_enveloped
    def episode_speaker_roster(metadata_path: str) -> dict:
        """Diarized speaker roster + talk-share for one episode — who spoke, %, host/guest.

        Reads the pipeline's ``.speakers.diagnostics.json`` (talk_share, unattributed share,
        per-voice_type counts). This has no HTTP route — net-new capability. Distinct from the
        knowledge-graph person tools: this is the diarized-voice layer. ``diagnostics: None``
        when the episode has no persisted diarization diagnostics.
        """
        return _enrichment.episode_speaker_roster(ctx, metadata_path)


def _register_connectivity(server: Any, ctx: CorpusContext) -> None:
    """Connectivity / neighborhood tools (#1054): one-call multi-faceted exploration."""

    @server.tool()
    @_enveloped
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
    @_enveloped
    def person_topics(person_id: str, k: int = 20) -> dict:
        """The topics a person engages, ranked by how much of their output touches each.

        ``person_id`` is a canonical ``person:`` id. Closes the person→topic traversal
        (person → stated insights → their topics) — pair with ``cross_show_synthesis`` /
        ``who_said_about_topic`` to jump from a person to the wider conversation.
        """
        return _connectivity.person_topics(ctx, person_id, k=k)

    @server.tool()
    @_enveloped
    def co_occurring_entities(entity_id: str, k: int = 20) -> dict:
        """Who is discussed *alongside* an entity — the social graph (the connectivity link).

        For a ``person:`` id: people who speak on the same topics, ranked by shared-topic
        count. Use to fan out from one voice to the others in the same conversation.
        """
        return _connectivity.co_occurring_entities(ctx, entity_id, k=k)

    @server.tool()
    @_enveloped
    def bridge(entity_a: str, entity_b: str) -> dict:
        """How two entities connect — *"how are X and Y related?"* in one call.

        Two ``person:`` ids → the topics they BOTH engage + whether they directly co-occur.
        Use after resolving two names to see what links two voices.
        """
        return _connectivity.bridge(ctx, entity_a, entity_b)

    @server.tool()
    @_enveloped
    def related_topics(topic_id: str, k: int = 20) -> dict:
        """Topics that co-occur with a topic (share insights) — topic↔topic connectivity.

        ``topic_id`` is a canonical ``topic:`` id. Use to widen from one theme to adjacent
        ones the corpus discusses together.
        """
        return _connectivity.related_topics(ctx, topic_id, k=k)

    @server.tool()
    @_enveloped
    def ego_network(entity_id: str, max_hops: int = 2, k: int = 20) -> dict:
        """The multi-hop insight/segment neighborhood around an entity (KG proximity).

        A variable-depth BFS (``max_hops`` 1-3) from an entity, returning reachable
        insight/segment nodes scored by hop-distance — unlike ``entity_neighborhood`` (a
        curated 1-hop entity projection). Use to gather everything "near" a person/topic/org.
        Each node ``id`` pivots into ``insight_detail`` / ``episode_detail``.
        """
        return _connectivity.ego_network(ctx, entity_id, max_hops=max_hops, k=k)

    @server.tool()
    @_enveloped
    def topic_clusters(topic_id: str) -> dict:
        """A topic's cluster siblings — semantic + theme cluster neighbours.

        ``topic_id`` is a canonical ``topic:`` id. Cluster *membership* (vs ``related_topics``
        co-occurrence): the ``semantic`` (embedding) and ``theme`` siblings sharing its group,
        each ``{id, label}`` a pivot back into the topic tools.
        """
        return _connectivity.topic_clusters(ctx, topic_id)


def _register_catalog(server: Any, ctx: CorpusContext) -> None:
    """Catalog / navigation tools (RFC-095 slice 3)."""

    @server.tool()
    @_enveloped
    def list_feeds() -> dict:
        """List the shows (feeds) in the corpus, with display titles and episode counts."""
        return _catalog.list_feeds(ctx)

    @server.tool()
    @_enveloped
    def list_episodes(
        feed: Optional[str] = None, since: Optional[str] = None, limit: int = 50
    ) -> dict:
        """List episodes newest-first, optionally filtered by ``feed`` substring and ``since`` date.

        ``since`` is a ``YYYY-MM-DD`` lower bound. Returns compact rows; use ``episode_detail``
        (with a row's ``metadata_path``) for one episode's full summary.
        """
        return _catalog.list_episodes(ctx, feed=feed, since=since, limit=limit)

    @server.tool()
    @_enveloped
    def episode_detail(metadata_path: str) -> dict:
        """Full detail for one episode by its ``metadata_path`` (from a list or search result)."""
        return _catalog.episode_detail(ctx, metadata_path)

    @server.tool()
    @_enveloped
    def top_people(limit: int = 10) -> dict:
        """The corpus's top voices — people ranked by grounded (quote-backed) insight count."""
        return _catalog.top_people(ctx, limit=limit)


def run_stdio(corpus_dir: Path | str) -> None:
    """Build and run the MCP server over stdio (the default agent-client transport)."""
    build_server(corpus_dir).run()
