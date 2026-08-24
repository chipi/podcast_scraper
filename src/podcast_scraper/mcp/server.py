"""FastMCP server construction (RFC-095).

Registers the plain ``tools/`` functions as MCP tools and runs over stdio. The MCP SDK is
imported inside :func:`build_server` so the rest of the package (and its tests) import
without it installed (the SDK rides in the ``[dev]`` extra).
"""

from __future__ import annotations

import functools
import logging
import os
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Loopback patterns are ALWAYS allowed so the container healthcheck (127.0.0.1) and local dev
# keep working regardless of the public host config.
_LOOPBACK_HOSTS = ["127.0.0.1:*", "localhost:*", "[::1]:*", "127.0.0.1", "localhost"]
_LOOPBACK_ORIGINS = ["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"]


def _transport_security() -> Any:
    """SDK-level DNS-rebinding protection tuned for serving behind the public edge (RFC-112).

    FastMCP defaults ``host`` to ``127.0.0.1`` and, seeing loopback, auto-enables DNS-rebind
    protection whose ``allowed_hosts`` is loopback-only. Behind Caddy/Cloudflare the forwarded
    ``Host`` is the public name (e.g. ``mcp.closelistening.app``), so every request would 421
    ("Invalid Host header"). We keep protection ON but add the public host(s) — derived from
    ``APP_MCP_RESOURCE_URL`` plus an explicit ``APP_MCP_ALLOWED_HOSTS`` override — alongside the
    always-allowed loopback set. Origins fold in the same ``APP_MCP_ALLOWED_ORIGINS`` allow-list
    the auth gate already honours. This is defence-in-depth: bearer-token auth + our own Origin
    guard (:mod:`mcp.auth`) still run in front.
    """
    from mcp.server.transport_security import TransportSecuritySettings

    hosts = list(_LOOPBACK_HOSTS)
    origins = list(_LOOPBACK_ORIGINS)

    resource = os.environ.get("APP_MCP_RESOURCE_URL", "").strip()
    if resource:
        host = urlparse(resource).hostname
        if host:
            hosts += [host, f"{host}:*"]
            origins += [f"https://{host}", f"https://{host}:*"]

    for extra in os.environ.get("APP_MCP_ALLOWED_HOSTS", "").split(","):
        extra = extra.strip()
        if extra and extra not in hosts:
            hosts.append(extra)

    for origin in os.environ.get("APP_MCP_ALLOWED_ORIGINS", "").split(","):
        origin = origin.strip()
        if origin and origin not in origins:
            origins.append(origin)

    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=hosts,
        allowed_origins=origins,
    )


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
        # Per-tool observability (#1505): span + structured log + metric, best-effort. The tool
        # NAME comes from the wrapped fn; user_id from the auth contextvar. Never breaks the call.
        from .telemetry import observe_tool_call

        with observe_tool_call(fn.__name__) as call:
            result = _safe(lambda: fn(*args, **kwargs))
            call.set_result(result)
            return result

    return wrapper


from .context import CorpusContext
from .tools import (
    catalog as _catalog,
    cil as _cil,
    composites as _composites,
    connectivity as _connectivity,
    enrichment as _enrichment,
    gi as _gi,
    operators as _operators,
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
    server = FastMCP("podcast-scraper", transport_security=_transport_security())
    _register_core(server, ctx)
    _register_relational(server, ctx)
    _register_cil(server, ctx)
    _register_gi(server, ctx)
    _register_operators(server, ctx)
    _register_enrichment(server, ctx)
    _register_connectivity(server, ctx)
    _register_catalog(server, ctx)
    _register_composites(server, ctx)
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


def _register_operators(server: Any, ctx: CorpusContext) -> None:
    """Search result-set operators: cluster / consensus over a query's hits."""

    @server.tool()
    @_enveloped
    def cluster_search(query: str, tier: str = "both", top_k: int = 20) -> dict:
        """Search, then group the hits by topic/theme cluster (``operator=cluster`` parity).

        Returns clustered ``groups`` (largest first) — "what themes does this query surface"
        instead of a flat ranked list.
        """
        return _operators.cluster_search(ctx, query, tier=tier, top_k=top_k)

    @server.tool()
    @_enveloped
    def consensus_search(
        query: str, tier: str = "both", top_k: int = 20, max_pairs: int = 20
    ) -> dict:
        """Search, then surface cross-speaker consensus pairs among the hit topics.

        "Where do speakers agree on what this query is about" — filters the topic_consensus
        enricher to the surfaced topics. Empty when that enricher wasn't run.
        """
        return _operators.consensus_search(ctx, query, tier=tier, top_k=top_k, max_pairs=max_pairs)


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

    from .tools import admin as _admin

    @server.tool()
    @_enveloped
    def corpus_status() -> dict:
        """Corpus derivation freshness (RFC-118): enrichment staleness + index facts.

        Per-enricher freshness rows with typed reasons (never_ran /
        enricher_version_changed / last_run_failed_or_timed_out /
        corpus_artifacts_newer), a rolled-up ``reenrich_recommended`` flag, and the
        vector index's presence + embedding model. Read-only; the ``reenrich`` /
        ``reindex`` tools are the matching levers.
        """
        return _admin.corpus_status(ctx)

    @server.tool()
    @_enveloped
    def reenrich(force: bool = False) -> dict:
        """WRITE: enqueue a corpus enrichment pass (``force=True`` = full re-derive).

        Appends a QUEUED job to the shared registry; the API server promotes and runs
        it (nothing spawns here). ``force`` bypasses staleness gates and the RFC-118
        incremental caches — use after a model/threshold change or when
        ``corpus_status`` reports drift.
        """
        return _admin.reenrich(ctx, force=force)

    @server.tool()
    @_enveloped
    def reindex(rebuild: bool = False) -> dict:
        """WRITE: enqueue a corpus vector reindex (``rebuild=True`` = drop and rebuild).

        Appends a QUEUED ``corpus_reindex`` job the API server promotes; the child is
        the subprocess-isolated standalone reindex entry point.
        """
        return _admin.reindex(ctx, rebuild=rebuild)


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


def _register_composites(server: Any, ctx: CorpusContext) -> None:
    """Composite / dossier tools: one call that fuses many surfaces (the multipliers)."""

    @server.tool()
    @_enveloped
    def entity_dossier(entity_id: str, k: int = 8) -> dict:
        """The full picture on one entity in a single call — the person/topic page fan-out.

        Kind-dispatched. person → profile + stated positions + neighborhood; topic → timeline
        + conversation arc + clusters + neighborhood. Replaces the 5-6 call person/topic-page
        chain; every nested item keeps its id so you can still drill any thread. ``k`` bounds
        each section.
        """
        return _composites.entity_dossier(ctx, entity_id, k=k)

    @server.tool()
    @_enveloped
    def episode_digest(metadata_path: str, insight_limit: int = 10) -> dict:
        """Everything about one episode in a single call — detail + insights + signals + speakers.

        Collapses the episode-page fan-out: catalog detail, salience-ranked grounded insights
        (with quotes), per-episode enrichment signals, and the diarized speaker roster.
        ``metadata_path`` from a ``list_episodes`` / ``search_corpus`` hit.
        """
        return _composites.episode_digest(ctx, metadata_path, insight_limit=insight_limit)


def run_stdio(corpus_dir: Path | str) -> None:
    """Build and run the MCP server over stdio (the default agent-client transport)."""
    build_server(corpus_dir).run()


def build_http_app(corpus_dir: Path | str) -> Any:
    """The Streamable-HTTP ASGI app for the corpus MCP, wrapped in the auth gate (RFC-112 slice 2).

    Every HTTP connection must present a valid MCP bearer token (verified against the app's internal
    endpoint) — see ``mcp.auth``. Returned as an ASGI app so it can be served by uvicorn behind the
    shared edge, or exercised in tests without a live socket.
    """
    from .auth import McpAuthMiddleware

    server = build_server(corpus_dir)
    return McpAuthMiddleware(server.streamable_http_app())


def run_server(
    corpus_dir: Path | str,
    *,
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 8009,
) -> None:
    """Run the corpus MCP over *transport*.

    ``stdio`` (local, auth-free) or ``http`` (Streamable HTTP on ``host:port``, auth-gated per
    RFC-112). The tool registry is identical across transports (RFC-095 transport-agnostic design).
    """
    if transport == "stdio":
        run_stdio(corpus_dir)
        return
    if transport != "http":
        raise ValueError(f"unsupported transport: {transport!r} (use 'stdio' or 'http')")
    import uvicorn

    # o11y parity with the api (#1505): OTel traces + GlitchTip errors + inbound request spans.
    # All best-effort — telemetry never blocks the server from starting (ADR-120). No-ops unless
    # the OTEL_*/PODCAST_SENTRY_DSN_MCP env is set (dev + tests stay silent).
    try:
        from podcast_scraper.utils.otel_init import init_otel

        init_otel()
    except Exception:  # noqa: BLE001 - never block serving on tracing
        logger.debug("mcp otel init skipped", exc_info=True)
    try:
        from podcast_scraper.utils.sentry_init import init_sentry

        init_sentry("mcp")
    except Exception:  # noqa: BLE001 - never block serving on error reporting
        logger.debug("mcp sentry init skipped", exc_info=True)

    uvicorn.run(_instrument_asgi(_with_metrics(build_http_app(corpus_dir))), host=host, port=port)


def _with_metrics(app: Any) -> Any:
    """Serve Prometheus ``/metrics`` (the per-tool counters) UNGATED, delegating everything else.

    Wrapped OUTSIDE the auth middleware so a scraper needs no bearer; the mcp binds loopback
    (:8009, Caddy fronts ``/mcp`` publicly), so ``/metrics`` is internal-only. No-op if
    prometheus_client isn't installed.
    """
    try:
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
    except Exception:  # noqa: BLE001 - no client → no /metrics endpoint
        return app

    async def _asgi(scope: dict, receive: Any, send: Any) -> None:
        if scope.get("type") == "http" and scope.get("path") == "/metrics":
            body = generate_latest()
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", CONTENT_TYPE_LATEST.encode())],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return
        await app(scope, receive, send)

    return _asgi


def _instrument_asgi(app: Any) -> Any:
    """Wrap the ASGI app with OTel inbound-request spans (so every ``/mcp`` gets a ``trace_id``).

    No-op if the OTel ASGI instrumentation isn't installed or tracing isn't configured. The tool
    spans (:mod:`mcp.telemetry`) nest under these server spans, and the trace propagates onward to
    the api's ``/internal/mcp/verify`` via the auto-instrumented outbound HTTP client.
    """
    try:
        from opentelemetry.instrumentation.asgi import OpenTelemetryMiddleware

        return OpenTelemetryMiddleware(app)
    except Exception:  # noqa: BLE001 - no ASGI instrumentor → serve un-instrumented
        logger.debug("mcp ASGI instrumentation skipped", exc_info=True)
        return app
