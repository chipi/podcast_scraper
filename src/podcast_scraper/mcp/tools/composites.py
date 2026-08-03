"""Composite / dossier MCP tools — one call that fuses many surfaces.

The multiplier layer: instead of an agent chaining resolve -> profile -> neighborhood ->
positions -> insights -> trending (the person-page fan-out) by hand, a dossier does the
*right* fan-out in one call. Each composite reuses the already-validated single-surface tool
functions (no new capability, just the correct join), and every nested result keeps its ids
so the agent can still drill further. Kept bounded by ``k`` caps (RFC-095 endorses synthesized
tools; corpus_briefing_pack is the precedent).
"""

from __future__ import annotations

from typing import Any, Dict

from ..context import CorpusContext
from . import (
    cil as _cil,
    connectivity as _connectivity,
    enrichment as _enrichment,
    gi as _gi,
    relational as _relational,
)


def _kind_of(entity_id: str) -> str:
    return entity_id.split(":", 1)[0] if ":" in entity_id else ""


def entity_dossier(ctx: CorpusContext, entity_id: str, k: int = 8) -> Dict[str, Any]:
    """The full picture on one entity in a single call — the person/topic page fan-out.

    Kind-dispatched (``person`` / ``topic`` / ``org``): fuses the connected neighborhood with
    the grounding + temporal surfaces the web person/topic pages render. person → profile +
    stated positions + neighborhood; topic → timeline + conversation arc + clusters +
    neighborhood. Every nested item keeps its id, so an agent can drill any thread further.
    ``k`` bounds each section.
    """
    kind = _kind_of(entity_id)
    dossier: Dict[str, Any] = {"entity_id": entity_id, "kind": kind}

    neighborhood = _connectivity.entity_neighborhood(ctx, entity_id, k=k)
    dossier["neighborhood"] = neighborhood.get("data") if neighborhood.get("ok") else None
    dossier["note"] = neighborhood.get("note", "")

    if kind == "person":
        dossier["profile"] = _cil.person_profile(ctx, entity_id).get("profile")
        dossier["positions"] = _relational.person_positions(ctx, entity_id, k=k).get("results")
    elif kind == "topic":
        dossier["timeline"] = _cil.topic_timeline(ctx, entity_id).get("timeline")
        dossier["conversation_arc"] = _cil.topic_conversation_arc(ctx, entity_id).get("arc")
        clusters = _connectivity.topic_clusters(ctx, entity_id)
        dossier["clusters"] = clusters.get("data") if clusters.get("ok") else None

    return dossier


def episode_digest(
    ctx: CorpusContext, metadata_path: str, insight_limit: int = 10
) -> Dict[str, Any]:
    """Everything about one episode in a single call — detail + insights + signals + speakers.

    Collapses the episode-page fan-out: catalog ``detail``, salience-ranked grounded
    ``insights`` (with quotes), per-episode enrichment ``signals`` (sentiment/density), and the
    diarized ``speaker_roster`` (talk-share, when present). ``metadata_path`` from a
    ``list_episodes`` / ``search_corpus`` hit; ``insight_limit`` caps the insights.
    """
    from . import catalog as _catalog

    return {
        "episode": metadata_path,
        "detail": _catalog.episode_detail(ctx, metadata_path),
        "insights": _gi.episode_insights(ctx, metadata_path, limit=insight_limit).get("insights"),
        "enrichment_signals": _enrichment.episode_enrichment_signals(ctx, metadata_path).get(
            "signals"
        ),
        "speaker_roster": _enrichment.episode_speaker_roster(ctx, metadata_path).get("diagnostics"),
    }
