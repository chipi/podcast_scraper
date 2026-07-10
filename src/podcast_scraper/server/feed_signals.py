"""Show-level ("feed") signal aggregation, shared by the operator + consumer surfaces.

Counts the Topic + Person nodes across a feed's per-episode KGs (a per-episode KG
carries only that episode's entities, so counting nodes = "mentions in that episode"),
then projects corpus-scope enrichment onto the show's entities: recurring guests
(≥2 episodes), dominant themes (``topic_theme_clusters``), trending topics
(``temporal_velocity``, gated on total ≥ 3), and a pooled grounding score
(``grounding_rate``). Every enrichment fold is best-effort — absent envelopes yield
empty/None. The operator route (``/api/corpus/feed-signals``) returns the full result;
the consumer route (``/api/app/podcasts/{feed_id}/signals``) projects a listener-shaped
subset over the same computation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from podcast_scraper.enrichment.enrichers._loaders import (
    is_unresolved_speaker_placeholder,
    node_label,
    nodes_of_type,
)
from podcast_scraper.server.corpus_catalog import (
    build_catalog_rows_cumulative,
    filter_rows,
)
from podcast_scraper.server.schemas import (
    CorpusFeedSignalsResponse,
    FeedGroundingSummary,
    FeedSignalPerson,
    FeedSignalTheme,
    FeedSignalTopic,
    FeedSignalTrend,
)


def _read_kg_artifact(root: str, relpath: str) -> dict[str, Any] | None:
    """Read a catalog-derived KG relpath under the corpus root; None if unreadable.

    ``relpath`` comes from the corpus scan (trusted), but the realpath-under-root
    check is a cheap defensive guard against a traversal in a malformed row.
    """
    try:
        root_real = os.path.realpath(root)
        target = os.path.realpath(os.path.join(root, relpath))
        if not (target == root_real or target.startswith(root_real + os.sep)):
            return None
        obj = json.loads(Path(target).read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except (OSError, ValueError):
        return None


def _person_like_nodes(art: dict[str, Any]) -> list[dict[str, Any]]:
    """People in a per-episode KG.

    Real KGs slug people as ``type:"Entity"`` with ``properties.kind == "person"``
    (id ``person:…``); a dedicated ``type:"Person"`` is also accepted for forward
    compatibility. Orgs (``kind:"org"``) are excluded.
    """
    out: list[dict[str, Any]] = []
    for n in art.get("nodes") or []:
        if not isinstance(n, dict):
            continue
        t = n.get("type")
        kind = (n.get("properties") or {}).get("kind")
        if t == "Person" or (t == "Entity" and kind == "person"):
            out.append(n)
    return out


def _accumulate_kg_entities(
    art: dict[str, Any],
    ep_key: str,
    topic_eps: dict[str, tuple[str, set[str]]],
    person_eps: dict[str, tuple[str, set[str]]],
) -> None:
    """Fold one episode KG's Topic + Person nodes into the running per-feed aggregates.

    A per-episode KG carries only that episode's entities, so each Topic/Person node
    counts as one "mention in this episode" (``ep_key`` is de-duped in the set).
    Diarization placeholders (``SPEAKER_NN``) are dropped from people.
    """
    for n in nodes_of_type(art, "Topic"):
        tid = str(n.get("id") or "")
        if tid:
            _, eps = topic_eps.setdefault(tid, (node_label(n) or tid, set()))
            eps.add(ep_key)
    for n in _person_like_nodes(art):
        pid = str(n.get("id") or "")
        if not pid:
            continue
        name = node_label(n) or pid
        if is_unresolved_speaker_placeholder(pid, name):
            continue
        _, eps = person_eps.setdefault(pid, (name, set()))
        eps.add(ep_key)


def _read_enrichment_data(root: str, enricher_id: str) -> dict[str, Any] | None:
    """The ``data`` payload of a corpus-scope enricher envelope, or None (absent/not-ok)."""
    try:
        obj = json.loads(
            Path(os.path.join(root, "enrichments", f"{enricher_id}.json")).read_text(
                encoding="utf-8"
            )
        )
    except (OSError, ValueError):
        return None
    if not isinstance(obj, dict) or obj.get("status") not in (None, "ok"):
        return None
    data = obj.get("data")
    return data if isinstance(data, dict) else None


def _recurring_guests(
    person_eps: dict[str, tuple[str, set[str]]], top_k: int
) -> list[FeedSignalPerson]:
    """People in ≥2 of the show's episodes (regulars vs one-off guests)."""
    out = [
        FeedSignalPerson(person_id=pid, name=name, episode_count=len(eps))
        for pid, (name, eps) in person_eps.items()
        if len(eps) >= 2
    ]
    out.sort(key=lambda p: (-p.episode_count, p.name))
    return out[:top_k]


def _dominant_themes(root: str, show_topic_ids: set[str], top_k: int) -> list[FeedSignalTheme]:
    """Theme clusters (topic_theme_clusters) that the show's topics fall into, by overlap."""
    data = _read_enrichment_data(root, "topic_theme_clusters")
    if not data:
        return []
    out: list[FeedSignalTheme] = []
    for c in data.get("clusters") or []:
        if not isinstance(c, dict):
            continue
        matched_ids = [
            str(m.get("topic_id") or "")
            for m in (c.get("members") or [])
            if isinstance(m, dict) and str(m.get("topic_id") or "") in show_topic_ids
        ]
        tid = str(c.get("graph_compound_parent_id") or "")
        label = str(c.get("canonical_label") or "").strip()
        if matched_ids and tid and label:
            out.append(
                FeedSignalTheme(
                    theme_id=tid,
                    label=label,
                    topic_count=len(matched_ids),
                    anchor_topic_id=matched_ids[0],
                )
            )
    out.sort(key=lambda t: (-t.topic_count, t.label))
    return out[:top_k]


def _topic_velocity_map(root: str) -> dict[str, tuple[float, int]]:
    """``topic_id → (velocity_last_over_6mo, total)`` from the temporal_velocity envelope."""
    data = _read_enrichment_data(root, "temporal_velocity")
    if not data:
        return {}
    vel: dict[str, tuple[float, int]] = {}
    for t in data.get("topics") or []:
        if isinstance(t, dict) and t.get("topic_id") is not None:
            v = t.get("velocity_last_over_6mo")
            total = t.get("total")
            if isinstance(v, (int, float)) and isinstance(total, int):
                vel[str(t["topic_id"])] = (float(v), total)
    return vel


def _trending_topics(
    vel: dict[str, tuple[float, int]],
    topic_eps: dict[str, tuple[str, set[str]]],
    top_k: int,
    min_velocity: float = 1.5,
    min_total: int = 3,
) -> list[FeedSignalTrend]:
    """Show topics that are genuinely heating up (temporal_velocity).

    Requires velocity ≥ ``min_velocity`` AND corpus ``total`` ≥ ``min_total`` — the
    same total gate the Home trending chips use — so a topic mentioned twice in one
    month (velocity math inflates it to ~6×) doesn't crowd out real momentum.
    """
    out: list[FeedSignalTrend] = []
    for tid, (label, eps) in topic_eps.items():
        hit = vel.get(tid)
        if hit is not None and hit[0] >= min_velocity and hit[1] >= min_total:
            out.append(
                FeedSignalTrend(
                    topic_id=tid, label=label, velocity=round(hit[0], 2), episode_count=len(eps)
                )
            )
    out.sort(key=lambda t: (-t.velocity, t.label))
    return out[:top_k]


def _show_grounding(root: str, show_person_ids: set[str]) -> FeedGroundingSummary | None:
    """Pooled quote-backing rate across the show's people (grounding_rate)."""
    data = _read_enrichment_data(root, "grounding_rate")
    if not data:
        return None
    grounded = total = people = 0
    for p in data.get("persons") or []:
        if not isinstance(p, dict) or str(p.get("person_id") or "") not in show_person_ids:
            continue
        gi = p.get("grounded_insights")
        ti = p.get("total_insights")
        if isinstance(gi, int) and isinstance(ti, int) and ti > 0:
            grounded += gi
            total += ti
            people += 1
    if total == 0:
        return None
    return FeedGroundingSummary(
        grounded_insights=grounded,
        total_insights=total,
        rate=round(grounded / total, 4),
        people_count=people,
    )


def compute_feed_signals(
    root: Path,
    feed_id: str,
    *,
    top_k: int = 8,
    max_episodes: int = 500,
) -> CorpusFeedSignalsResponse:
    """Aggregate a show's Topic/Person KG entities + projected enrichment (see module doc)."""
    rows = filter_rows(build_catalog_rows_cumulative(root), feed_id=feed_id)

    topic_eps: dict[str, tuple[str, set[str]]] = {}
    person_eps: dict[str, tuple[str, set[str]]] = {}
    scanned = 0
    for r in rows[:max_episodes]:
        if not r.has_kg or not r.kg_relative_path:
            continue
        art = _read_kg_artifact(str(root), r.kg_relative_path)
        if art is None:
            continue
        scanned += 1
        _accumulate_kg_entities(
            art, r.episode_id or r.metadata_relative_path, topic_eps, person_eps
        )

    root_s = str(root)
    vel = _topic_velocity_map(root_s)
    top_topics = [
        FeedSignalTopic(
            topic_id=tid,
            label=label,
            episode_count=len(eps),
            velocity=(round(vel[tid][0], 2) if tid in vel else None),
        )
        for tid, (label, eps) in sorted(
            topic_eps.items(), key=lambda kv: (-len(kv[1][1]), kv[1][0])
        )[:top_k]
    ]
    key_people = [
        FeedSignalPerson(person_id=pid, name=name, episode_count=len(eps))
        for pid, (name, eps) in sorted(
            person_eps.items(), key=lambda kv: (-len(kv[1][1]), kv[1][0])
        )[:top_k]
    ]
    return CorpusFeedSignalsResponse(
        path=root_s,
        feed_id=feed_id,
        episode_count=scanned,
        top_topics=top_topics,
        key_people=key_people,
        recurring_guests=_recurring_guests(person_eps, top_k),
        dominant_themes=_dominant_themes(root_s, set(topic_eps.keys()), top_k),
        trending_topics=_trending_topics(vel, topic_eps, top_k),
        grounding=_show_grounding(root_s, set(person_eps.keys())),
    )
