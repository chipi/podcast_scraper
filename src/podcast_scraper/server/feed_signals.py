"""Show-level ("feed") signal aggregation, shared by the operator + consumer surfaces.

Counts the Topic + Person nodes across a feed's per-episode KGs (a per-episode KG
carries only that episode's entities, so counting nodes = "mentions in that episode"),
then projects corpus-scope enrichment onto the show's entities: recurring guests
(≥2 episodes), dominant themes (``topic_theme_clusters``), trending topics
(``temporal_velocity``, gated on total ≥ 3), per-topic distinctiveness against the
corpus base rate (``topic_cooccurrence_corpus``), and a pooled grounding score
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
from podcast_scraper.kg.filters import is_filler_topic
from podcast_scraper.server.app_catalog_cache import cached_catalog
from podcast_scraper.server.app_corpus_access import cached_json_artifact
from podcast_scraper.server.corpus_catalog import (
    filter_rows,
)
from podcast_scraper.server.schemas import (
    CorpusFeedSignalsResponse,
    FeedConnectivity,
    FeedGroundingSummary,
    FeedRecurringPair,
    FeedSignalPerson,
    FeedSignalTheme,
    FeedSignalTopic,
    FeedSignalTrend,
)
from podcast_scraper.speaker_detectors.hosts import looks_like_publisher


def _read_kg_artifact(root: str, relpath: str) -> dict[str, Any] | None:
    """Read a catalog-derived KG relpath under the corpus root; None if unreadable.

    ``relpath`` comes from the corpus scan (trusted), but the realpath-under-root
    check is a cheap defensive guard against a traversal in a malformed row.
    """
    # Reject an absolute path or any ``..`` segment before it reaches os.path.join, so a
    # malformed catalog row can never escape the corpus root (defense-in-depth; the
    # realpath-under-root check below is the actual containment guard) (#1172).
    if os.path.isabs(relpath) or os.pardir in Path(relpath).parts:
        return None
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
        if not tid:
            continue
        label = node_label(n) or tid
        # Filler must not reach `top_topics` OR `_feed_connectivity`. A host catchphrase appearing
        # in every episode would otherwise become the show's #1 topic chip AND pair with every
        # real topic, inflating `recurring_pairs` — corrupting the exact metric #1932 added to
        # decide whether deepening a feed is worth the ingest budget.
        if is_filler_topic(label, tid):
            continue
        _, eps = topic_eps.setdefault(tid, (label, set()))
        eps.add(ep_key)
    for n in _person_like_nodes(art):
        pid = str(n.get("id") or "")
        if not pid:
            continue
        name = node_label(n) or pid
        if is_unresolved_speaker_placeholder(pid, name):
            continue
        if looks_like_publisher(name):  # publisher mislabelled a person on older data (#2)
            continue
        _, eps = person_eps.setdefault(pid, (name, set()))
        eps.add(ep_key)


def _read_enrichment_data(root: str, enricher_id: str) -> dict[str, Any] | None:
    """The ``data`` payload of a corpus-scope enricher envelope, or None (absent/not-ok)."""
    # enricher_id must be a bare token (never a path fragment): reject anything with a
    # path separator or traversal before it is used to build a path (defense-in-depth; #1172).
    if not enricher_id or not enricher_id.replace("_", "").replace("-", "").isalnum():
        return None
    # Shared, corpus-mtime-cached envelope read — a show page folds four corpus artifacts
    # (theme clusters, velocity, cooccurrence, grounding) and several shows share them.
    obj = cached_json_artifact(Path(root), os.path.join("enrichments", f"{enricher_id}.json"))
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


def _topic_velocity_map(root: str) -> dict[str, tuple[float, int, float]]:
    """``topic_id → (velocity_last_over_6mo, total, trend_score)`` from temporal_velocity.

    ``trend_score`` is the RANKING signal (#1931); ``velocity`` is kept because the chip displays
    it. Envelopes written before #1931 carry no ``trend_score`` — those rows get ``0.0`` and
    ``_trending_topics`` falls back to velocity for ordering, so an un-re-enriched corpus degrades
    to the old behaviour instead of collapsing to an arbitrary order.
    """
    data = _read_enrichment_data(root, "temporal_velocity")
    if not data:
        return {}
    vel: dict[str, tuple[float, int, float]] = {}
    for t in data.get("topics") or []:
        if isinstance(t, dict) and t.get("topic_id") is not None:
            v = t.get("velocity_last_over_6mo")
            total = t.get("total")
            if isinstance(v, (int, float)) and isinstance(total, int):
                raw_ts = t.get("trend_score")
                ts = float(raw_ts) if isinstance(raw_ts, (int, float)) else 0.0
                vel[str(t["topic_id"])] = (float(v), total, ts)
    return vel


def _trending_topics(
    vel: dict[str, tuple[float, int, float]],
    topic_eps: dict[str, tuple[str, set[str]]],
    top_k: int,
    min_velocity: float = 0.0,
    min_total: int = 3,
) -> list[FeedSignalTrend]:
    """Show topics the corpus is currently talking about (temporal_velocity).

    Requires corpus ``total`` >= ``min_total`` and ranks on ``trend_score``.

    ``min_velocity`` defaults to 0.0 — OFF — for the same reason the Home rail's gate does
    (#1931), and this surface had to be fixed separately because it computes its own projection:

    Velocity is an acceleration RATIO. After #1930's shrinkage pulled thin topics toward 1.0,
    exactly **2 of 602** corpus topics cleared 1.5. This function intersects a show's topics with
    that set, so the "Trending" strip on the Show rail — operator AND consumer, since
    ``/api/app/podcasts/{feed_id}/signals`` passes these rows straight through — would render
    empty for every show once the corpus is re-enriched. That is #1668's "fully built, mounted,
    fetching, and never renders" failure, one surface over.

    The gate was doing real work BEFORE the shrinkage change (a topic mentioned twice in one month
    inflated to ~6x and crowded out real momentum), which is why it was written. ``min_total``
    is what suppresses that case now, and it does it on evidence rather than on a ratio.
    """
    out: list[FeedSignalTrend] = []
    for tid, (label, eps) in topic_eps.items():
        hit = vel.get(tid)
        if hit is not None and hit[0] >= min_velocity and hit[1] >= min_total:
            out.append(
                FeedSignalTrend(
                    topic_id=tid,
                    label=label,
                    velocity=round(hit[0], 2),
                    trend_score=round(hit[2], 4),
                    episode_count=len(eps),
                )
            )
    # Rank on trend_score, velocity as the tie-break so pre-#1931 envelopes (trend_score 0.0
    # for every row) keep their old, meaningful ordering instead of collapsing to label order.
    out.sort(key=lambda t: (-t.trend_score, -t.velocity, t.label))
    return out[:top_k]


def _topic_base_rate_map(root: str) -> tuple[dict[str, int], int]:
    """``(topic_id → episodes in the WHOLE corpus, corpus episode total)``.

    Read off the ``topic_cooccurrence_corpus`` envelope, which already carries a document
    frequency per topic (``topic_{a,b}_episode_count``, the enricher's ``topic_df``) plus the
    corpus episode total it divided by. Reusing it keeps this route's cost flat: computing the
    base rate here would mean scanning every episode KG in the corpus, where today we scan only
    the show's.

    Coverage caveat: the envelope stores frequencies on *pairs*, so a topic that never shares an
    episode with another topic has no entry. That yields no base rate for it, which callers
    surface as an unknown lift rather than a fabricated one.
    """
    data = _read_enrichment_data(root, "topic_cooccurrence_corpus")
    if not data:
        return {}, 0
    total = data.get("episode_count")
    if not isinstance(total, int) or total <= 0:
        return {}, 0
    df: dict[str, int] = {}
    for p in data.get("pairs") or []:
        if not isinstance(p, dict):
            continue
        for id_key, count_key in (
            ("topic_a_id", "topic_a_episode_count"),
            ("topic_b_id", "topic_b_episode_count"),
        ):
            tid = str(p.get(id_key) or "")
            cnt = p.get(count_key)
            if tid and isinstance(cnt, int) and 0 < cnt <= total:
                df.setdefault(tid, cnt)
    return df, total


def _topic_lift(
    show_eps: int, topic_eps: int, corpus_df: int | None, corpus_eps: int
) -> float | None:
    """How over-represented a topic is on this show versus the corpus, or None if unknowable.

    ``lift = (topic's share of THIS show's episodes) / (its share of ALL episodes)``. 1.0 means
    the show talks about it exactly as much as the corpus does — i.e. it says nothing about this
    show in particular; above 1.0 means the show is unusually focused on it. This is what
    separates a distinguishing topic from wallpaper: in the validation corpus every show covers
    "expert interviews" in all four of its episodes, so raw coverage ranks it level with the one
    topic that actually identifies the show.

    Same shape as the pair lift in ``topic_cooccurrence_corpus`` (observed / expected-under-
    independence), with "appears on this show" substituted for "co-occurs with topic B".
    """
    if not show_eps or not corpus_eps or not corpus_df:
        return None
    base_share = corpus_df / corpus_eps
    if base_share <= 0:
        return None
    return round((topic_eps / show_eps) / base_share, 2)


def _show_grounding(root: str, show_episode_ids: set[str]) -> FeedGroundingSummary | None:
    """Pooled quote-backing rate across the show's EPISODES (``grounding_rate``).

    Keyed by episode since #1927. It used to pool across the show's PEOPLE, which returned 1.0 for
    everyone: an insight is grounded exactly when a supporting quote exists, the quote carries the
    speaker, so ungrounded insights have no speaker and could never enter a person's denominator.
    Pooling over episodes counts every insight the show produced, grounded or not, which is the
    number a show-level QA signal was always meant to report.
    """
    data = _read_enrichment_data(root, "grounding_rate")
    if not data:
        return None
    grounded = total = episodes = 0
    for row in data.get("episodes") or []:
        if not isinstance(row, dict) or str(row.get("episode_id") or "") not in show_episode_ids:
            continue
        gi = row.get("grounded_insights")
        ti = row.get("total_insights")
        if isinstance(gi, int) and isinstance(ti, int) and ti > 0:
            grounded += gi
            total += ti
            episodes += 1
    if total == 0:
        return None
    return FeedGroundingSummary(
        grounded_insights=grounded,
        total_insights=total,
        rate=round(grounded / total, 4),
        episode_count=episodes,
    )


def _feed_connectivity(
    topic_eps: dict[str, tuple[str, set[str]]],
    scanned: int,
    top_k: int,
) -> "FeedConnectivity":
    """How much this show RETURNS to the same topic combinations (#1932).

    Counts topic pairs that appear together in >= 2 of the show's own episodes. Measured across
    the 1,066-episode corpus, this separates shows by FORMAT far more sharply than by episode
    count — Latent Space produces 51 recurring pairs from 41 episodes, Planet Money produces 1
    from 70. Technical / thesis-driven interview shows return to a fixed concept vocabulary and
    compound; narrative journalism tells a new story each week by design and structurally cannot.

    OPERATOR-ONLY, deliberately. This measures the CORPUS, not the content: it says how a show
    interacts with our extraction over the episodes we happen to have sampled, and it moves when
    we deepen a feed, merge label variants, or retune a floor. An operator reads that as "we
    ingested more"; a listener would read ``0.014`` as a quality rating on a show that is doing
    exactly what good narrative journalism does. The consumer projection
    (``AppPodcastSignalsResponse``) omits it for the same reason it omits the grounding score.

    ``recurring_pair_rate`` is the comparable number — raw counts scale with episodes scanned, so
    comparing a 41-episode feed's 51 against a 70-episode feed's 1 is only fair per-episode.
    """
    # Invert to episode -> topics, then count pairs per episode. Bounded by the per-episode topic
    # count (single digits in practice), not by the corpus.
    eps_topics: dict[str, list[str]] = {}
    for tid, (_label, eps) in topic_eps.items():
        for ep in eps:
            eps_topics.setdefault(ep, []).append(tid)

    pair_eps: dict[tuple[str, str], int] = {}
    for tids in eps_topics.values():
        ordered = sorted(tids)
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                key = (ordered[i], ordered[j])
                pair_eps[key] = pair_eps.get(key, 0) + 1

    recurring = {k: v for k, v in pair_eps.items() if v >= 2}
    labels = {tid: lab for tid, (lab, _eps) in topic_eps.items()}
    top = sorted(recurring.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
    return FeedConnectivity(
        recurring_pairs=len(recurring),
        recurring_pair_rate=round(len(recurring) / scanned, 4) if scanned else 0.0,
        episodes_scanned=scanned,
        top_recurring_pairs=[
            FeedRecurringPair(
                topic_a_id=a,
                topic_b_id=b,
                topic_a_label=labels.get(a, a),
                topic_b_label=labels.get(b, b),
                episode_count=n,
            )
            for (a, b), n in top
        ],
    )


def compute_feed_signals(
    root: Path,
    feed_id: str,
    *,
    top_k: int = 8,
    max_episodes: int = 500,
) -> CorpusFeedSignalsResponse:
    """Aggregate a show's Topic/Person KG entities + projected enrichment (see module doc)."""
    rows = filter_rows(cached_catalog(root), feed_id=feed_id)

    topic_eps: dict[str, tuple[str, set[str]]] = {}
    person_eps: dict[str, tuple[str, set[str]]] = {}
    show_episode_ids: set[str] = set()
    scanned = 0
    for r in rows[:max_episodes]:
        if not r.has_kg or not r.kg_relative_path:
            continue
        art = _read_kg_artifact(str(root), r.kg_relative_path)
        if art is None:
            continue
        scanned += 1
        ep_key = r.episode_id or r.metadata_relative_path
        show_episode_ids.add(ep_key)
        _accumulate_kg_entities(art, ep_key, topic_eps, person_eps)

    # None, not a zero-valued row, when there was nothing to measure. A show with no KG episodes
    # would otherwise render "0.00 pairs/episode · 0 scanned", which an operator reads as a
    # measured verdict ("this show never returns to anything") when the truth is "not measured".
    # It also made ShowRailPanel's no-signals empty state unreachable, since `connectivity` was
    # always truthy. Same rule the grounding block follows.
    connectivity = _feed_connectivity(topic_eps, scanned, top_k) if scanned else None

    root_s = str(root)
    vel = _topic_velocity_map(root_s)
    corpus_df, corpus_eps = _topic_base_rate_map(root_s)
    # Selection stays "most-covered first" — that is what "top topics" means, and the operator
    # Show rail depends on it. Lift rides along as a field so a consumer can rank by
    # distinctiveness without changing what the operator sees.
    top_topics = [
        FeedSignalTopic(
            topic_id=tid,
            label=label,
            episode_count=len(eps),
            velocity=(round(vel[tid][0], 2) if tid in vel else None),
            corpus_episode_count=corpus_df.get(tid),
            corpus_episode_total=(corpus_eps or None),
            lift=_topic_lift(scanned, len(eps), corpus_df.get(tid), corpus_eps),
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
        grounding=_show_grounding(root_s, show_episode_ids),
        connectivity=connectivity,
    )
