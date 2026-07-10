"""Consumer relational projections — person/topic cards from KG co-occurrence (PRD-043 FR2/FR3).

Read-only aggregation over each episode's ``*.kg.json`` (the same view the entities endpoint
uses via :func:`entities_from_kg`). *KG-grounded*: an entity *appears in* — and a topic *is
about* — an episode iff that episode's KG asserts the node; relatedness is co-occurrence within
those episodes. No ``CorpusGraph`` build, no LLM, no operator-route coupling.

One corpus scan per request (consistent with the existing ``/related`` and ``/search`` consumer
endpoints). The scan is the cost; cache later if the corpus grows large enough to matter.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Sequence

from podcast_scraper.search.theme_clusters import (
    consumer_theme_cluster_map,
    consumer_theme_cluster_siblings,
)
from podcast_scraper.search.topic_clusters import (
    consumer_cluster_siblings,
    consumer_topic_cluster_map,
)
from podcast_scraper.server.app_content_source import row_to_summary
from podcast_scraper.server.app_corpus_access import load_json_artifact
from podcast_scraper.server.app_kg_view import entities_from_kg
from podcast_scraper.server.cil_queries import topic_perspectives
from podcast_scraper.server.corpus_catalog import (
    build_catalog_rows_cumulative,
    CatalogEpisodeRow,
)
from podcast_scraper.server.schemas import (
    AppEntity,
    AppEntityRef,
    AppEpisodeSummary,
    AppInsight,
    AppPersonCard,
    AppTopic,
    AppTopicCard,
    AppTopicPerspective,
    AppTopicPerspectivesResponse,
)

_DEFAULT_TOP_K = 12

# topic_id -> {cluster_id, cluster_label, cluster_size}; from search/topic_clusters.json.
ClusterMap = dict[str, dict[str, object]]


def _iter_kg_entities(
    root: Path, rows: Sequence[CatalogEpisodeRow]
) -> Iterator[tuple[CatalogEpisodeRow, list[AppEntity], list[AppEntity], list[AppTopic]]]:
    """Yield ``(row, persons, orgs, topics)`` for each episode with a readable KG artifact."""
    for row in rows:
        if not row.has_kg:
            continue
        artifact = load_json_artifact(root, row.kg_relative_path)
        if artifact is None:
            continue
        persons, orgs, topics = entities_from_kg(artifact)
        yield row, persons, orgs, topics


def _normalize_label(text: str) -> str:
    """Fold a label/query to a comparison key: punctuation→space, collapse, lower.

    "Matthew Walker." / "matthew-walker" / "MATTHEW  WALKER" all map to "matthew walker",
    giving exact/near-exact matching (case / punctuation / spacing insensitive) without the
    false positives of fuzzy distance matching.
    """
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text)).strip().lower()


def resolve_entity(
    root: Path,
    query: str,
    *,
    rows: Sequence[CatalogEpisodeRow] | None = None,
) -> AppEntityRef | None:
    """Resolve an exact/near-exact person/topic name match for ``query``, else ``None`` (3.4).

    Persons take precedence over topics on a tie. Only persons/topics are resolved (those are the
    entities with cards). One corpus KG scan per call — cache later if search traffic warrants it.
    """
    norm = _normalize_label(query)
    if not norm:
        return None
    catalog = list(rows) if rows is not None else build_catalog_rows_cumulative(root)
    persons_idx: dict[str, AppEntityRef] = {}
    topics_idx: dict[str, AppEntityRef] = {}
    for _row, persons, _orgs, topics in _iter_kg_entities(root, catalog):
        for p in persons:
            persons_idx.setdefault(
                _normalize_label(p.name), AppEntityRef(id=p.id, kind="person", label=p.name)
            )
        for t in topics:
            topics_idx.setdefault(
                _normalize_label(t.label), AppEntityRef(id=t.id, kind="topic", label=t.label)
            )
    return persons_idx.get(norm) or topics_idx.get(norm)


def _enrich_topic(
    topic: AppTopic, cluster_map: ClusterMap, theme_map: ClusterMap | None = None
) -> AppTopic:
    """Attach semantic + theme cluster identity to a topic (no-op when unclustered)."""
    update: dict[str, object] = {}
    info = cluster_map.get(topic.id)
    if info:
        update.update(info)
    if theme_map:
        tinfo = theme_map.get(topic.id)
        if tinfo:
            update.update(tinfo)
    return topic.model_copy(update=update) if update else topic


def _sorted_episode_cards(root: Path, rows: list[CatalogEpisodeRow]) -> list[AppEpisodeSummary]:
    """Project rows to episode cards, newest-first (undated episodes sort last)."""
    cards = [row_to_summary(root, r) for r in rows]
    cards.sort(key=lambda c: (c.publish_date is not None, c.publish_date or ""), reverse=True)
    return cards


# Role precedence for the aggregate card badge: a person who hosts anywhere is a "host",
# a guest anywhere (but never a host) is a "guest", otherwise the weakest role seen.
_ROLE_RANK = {"host": 3, "guest": 2, "mentioned": 1}


def _aggregate_role(roles: Sequence[str | None]) -> str | None:
    """The strongest speaker role across a person's episode nodes (host > guest > mentioned)."""
    ranked = [r for r in roles if r]
    if not ranked:
        return None
    return max(ranked, key=lambda r: _ROLE_RANK.get(r, 0))


def build_person_card(
    root: Path,
    person_id: str,
    *,
    rows: Sequence[CatalogEpisodeRow] | None = None,
    top_k: int = _DEFAULT_TOP_K,
) -> AppPersonCard | None:
    """Project the person's corpus footprint to a card, or ``None`` if they appear nowhere."""
    catalog = list(rows) if rows is not None else build_catalog_rows_cumulative(root)
    cluster_map: ClusterMap = consumer_topic_cluster_map(root)
    theme_map: ClusterMap = consumer_theme_cluster_map(root)

    label = ""
    roles: list[str | None] = []
    appears_in: list[CatalogEpisodeRow] = []
    people_by_id: dict[str, AppEntity] = {}
    topics_by_id: dict[str, AppTopic] = {}
    person_counts: Counter[str] = Counter()
    topic_counts: Counter[str] = Counter()

    for row, persons, _orgs, topics in _iter_kg_entities(root, catalog):
        match = next((p for p in persons if p.id == person_id), None)
        if match is None:
            continue
        if not label:
            label = match.name
        roles.append(match.role)
        appears_in.append(row)
        for p in persons:
            if p.id == person_id:
                continue
            people_by_id[p.id] = p
            person_counts[p.id] += 1
        for t in topics:
            topics_by_id[t.id] = t
            topic_counts[t.id] += 1

    if not appears_in:
        return None

    related_people = [people_by_id[i] for i, _ in person_counts.most_common(top_k)]
    related_topics = [
        _enrich_topic(topics_by_id[i], cluster_map, theme_map)
        for i, _ in topic_counts.most_common(top_k)
    ]
    return AppPersonCard(
        id=person_id,
        label=label or person_id.split(":", 1)[-1],
        role=_aggregate_role(roles),
        episode_count=len(appears_in),
        episodes=_sorted_episode_cards(root, appears_in),
        related_people=related_people,
        related_topics=related_topics,
    )


def build_topic_card(
    root: Path,
    topic_id: str,
    *,
    rows: Sequence[CatalogEpisodeRow] | None = None,
    top_k: int = _DEFAULT_TOP_K,
) -> AppTopicCard | None:
    """Project the topic's corpus footprint + cluster siblings to a card, or ``None`` if absent."""
    catalog = list(rows) if rows is not None else build_catalog_rows_cumulative(root)
    cluster_map: ClusterMap = consumer_topic_cluster_map(root)
    theme_map: ClusterMap = consumer_theme_cluster_map(root)

    label = ""
    about: list[CatalogEpisodeRow] = []
    people_by_id: dict[str, AppEntity] = {}
    person_counts: Counter[str] = Counter()

    for row, persons, _orgs, topics in _iter_kg_entities(root, catalog):
        match = next((t for t in topics if t.id == topic_id), None)
        if match is None:
            continue
        if not label:
            label = match.label
        about.append(row)
        for p in persons:
            people_by_id[p.id] = p
            person_counts[p.id] += 1

    if not about:
        return None

    related_people = [people_by_id[i] for i, _ in person_counts.most_common(top_k)]
    info = cluster_map.get(topic_id) or {}
    cid, clabel, csize = info.get("cluster_id"), info.get("cluster_label"), info.get("cluster_size")
    tinfo = theme_map.get(topic_id) or {}
    tcid, tclabel, tcsize = (
        tinfo.get("theme_cluster_id"),
        tinfo.get("theme_cluster_label"),
        tinfo.get("theme_cluster_size"),
    )
    siblings = [
        _enrich_topic(AppTopic(id=s["id"], label=s["label"]), cluster_map, theme_map)
        for s in consumer_cluster_siblings(root, topic_id)[:top_k]
    ]
    theme_siblings = [
        _enrich_topic(AppTopic(id=s["id"], label=s["label"]), cluster_map, theme_map)
        for s in consumer_theme_cluster_siblings(root, topic_id)[:top_k]
    ]
    return AppTopicCard(
        id=topic_id,
        label=label or topic_id.split(":", 1)[-1],
        cluster_id=cid if isinstance(cid, str) else None,
        cluster_label=clabel if isinstance(clabel, str) else None,
        cluster_size=csize if isinstance(csize, int) else 0,
        sibling_topics=siblings,
        theme_cluster_id=tcid if isinstance(tcid, str) else None,
        theme_cluster_label=tclabel if isinstance(tclabel, str) else None,
        theme_cluster_size=tcsize if isinstance(tcsize, int) else 0,
        theme_sibling_topics=theme_siblings,
        episode_count=len(about),
        episodes=_sorted_episode_cards(root, about),
        related_people=related_people,
    )


def _node_to_app_insight(node: dict[str, Any]) -> AppInsight:
    """Project a GI Insight node to AppInsight (grounded; quotes omitted here)."""
    props = node.get("properties") or {}
    text = props.get("text") or props.get("title") or ""
    conf = props.get("confidence")
    itype = props.get("insight_type") or props.get("type")
    phint = props.get("position_hint")
    return AppInsight(
        id=str(node.get("id") or ""),
        text=str(text),
        grounded=True,
        insight_type=str(itype) if isinstance(itype, str) and itype.strip() else None,
        confidence=float(conf) if isinstance(conf, (int, float)) else None,
        position_hint=str(phint) if phint is not None else None,
        quotes=[],
    )


def build_topic_perspectives(
    root: Path, topic_id: str, *, mine_slugs: set[str] | None = None
) -> AppTopicPerspectivesResponse | None:
    """Group a topic's grounded insights by speaker — one take per speaker (#1146).

    ``mine_slugs`` (scope=mine, #1149) restricts to episodes in the user's heard∪captured
    set; an empty set yields no perspectives (honest-empty). Returns ``None`` when the topic
    has no speaker-attributable insight in the (scoped) GI.
    """
    keep: set[str] | None = None
    if mine_slugs is not None:
        rows = build_catalog_rows_cumulative(root)
        keep = {
            r.episode_id
            for r in rows
            if r.episode_id and row_to_summary(root, r).slug in mine_slugs
        }
    groups = topic_perspectives(str(root), str(root), topic_id, keep_episode_ids=keep)
    if not groups:
        return None
    perspectives = [
        AppTopicPerspective(
            person_id=str(g["person_id"]),
            person_name=str(g["person_name"]),
            insight_count=int(g["insight_count"]),
            episode_count=int(g["episode_count"]),
            insights=[_node_to_app_insight(n) for n in g["insights"]],
        )
        for g in groups
    ]
    return AppTopicPerspectivesResponse(
        topic_id=topic_id,
        topic_label=topic_id.split(":", 1)[-1],
        perspective_count=len(perspectives),
        perspectives=perspectives,
    )
