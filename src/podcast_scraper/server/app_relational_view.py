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
from typing import Iterator, Sequence

from podcast_scraper.search.topic_clusters import (
    consumer_cluster_siblings,
    consumer_topic_cluster_map,
)
from podcast_scraper.server.app_content_source import row_to_summary
from podcast_scraper.server.app_corpus_access import load_json_artifact
from podcast_scraper.server.app_kg_view import entities_from_kg
from podcast_scraper.server.corpus_catalog import (
    build_catalog_rows_cumulative,
    CatalogEpisodeRow,
)
from podcast_scraper.server.schemas import (
    AppEntity,
    AppEntityRef,
    AppEpisodeSummary,
    AppPersonCard,
    AppTopic,
    AppTopicCard,
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


def _enrich_topic(topic: AppTopic, cluster_map: ClusterMap) -> AppTopic:
    """Attach corpus-cluster identity to a topic (no-op when the topic is unclustered)."""
    info = cluster_map.get(topic.id)
    return topic.model_copy(update=info) if info else topic


def _sorted_episode_cards(root: Path, rows: list[CatalogEpisodeRow]) -> list[AppEpisodeSummary]:
    """Project rows to episode cards, newest-first (undated episodes sort last)."""
    cards = [row_to_summary(root, r) for r in rows]
    cards.sort(key=lambda c: (c.publish_date is not None, c.publish_date or ""), reverse=True)
    return cards


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

    label = ""
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
        _enrich_topic(topics_by_id[i], cluster_map) for i, _ in topic_counts.most_common(top_k)
    ]
    return AppPersonCard(
        id=person_id,
        label=label or person_id.split(":", 1)[-1],
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
    siblings = [
        _enrich_topic(AppTopic(id=s["id"], label=s["label"]), cluster_map)
        for s in consumer_cluster_siblings(root, topic_id)[:top_k]
    ]
    return AppTopicCard(
        id=topic_id,
        label=label or topic_id.split(":", 1)[-1],
        cluster_id=cid if isinstance(cid, str) else None,
        cluster_label=clabel if isinstance(clabel, str) else None,
        cluster_size=csize if isinstance(csize, int) else 0,
        sibling_topics=siblings,
        episode_count=len(about),
        episodes=_sorted_episode_cards(root, about),
        related_people=related_people,
    )
