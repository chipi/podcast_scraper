"""Inverted KG entity index for the consumer relational cards (perf remediation, follow-up).

``build_person_card`` / ``build_topic_card`` / ``resolve_entity`` each need "which episodes does
this entity appear in, and who/what co-occurs there". The straightforward implementation parsed
**every** episode ``*.kg.json`` on every request (``_iter_kg_entities`` over the whole catalog) —
O(corpus) JSON parsing per card, the last O(corpus) hot spot after the catalog cache landed.

This builds that projection **once per ingest** and caches it on the shared :mod:`perf_cache`
(corpus-mtime token, same as the catalog cache): a per-KG-episode entity list plus an inverted
``entity_id → episode indices`` map and a ``normalized-label → ref`` map for search. A card request
then reads only the episodes an entity is in — O(matches) not O(corpus) — with no KG file re-read.

The one full KG pass is paid by the first card request after an ingest (the same parse cost one card
paid before), then amortized to zero until the next ingest. Entities are shared read-only, the same
convention as :func:`app_catalog_cache.cached_catalog`.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, NamedTuple, Sequence

from podcast_scraper import perf_cache
from podcast_scraper.server.app_catalog_cache import cached_catalog
from podcast_scraper.server.app_corpus_access import load_json_artifact
from podcast_scraper.server.app_kg_view import entities_from_kg
from podcast_scraper.server.corpus_catalog import CatalogEpisodeRow
from podcast_scraper.server.schemas import AppEntity, AppEntityRef, AppTopic

_INDEX_NS = "app_kg_entity_index"


def normalize_label(text: str) -> str:
    """Fold a label/query to a comparison key: punctuation→space, collapse, lower.

    "Matthew Walker." / "matthew-walker" / "MATTHEW  WALKER" all map to "matthew walker",
    giving exact/near-exact matching (case / punctuation / spacing insensitive) without the
    false positives of fuzzy distance matching.
    """
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text)).strip().lower()


def iter_kg_entities(
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


class EpisodeEntities(NamedTuple):
    """One KG episode's card-relevant entities (orgs dropped — no card reads them)."""

    row: CatalogEpisodeRow
    persons: list[AppEntity]
    topics: list[AppTopic]


@dataclass(frozen=True)
class KgEntityIndex:
    """The corpus's KG entities, inverted for O(matches) card lookups.

    ``episodes`` is in catalog order; the ``*_to_eps`` maps hold indices into it in that same order,
    so a consumer iterating an entity's episodes sees them in the exact order the old full-catalog
    scan did (co-occurrence counts + first-seen label wins stay identical).
    """

    episodes: list[EpisodeEntities]
    person_to_eps: dict[str, list[int]]
    topic_to_eps: dict[str, list[int]]
    person_ref_by_norm: dict[str, AppEntityRef]
    topic_ref_by_norm: dict[str, AppEntityRef]

    def person_episodes(self, person_id: str) -> list[EpisodeEntities]:
        """Episodes ``person_id`` appears in, in catalog order (empty when unknown)."""
        return [self.episodes[i] for i in self.person_to_eps.get(person_id, ())]

    def topic_episodes(self, topic_id: str) -> list[EpisodeEntities]:
        """Episodes about ``topic_id``, in catalog order (empty when unknown)."""
        return [self.episodes[i] for i in self.topic_to_eps.get(topic_id, ())]


def build_kg_index(root: Path) -> KgEntityIndex:
    """One full pass over the corpus KGs → the inverted index (called once per ingest via cache)."""
    episodes: list[EpisodeEntities] = []
    person_to_eps: dict[str, list[int]] = defaultdict(list)
    topic_to_eps: dict[str, list[int]] = defaultdict(list)
    person_ref_by_norm: dict[str, AppEntityRef] = {}
    topic_ref_by_norm: dict[str, AppEntityRef] = {}

    for row, persons, _orgs, topics in iter_kg_entities(root, cached_catalog(root)):
        i = len(episodes)
        episodes.append(EpisodeEntities(row=row, persons=persons, topics=topics))
        for p in persons:
            person_to_eps[p.id].append(i)
            person_ref_by_norm.setdefault(
                normalize_label(p.name), AppEntityRef(id=p.id, kind="person", label=p.name)
            )
        for t in topics:
            topic_to_eps[t.id].append(i)
            topic_ref_by_norm.setdefault(
                normalize_label(t.label), AppEntityRef(id=t.id, kind="topic", label=t.label)
            )

    return KgEntityIndex(
        episodes=episodes,
        person_to_eps=dict(person_to_eps),
        topic_to_eps=dict(topic_to_eps),
        person_ref_by_norm=person_ref_by_norm,
        topic_ref_by_norm=topic_ref_by_norm,
    )


def get_kg_index(root: Path) -> KgEntityIndex:
    """The KG entity index, cached by corpus mtime (built once per ingest); shared read-only."""
    index: KgEntityIndex = perf_cache.get_or_compute(
        _INDEX_NS,
        str(Path(root).resolve()),
        perf_cache.corpus_mtime(root),
        lambda: build_kg_index(root),
    )
    return index
