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

import logging
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
    """One KG episode's card-relevant entities (persons, topics, and orgs — #2031)."""

    row: CatalogEpisodeRow
    persons: list[AppEntity]
    topics: list[AppTopic]
    orgs: list[AppEntity]


logger = logging.getLogger(__name__)


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
    org_to_eps: dict[str, list[int]]
    person_ref_by_norm: dict[str, AppEntityRef]
    topic_ref_by_norm: dict[str, AppEntityRef]
    org_ref_by_norm: dict[str, AppEntityRef]

    def person_episodes(self, person_id: str) -> list[EpisodeEntities]:
        """Episodes ``person_id`` appears in, in catalog order (empty when unknown)."""
        return [self.episodes[i] for i in self.person_to_eps.get(person_id, ())]

    def topic_episodes(self, topic_id: str) -> list[EpisodeEntities]:
        """Episodes about ``topic_id``, in catalog order (empty when unknown)."""
        return [self.episodes[i] for i in self.topic_to_eps.get(topic_id, ())]

    def org_episodes(self, org_id: str) -> list[EpisodeEntities]:
        """Episodes mentioning ``org_id``, in catalog order (empty when unknown) — #2031."""
        return [self.episodes[i] for i in self.org_to_eps.get(org_id, ())]


def _canonical_person_ids(root: Path) -> dict[str, str]:
    """``{variant_id: canonical_id}`` for people, or ``{}`` if the map cannot be built.

    #2056: the consumer cards showed ``Theo Jaffee`` and ``Theo Jaffe`` as two people on *The a16z
    Show*, and ``Lucas Kaiser`` beside ``Lukasz Kaiser``. The resolver already matches those pairs
    and ``same_show_required=True`` already permits them — the two ids share a show. Nothing was
    wrong with the matcher. It was simply never consulted on this surface: ``iter_kg_entities``
    reads ids straight out of each ``*.kg.json``, and only ``search/corpus_graph`` and
    ``server/cil_queries`` apply ``build_entity_id_map``.

    IT IS NOT FREE, and an earlier version of this docstring said it was ("costs nothing per
    request"). The map is a full corpus scan — 90 seconds over the 2,257-episode production
    snapshot — and ``perf_cache`` runs ``compute()`` outside its lock, so on a MISS every arriving
    request thread started its own. It goes through ``cached_entity_id_map`` now: one shared,
    single-flighted cache for all three surfaces that need it, warmed by ``app_cache_warm`` so the
    card path sees a hit rather than a 90-second build.

    Failure is non-fatal and returns ``{}``: an un-canonicalised index shows a duplicate, which is
    the status quo. Failing the whole index would take the cards down with it.
    """
    try:
        from ..kg.entity_clusters import cached_entity_id_map

        return {k: v for k, v in cached_entity_id_map(root).items() if k.startswith("person:")}
    except Exception:  # noqa: BLE001 — a duplicate person must not cost the reader their cards
        logger.warning("entity canonicalisation unavailable; cards may show variant duplicates")
        return {}


def build_kg_index(root: Path) -> KgEntityIndex:
    """One full pass over the corpus KGs → the inverted index (called once per ingest via cache)."""
    canonical = _canonical_person_ids(root)
    episodes: list[EpisodeEntities] = []
    person_to_eps: dict[str, list[int]] = defaultdict(list)
    topic_to_eps: dict[str, list[int]] = defaultdict(list)
    org_to_eps: dict[str, list[int]] = defaultdict(list)
    person_ref_by_norm: dict[str, AppEntityRef] = {}
    topic_ref_by_norm: dict[str, AppEntityRef] = {}
    org_ref_by_norm: dict[str, AppEntityRef] = {}

    for row, persons, orgs, topics in iter_kg_entities(root, cached_catalog(root)):
        i = len(episodes)
        if canonical:
            # Rewrite to the canonical id BEFORE the node lands in the projection, and dedupe
            # within the episode: two variant spellings in one episode would otherwise become the
            # same id twice, listing the person on their own card as two co-appearances.
            seen: set[str] = set()
            rewritten = []
            for p in persons:
                cid = canonical.get(p.id, p.id)
                if cid in seen:
                    continue
                seen.add(cid)
                rewritten.append(p if cid == p.id else p.model_copy(update={"id": cid}))
            persons = rewritten
        episodes.append(EpisodeEntities(row=row, persons=persons, topics=topics, orgs=orgs))
        for p in persons:
            person_to_eps[p.id].append(i)
            # Both spellings are registered so a search for EITHER finds the surviving person —
            # the variant label is how the reader knows them, even once its id is gone.
            person_ref_by_norm.setdefault(
                normalize_label(p.name), AppEntityRef(id=p.id, kind="person", label=p.name)
            )
        for t in topics:
            topic_to_eps[t.id].append(i)
            topic_ref_by_norm.setdefault(
                normalize_label(t.label), AppEntityRef(id=t.id, kind="topic", label=t.label)
            )
        for o in orgs:
            org_to_eps[o.id].append(i)
            org_ref_by_norm.setdefault(
                normalize_label(o.name), AppEntityRef(id=o.id, kind="organization", label=o.name)
            )

    return KgEntityIndex(
        episodes=episodes,
        person_to_eps=dict(person_to_eps),
        topic_to_eps=dict(topic_to_eps),
        org_to_eps=dict(org_to_eps),
        person_ref_by_norm=person_ref_by_norm,
        topic_ref_by_norm=topic_ref_by_norm,
        org_ref_by_norm=org_ref_by_norm,
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
