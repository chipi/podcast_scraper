"""Themes and storylines, read through whichever installed extension provides them (ADR-162).

A THEME (``tc:``) groups topics that mean the same thing; a STORYLINE (``thc:``) groups topics that
keep coming up together. Both are private features. The platform's search, index, read models and
audits ask this module, never the implementation: with no extension installed every reader returns
nothing, the builder builds nothing, and the search operators are not offered.

The names are the implementation's own, so a caller reads the same as it did when it imported
``search.topic_clusters`` and ``search.storylines`` directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from podcast_scraper.extensions import load_extensions

#: The aux-row ``doc_type`` a storyline is indexed under; a row-schema name, owned by the platform.
STORYLINE_DOC_TYPE = "storyline"

#: The themes artifact, under ``<corpus>/search/``.
TOPIC_CLUSTERS_FILENAME = "topic_clusters.json"

#: The storylines artifact, relative to the corpus root.
STORYLINES_REL = os.path.join("enrichments", "topic_theme_clusters.json")

#: ``(doc_id, text, metadata)``, as the indexer embeds an aux row.
IndexRow = Tuple[str, str, Dict[str, Any]]

#: ``(hits, corpus_root) -> rows``. A hit is ``{"doc_id": ..., "metadata": {...}}``.
SearchOperator = Callable[[List[Dict[str, Any]], Path], List[Dict[str, Any]]]


def _no_map(corpus_root: Path) -> Dict[str, Dict[str, Any]]:
    return {}


def _no_siblings(corpus_root: Path, topic_id: str) -> List[Dict[str, str]]:
    return []


def _no_top(corpus_root: Path, top_n: int = 12, **kwargs: Any) -> List[Dict[str, Any]]:
    return []


def _no_payload(corpus_root: Path) -> Optional[Dict[str, Any]]:
    return None


def _no_lift(corpus_root: Path) -> Dict[str, Dict[str, float]]:
    return {}


def _no_episodes(corpus_root: Path) -> Dict[str, frozenset]:
    return {}


def _no_rows(corpus_root: Path) -> List[IndexRow]:
    return []


def _no_build(output_dir: str | Path, **kwargs: Any) -> Optional[Dict[str, Any]]:
    return None


@dataclass(frozen=True)
class TopicGroupings:
    """What an extension provides for themes and storylines. Each reader takes the corpus root."""

    theme_map_by_topic: Callable[[Path], Dict[str, Dict[str, Any]]] = _no_map
    theme_siblings_by_topic: Callable[[Path, str], List[Dict[str, str]]] = _no_siblings
    top_themes_by_member_count: Callable[..., List[Dict[str, Any]]] = _no_top
    #: ``topic_id -> {cluster_id, cluster_label, ...}`` as search hits carry it.
    load_theme_enrichment_map: Callable[[Path], Dict[str, Dict[str, Any]]] = _no_map
    #: The raw themes artifact, for audits that need more than the summaries.
    load_theme_payload: Callable[[Path], Optional[Dict[str, Any]]] = _no_payload
    #: ``(output_dir, *, index_dir=, threshold=) -> payload``; builds the themes artifact.
    build_topic_clusters_for_corpus: Callable[..., Optional[Dict[str, Any]]] = _no_build

    storyline_map_by_topic: Callable[[Path], Dict[str, Dict[str, Any]]] = _no_map
    storyline_siblings_by_topic: Callable[[Path, str], List[Dict[str, str]]] = _no_siblings
    top_storylines_by_member_count: Callable[..., List[Dict[str, Any]]] = _no_top
    storyline_member_lift: Callable[[Path], Dict[str, Dict[str, float]]] = _no_lift
    storyline_episode_ids: Callable[[Path], Dict[str, frozenset]] = _no_episodes
    #: Aux rows the indexer embeds under :data:`STORYLINE_DOC_TYPE`.
    storyline_index_rows: Callable[[Path], List[IndexRow]] = _no_rows

    #: Result-set operators ``/api/search`` offers, by the name a caller passes as ``operator``.
    search_operators: Callable[[], Dict[str, SearchOperator]] = dict


_NONE = TopicGroupings()


def installed() -> TopicGroupings:
    """The first installed extension's groupings, else the empty ones."""
    for ext in load_extensions():
        if ext.groupings is not None:
            return ext.groupings
    return _NONE


def available() -> bool:
    """True when an installed extension provides themes and storylines."""
    return installed() is not _NONE


def theme_map_by_topic(corpus_root: Path) -> Dict[str, Dict[str, Any]]:
    """``topic_id -> theme`` for every topic in a theme."""
    return installed().theme_map_by_topic(corpus_root)


def theme_siblings_by_topic(corpus_root: Path, topic_id: str) -> List[Dict[str, str]]:
    """The other topics in *topic_id*'s theme."""
    return installed().theme_siblings_by_topic(corpus_root, topic_id)


def top_themes_by_member_count(corpus_root: Path, top_n: int = 12) -> List[Dict[str, Any]]:
    """The largest themes, ``{id, label, size}``."""
    return installed().top_themes_by_member_count(corpus_root, top_n)


def load_theme_enrichment_map(corpus_root: Path) -> Dict[str, Dict[str, Any]]:
    """The theme fields a ``kg_topic`` search hit is decorated with, by topic id."""
    return installed().load_theme_enrichment_map(corpus_root)


def load_theme_payload(corpus_root: Path) -> Optional[Dict[str, Any]]:
    """The themes artifact as stored, or ``None``."""
    return installed().load_theme_payload(corpus_root)


def build_topic_clusters_for_corpus(
    output_dir: str | Path, **kwargs: Any
) -> Optional[Dict[str, Any]]:
    """Build the themes artifact; ``None`` when no extension provides themes."""
    return installed().build_topic_clusters_for_corpus(output_dir, **kwargs)


def storyline_map_by_topic(corpus_root: Path) -> Dict[str, Dict[str, Any]]:
    """``topic_id -> storyline`` for every topic in a storyline."""
    return installed().storyline_map_by_topic(corpus_root)


def storyline_siblings_by_topic(corpus_root: Path, topic_id: str) -> List[Dict[str, str]]:
    """The other topics in *topic_id*'s storyline."""
    return installed().storyline_siblings_by_topic(corpus_root, topic_id)


def top_storylines_by_member_count(
    corpus_root: Path, top_n: int = 12, **kwargs: Any
) -> List[Dict[str, Any]]:
    """The largest storylines (``min_members=`` passes through)."""
    return installed().top_storylines_by_member_count(corpus_root, top_n, **kwargs)


def storyline_member_lift(corpus_root: Path) -> Dict[str, Dict[str, float]]:
    """``storyline_id -> {topic_id: lift}``."""
    return installed().storyline_member_lift(corpus_root)


def storyline_episode_ids(corpus_root: Path) -> Dict[str, frozenset]:
    """``storyline_id -> episode ids`` it draws on."""
    return installed().storyline_episode_ids(corpus_root)


def storyline_index_rows(corpus_root: Path) -> List[IndexRow]:
    """The storyline rows the indexer embeds."""
    return installed().storyline_index_rows(corpus_root)


def search_operators() -> Dict[str, SearchOperator]:
    """Result-set operators by name; empty when none is installed."""
    return installed().search_operators()


__all__ = [
    "IndexRow",
    "STORYLINE_DOC_TYPE",
    "STORYLINES_REL",
    "SearchOperator",
    "TOPIC_CLUSTERS_FILENAME",
    "TopicGroupings",
    "available",
    "build_topic_clusters_for_corpus",
    "installed",
    "load_theme_enrichment_map",
    "load_theme_payload",
    "search_operators",
    "storyline_episode_ids",
    "storyline_index_rows",
    "storyline_map_by_topic",
    "storyline_member_lift",
    "storyline_siblings_by_topic",
    "theme_map_by_topic",
    "theme_siblings_by_topic",
    "top_storylines_by_member_count",
    "top_themes_by_member_count",
]
