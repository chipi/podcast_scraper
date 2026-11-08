"""Today's in-tree read-side code on the read hooks (E04 S3).

Each entry imports its code inside its body, so this module loads without it. An entry goes when
its code moves to an extension, which then contributes it through ``Extension.read_hooks``.
"""

from __future__ import annotations

from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from .read_hooks import NodeEpisodes, ReadHooks


def _segment_speaker(raw: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    from podcast_scraper.speaker_stage import display

    return display._segment_speaker(raw)


def _unidentified_label() -> str:
    from podcast_scraper.speaker_stage import display

    return display._UNIDENTIFIED_TEXT


def _ad_reads(raw: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    from podcast_scraper.insight_stage import read_side

    return read_side._detected_ad_reads(raw)


def _looks_like_publisher(name: str, language: Optional[str]) -> bool:
    from podcast_scraper.speaker_detectors.hosts import looks_like_publisher

    return looks_like_publisher(name, language)


def _entity_id_map(root: Union[Path, str]) -> Dict[str, str]:
    from podcast_scraper.kg.entity_clusters import cached_entity_id_map

    return cached_entity_id_map(root)


def _gi_integrity(root: Path) -> Dict[str, Any]:
    from podcast_scraper.gi.integrity import assess_gi_integrity

    return assess_gi_integrity(root)


def _topic_perspectives(
    root: Path,
    topic_ids: Sequence[str],
    *,
    keep_episode_ids: Optional[Set[str]],
    slug_by_episode: Dict[str, str],
) -> List[Dict[str, Any]]:
    from podcast_scraper.insight_stage.perspectives import speaker_perspectives

    return speaker_perspectives(
        root, topic_ids, keep_episode_ids=keep_episode_ids, slug_by_episode=slug_by_episode
    )


def _group_borrows(root: Path, rows: List[Any]) -> Tuple[Dict[str, str], Dict[str, str]]:
    from podcast_scraper.identity.web_borrows import _group_borrows as borrows

    return borrows(root, rows)


def _insights_from_gi(
    artifact: Any, *, limit: Optional[int], unattributed_fallback: bool, people: Any
) -> List[Any]:
    from podcast_scraper.insight_stage import gi_view

    return gi_view.insights_from_gi(
        artifact, limit=limit, unattributed_fallback=unattributed_fallback, people=people
    )


def _quote_speaker_name(
    artifact: Any, quote_props: Any, spoken_by_id: Any, people: Any
) -> Optional[str]:
    from podcast_scraper.insight_stage import gi_view

    return gi_view.quote_speaker_name(artifact, quote_props, spoken_by_id, people)


def _served_id_map(root: Path) -> Dict[str, str]:
    from podcast_scraper.identity.ledger import served_id_map

    return served_id_map(root)


def _bridge_node_episodes(
    root: str, anchor: str, node_id: str, *, max_episodes: Optional[int]
) -> NodeEpisodes:
    from podcast_scraper.insight_stage.cil_queries import (
        canonical_cil_entity_id,
        episodes_for_bridge_node_id,
    )

    rows, truncated, total = episodes_for_bridge_node_id(
        root, anchor, node_id, max_episodes=max_episodes
    )
    return canonical_cil_entity_id(node_id), rows, truncated, total


def _lift_transcript_hits(
    rows: List[Dict[str, Any]],
    corpus_root: Path,
    gi_paths: Mapping[str, Path],
    *,
    visible_for: Optional[Callable[[str], Callable[[str], bool]]],
    people: Any,
) -> None:
    from podcast_scraper.insight_stage.transcript_chunk_lift import lift_transcript_rows

    lift_transcript_rows(rows, corpus_root, gi_paths, visible_for=visible_for, people=people)


def _graph_artifacts(corpus_dir: Path, *, validate: bool) -> Iterator[Tuple[str, Dict[str, Any]]]:
    from podcast_scraper.gi.corpus import iter_gi_artifacts
    from podcast_scraper.gi.explore import scan_artifact_paths as scan_gi_paths
    from podcast_scraper.kg.corpus import newest_run_artifact_paths

    gi_paths = newest_run_artifact_paths(corpus_dir, scan_gi_paths(corpus_dir), ".gi.json")
    for _path, data in iter_gi_artifacts(gi_paths, validate=validate):
        yield "gi", data


def _host_variants(a: str, b: str) -> bool:
    from podcast_scraper.identity.person_match import host_variants

    return host_variants(a, b)


def ROUTERS() -> Sequence[Any]:
    """The operator-plane routers of the insight / identity stages: GI explore, person pages, CIL.

    All three are on the curated public operator surface, as they were in ``server.app``'s
    tuples."""
    from podcast_scraper.extensions import RouterMount
    from podcast_scraper.insight_stage.routes import cil, corpus_persons, explore

    return tuple(
        RouterMount(router=m.router, plane="operator", operator_public=True)
        for m in (explore, corpus_persons, cil)
    )


BUILTIN = ReadHooks(
    name="builtin",
    segment_speaker=_segment_speaker,
    unidentified_label=_unidentified_label,
    ad_reads=_ad_reads,
    looks_like_publisher=_looks_like_publisher,
    entity_id_map=_entity_id_map,
    gi_integrity=_gi_integrity,
    topic_perspectives=_topic_perspectives,
    group_borrows=_group_borrows,
    insights_from_gi=_insights_from_gi,
    quote_speaker_name=_quote_speaker_name,
    served_id_map=_served_id_map,
    bridge_node_episodes=_bridge_node_episodes,
    lift_transcript_hits=_lift_transcript_hits,
    graph_artifacts=_graph_artifacts,
    host_variants=_host_variants,
)
