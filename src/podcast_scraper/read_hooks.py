"""Where the server's read side reaches a stage it does not own (E04 S3).

Apart from :mod:`podcast_scraper.stage_hooks`, so the pipeline's registry does not import the
server's read models (the pipeline does not show anything; a request does).

Each point has one owner: the built-in one (:mod:`.builtin_read_hooks`, today's in-tree code),
else the first installed extension whose ``read_hooks`` sets it, else the plain default below.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

#: ``(canonical_id, rows, truncated, total_matched)``.
NodeEpisodes = Tuple[str, List[Dict[str, Any]], bool, Optional[int]]


@dataclass(frozen=True)
class ReadHooks:
    """The read-side hooks one owner contributes; every part is optional."""

    name: str
    #: ``(raw_segment) -> (kind, text)``: how a transcript segment's speaker is shown.
    segment_speaker: Optional[Callable[[Dict[str, Any]], Tuple[Optional[str], Optional[str]]]] = (
        None
    )
    #: ``() -> text``: the label an unidentified voice is numbered under.
    unidentified_label: Optional[Callable[[], str]] = None
    #: ``(raw_segments) -> [(start_ms, end_ms)]``: the ad reads a clip must never play.
    ad_reads: Optional[Callable[[List[Dict[str, Any]]], List[Tuple[int, int]]]] = None
    #: ``(name, language) -> bool``: whether a name is a publisher rather than a person.
    looks_like_publisher: Optional[Callable[[str, Optional[str]], bool]] = None
    #: ``(corpus_root) -> {entity_id: canonical_id}``: joined entities, for de-duplicated cards.
    entity_id_map: Optional[Callable[[Union[Path, str]], Dict[str, str]]] = None
    #: ``(corpus_root) -> assessment``: the GI integrity report the ops route returns.
    gi_integrity: Optional[Callable[[Path], Dict[str, Any]]] = None
    #: ``(root, topic_ids, *, keep_episode_ids, slug_by_episode) -> [group]``: what speakers said
    #: about the topics, one group per speaker with its insights as ``AppInsight``.
    topic_perspectives: Optional[Callable[..., List[Dict[str, Any]]]] = None
    #: ``(root, person_rows) -> (bio_from, photo_from)``: which joined member lends its bio / photo.
    group_borrows: Optional[Callable[[Path, List[Any]], Tuple[Dict[str, str], Dict[str, str]]]] = (
        None
    )
    #: ``(gi_artifact, *, limit, unattributed_fallback, people) -> [AppInsight]``: an episode's
    #: insights as its surfaces show them.
    insights_from_gi: Optional[Callable[..., List[Any]]] = None
    #: ``(gi_artifact, quote_props, spoken_by_id, people) -> name``: who a quote names as speaker.
    quote_speaker_name: Optional[Callable[..., Optional[str]]] = None
    #: ``(root) -> {entity_id: canonical_id}``: the identity map the server applies (ADR-166).
    served_id_map: Optional[Callable[[Path], Dict[str, str]]] = None
    #: ``(root, anchor, node_id, *, max_episodes) -> (canonical_id, rows, truncated, total)``: the
    #: episodes whose cross-layer bridge lists a node.
    bridge_node_episodes: Optional[Callable[..., NodeEpisodes]] = None
    #: ``(rows, corpus_root, gi_paths, *, visible_for, people) -> None``: set ``lifted`` on the
    #: transcript hits a GI quote covers (in place).
    lift_transcript_hits: Optional[Callable[..., None]] = None
    #: ``(corpus_dir, *, validate) -> [(source, artifact)]``: graph layers beyond KG, ingested
    #: after it (GI: newest run per episode).
    graph_artifacts: Optional[Callable[..., Iterable[Tuple[str, Dict[str, Any]]]]] = None
    #: ``(name_a, name_b) -> bool``: whether two host spellings name one person.
    host_variants: Optional[Callable[[str, str], bool]] = None
    #: ``(corpus_root) -> PersonDisplay``: the display builder (``identity.display``); without an
    #: owner, ``plain_display``, where every person shows their own episode's name.
    person_display_builder: Optional[Callable[[Path], Any]] = None


def _all() -> Iterator[ReadHooks]:
    from .builtin_read_hooks import BUILTIN
    from .extensions import load_extensions

    yield BUILTIN
    for ext in load_extensions():
        if ext.read_hooks is not None:
            yield ext.read_hooks


def _owner(field: str) -> Any:
    for hooks in _all():
        fn = getattr(hooks, field)
        if fn is not None:
            return fn
    return None


def person_display_builder() -> Optional[Callable[[Path], Any]]:
    """The owner's display builder, or ``None`` (``identity.display`` then shows each person's own
    episode name)."""
    fn: Optional[Callable[[Path], Any]] = _owner("person_display_builder")
    return fn


def segment_speaker(raw: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """How a segment's speaker is shown; without a speaker stage, no speaker."""
    fn = _owner("segment_speaker")
    return fn(raw) if fn is not None else (None, None)


def unidentified_label() -> str:
    """The label an unidentified voice is numbered under."""
    fn = _owner("unidentified_label")
    return fn() if fn is not None else "Unidentified speaker"


def ad_reads(raw: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    """The ad reads a clip must never play; without an ad stage, none are known."""
    fn = _owner("ad_reads")
    return fn(raw) if fn is not None else []


def looks_like_publisher(name: str, language: Optional[str] = None) -> bool:
    """Whether *name* is a publisher, read in *language*; without a speaker stage, none is."""
    fn = _owner("looks_like_publisher")
    return bool(fn(name, language)) if fn is not None else False


def entity_id_map(root: Union[Path, str]) -> Dict[str, str]:
    """Joined entities' canonical ids; without the identity stage, none (each shows as itself)."""
    fn = _owner("entity_id_map")
    return fn(root) if fn is not None else {}


def gi_integrity(root: Path) -> Optional[Dict[str, Any]]:
    """The GI integrity report, or None when no stage builds GI."""
    fn = _owner("gi_integrity")
    return fn(root) if fn is not None else None


def topic_perspectives(
    root: Path,
    topic_ids: Sequence[str],
    *,
    keep_episode_ids: Optional[Set[str]],
    slug_by_episode: Dict[str, str],
) -> List[Dict[str, Any]]:
    """What speakers said about the topics; without an insight stage, nothing (honest-empty)."""
    fn = _owner("topic_perspectives")
    if fn is None:
        return []
    out: List[Dict[str, Any]] = fn(
        root, topic_ids, keep_episode_ids=keep_episode_ids, slug_by_episode=slug_by_episode
    )
    return out


def group_borrows(root: Path, rows: List[Any]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Which joined member lends its bio / photo; without the identity stage, none is borrowed."""
    fn = _owner("group_borrows")
    return fn(root, rows) if fn is not None else ({}, {})


def insights_from_gi(
    artifact: Any,
    *,
    limit: Optional[int] = None,
    unattributed_fallback: bool = False,
    people: Any = None,
) -> List[Any]:
    """An episode's insights as its surfaces show them; without an insight stage, none."""
    fn = _owner("insights_from_gi")
    if fn is None:
        return []
    out: List[Any] = fn(
        artifact, limit=limit, unattributed_fallback=unattributed_fallback, people=people
    )
    return out


def quote_speaker_name(
    artifact: Any, quote_props: Any, spoken_by_id: Any = None, people: Any = None
) -> Optional[str]:
    """Who a quote names as its speaker; without an insight stage, nobody."""
    fn = _owner("quote_speaker_name")
    return fn(artifact, quote_props, spoken_by_id, people) if fn is not None else None


def served_id_map(root: Path) -> Dict[str, str]:
    """The identity map the server applies; without the identity stage, no joins."""
    fn = _owner("served_id_map")
    return fn(root) if fn is not None else {}


def bridge_node_episodes(
    root: str, anchor: str, node_id: str, *, max_episodes: Optional[int] = None
) -> NodeEpisodes:
    """Episodes whose bridge lists *node_id*; without the cross-layer stage, the id as given and
    none."""
    fn = _owner("bridge_node_episodes")
    if fn is None:
        return node_id, [], False, None
    out: NodeEpisodes = fn(root, anchor, node_id, max_episodes=max_episodes)
    return out


def lift_transcript_hits(
    rows: List[Dict[str, Any]],
    corpus_root: Path,
    gi_paths: Mapping[str, Path],
    *,
    visible_for: Optional[Callable[[str], Callable[[str], bool]]] = None,
    people: Any = None,
) -> None:
    """Lift transcript hits to the insights that quote them; without an insight stage, none are."""
    fn = _owner("lift_transcript_hits")
    if fn is not None:
        fn(rows, corpus_root, gi_paths, visible_for=visible_for, people=people)


def graph_artifacts(
    corpus_dir: Path, *, validate: bool = False
) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """Graph layers beyond KG, in ingest order and streamed (never held together); without an
    insight stage, none (the graph is KG)."""
    fn = _owner("graph_artifacts")
    if fn is None:
        return ()
    out: Iterable[Tuple[str, Dict[str, Any]]] = fn(corpus_dir, validate=validate)
    return out


def host_variants(a: str, b: str) -> bool:
    """Whether two host spellings name one person; without the identity stage, never."""
    fn = _owner("host_variants")
    return bool(fn(a, b)) if fn is not None else False


def builtin_routers() -> Sequence[Any]:
    """The in-tree stage routers (``RouterMount``), mounted like an extension's, before them."""
    from .builtin_read_hooks import ROUTERS

    return ROUTERS()


__all__ = [
    "ReadHooks",
    "NodeEpisodes",
    "ad_reads",
    "bridge_node_episodes",
    "builtin_routers",
    "entity_id_map",
    "gi_integrity",
    "graph_artifacts",
    "group_borrows",
    "host_variants",
    "insights_from_gi",
    "lift_transcript_hits",
    "looks_like_publisher",
    "quote_speaker_name",
    "segment_speaker",
    "served_id_map",
    "topic_perspectives",
    "unidentified_label",
]
