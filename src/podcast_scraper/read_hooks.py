"""Where the server's read side reaches a stage it does not own (E04 S3).

Apart from :mod:`podcast_scraper.stage_hooks` for the same reason as :mod:`.cli_hooks`: corpus
migrations reach the pipeline's registry, and a migration's measured code is everything it
reaches. What a request shows is not something a migration runs.

Each point has one owner: the built-in one (:mod:`.builtin_read_hooks`, today's in-tree code),
else the first installed extension whose ``read_hooks`` sets it, else the plain default below.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple


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
    entity_id_map: Optional[Callable[[Path], Dict[str, str]]] = None
    #: ``(corpus_root) -> assessment``: the GI integrity report the ops route returns.
    gi_integrity: Optional[Callable[[Path], Dict[str, Any]]] = None
    #: ``(root, topic_ids, *, keep_episode_ids, slug_by_episode) -> [group]``: what speakers said
    #: about the topics, one group per speaker with its insights as ``AppInsight``.
    topic_perspectives: Optional[Callable[..., List[Dict[str, Any]]]] = None
    #: ``(root, person_rows) -> (bio_from, photo_from)``: which joined member lends its bio / photo.
    group_borrows: Optional[Callable[[Path, List[Any]], Tuple[Dict[str, str], Dict[str, str]]]] = (
        None
    )


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


def entity_id_map(root: Path) -> Dict[str, str]:
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


__all__ = [
    "ReadHooks",
    "ad_reads",
    "entity_id_map",
    "gi_integrity",
    "group_borrows",
    "looks_like_publisher",
    "segment_speaker",
    "topic_perspectives",
    "unidentified_label",
]
