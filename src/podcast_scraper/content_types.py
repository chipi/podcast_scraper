"""Content types and transports: what an item is, and how its source lists new ones (ADR-177).

A **content type** names its id scheme, its stage profile (which stages run for it), its prompt
profile and its viewer card kind. A **transport** lists the new items of a source and owns what
that takes: its cursor, its request headers and its per-host rate limit.

Podcast is the first, built-in type, over RSS enclosures; its path is ``run_pipeline``, unchanged.
Installed extensions add types and transports through ``Extension.content_types`` and
``Extension.transports``; a name registered twice is refused when the registry loads.

``run_source`` is the one dispatch point both multi-feed loops (``cli``, ``service``) call per
source: ``run_pipeline`` for a podcast, the text-in entry point for a type whose items arrive as
text.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

PODCAST_EPISODE = "podcast_episode"
RSS_ENCLOSURE = "rss_enclosure"

#: The default for every shared read: podcast episodes only.
PODCAST_ONLY: FrozenSet[str] = frozenset({PODCAST_EPISODE})


def item_content_type(doc: Any) -> str:
    """The type a metadata document declares; none declared means a podcast episode."""
    if isinstance(doc, dict):
        declared = doc.get("content_type")
        if isinstance(declared, str) and declared.strip():
            return declared.strip()
    return PODCAST_EPISODE


def is_wanted(doc: Any, content_types: Collection[str]) -> bool:
    """Whether a reader that takes ``content_types`` reads this document."""
    return item_content_type(doc) in content_types


@dataclass(frozen=True)
class StageProfile:
    """The stages a type runs. ``stages`` of ``None`` is the full podcast pipeline."""

    name: str
    stages: Optional[FrozenSet[str]] = None

    def runs(self, stage: str) -> bool:
        """True when this profile runs ``stage`` (the full pipeline runs every stage)."""
        return self.stages is None or stage in self.stages


@dataclass(frozen=True)
class ContentType:
    """One kind of item: its id scheme, stage profile, prompt profile and viewer card kind."""

    name: str
    #: ``episode`` for podcast ids, which never change (T-8); ``item`` for new types.
    id_scheme: str
    stage_profile: StageProfile
    #: The transport a source of this type uses when it names none.
    default_transport: str
    #: A prompt-name prefix or directory for this type's prompts (``None``: the stage's own).
    prompt_profile: Optional[str] = None
    card_kind: str = ""


@dataclass(frozen=True)
class SourceItem:
    """One new item a transport listed: its stable id, and its text or a URL to fetch."""

    item_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    text: Optional[str] = None
    url: Optional[str] = None


#: ``(source, cursor) -> (items, next_cursor)``: the new items after ``cursor``.
ListItems = Callable[[Any, Optional[str]], Tuple[List[SourceItem], Optional[str]]]


@dataclass(frozen=True)
class Transport:
    """How a source's new items are listed. ``list_items`` of ``None``: the type's own runner
    fetches them (podcast: ``run_pipeline`` reads the feed itself)."""

    name: str
    list_items: Optional[ListItems] = None
    request_headers: Mapping[str, str] = field(default_factory=dict)
    #: Minimum seconds between two requests to one host (0: the run's own pacing).
    min_host_interval_s: float = 0.0


PODCAST = ContentType(
    name=PODCAST_EPISODE,
    id_scheme="episode",
    stage_profile=StageProfile(name="podcast"),
    default_transport=RSS_ENCLOSURE,
    card_kind="episode",
)
BUILTIN_TYPES: Tuple[ContentType, ...] = (PODCAST,)
BUILTIN_TRANSPORTS: Tuple[Transport, ...] = (Transport(name=RSS_ENCLOSURE),)


class RegistryError(ValueError):
    """A content type or transport registered twice, or a type naming an unknown transport."""


class UnknownContentType(KeyError):
    """A source names a content type nothing registered."""


class NoTextRunner(RuntimeError):
    """A non-podcast source reached the dispatch before a text-in runner exists for it."""


@dataclass(frozen=True)
class ItemContext:
    """The item a processing job is about: its type, id and body file (ADR-177 decision 7).

    A podcast job carries none and reads as :data:`PODCAST_ITEM`. Hooks gain no type argument:
    one that needs the type reads :func:`current_item` while the job is processed.
    """

    content_type: ContentType
    item_id: Optional[str] = None
    body_path: Optional[str] = None


PODCAST_ITEM = ItemContext(content_type=PODCAST)

_CURRENT_ITEM: ContextVar[ItemContext] = ContextVar("current_item", default=PODCAST_ITEM)


def current_item() -> ItemContext:
    """The item being processed on this thread; a podcast when no job set one."""
    return _CURRENT_ITEM.get()


@contextmanager
def processing_item(item: Optional[ItemContext]) -> Iterator[ItemContext]:
    """Make ``item`` the current item for the block (``None``: a podcast, as before)."""
    ctx = item or PODCAST_ITEM
    token = _CURRENT_ITEM.set(ctx)
    try:
        yield ctx
    finally:
        _CURRENT_ITEM.reset(token)


def _merge(builtins: Sequence[Any], extra: Sequence[Any], what: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for item in [*builtins, *extra]:
        if item.name in out:
            raise RegistryError(f"{what} {item.name!r} is registered twice")
        out[item.name] = item
    return out


def _from_extensions() -> Tuple[List[ContentType], List[Transport]]:
    from .extensions import load_extensions

    types: List[ContentType] = []
    transports: List[Transport] = []
    for ext in load_extensions():
        types.extend(ext.content_types())
        transports.extend(ext.transports())
    return types, transports


def registry() -> Tuple[Dict[str, ContentType], Dict[str, Transport]]:
    """Every registered type and transport, built-in first; refuses duplicates and a type whose
    default transport nothing registered."""
    ext_types, ext_transports = _from_extensions()
    types: Dict[str, ContentType] = _merge(BUILTIN_TYPES, ext_types, "content type")
    transports: Dict[str, Transport] = _merge(BUILTIN_TRANSPORTS, ext_transports, "transport")
    for t in types.values():
        if t.default_transport not in transports:
            raise RegistryError(
                f"content type {t.name!r} names transport {t.default_transport!r}, "
                "which nothing registers"
            )
    return types, transports


def content_type(name: Optional[str]) -> ContentType:
    """The registered type ``name``; ``None`` (a source or item that names none) is a podcast."""
    types, _ = registry()
    key = name or PODCAST_EPISODE
    if key not in types:
        raise UnknownContentType(key)
    return types[key]


def run_source(
    cfg: Any,
    entry: Any = None,
    *,
    run_pipeline_fn: Optional[Callable[[Any], Tuple[int, str]]] = None,
) -> Tuple[int, str]:
    """Run one source: ``run_pipeline`` for a podcast, the text-in entry point otherwise.

    ``entry`` is the source's ``RssFeedEntry`` (``None`` for a single-feed run, a podcast).
    ``run_pipeline_fn`` lets the CLI keep its injected runner.
    """
    ctype = content_type(getattr(entry, "content_type", None))
    if ctype.name == PODCAST_EPISODE:
        if run_pipeline_fn is None:
            from . import workflow

            run_pipeline_fn = workflow.run_pipeline
        return run_pipeline_fn(cfg)
    raise NoTextRunner(f"content type {ctype.name!r}: no text-in runner yet (ADR-177 decision 4)")


__all__ = [
    "PODCAST_ONLY",
    "item_content_type",
    "is_wanted",
    "BUILTIN_TRANSPORTS",
    "BUILTIN_TYPES",
    "ContentType",
    "ItemContext",
    "ListItems",
    "NoTextRunner",
    "PODCAST",
    "PODCAST_EPISODE",
    "PODCAST_ITEM",
    "RSS_ENCLOSURE",
    "RegistryError",
    "SourceItem",
    "StageProfile",
    "Transport",
    "UnknownContentType",
    "content_type",
    "current_item",
    "processing_item",
    "registry",
    "run_source",
]
