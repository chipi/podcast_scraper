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

Podcast metadata carries no ``content_type`` field until R3b (ADR-178): an absent field reads as a
podcast episode. Shared reads (the catalog builders, the search indexer's walk) take only podcast
episodes unless a caller asks for other types, so a surface sees a new type only when it opts in
(ADR-177 decision 8).
"""

from __future__ import annotations

import hashlib
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

from podcast_scraper.utils.canonical_url import canonical_url

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


def asks_other_types(content_types: Collection[str]) -> bool:
    """Whether a reader taking ``content_types`` must also walk ``sources/``, where every type but
    podcasts lives (RFC-134 §5)."""
    return any(t != PODCAST_EPISODE for t in content_types)


#: Stage names a profile can list. A podcast's profile lists none, which runs every stage.
TRANSLATION = "translation"
SUMMARY = "summary"
GI = "gi"
KG = "kg"
AD_CLEANING = "ad_cleaning"
STAGES: FrozenSet[str] = frozenset({TRANSLATION, SUMMARY, GI, KG, AD_CLEANING})

#: The kinds of source (ontology v3, a fixed vocabulary; ``test`` is the test-only type's).
SOURCE_KINDS: FrozenSet[str] = frozenset(
    {"podcast", "news", "social", "advisory", "report", "test"}
)


class RegistryError(ValueError):
    """A content type or transport registered twice, a type naming an unknown transport or source
    kind, or a profile naming an unknown stage."""


@dataclass(frozen=True)
class StageProfile:
    """The stages a type runs. ``stages`` of ``None`` is the full podcast pipeline."""

    name: str
    stages: Optional[FrozenSet[str]] = None

    def __post_init__(self) -> None:
        unknown = (self.stages or frozenset()) - STAGES
        if unknown:
            raise RegistryError(
                f"stage profile {self.name!r} names unknown stages {sorted(unknown)}"
            )

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
    #: The item's stable identity, from which :func:`item_id` mints its id (``None``: the
    #: transport's own id). A URL-identified type returns :func:`url_identity` of its normalised
    #: canonical URL, so one item over two transports is one item.
    identity: Optional[Callable[[SourceItem], str]] = None
    #: What kind of source its items come from, from :data:`SOURCE_KINDS` (T-1 keeps it apart
    #: from the item noun, ``name``). KG 3.0 records it on the ``Source`` node.
    source_kind: str = ""
    #: The type's own step for an item its transport listed as a URL only (ADR-177 decision 2):
    #: ``(item, source) -> item with its text``, or ``None`` to skip it (``None``: such items are
    #: skipped). News fetches and extracts the page here.
    fetch_body: Optional[Callable[["SourceItem", Any], Optional["SourceItem"]]] = None
    #: The item's metadata document as the type's contract has it, from the document the shared
    #: metadata stage built and the item's context (``None``: that document with ``content_type``
    #: first).
    shape_metadata: Optional[Callable[[Dict[str, Any], "ItemContext"], Dict[str, Any]]] = None
    #: The type's own nodes and edges added to an item's KG 3.0 artifact before it is validated
    #: (``None``: the artifact as extraction built it). News adds its publisher and bylines.
    kg_extend: Optional[Callable[[Dict[str, Any], "ItemContext"], Dict[str, Any]]] = None
    #: A stored item as its transport would list it again, from its metadata document and its
    #: stored body: what ``rederive_only`` re-runs the stages on, so no source is fetched
    #: (``None``: the item's title and URL only).
    restore_item: Optional[Callable[[Dict[str, Any], str], "SourceItem"]] = None
    #: ``(item, stage) -> reason`` when the type's own rule skips a stage its profile runs, for
    #: this item, else ``None`` (``None`` hook: the profile decides). The reason is recorded in
    #: the item's processing manifest. News skips the summary of a teaser or video, and every
    #: enrichment stage of an article not in English while its source's translation is off.
    stage_skip_reason: Optional[Callable[["ItemContext", str], Optional[str]]] = None

    def __post_init__(self) -> None:
        if self.source_kind not in SOURCE_KINDS:
            raise RegistryError(
                f"content type {self.name!r}: source_kind {self.source_kind!r} is not one of "
                f"{sorted(SOURCE_KINDS)}"
            )


@dataclass(frozen=True)
class SourceItem:
    """One new item a transport listed: the transport's own id for it, and its text or a URL to
    fetch. Its times go in ``metadata`` as ``published_at``, ``filed_at``, ``period_start``,
    ``period_end`` and ``observed_at`` (ontology v3)."""

    item_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    text: Optional[str] = None
    url: Optional[str] = None
    #: What was fetched to get ``text`` (a page's HTML, an API's JSON record), kept beside the body
    #: as its raw cache (RFC-134 §5), with ``raw_suffix`` as its file suffix.
    raw: Optional[str] = None
    raw_suffix: str = ".html"


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
    source_kind="podcast",
)
BUILTIN_TYPES: Tuple[ContentType, ...] = (PODCAST,)
BUILTIN_TRANSPORTS: Tuple[Transport, ...] = (Transport(name=RSS_ENCLOSURE),)


class UnknownContentType(KeyError):
    """A source names a content type nothing registered."""


class NoTextRunner(RuntimeError):
    """A text source's transport lists no items itself, so the text-in entry point cannot run it."""


@dataclass(frozen=True)
class ItemContext:
    """The item a processing job is about: its type, id and body file (ADR-177 decision 7).

    A podcast job carries none and reads as :data:`PODCAST_ITEM`. Hooks gain no type argument:
    one that needs the type reads :func:`current_item` while the job is processed.
    """

    content_type: ContentType
    item_id: Optional[str] = None
    body_path: Optional[str] = None
    #: The item's source (``source:`` id from :func:`source_id`) and the transport that listed it.
    source_id: Optional[str] = None
    transport: Optional[str] = None
    #: The item's times by ontology v3 name (``published_at``, ...), ISO 8601; absent when unknown.
    times: Mapping[str, str] = field(default_factory=dict)
    #: The item's own fields as its transport and its type's fetch gave them, and the body's hash
    #: (``body_sha256``); the type's ``shape_metadata`` reads them.
    fields: Mapping[str, Any] = field(default_factory=dict)


PODCAST_ITEM = ItemContext(content_type=PODCAST)

_CURRENT_ITEM: ContextVar[ItemContext] = ContextVar("current_item", default=PODCAST_ITEM)


def current_item() -> ItemContext:
    """The item being processed on this thread; a podcast when no job set one."""
    return _CURRENT_ITEM.get()


def url_identity(url: str) -> str:
    """A URL-identified item's or source's identity: the first 20 hex characters of the sha256 of
    its canonical URL (ontology v3; the rules in ``utils.canonical_url``), so one page reached by
    two forms of its URL is one id. Raises ``ValueError`` for a URL that is not absolute."""
    return hashlib.sha256(canonical_url(url).encode("utf-8")).hexdigest()[:20]


def item_id(ctype: ContentType, item: SourceItem) -> str:
    """The item's id, ``item:{type}:{identity}``: minted by its type, never by the transport."""
    key = ctype.identity(item) if ctype.identity is not None else item.item_id
    if not key:
        raise ValueError(f"{ctype.name}: an item has no identity (transport id {item.item_id!r})")
    return f"item:{ctype.name}:{key}"


def source_id(ctype: ContentType, source_url: str) -> str:
    """A non-podcast source's id, ``source:{url_identity(url)}``: minted by its type from the
    source's URL (podcasts keep ``podcast:{slug}``)."""
    if not source_url:
        raise ValueError(f"{ctype.name}: a source has no URL to mint its id from")
    return f"source:{url_identity(source_url)}"


#: The stages that derive from an item's text: an item its source marks not minable runs none.
ENRICHMENT_STAGES: FrozenSet[str] = frozenset({TRANSLATION, SUMMARY, GI, KG})


def runs_for_item(item: "ItemContext", stage: str) -> bool:
    """True when ``item`` runs ``stage``: its type's profile runs it, and, for an enrichment stage,
    its source lets it be mined (``mineable`` is not ``False``; ontology v3 "Rights, per item"),
    and its type's own rule does not skip it (:func:`stage_skip_reason`)."""
    ctype = item.content_type
    if not ctype.stage_profile.runs(stage):
        return False
    if stage in ENRICHMENT_STAGES and item.fields.get("mineable") is False:
        return False
    return stage_skip_reason(item, stage) is None


def stage_skip_reason(item: "ItemContext", stage: str) -> Optional[str]:
    """Why the item's type skips ``stage`` for this item (``stage_skip_reason``), or ``None``."""
    hook = item.content_type.stage_skip_reason
    return hook(item, stage) if hook is not None else None


def stage_runs(stage: str) -> bool:
    """True when the current item runs ``stage`` (always, for a podcast)."""
    return runs_for_item(current_item(), stage)


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


def _public_builtins() -> Tuple[List[ContentType], List[Transport]]:
    """Public's built-in types after podcast: news (ADR-177 decision 5). Imported here, not at
    module load, because the news package imports this module."""
    from . import news

    return list(news.TYPES), list(news.TRANSPORTS)


def registry() -> Tuple[Dict[str, ContentType], Dict[str, Transport]]:
    """Every registered type and transport, built-in first; refuses duplicates and a type whose
    default transport nothing registered."""
    ext_types, ext_transports = _from_extensions()
    pub_types, pub_transports = _public_builtins()
    types: Dict[str, ContentType] = _merge([*BUILTIN_TYPES, *pub_types], ext_types, "content type")
    transports: Dict[str, Transport] = _merge(
        [*BUILTIN_TRANSPORTS, *pub_transports], ext_transports, "transport"
    )
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
    _, transports = registry()
    transport_name = getattr(entry, "transport", None) or ctype.default_transport
    if transport_name not in transports:
        raise RegistryError(f"source names transport {transport_name!r}, which nothing registers")
    from .workflow.text_in import run_text_source

    return run_text_source(cfg, entry, ctype, transports[transport_name])


__all__ = [
    "AD_CLEANING",
    "BUILTIN_TRANSPORTS",
    "BUILTIN_TYPES",
    "ContentType",
    "GI",
    "ItemContext",
    "KG",
    "ListItems",
    "NoTextRunner",
    "PODCAST",
    "PODCAST_EPISODE",
    "PODCAST_ITEM",
    "PODCAST_ONLY",
    "RSS_ENCLOSURE",
    "RegistryError",
    "SOURCE_KINDS",
    "STAGES",
    "SUMMARY",
    "SourceItem",
    "StageProfile",
    "TRANSLATION",
    "Transport",
    "UnknownContentType",
    "content_type",
    "current_item",
    "is_wanted",
    "item_content_type",
    "item_id",
    "processing_item",
    "registry",
    "run_source",
    "source_id",
    "stage_runs",
    "url_identity",
]
