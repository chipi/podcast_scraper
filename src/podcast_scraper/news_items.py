"""News items as surfaces show them (News 1.0, RFC-134 §10; the item contract
``docs/architecture/corpus/news-item-1.0.md``).

Every field name a surface reads from a news item's metadata is named here and nowhere else, so a
change to the contract is a change to this module. A served news item never carries the body or
anything cut from it (ADR-179): this module never opens ``transcripts/``, the segments file or the
raw page, and :func:`served_item` builds its output only from the metadata fields below.

``serveable`` decides what is shown (ontology v3, "Rights, per item"): ``metadata`` (also when
absent) shows the title, link, publisher and times; ``summary`` adds the platform's summary;
``none`` shows nothing, so such an item is left out of every surface.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional
from urllib.parse import urlsplit

from podcast_scraper.languages import UNUSABLE_FIELD
from podcast_scraper.news_classify import is_kept
from podcast_scraper.publishers import load_registry, Registry
from podcast_scraper.search.corpus_scope import discover_metadata_files
from podcast_scraper.utils.corpus_walk import SOURCES_DIR
from podcast_scraper.utils.utc import utc_sort_key

NEWS_ARTICLE = "news_article"


def is_news_item_file(file_path: os.PathLike[str] | str, root: os.PathLike[str] | str) -> bool:
    """Whether *file_path* is one of a news item's files (``sources/news_article/…``), by its path
    and by where it really is, so a symlink elsewhere in the corpus does not hide one. File routes
    refuse these (RFC-134 §10): the body is never served, and a withheld item's metadata and KG name
    what no surface shows; the news routes serve what may be shown."""
    rels = (
        os.path.relpath(file_path, root),
        os.path.relpath(os.path.realpath(file_path), os.path.realpath(root)),
    )
    return any(Path(rel).parts[:2] == (SOURCES_DIR, NEWS_ARTICLE) for rel in rels)


# --- the contract's field names (news-item-1.0.md) ------------------------------------------
# Decided (ontology v3).
F_CONTENT_TYPE = "content_type"
F_ITEM_ID = "item_id"
F_URL = "url"
F_SOURCE_ID = "source_id"
F_TITLE = "title"
F_PUBLISHED_AT = "published_at"
F_PAYWALLED = "paywalled"
F_SERVEABLE = "serveable"
# Accepted with the contract (the operator, 2026-11-18), top level.
F_LANGUAGE = "language"
F_PUBLISHER_ID = "publisher_id"
F_SUMMARY = "summary"
# Accepted with the contract, the type's own fields under ``news_article`` (ADR-181).
F_BYLINE = "byline"
F_EDITORIAL_TYPE = "editorial_type"
F_TEXT_STATUS = "text_status"
F_FETCHED_AT = "fetched_at"

SERVEABLE_METADATA = "metadata"
SERVEABLE_SUMMARY = "summary"
SERVEABLE_NONE = "none"
_SERVEABLE = (SERVEABLE_METADATA, SERVEABLE_SUMMARY, SERVEABLE_NONE)
EDITORIAL_TYPES = ("reporting", "opinion", "analysis", "paid")


def serveable(doc: Mapping[str, Any]) -> str:
    """What a surface may show; absent or unknown reads as ``metadata``, the narrowest value that
    still shows the item."""
    value = doc.get(F_SERVEABLE)
    return value if value in _SERVEABLE else SERVEABLE_METADATA


def is_news(doc: Any) -> bool:
    """Whether ``doc`` is a news item's metadata (``content_type: news_article``)."""
    return isinstance(doc, Mapping) and doc.get(F_CONTENT_TYPE) == NEWS_ARTICLE


#: The class the writer gives a non-full item (:mod:`podcast_scraper.news_classify`).
F_ITEM_CLASS = "item_class"


def _text_kept(doc: Mapping[str, Any]) -> bool:
    """Whether the item's text makes it worth showing (the operator, 2026-11-19): a full article;
    a teaser only when the writer classed it a news summary (one without a class, written before
    the classifier, stays hidden); a video page never, for now (it stays written)."""
    own = _own(doc)
    status = own.get(F_TEXT_STATUS)
    if status == "video":
        return False
    if status == "teaser":
        return is_kept(_str(own.get(F_ITEM_CLASS)))
    return True


def is_shown(doc: Any) -> bool:
    """Whether a surface may show the item: a news item whose ``serveable`` is not ``none``, that
    its writer did not mark ``unusable`` (top-level on an item, news-item-1.0; D-44: an item not in
    English that is not translated), and whose text is worth showing (:func:`_text_kept`)."""
    return (
        is_news(doc)
        and serveable(doc) != SERVEABLE_NONE
        and doc.get(UNUSABLE_FIELD) is not True
        and _text_kept(doc)
    )


def news_kg_item_id(kg_path: os.PathLike[str] | str) -> Optional[str]:
    """The ``item_id`` of the news item whose KG *kg_path* is (from the metadata beside it), or
    ``None`` when it is not one. Whether to serve it is the item's current state, not this copy's
    (:func:`current_copies`): an older run's KG of an item withheld since must not be served."""
    real = Path(os.path.realpath(kg_path))
    if not real.name.endswith(".kg.json"):
        return None
    meta = real.with_name(real.name[: -len(".kg.json")] + ".metadata.json")
    try:
        doc = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return _str(doc.get(F_ITEM_ID)) if is_news(doc) else None


def _own(doc: Mapping[str, Any]) -> Mapping[str, Any]:
    own = doc.get(NEWS_ARTICLE)
    return own if isinstance(own, Mapping) else {}


def _str(value: Any) -> Optional[str]:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _publisher(doc: Mapping[str, Any], registry: Registry) -> Optional[Dict[str, str]]:
    """The item's own ``publisher_id``, else its source's publisher, else the publisher whose
    domain the URL's host matches; none of them, no publisher."""
    host = urlsplit(str(doc.get(F_URL) or "")).hostname or ""
    pid = _str(doc.get(F_PUBLISHER_ID))
    pub = registry.by_id(pid) if pid else None
    if pub is None:
        sid = _str(doc.get(F_SOURCE_ID))
        pub = registry.for_source(sid) if sid else None
    if pub is None and host:
        pub = registry.for_domain(host)
    return pub.served(host) if pub else None


def _site(doc: Mapping[str, Any]) -> Optional[str]:
    """The article URL's host without a leading ``www.``: what a surface names as the item's origin
    when no publisher is known (an aggregator's link to an unregistered site)."""
    host = (urlsplit(str(doc.get(F_URL) or "")).hostname or "").lower()
    host = host[4:] if host.startswith("www.") else host
    return host or None


def served_item(
    doc: Mapping[str, Any], registry: Registry, *, metadata_relative_path: str = ""
) -> Optional[Dict[str, Any]]:
    """A news item as a surface shows it, or ``None`` when no surface may show it
    (:func:`is_shown`). Built from metadata fields only; never the body."""
    if not is_shown(doc):
        return None
    shows = serveable(doc)
    own = _own(doc)
    byline = own.get(F_BYLINE)
    editorial = own.get(F_EDITORIAL_TYPE)
    paywalled = doc.get(F_PAYWALLED)
    return {
        "content_type": NEWS_ARTICLE,
        "item_id": _str(doc.get(F_ITEM_ID)) or "",
        "title": _str(doc.get(F_TITLE)) or "",
        "url": _str(doc.get(F_URL)),
        "source_id": _str(doc.get(F_SOURCE_ID)),
        "publisher": _publisher(doc, registry),
        "site": _site(doc),
        "published_at": _str(doc.get(F_PUBLISHED_AT)),
        "language": _str(doc.get(F_LANGUAGE)),
        "byline": (
            [n for n in byline if isinstance(n, str) and n.strip()]
            if isinstance(byline, list)
            else []
        ),
        "editorial_type": editorial if editorial in EDITORIAL_TYPES else None,
        "paywalled": paywalled if isinstance(paywalled, bool) else None,
        "text_status": _str(own.get(F_TEXT_STATUS)),
        # A newsletter issue covering several stories (the source's roundup_title_pattern matched
        # its title at ingest); storyline attach asks more of it.
        "roundup": own.get("roundup") is True,
        "serveable": shows,
        "summary": _str(doc.get(F_SUMMARY)) if shows == SERVEABLE_SUMMARY else None,
        "metadata_relative_path": metadata_relative_path,
    }


@dataclass(frozen=True)
class NewsItemFile:
    """One news item's metadata file: its path, its corpus-relative path and its document."""

    path: Path
    relative_path: str
    doc: Mapping[str, Any]


def iter_news_metadata(corpus_root: Path) -> Iterator[NewsItemFile]:
    """Every news item's metadata under ``sources/``, newest run per item (the shared membership
    rule, opted in with ``include_sources``)."""
    root = Path(corpus_root).resolve()
    for path in discover_metadata_files(root, include_sources=True):
        if path.suffix.lower() != ".json":
            continue
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if is_news(doc):
            yield NewsItemFile(path, path.relative_to(root).as_posix(), doc)


def current_copies(corpus_root: Path) -> Dict[str, NewsItemFile]:
    """The copy that decides each item, by ``item_id``: one article reached by two sources or two
    runs is one item (the contract). The newest copy by ``fetched_at`` decides, then the run path
    (run directories sort by time); on a tie a copy that may not be shown wins. So a newer run that
    withholds an item withdraws every older copy, and one that restores it brings it back."""
    best: Dict[str, Any] = {}
    for f in iter_news_metadata(corpus_root):
        item_id = _str(f.doc.get(F_ITEM_ID))
        if not item_id:
            continue
        key = (_str(f.doc.get(F_FETCHED_AT)) or "", not is_shown(f.doc), f.relative_path)
        if item_id not in best or key > best[item_id][0]:
            best[item_id] = (key, f)
    return {item_id: f for item_id, (_key, f) in best.items()}


def served_items(corpus_root: Path, registry: Optional[Registry] = None) -> List[Dict[str, Any]]:
    """Every news item a surface may show, newest first, once per ``item_id``, as its current copy
    (:func:`current_copies`) says; an item whose current copy may not be shown is left out."""
    reg = registry if registry is not None else load_registry()
    out: List[Dict[str, Any]] = []
    for f in current_copies(corpus_root).values():
        item = served_item(f.doc, reg, metadata_relative_path=f.relative_path)
        if item is not None:
            out.append(item)
    out.sort(key=lambda i: utc_sort_key(i["published_at"]), reverse=True)
    return out


#: KG node types a news item's card lists (its own Source and ContentItem are the item itself).
_KG_LISTED = ("Topic", "Person", "Organization", "Object")


def item_kg(corpus_root: Path, metadata_relative_path: str) -> Dict[str, Any]:
    """The item's KG beside its metadata: its relative path, and the topics, people, organisations
    and objects it names (``{id, type, label, role}``; ``role`` only for an author). Empty when the
    item has no KG file."""
    meta = Path(corpus_root) / metadata_relative_path
    kg_path = meta.with_name(meta.name.replace(".metadata.json", ".kg.json"))
    if not kg_path.is_file():
        return {"kg_relative_path": None, "kg_nodes": []}
    try:
        art = json.loads(kg_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"kg_relative_path": None, "kg_nodes": []}
    authors = {
        e.get("from")
        for e in art.get("edges") or []
        if isinstance(e, dict)
        and e.get("type") == "PARTICIPATES_IN"
        and (e.get("properties") or {}).get("role") == "author"
    }
    nodes = []
    for n in art.get("nodes") or []:
        if not isinstance(n, dict) or n.get("type") not in _KG_LISTED:
            continue
        raw_props = n.get("properties")
        props: Dict[str, Any] = raw_props if isinstance(raw_props, dict) else {}
        label = props.get("label") or props.get("name") or n.get("id")
        node = {"id": n.get("id"), "type": n.get("type"), "label": label}
        if n.get("id") in authors:
            node["role"] = "author"
        nodes.append(node)
    return {
        "kg_relative_path": kg_path.relative_to(Path(corpus_root)).as_posix(),
        "kg_nodes": nodes,
    }
