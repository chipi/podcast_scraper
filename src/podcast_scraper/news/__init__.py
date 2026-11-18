"""News, public's second built-in content type (ADR-177 decision 5; E05, News 1.0).

An article is identified by its normalised canonical URL, so one article over two transports is
one item. Its transport lists URLs; the type fetches each page and extracts its main text
(``fetch``), and writes its metadata in the News 1.0 contract's shape (``document``). It runs
summary and KG, never GI (E05 decisions), and never audio, transcription or ad cleaning.
"""

from __future__ import annotations

from typing import Tuple

from ..content_types import (
    ContentType,
    ItemContext,
    KG,
    SourceItem,
    StageProfile,
    SUMMARY,
    TRANSLATION,
    Transport,
    url_identity,
)
from ..news_items import NEWS_ARTICLE
from . import document, extract, fetch, graph, guardian, hn, rss


def _skips_stage(item: ItemContext, stage: str) -> bool:
    """A teaser or video body is never enriched as if whole (contract ``text_status``): it gets
    its KG, not a summary of its one to three sentences."""
    return stage == SUMMARY and item.fields.get("text_status") in (
        extract.TEXT_TEASER,
        extract.TEXT_VIDEO,
    )


def _identity(item: SourceItem) -> str:
    if not item.url:
        raise ValueError(f"a news item has no URL (transport id {item.item_id!r})")
    return url_identity(item.url)


NEWS = ContentType(
    name=NEWS_ARTICLE,
    id_scheme="item",
    stage_profile=StageProfile(name=NEWS_ARTICLE, stages=frozenset({TRANSLATION, SUMMARY, KG})),
    default_transport=rss.NAME,
    card_kind="article",
    identity=_identity,
    source_kind="news",
    fetch_body=fetch.fetch_body,
    shape_metadata=document.shape,
    kg_extend=graph.extend,
    restore_item=document.restore_item,
    skips_stage=_skips_stage,
)

TYPES: Tuple[ContentType, ...] = (NEWS,)
TRANSPORTS: Tuple[Transport, ...] = (rss.TRANSPORT, hn.TRANSPORT, guardian.TRANSPORT)
