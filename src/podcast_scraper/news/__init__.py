"""News, public's second built-in content type (ADR-177 decision 5; E05, News 1.0).

An article is identified by its normalised canonical URL, so one article over two transports is
one item. Its transport lists URLs; the type fetches each page and extracts its main text
(``fetch``), and writes its metadata in the News 1.0 contract's shape (``document``). It runs
summary and KG, never GI (E05 decisions), and never audio, transcription or ad cleaning.
"""

from __future__ import annotations

from typing import Optional, Tuple

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
from ..languages import primary_language
from ..news_items import NEWS_ARTICLE
from . import document, extract, fetch, graph, guardian, hn, rss

#: Why a stage is skipped, as the processing manifest records it.
SKIP_TRANSLATION_OFF = "translation_off"
SKIP_NOT_WHOLE_TEXT = "not_whole_text"


def _stage_skip_reason(item: ItemContext, stage: str) -> Optional[str]:
    """News 1.0 does not translate unless the source's ``translate`` switch is on (operator,
    2026-11-18): an article not in English (or in a language nobody could tell) is written as it
    is, and no stage runs English prompts over it. A teaser or video body is never enriched as if
    whole (contract ``text_status``): it gets its KG, not a summary of its one to three
    sentences."""
    english = primary_language(item.fields.get("language")) == "en"
    if not english and item.fields.get("translate") is not True:
        return SKIP_TRANSLATION_OFF if stage in (TRANSLATION, SUMMARY, KG) else None
    if stage == SUMMARY and item.fields.get("text_status") in (
        extract.TEXT_TEASER,
        extract.TEXT_VIDEO,
    ):
        return SKIP_NOT_WHOLE_TEXT
    return None


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
    stage_skip_reason=_stage_skip_reason,
)

TYPES: Tuple[ContentType, ...] = (NEWS,)
TRANSPORTS: Tuple[Transport, ...] = (rss.TRANSPORT, hn.TRANSPORT, guardian.TRANSPORT)
