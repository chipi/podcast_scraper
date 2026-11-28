"""The Guardian Content API transport: the articles after the cursor, oldest first (E05).

The API gives each article's body, byline and language, so its items arrive with their text and
no page is fetched; the JSON record is kept as the raw cache (``.raw.json``). The key is read from
the variable the source's ``api_key_env`` names, else ``GUARDIAN_API_KEY``, and never written to a
log (the shared downloader redacts ``api-key`` in every URL it logs). Without a key the source
lists nothing, and says so (naming the source, not the variable). An article's cursor value is its
``webPublicationDate``; the run moves the cursor only past articles that wrote
(``workflow.text_in``). One request asks for the API's largest page, oldest first: an article
past it is newer than all listed, so the cursor never passes it.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode, urlsplit

from ..content_types import SourceItem, Transport
from ..rss import downloader
from . import extract
from .fetch import complete, source_spec, TIMEOUT_S, user_agent

logger = logging.getLogger(__name__)

NAME = "guardian"
KEY_ENV = "GUARDIAN_API_KEY"
FIELDS = "body,byline,trailText,lang"
#: The Content API's largest ``page-size``.
PAGE_SIZE = 200


def _request_url(source: Any, page_size: int, cursor: Optional[str], key: str) -> str:
    base = str(getattr(source, "url", "") or "")
    params: Dict[str, Any] = {
        "order-by": "oldest",
        "page-size": page_size,
        "show-fields": FIELDS,
        "api-key": key,
    }
    if cursor:
        params["from-date"] = cursor[:10]
    joiner = "&" if urlsplit(base).query else "?"
    return base + joiner + urlencode(params)


def _item(result: Dict[str, Any], source: Any) -> Optional[SourceItem]:
    """One API result as a written-ready item, or ``None`` (no text, or a language we do not
    ingest)."""
    fields = result.get("fields") or {}
    text = extract.strip_boilerplate(extract._blocks_text(str(fields.get("body") or "")))
    if not text:
        return None
    listed = SourceItem(
        item_id=str(result.get("id") or ""),
        url=str(result.get("webUrl") or ""),
        metadata={
            "title": result.get("webTitle"),
            "published_at": result.get("webPublicationDate"),
            "transport_ids": {NAME: result.get("id")},
        },
    )
    return complete(
        listed,
        source,
        text,
        method=NAME,
        declared_language=fields.get("lang"),
        byline_raw=fields.get("byline") or None,
        raw=json.dumps(result, ensure_ascii=False, indent=2),
        raw_suffix=".json",
    )


def list_items(source: Any, cursor: Optional[str]) -> Tuple[List[SourceItem], Optional[str]]:
    """The articles published after ``cursor``, oldest first, one page of them: an article past
    the page is newer than every one listed, so the cursor never passes it. The run caps what it
    writes at the source's ``max_items``, counting only articles not yet written."""
    key_env = source_spec(source).api_key_env or KEY_ENV
    key = os.environ.get(key_env, "").strip()
    if not key:
        # The variable is not named here: the config and docs/guides/NEWS_API_KEYS.md name it.
        logger.warning(
            "news guardian: %s: the API key variable is not set; the source lists nothing",
            getattr(source, "url", ""),
        )
        return [], cursor
    spec = source_spec(source)
    include = re.compile(spec.include_url_pattern) if spec.include_url_pattern else None
    exclude = re.compile(spec.exclude_url_pattern) if spec.exclude_url_pattern else None
    resp = downloader.fetch_url(
        _request_url(source, PAGE_SIZE, cursor, key), user_agent(source), TIMEOUT_S
    )
    if resp is None:
        # As a podcast feed's (stages/scraping.py): a soft feed incident, the cursor kept. The
        # URL is not named: it carries the key (the downloader's own log redacts it).
        raise ValueError("Failed to fetch RSS feed. (news guardian: the API could not be reached)")
    try:
        results = (resp.json().get("response") or {}).get("results") or []
    except ValueError as exc:
        raise ValueError(
            "Failed to parse RSS XML: news guardian: the API did not answer JSON"
        ) from exc
    items: List[SourceItem] = []
    newest = cursor
    for result in results:
        published = str(result.get("webPublicationDate") or "")
        link = str(result.get("webUrl") or "")
        if cursor and published and published <= cursor:
            continue
        if (include and not include.search(link)) or (exclude and exclude.search(link)):
            continue
        item = _item(result, source)
        if item is not None:
            items.append(dataclasses.replace(item, cursor=published or None))
        if published and (newest is None or published > newest):
            newest = published
    return items, newest


TRANSPORT = Transport(
    name=NAME, list_items=list_items, max_items=lambda source: source_spec(source).max_items
)
