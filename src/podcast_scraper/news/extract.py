"""A news page's main text: trafilatura first, readability as the fallback (E05 "Extraction").

Paragraphs are separated by a blank line: an untimed document's segments are the text between
blank lines in the body, and no segments file is written (ontology v3, "Segments").
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, List, Optional
from urllib.parse import urlsplit

#: trafilatura's text shorter than this share of readability's means trafilatura missed part of
#: the article; readability's is taken instead.
READABILITY_RATIO = 0.5
#: A body shorter than this is a teaser, never chunked or enriched as if whole (E05).
TEASER_CHARS = 600
#: A URL path with one of these segments is a video page; its text is a caption at most.
VIDEO_PATH_SEGMENTS = frozenset({"video", "videos", "av"})

TRAFILATURA = "trafilatura"
READABILITY = "readability"
#: The feed's own text for the item (``content:encoded``, else its description).
FEED = "feed"
#: The page's own summary line (its description metadata).
STANDFIRST = "standfirst"

#: Where a source's item text comes from (``news.text_from`` in feeds.spec).
TEXT_FROM_PAGE = "page"
TEXT_FROM_FEED = "feed"
TEXT_FROM_STANDFIRST = "standfirst"
#: Page text shorter than this, after boilerplate is dropped, is no article: the standfirst is
#: taken instead when the page has one.
MIN_USABLE_CHARS = 200

#: A paragraph shorter than this that matches ``BOILERPLATE`` is page furniture.
BOILERPLATE_MAX_CHARS = 240
BOILERPLATE = re.compile(
    r"subscri(be|ption)|rent or buy|access options|prices (vary|may be subject)|"
    r"\blog ?in\b|\bsign (up|in)\b|reset your password|limited support for css|"
    r"compatibility mode|browser version|first appeared on|\baccess .{0,40}\bjournals\b|"
    r"\bprint issues\b|^competing interests|^doi:|declare no competing interests|\bpassword\b|"
    r"^nature, published online",
    re.IGNORECASE,
)

TEXT_FULL = "full"
TEXT_TEASER = "teaser"
TEXT_VIDEO = "video"

_BLOCKS = ("p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "pre")


@dataclass(frozen=True)
class Extracted:
    """A page's main text, the extractor that produced it and the page's declared language."""

    text: str
    method: str
    language: Optional[str]


def _lines(text: str) -> str:
    """One paragraph per non-empty line in, paragraphs separated by a blank line out."""
    return "\n\n".join(" ".join(line.split()) for line in text.splitlines() if line.strip())


def _blocks_text(fragment: str) -> str:
    """An HTML fragment's block elements as paragraphs, and the text between them as paragraphs
    of its own (Nature's feed puts the standfirst after its date line's ``<p>``, Common #50)."""
    import lxml.html

    try:
        root = lxml.html.fromstring(fragment)
    except (ValueError, lxml.etree.ParserError):
        return ""
    blocks: List[str] = []
    loose: List[str] = []

    def flush() -> None:
        blocks.append(" ".join("".join(loose).split()))
        loose.clear()

    def walk(el: Any) -> None:
        loose.append(el.text or "")
        for child in el:
            if not isinstance(child.tag, str):
                pass
            elif child.tag in _BLOCKS:
                # A block inside another block is read with its parent.
                flush()
                blocks.append(" ".join(child.text_content().split()))
            else:
                walk(child)
            loose.append(child.tail or "")

    if root.tag in _BLOCKS:
        blocks.append(" ".join(root.text_content().split()))
    else:
        walk(root)
        flush()
    return _lines("\n".join(blocks))


def _readability_text(html: str) -> str:
    from readability import Document

    try:
        summary = Document(html).summary(html_partial=True)
    except Exception:  # readability raises on pages it cannot parse; trafilatura's text stands
        return ""
    return _blocks_text(summary)


def strip_boilerplate(body: str) -> str:
    """``body`` without the short paragraphs that are a page's furniture, not its article:
    subscription offers, log-in prompts, browser notices (a paywalled page's extraction is mostly
    these). A long paragraph is kept whatever it mentions."""
    kept = [
        p for p in paragraphs(body) if len(p) >= BOILERPLATE_MAX_CHARS or not BOILERPLATE.search(p)
    ]
    return "\n\n".join(kept)


def paragraphs(body: str) -> List[str]:
    """The body's paragraphs: the text between blank lines (ontology v3, "Segments")."""
    return [" ".join(p.split()) for p in body.split("\n\n") if p.strip()]


def page_author(html: str, url: str) -> Optional[str]:
    """The page's own byline: its ``<meta name="author">``, else trafilatura's author (E05: the
    feed's author field first, the page's second). ``None`` when the page names none."""
    import lxml.html
    import trafilatura

    try:
        tags = lxml.html.fromstring(html).xpath('//meta[@name="author"]/@content')
    except (ValueError, lxml.etree.ParserError):
        tags = []
    tag = " ".join(str(tags[0]).split()) if tags else ""
    if tag:
        return tag
    md = trafilatura.extract_metadata(html, default_url=url)
    author = " ".join(str(getattr(md, "author", "") or "").split())
    return author or None


def page_language(html: str) -> Optional[str]:
    """The primary subtag of the page's ``<html lang>`` (``en-GB`` -> ``en``), or ``None``."""
    import lxml.html

    try:
        root = lxml.html.fromstring(html)
    except (ValueError, lxml.etree.ParserError):
        return None
    lang = (root.get("lang") or root.get("xml:lang") or "").strip()
    primary = lang.split("-")[0].split("_")[0].lower()
    return primary if primary.isalpha() and 2 <= len(primary) <= 3 else None


def page_standfirst(html: str, url: str) -> str:
    """The page's own summary line (its description metadata, ``og:description``): what a
    paywalled or script-rendered page still says about its article."""
    import trafilatura

    md = trafilatura.extract_metadata(html, default_url=url)
    return " ".join(str(getattr(md, "description", "") or "").split())


def main_text(
    html: str, url: str, feed_html: Optional[str] = None, *, text_from: str = TEXT_FROM_PAGE
) -> Optional[Extracted]:
    """The item's text, from where its source says (``text_from``):

    - ``page`` (the default): the page's main text, or the feed's own text for the item (its
      full content or its description) when that is longer; and when neither has
      :data:`MIN_USABLE_CHARS` (a paywalled or script-rendered page), the page's standfirst;
    - ``feed``: the feed's own text;
    - ``standfirst``: the page's standfirst (a source whose article pages are paywalled).

    ``None`` when there is none."""
    import trafilatura

    lang = page_language(html)
    feed = strip_boilerplate(_blocks_text(feed_html)) if feed_html and feed_html.strip() else ""
    if text_from == TEXT_FROM_FEED:
        return Extracted(feed, FEED, lang) if feed else None
    standfirst = page_standfirst(html, url)
    if text_from == TEXT_FROM_STANDFIRST:
        return Extracted(standfirst, STANDFIRST, lang) if standfirst else None

    traf = strip_boilerplate(
        _lines(
            trafilatura.extract(
                html,
                url=url,
                output_format="txt",
                include_comments=False,
                include_tables=False,
                favor_precision=True,
            )
            or ""
        )
    )
    read = strip_boilerplate(_readability_text(html))
    best: Optional[Extracted] = None
    if traf and len(traf) >= READABILITY_RATIO * len(read):
        best = Extracted(traf, TRAFILATURA, lang)
    elif read:
        best = Extracted(read, READABILITY, lang)
    if feed and (best is None or len(feed) > len(best.text)):
        best = Extracted(feed, FEED, lang)
    if standfirst and (best is None or len(best.text) < MIN_USABLE_CHARS):
        best = Extracted(standfirst, STANDFIRST, lang)
    return best


def text_status(text: str, url: str) -> str:
    """``video`` for a video page, ``teaser`` for a body too short to be the article, else
    ``full``."""
    segments = {s.lower() for s in urlsplit(url).path.split("/") if s}
    if segments & VIDEO_PATH_SEGMENTS:
        return TEXT_VIDEO
    if len(text) < TEASER_CHARS:
        return TEXT_TEASER
    return TEXT_FULL
