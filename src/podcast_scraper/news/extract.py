"""A news page's main text: trafilatura first, readability as the fallback (E05 "Extraction").

Paragraphs are separated by a blank line: an untimed document's segments are the text between
blank lines in the body, and no segments file is written (ontology v3, "Segments").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
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


def _readability_text(html: str) -> str:
    import lxml.html
    from readability import Document

    try:
        summary = Document(html).summary(html_partial=True)
    except Exception:  # readability raises on pages it cannot parse; trafilatura's text stands
        return ""
    root = lxml.html.fromstring(summary)
    blocks: List[str] = []
    for el in root.iter(*_BLOCKS):
        # A block inside another block is read with its parent.
        if any(a.tag in _BLOCKS for a in el.iterancestors()):
            continue
        blocks.append(" ".join(el.text_content().split()))
    return _lines("\n".join(blocks))


def paragraphs(body: str) -> List[str]:
    """The body's paragraphs: the text between blank lines (ontology v3, "Segments")."""
    return [" ".join(p.split()) for p in body.split("\n\n") if p.strip()]


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


def main_text(html: str, url: str) -> Optional[Extracted]:
    """The page's main text, or ``None`` when neither extractor finds any."""
    import trafilatura

    traf = _lines(
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
    read = _readability_text(html)
    lang = page_language(html)
    if traf and len(traf) >= READABILITY_RATIO * len(read):
        return Extracted(traf, TRAFILATURA, lang)
    if read:
        return Extracted(read, READABILITY, lang)
    return None


def text_status(text: str, url: str) -> str:
    """``video`` for a video page, ``teaser`` for a body too short to be the article, else
    ``full``."""
    segments = {s.lower() for s in urlsplit(url).path.split("/") if s}
    if segments & VIDEO_PATH_SEGMENTS:
        return TEXT_VIDEO
    if len(text) < TEASER_CHARS:
        return TEXT_TEASER
    return TEXT_FULL
