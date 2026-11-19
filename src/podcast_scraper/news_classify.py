"""What kind of item a non-full news page is (the operator, 2026-11-19).

A feed often gives only part of an article: its opening paragraphs, one sentence, a day's brief, a
series promotion, or something that is not an article at all. Only a ``news_summary`` (an item that
reports what happened) is worth keeping; the other classes are skipped. Rules only, no model: the
operator chose them after judging twelve real teasers, and the tests check those twelve, rewritten
(real feed text is not ours to publish). The writer stores the class as ``news_article.item_class``
for every non-full item; full articles are not classified.
"""

from __future__ import annotations

import re
from typing import Optional

NEWS_SUMMARY = "news_summary"
SMALL_TEASER = "small_teaser"
DAY_BRIEF = "day_brief"
PROMO = "promo"
NOT_ARTICLE = "not_article"

#: The classes kept; every other class is skipped.
KEPT = frozenset({NEWS_SUMMARY})

#: Fewer words than this is a one-line standfirst, not a summary (the twelve: 22-27 skipped,
#: 55 and up kept or classed otherwise).
MIN_SUMMARY_WORDS = 40
#: Fewer words than this is no article at all ("Letter to the Editor").
MIN_ARTICLE_WORDS = 10

#: A newsletter or day's brief: several stories, or the writer talking to the reader.
_DAY_BRIEF = re.compile(
    r"\b(good morning|need-to-know|send (news )?tips|in today's (newsletter|edition)|"
    r"this (week|morning)'s (newsletter|briefing))\b",
    re.IGNORECASE,
)
#: A promotion of a series or a section rather than a report.
_PROMO = re.compile(
    r"\b(welcome to\b.*\bseries|special series|introducing our new)\b", re.IGNORECASE
)
#: A page that is a stub for something else.
_STUB = re.compile(
    r"^\s*(letter to the editor|correction|editorial|podcast|video)\s*\.?\s*$", re.IGNORECASE
)
#: A social or status post (Mastodon-style /@account/<id>), not an article.
_SOCIAL_POST = re.compile(r"/@[^/\s]+/\d+")


def classify(title: str, body: str, url: Optional[str] = None) -> str:
    """The class of a non-full news item from its title, the text its feed gave and its URL."""
    text = (body or "").strip()
    words = len(text.split())
    if url and _SOCIAL_POST.search(url):
        return NOT_ARTICLE
    if words < MIN_ARTICLE_WORDS or _STUB.match(text):
        return NOT_ARTICLE
    if _DAY_BRIEF.search(text):
        return DAY_BRIEF
    if _PROMO.search(text):
        return PROMO
    if words < MIN_SUMMARY_WORDS:
        return SMALL_TEASER
    return NEWS_SUMMARY


def is_kept(item_class: Optional[str]) -> bool:
    """Whether an item of this class is kept (only a news summary is)."""
    return item_class in KEPT
