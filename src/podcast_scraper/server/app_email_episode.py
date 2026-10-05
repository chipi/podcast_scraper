"""What an email shows next to an episode item (operator 2026-10-05).

The digest, recommendations, revisit-nudge and new-episode emails carried a slug, a deep link and
graph refs — enough to link, not enough to look like the app. The app's episode header shows the
artwork, the show's name above the title, a duration · date line and the episode's summary; the
emails now can too.

Filled at payload assembly from the catalog — one lookup per distinct slug — so every producer gets
it without each one re-deriving it, and a field the producer already set is never overwritten.
``artwork_url`` is whatever the app itself would render: our locally-stored thumb (an app-relative
``/api/app/artwork?…`` path the email renderer makes absolute; the edge serves it without a
session), else the feed-hosted image. Unknown slugs and missing values are simply left out.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from podcast_scraper.server.app_content_source import row_to_summary
from podcast_scraper.server.app_slugs import resolve_slug

#: Email clients cannot clamp by line, so the summary is cut server-side, at a word, to roughly
#: the three lines the app's episode card shows before "Show more".
SUMMARY_CHARS = 240


def _short(text: str | None, limit: int = SUMMARY_CHARS) -> str | None:
    t = " ".join((text or "").split())
    if len(t) <= limit:
        return t or None
    cut = t[:limit].rsplit(" ", 1)[0].rstrip(",;:—-")
    return f"{cut}…"


def episode_display(root: Path, slug: str) -> dict[str, Any]:
    """Display facts for one episode, or {} when the slug no longer resolves."""
    row = resolve_slug(root, slug)
    if row is None:
        return {}
    s = row_to_summary(root, row)
    facts: dict[str, Any] = {
        "episode_title": s.title,
        "podcast_title": s.podcast_title,
        "artwork_url": s.artwork_url or s.episode_image_url or s.feed_image_url,
        "duration_seconds": s.duration_seconds,
        "publish_date": s.publish_date,
        # The publisher's own blurb, and OUR summary — both shown, ours labelled "Summary", so a
        # reader can tell which is which (operator 2026-10-05).
        "description": _short(row.episode_description),
        "summary": _short(s.summary_text or s.summary_preview),
    }
    return {k: v for k, v in facts.items() if v not in (None, "")}


def enrich_items(root: Path, items: Iterable[dict[str, Any]]) -> None:
    """Fill display facts into each item IN PLACE; never overwrite what a producer already set.

    Only items that LINK to their episode: a trending item carries a representative episode's slug
    but links to the TOPIC, and that episode's title and artwork would mislabel the link.
    """
    cache: dict[str, dict[str, Any]] = {}
    for item in items:
        if not str(item.get("deep_link") or "").startswith("/episode/"):
            continue
        slug = item.get("episode_slug")
        if not isinstance(slug, str) or not slug:
            continue
        if slug not in cache:
            cache[slug] = episode_display(root, slug)
        for key, value in cache[slug].items():
            if not item.get(key):
                item[key] = value
