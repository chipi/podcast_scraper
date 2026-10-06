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

from podcast_scraper.search.storylines import (
    storyline_map_by_topic,
    top_storylines_by_member_count,
)
from podcast_scraper.search.topic_clusters import theme_map_by_topic
from podcast_scraper.server.app_content_source import row_to_summary
from podcast_scraper.server.app_slugs import resolve_slug

#: Email clients cannot clamp by line, so the summary is cut server-side, at a word, to roughly
#: the three lines the app's episode card shows before "Show more".
SUMMARY_CHARS = 240


def short_text(text: str | None, limit: int = SUMMARY_CHARS) -> str | None:
    """Whitespace-collapsed ``text`` cut at a word to ``limit`` chars with an ellipsis, or None."""
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
        "description": short_text(row.episode_description),
        "summary": short_text(s.summary_text or s.summary_preview),
    }
    return {k: v for k, v in facts.items() if v not in (None, "")}


#: Chips per kind under an email episode — the app's chip rows are short too.
GROUPING_CHIPS = 2


class Groupings:
    """The themes and storylines an episode's topics belong to (operator 2026-10-05: topic, theme
    and storyline chips wherever an email shows an episode). The two corpus maps load ONCE per
    email batch, not per item.

    A theme is addressed by its own ``tc:`` id (the /theme/:id route); a storyline by its ANCHOR
    topic id (the /storyline/:id route) — and only storylines that clear the corpus surfacing floor
    have an anchor, so every chip opens a real page (the rule app_recap_view.episode_storylines
    applies).
    """

    def __init__(self, root: Path) -> None:
        self._themes = theme_map_by_topic(root)
        self._storyline_of = storyline_map_by_topic(root)
        self._storylines = {c["id"]: c for c in top_storylines_by_member_count(root, top_n=1000)}

    def for_topics(self, topic_ids: Iterable[str]) -> dict[str, list[dict[str, str]]]:
        """``{"themes": [...], "storylines": [...]}`` for these topics, each list capped; an empty
        kind is left out."""
        themes: list[dict[str, str]] = []
        storylines: list[dict[str, str]] = []
        seen_t: set[str] = set()
        seen_s: set[str] = set()
        for tid in topic_ids:
            th = self._themes.get(tid)
            if th and th["cluster_id"] not in seen_t and len(themes) < GROUPING_CHIPS:
                seen_t.add(th["cluster_id"])
                themes.append({"id": th["cluster_id"], "label": str(th["cluster_label"])})
            info = self._storyline_of.get(tid)
            summary = self._storylines.get(info.get("storyline_id")) if info else None
            if summary and len(storylines) < GROUPING_CHIPS:
                anchor = str(summary["anchor_topic_id"])
                if anchor not in seen_s:
                    seen_s.add(anchor)
                    storylines.append({"id": anchor, "label": str(summary["label"])})
        out: dict[str, list[dict[str, str]]] = {}
        if themes:
            out["themes"] = themes
        if storylines:
            out["storylines"] = storylines
        return out


def enrich_items(root: Path, items: Iterable[dict[str, Any]]) -> None:
    """Fill display facts into each item IN PLACE; never overwrite what a producer already set.

    Only items that LINK to their episode: a trending item carries a representative episode's slug
    but links to the TOPIC, and that episode's title and artwork would mislabel the link.
    """
    cache: dict[str, dict[str, Any]] = {}
    groupings: Groupings | None = None
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
        topic_ids = [
            r["id"]
            for r in item.get("graph_refs") or []
            if r.get("kind") == "topic" and r.get("id")
        ]
        if topic_ids:
            groupings = groupings or Groupings(root)
            for key, value in groupings.for_topics(topic_ids).items():
                item.setdefault(key, value)
