"""Trending episodes — quote cards, two reasons interleaved (operator 2026-10-10, option "c").

An episode has no weekly "mention count" the way a topic does, so it trends for one of two
reasons, and each card says which:

* ``rising_topic`` — a topic that is rising (topic momentum), shown through the strongest speaker
  take on it: that take's episode, quote and moment.
* ``most_engaged`` — the episodes listeners listened to, opened and saved most over the last four
  weeks (the engagement series; highlights are not in it, so the reason does not claim them).

Under ``scope=mine`` both halves keep to the listener's world (ADR-162): rising topics of their
own world, episodes in their world only. Built from what is on disk — no LLM, no fetch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from podcast_scraper.server.app_corpus_access import load_json_artifact
from podcast_scraper.server.app_engagement_series import engagement_series
from podcast_scraper.server.app_gi_view import insights_from_gi
from podcast_scraper.server.app_momentum import (
    _weeks_ending,
    MomentumConfig,
    resolve_as_of_week,
    trending,
)
from podcast_scraper.server.app_relational_view import build_topic_perspectives
from podcast_scraper.server.app_slugs import resolve_slug

#: How far back "most engaged" looks, in ISO weeks.
ENGAGED_WINDOW_WEEKS = 4
#: Fewer events than this is one person's afternoon, not a trend.
ENGAGED_MIN_EVENTS = 2
#: Quotes longer than this are cut for the card (the episode page has the whole passage).
QUOTE_MAX_CHARS = 280


@dataclass
class TrendingEpisodeCard:
    """One card: the episode, why it is here, and the grounded quote that leads it."""

    slug: str
    reason: str  # "rising_topic" | "most_engaged"
    quote: str
    speaker: str | None
    start_ms: int | None
    topic_id: str | None = None
    topic_label: str | None = None
    events: int | None = None


def _cut(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= QUOTE_MAX_CHARS else text[: QUOTE_MAX_CHARS - 1].rstrip() + "…"


def _rising_topic_cards(
    root: Path,
    data_dir: Path | None,
    *,
    scope: str,
    user_id: str | None,
    restrict_entities: set[str] | None,
    world: set[str] | None,
    limit: int,
    config: MomentumConfig | None,
    now: str | None,
) -> list[TrendingEpisodeCard]:
    topics = trending(
        root,
        data_dir,
        kind="topic",
        scope=scope,
        user_id=user_id,
        limit=max(limit * 2, 6),
        config=config,
        restrict_to=restrict_entities,
        now=now,
    )
    cards: list[TrendingEpisodeCard] = []
    used: set[str] = set()
    for topic in topics:
        if len(cards) >= limit:
            break
        resp = build_topic_perspectives(root, topic.entity_id, mine_slugs=world)
        if resp is None:
            continue
        pick = None
        for person in resp.perspectives:
            for ins in person.insights:
                if ins.episode_slug and ins.episode_slug not in used:
                    pick = (person, ins)
                    break
            if pick:
                break
        if pick is None:
            continue
        person, ins = pick
        quote = (ins.quotes[0].text if ins.quotes else "") or ins.text
        used.add(str(ins.episode_slug))
        cards.append(
            TrendingEpisodeCard(
                slug=str(ins.episode_slug),
                reason="rising_topic",
                quote=_cut(quote),
                speaker=person.person_name,
                start_ms=ins.start_ms,
                topic_id=topic.entity_id,
                topic_label=topic.label,
            )
        )
    return cards


def _engaged_slugs(
    data_dir: Path | None, *, world: set[str] | None, now: str | None
) -> list[tuple[str, int]]:
    """``(slug, events)`` over the last ENGAGED_WINDOW_WEEKS, most first, floored."""
    if data_dir is None:
        return []
    weeks = set(_weeks_ending(resolve_as_of_week(now), ENGAGED_WINDOW_WEEKS))
    out: list[tuple[str, int]] = []
    for ent in engagement_series(data_dir).get("entities") or []:
        if ent.get("kind") != "episode":
            continue
        slug = str(ent.get("entity_id") or "")
        if not slug or (world is not None and slug not in world):
            continue
        weekly: dict[str, int] = ent.get("weekly_counts") or {}
        events = sum(int(c) for w, c in weekly.items() if w in weeks)
        if events >= ENGAGED_MIN_EVENTS:
            out.append((slug, events))
    out.sort(key=lambda se: (-se[1], se[0]))
    return out


def _engaged_cards(
    root: Path, data_dir: Path | None, *, world: set[str] | None, limit: int, now: str | None
) -> list[TrendingEpisodeCard]:
    cards: list[TrendingEpisodeCard] = []
    for slug, events in _engaged_slugs(data_dir, world=world, now=now):
        if len(cards) >= limit:
            break
        row = resolve_slug(root, slug)
        if row is None or not row.has_gi:
            continue
        insights = insights_from_gi(load_json_artifact(root, row.gi_relative_path), limit=1)
        if not insights:
            continue
        ins = insights[0]
        q = ins.quotes[0] if ins.quotes else None
        cards.append(
            TrendingEpisodeCard(
                slug=slug,
                reason="most_engaged",
                quote=_cut((q.text if q else "") or ins.text),
                speaker=(q.speaker if q else None),
                start_ms=(q.start_ms if q else None),
                events=events,
            )
        )
    return cards


def trending_episodes(
    root: Path,
    data_dir: Path | None,
    *,
    scope: str = "corpus",
    user_id: str | None = None,
    restrict_entities: set[str] | None = None,
    world: set[str] | None = None,
    limit: int = 8,
    config: MomentumConfig | None = None,
    now: str | None = None,
) -> list[TrendingEpisodeCard]:
    """The two halves, interleaved (rising, engaged, rising, …), one card per episode."""
    rising = _rising_topic_cards(
        root,
        data_dir,
        scope=scope,
        user_id=user_id,
        restrict_entities=restrict_entities,
        world=world,
        limit=limit,
        config=config,
        now=now,
    )
    engaged = _engaged_cards(root, data_dir, world=world, limit=limit, now=now)
    out: list[TrendingEpisodeCard] = []
    seen: set[str] = set()
    for i in range(max(len(rising), len(engaged))):
        for half in (rising, engaged):
            if i < len(half) and half[i].slug not in seen and len(out) < limit:
                seen.add(half[i].slug)
                out.append(half[i])
    return out


def card_to_dict(card: TrendingEpisodeCard) -> dict[str, Any]:
    """Plain dict for the response model."""
    return dict(card.__dict__)
