"""Trending episodes — quote cards, two reasons interleaved (operator 2026-10-10, option "c").

An episode has no weekly "mention count" the way a topic does, so it trends for one of two
reasons, and each card says which:

* ``rising_topic`` — a topic that is rising (topic momentum), shown through the strongest speaker
  take on it: that take's episode, quote and moment.
* ``most_engaged`` — the episodes the most DISTINCT listeners finished, captured, saved or opened
  over the last four weeks (the engagement series, ADR-163 §5); the card's ``events`` is that
  listener count.

Under ``scope=mine`` both halves keep to the listener's world (ADR-162): rising topics of their
own world, episodes in their world only. Built from what is on disk — no LLM, no fetch.

Each card carries its ``Reason`` (``app_reasons``, ADR-163): a card is a QUOTE card, so an insight
without a quote is passed over for one that has one — never filled from the insight's own text.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
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
from podcast_scraper.server.app_reasons import grounded_quote, Reason, ReasonSubject
from podcast_scraper.server.app_relational_view import build_topic_perspectives
from podcast_scraper.server.app_serendipity import (
    episode_entity_ids,
    Habits,
    interleave_outside,
    is_outside_habits,
)
from podcast_scraper.server.app_slugs import resolve_slug

#: How far back "most engaged" looks, in ISO weeks.
ENGAGED_WINDOW_WEEKS = 4
#: Fewer distinct listeners than this is one person's afternoon, not a trend.
ENGAGED_MIN_LISTENERS = 2


@dataclass
class TrendingEpisodeCard:
    """One card: the episode and why it is here (its reason carries the grounded quote)."""

    slug: str
    why: Reason


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
                if not ins.episode_slug or ins.episode_slug in used:
                    continue
                quote = grounded_quote(
                    ins.quotes, str(ins.episode_slug), speaker=person.person_name, insight_id=ins.id
                )
                if quote is not None:
                    pick = quote
                    break
            if pick:
                break
        if pick is None:
            continue
        used.add(pick.episode_slug)
        cards.append(
            TrendingEpisodeCard(
                slug=pick.episode_slug,
                why=Reason(
                    kind="rising_topic",
                    quote=pick,
                    subject=ReasonSubject(kind="topic", id=topic.entity_id, label=topic.label),
                ),
            )
        )
    return cards


def _engaged_slugs(
    data_dir: Path | None, *, world: set[str] | None, now: str | None
) -> list[tuple[str, int]]:
    """``(slug, distinct listeners)`` over the last ENGAGED_WINDOW_WEEKS, floored at
    ``ENGAGED_MIN_LISTENERS``; ranked by the weighted engagement score, then listeners."""
    if data_dir is None:
        return []
    weeks = _weeks_ending(resolve_as_of_week(now), ENGAGED_WINDOW_WEEKS)
    since = min(weeks) if weeks else None
    ranked: list[tuple[int, int, str]] = []
    for ent in engagement_series(data_dir, since_week=since).get("entities") or []:
        if ent.get("kind") != "episode":
            continue
        slug = str(ent.get("entity_id") or "")
        if not slug or (world is not None and slug not in world):
            continue
        listeners = int(ent.get("listeners") or 0)
        if listeners >= ENGAGED_MIN_LISTENERS:
            ranked.append((int(ent.get("total") or 0), listeners, slug))
    ranked.sort(key=lambda r: (-r[0], -r[1], r[2]))
    return [(slug, listeners) for _score, listeners, slug in ranked]


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
        # The episode's own panel (the Brief's insights): the first one with a quote leads.
        insights = insights_from_gi(
            load_json_artifact(root, row.gi_relative_path), unattributed_fallback=True
        )
        quote = next(
            (
                q
                for q in (grounded_quote(i.quotes, slug, insight_id=i.id) for i in insights)
                if q is not None
            ),
            None,
        )
        if quote is None:
            continue
        cards.append(
            TrendingEpisodeCard(
                slug=slug, why=Reason(kind="most_engaged", quote=quote, events=events)
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


def mix_outside_habits(
    root: Path,
    mine: list[TrendingEpisodeCard],
    everyone: list[TrendingEpisodeCard],
    habits: Habits,
    *,
    limit: int,
) -> list[TrendingEpisodeCard]:
    """Mine's cards with the serendipity share (ADR-163 §7): every 5th slot is one of Everyone's
    cards from OUTSIDE the listener's habits (``app_serendipity``), labelled ``outside_habits``."""
    taken = {c.slug for c in mine}
    outside: list[TrendingEpisodeCard] = []
    for card in everyone:
        if card.slug in taken:
            continue
        row = resolve_slug(root, card.slug)
        if row is None:
            continue
        if is_outside_habits(row.feed_id, episode_entity_ids(root, row.kg_relative_path), habits):
            outside.append(card)
            taken.add(card.slug)
    return interleave_outside(
        mine,
        outside,
        limit=limit,
        mark=lambda c: TrendingEpisodeCard(slug=c.slug, why=replace(c.why, outside_habits=True)),
    )


def card_to_dict(card: TrendingEpisodeCard) -> dict[str, Any]:
    """Plain dict for the response model."""
    return {"slug": card.slug, "why": card.why.to_dict()}
