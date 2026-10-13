"""Discover's rails, paged on the server (ADR-163; operator 2026-10-11).

Discover is a taste, not a scroll: every rail shows its first ``FIRST`` cards, and "Show more" asks
the server for the next page. This module owns the two things that must not live in the browser:

* **Each rail's full ranked list** (``rail_cards``), at most ``RAIL_MAX`` cards, with what the
  listener asked to see less of removed (ADR-163 §6) — so every page of a rail is a slice of ONE
  list.
* **One card per episode on the page** (``first_pages``): rails are taken in page order and an
  episode belongs to the highest rail that has it. Later pages of a rail leave out ``exclude``, the
  episodes the rails above it showed first.

**Under Mine everything on the page is the listener's own** (operator 2026-10-11): a corpus holds
babies, history, cars and sport, and a listener who cares about none of them is not shown them. No
rail mixes in Everyone's cards; breadth is the Everyone scope. And a rail says what to hear NEXT, so
no rail offers an episode the listener already heard, captured from or saved.

Cold start (ADR-163 §6) is decided here too: under Mine with nothing in the listener's world, the
quote rail is Everyone's cards, flagged ``fallback`` for the page to label — never presented as
theirs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from podcast_scraper.server.app_another_view_rail import another_view_rail
from podcast_scraper.server.app_captured_rail import captured_rail
from podcast_scraper.server.app_momentum import MomentumConfig
from podcast_scraper.server.app_people_rail import people_rail
from podcast_scraper.server.app_thread_rail import thread_rail
from podcast_scraper.server.app_trending_episodes import trending_episodes, TrendingEpisodeCard

#: The most cards a rail ranks; "Show more" pages through these.
RAIL_MAX = 12
#: Cards each rail shows before "Show more".
FIRST = 2
#: The rails of Discover, in page order. Everyone sees only the quote cards; the rest are a
#: signed-in listener's own (they read their captures, follows and listening).
EVERYONE_RAILS: tuple[str, ...] = ("quotes",)
MINE_RAILS: tuple[str, ...] = ("quotes", "captured", "thread", "people", "another_view")


@dataclass
class RailContext:
    """What every rail of one request reads; Everyone's cards are computed once and shared."""

    root: Path
    data_dir: Optional[Path]
    user_id: Optional[str]
    scope: str  # "mine" | "corpus"
    config: Optional[MomentumConfig] = None
    _everyone: Optional[list[TrendingEpisodeCard]] = field(default=None, repr=False)
    _less: Optional[set[str]] = field(default=None, repr=False)

    @property
    def mine(self) -> bool:
        return self.scope == "mine" and self.user_id is not None and self.data_dir is not None

    @property
    def uid(self) -> str:
        """The listener (Mine only)."""
        assert self.user_id is not None
        return self.user_id

    @property
    def dd(self) -> Path:
        """Their data dir (Mine only)."""
        assert self.data_dir is not None
        return self.data_dir

    def everyone(self) -> list[TrendingEpisodeCard]:
        """Everyone's trending episodes, read once per request (the cold-start rail's cards)."""
        if self._everyone is None:
            self._everyone = trending_episodes(
                self.root, self.data_dir, scope="corpus", limit=RAIL_MAX * 2, config=self.config
            )
        return self._everyone

    def less(self) -> set[str]:
        """What the listener asked to see less of. Theirs in EVERY scope (operator 2026-10-11): it
        is a preference about recommendations, not about whose cards these are."""
        if self._less is None:
            from podcast_scraper.server import app_user_state

            self._less = (
                app_user_state.suggest_less_tokens(self.data_dir, self.user_id)
                if self.user_id is not None and self.data_dir is not None
                else set()
            )
        return self._less


def rails_for(ctx: RailContext) -> tuple[str, ...]:
    """The rails this request's page shows, in order."""
    return MINE_RAILS if ctx.mine else EVERYONE_RAILS


def without_suggested_less(
    cards: list[TrendingEpisodeCard], tokens: set[str]
) -> list[TrendingEpisodeCard]:
    """Cards whose reason does not name a subject the listener asked to see less of (ADR-163 §6).
    A card with no subject (most engaged) is never removed: "less" has nothing to point at."""
    if not tokens:
        return cards
    return [c for c in cards if c.why.subject is None or c.why.subject.id not in tokens]


def _own(ctx: RailContext, cards: list[TrendingEpisodeCard]) -> list[TrendingEpisodeCard]:
    """A listener's own cards less what they asked to see less of."""
    return without_suggested_less(cards, ctx.less())[:RAIL_MAX]


def rail_cards(ctx: RailContext, rail: str) -> tuple[list[TrendingEpisodeCard], bool]:
    """``(the rail's full ranked list, whether it is the cold-start fallback)``."""
    if rail not in rails_for(ctx):
        return [], False
    if rail == "quotes":
        if not ctx.mine:
            return without_suggested_less(ctx.everyone(), ctx.less())[:RAIL_MAX], False
        from podcast_scraper.server.app_user_corpus import (
            experienced_episode_set,
            personal_entity_ids,
            saved_episode_set,
            world_episode_set,
        )

        world = world_episode_set(ctx.root, ctx.dd, ctx.uid)
        if not world:
            # Cold start: Everyone's, flagged, so the page says whose they are.
            return without_suggested_less(ctx.everyone(), ctx.less())[:RAIL_MAX], True
        entities = personal_entity_ids(ctx.root, ctx.dd, ctx.uid)
        # Rising topics of their world, through any episode about the topic they have not had yet:
        # the topic is what makes the card theirs, and the rail is what to hear next.
        own = trending_episodes(
            ctx.root,
            ctx.data_dir,
            scope="mine",
            user_id=ctx.user_id,
            restrict_entities=entities,
            world=world,
            topics_anywhere=True,
            exclude=experienced_episode_set(ctx.root, ctx.dd, ctx.uid)
            | saved_episode_set(ctx.dd, ctx.uid),
            limit=RAIL_MAX * 2,
            config=ctx.config,
        )
        return _own(ctx, own), False
    build = {
        "captured": captured_rail,
        "thread": thread_rail,
        "people": people_rail,
        "another_view": another_view_rail,
    }[rail]
    return _own(ctx, build(ctx.root, ctx.dd, ctx.uid, limit=RAIL_MAX * 2)), False


@dataclass(frozen=True)
class RailPage:
    """One page of a rail: its cards, how many it has in all, and whether it is the cold-start
    fallback (everyone's cards under Mine)."""

    rail: str
    items: list[TrendingEpisodeCard]
    total: int
    fallback: bool = False


def page_of(
    cards: list[TrendingEpisodeCard], exclude: Iterable[str], offset: int, limit: int
) -> tuple[list[TrendingEpisodeCard], int]:
    """``(cards[offset:offset+limit] after leaving out ``exclude``, how many there are in all)``."""
    skip = set(exclude)
    kept = [c for c in cards if c.slug not in skip]
    return kept[offset : offset + limit], len(kept)


def first_pages(ctx: RailContext, first: int = FIRST) -> list[RailPage]:
    """Every rail's first page, in page order, one card per episode across the page. A rail with
    nothing left after the rails above took their episodes is left out."""
    shown: list[str] = []
    out: list[RailPage] = []
    for rail in rails_for(ctx):
        cards, fallback = rail_cards(ctx, rail)
        items, total = page_of(cards, shown, 0, first)
        if items:
            out.append(RailPage(rail, items, total, fallback))
            shown += [c.slug for c in items]
    return out
