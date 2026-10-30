"""A board's Brief, its short listen, and boards in search (RFC-119 amendment 2026-10-12).

A board is a user's collection (RFC-119). All three read only what is saved on the board itself: a
saved show, topic or person is not expanded into its episodes (operator 2026-10-12). "The board's
episodes" are its saved episodes plus the episodes of its saved moments, in board order.

Built from what is on disk, like the episode Brief (``app_recap_view``) and short listens
(``app_goal_reel``): no LLM.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from podcast_scraper.server import app_collections_store, app_user_state
from podcast_scraper.server.app_goal_reel import episode_clips, hydrate, ReelClip
from podcast_scraper.server.app_moments import Moment, MomentsConfig
from podcast_scraper.server.app_recap_view import signature_quote
from podcast_scraper.server.app_umbrellas import umbrella_tokens
from podcast_scraper.server.catalog_cache import cached_catalog
from podcast_scraper.server.content_source import row_to_summary
from podcast_scraper.server.corpus_access import load_json_artifact
from podcast_scraper.server.corpus_catalog import aggregate_feeds
from podcast_scraper.server.gi_view import insights_from_gi
from podcast_scraper.server.kg_view import entities_from_kg
from podcast_scraper.server.schemas import (
    AppBoardBrief,
    AppBoardHolds,
    AppBoardPoint,
    AppBoardSubject,
    AppBoardThread,
    AppBoardVoice,
)
from podcast_scraper.server.slugs import resolve_slug

#: A board's short listen is addressed as ``board:<collection id>``, beside ``topic:`` and the rest.
BOARD_SUBJECT_PREFIX = "board:"
#: The Brief reads at most this many of a board's episodes, so a very large board stays one cheap
#: read. Key points come from the first ``MAX_POINTS`` of them, in board order.
MAX_BRIEF_EPISODES = 40
MAX_POINTS = 8
#: The topics, storylines and themes of the board's episodes, most episodes first, as an episode's
#: or a show's "about" lists them; the sheet folds them after the first few.
MAX_TOPICS = 30
MAX_THREADS = 12
#: The people who speak across the board's episodes (operator 2026-10-12: a board has no host or
#: guest of its own, it has its voices, as a topic does).
MAX_VOICES = 12
#: Roles of the people who SPEAK in an episode; "mentioned" people are talked about, not heard.
_SPEAKING_ROLES = frozenset({"host", "guest"})

_SUBJECT_KINDS = frozenset({"topic", "person", "theme", "storyline"})


@dataclass(frozen=True)
class Board:
    """One board as stored: its row, its live items in board order, and its saved moments."""

    meta: dict[str, Any]
    items: list[dict[str, Any]]
    highlights: dict[str, dict[str, Any]]


def load_board(data_dir: Path, user_id: str, collection_id: str) -> Optional[Board]:
    """The user's board, or None when they have no board by that id. A saved moment whose capture
    was deleted since is left out, as the board's own count leaves it out."""
    by_id = {
        str(h["id"]): h for h in app_user_state.get_highlights(data_dir, user_id) if h.get("id")
    }
    rows = app_collections_store.list_collections(data_dir, user_id, live_item_ids=set(by_id))
    meta = next((c for c in rows if c["id"] == collection_id), None)
    if meta is None:
        return None
    items = [
        m
        for m in app_collections_store.get_items(data_dir, user_id, collection_id)
        if m.get("kind") != "highlight" or str(m.get("ref")) in by_id
    ]
    held = {str(m.get("ref")) for m in items if m.get("kind") == "highlight"}
    return Board(meta=meta, items=items, highlights={k: v for k, v in by_id.items() if k in held})


def board_episode_slugs(board: Board) -> list[str]:
    """The board's episodes, in board order: each saved episode, and each saved moment's."""
    out: list[str] = []
    for item in board.items:
        kind, ref = str(item.get("kind")), str(item.get("ref"))
        slug = ref if kind == "episode" else None
        if kind == "highlight":
            slug = str(board.highlights.get(ref, {}).get("episode_slug") or "") or None
        if slug and slug not in out:
            out.append(slug)
    return out


def _holds(board: Board, episodes: int) -> AppBoardHolds:
    kinds = Counter(str(m.get("kind")) for m in board.items)
    return AppBoardHolds(
        episodes=episodes,
        moments=kinds["highlight"],
        shows=kinds["show"],
        subjects=sum(kinds[k] for k in _SUBJECT_KINDS),
        searches=kinds["search"],
        links=kinds["link"],
    )


def _point_text(text: str, quote: Optional[str]) -> str:
    """The insight, or its whole quote when the insight is that quote cut short with "…"."""
    if quote and text.endswith("…") and quote.startswith(text[:-1].rstrip()):
        return quote
    return text


def _point(root: Path, row: Any) -> Optional[AppBoardPoint]:
    """The episode's strongest insight and where its best quote is said, or None without GI."""
    if not row.has_gi:
        return None
    insights = insights_from_gi(load_json_artifact(root, row.gi_relative_path), limit=1)
    if not insights:
        return None
    top = insights[0]
    quote = signature_quote(insights)
    return AppBoardPoint(
        episode=row_to_summary(root, row),
        insight_id=top.id,
        text=_point_text(top.text, quote.text if quote else None),
        quote_text=quote.text if quote else None,
        speaker=quote.speaker if quote else None,
        start_ms=quote.start_ms if quote else None,
    )


def _same_label(a: str, b: str) -> bool:
    """ "Second-order effects" and "second order effects" name one thing."""

    def norm(x: str) -> str:
        return re.sub(r"[\W_]+", " ", x.casefold()).strip()

    return norm(a) == norm(b)


def build_board_brief(root: Path, board: Board) -> AppBoardBrief:
    """The board's Brief: what it holds, its voices, what its episodes are about, and each
    episode's key point. Everything is counted in the board's episodes, most first."""
    from podcast_scraper.search.storylines import (
        storyline_map_by_topic,
        top_storylines_by_member_count,
    )
    from podcast_scraper.search.topic_clusters import theme_map_by_topic
    from podcast_scraper.server.relational_view import hosted_photo_urls

    rows = [
        (slug, row)
        for slug in board_episode_slugs(board)[:MAX_BRIEF_EPISODES]
        if (row := resolve_slug(root, slug)) is not None
    ]
    umbrellas = umbrella_tokens(root)
    saved = {str(m.get("ref")) for m in board.items if m.get("kind") in ("topic", "person")}
    topic_eps: Counter[str] = Counter()
    voice_eps: Counter[str] = Counter()
    labels: dict[str, str] = {}
    storyline_eps: Counter[str] = Counter()
    theme_eps: Counter[str] = Counter()
    theme_labels: dict[str, str] = {}
    storyline_map = storyline_map_by_topic(root)
    theme_map = theme_map_by_topic(root)
    points: list[AppBoardPoint] = []
    for _slug, row in rows:
        if len(points) < MAX_POINTS and (point := _point(root, row)) is not None:
            points.append(point)
        if not row.has_kg:
            continue
        persons, _orgs, topics = entities_from_kg(load_json_artifact(root, row.kg_relative_path))
        topic_ids = {topic.id for topic in topics if topic.id not in umbrellas}
        for topic in topics:
            if topic.id in topic_ids:
                labels.setdefault(topic.id, topic.label)
        topic_eps.update(topic_ids)
        # Who SPEAKS in it. A name known only inside one episode has no card to open.
        speakers = {
            p.id: p.name
            for p in persons
            if p.kind == "person" and p.role in _SPEAKING_ROLES and not p.episode_scoped
        }
        for pid, name in speakers.items():
            labels.setdefault(pid, name)
        voice_eps.update(speakers.keys())
        storyline_eps.update(
            {
                str(storyline_map[t]["storyline_id"])
                for t in topic_ids
                if t in storyline_map and storyline_map[t].get("storyline_id")
            }
            - umbrellas
        )
        for t in topic_ids:
            info = theme_map.get(t)
            if info and info.get("cluster_id") and info["cluster_id"] not in umbrellas:
                theme_labels.setdefault(str(info["cluster_id"]), str(info.get("cluster_label")))
        theme_eps.update(
            {
                str(theme_map[t]["cluster_id"])
                for t in topic_ids
                if theme_map.get(t, {}).get("cluster_id")
            }
            - umbrellas
        )
    about = [
        AppBoardSubject(id=tid, kind="topic", label=labels[tid], episodes=n, saved=tid in saved)
        for tid, n in topic_eps.items()
        if labels.get(tid)
    ]
    about.sort(key=lambda s: (-s.episodes, s.label.casefold()))
    photos = hosted_photo_urls(root) if voice_eps else {}
    voices = [
        AppBoardVoice(
            id=pid, name=labels[pid], episodes=n, image_url=photos.get(pid), saved=pid in saved
        )
        for pid, n in voice_eps.items()
        if labels.get(pid)
    ]
    voices.sort(key=lambda v: (-v.episodes, v.name.casefold()))
    # A storyline opens at its anchor topic, and only one that clears the corpus's surfacing floor
    # has one, as in an episode's Brief (``app_recap_view.episode_storylines``).
    anchors = {c["id"]: c for c in top_storylines_by_member_count(root, top_n=1000)}
    storylines = [
        AppBoardThread(
            id=str(anchors[thc]["anchor_topic_id"]),
            kind="storyline",
            label=str(anchors[thc]["label"]),
            episodes=n,
        )
        for thc, n in storyline_eps.items()
        if thc in anchors
    ]
    # A theme with a storyline's name would read twice in one row: the storyline stays.
    themes = [
        AppBoardThread(id=tc, kind="theme", label=theme_labels[tc], episodes=n)
        for tc, n in theme_eps.items()
        if theme_labels.get(tc)
        and not any(_same_label(theme_labels[tc], s.label) for s in storylines)
    ]
    threads = storylines + themes
    threads.sort(key=lambda t: (-t.episodes, t.kind, t.label.casefold()))
    return AppBoardBrief(
        board_id=str(board.meta["id"]),
        name=str(board.meta.get("name") or ""),
        holds=_holds(board, len(rows)),
        voices=voices[:MAX_VOICES],
        about=about[:MAX_TOPICS],
        points=points,
        threads=threads[:MAX_THREADS],
    )


def _saved_moment_clip(root: Path, highlight: dict[str, Any]) -> Optional[ReelClip]:
    """A saved moment as a clip: it plays what the listener saved. None without a time span."""
    start, end = highlight.get("start_ms"), highlight.get("end_ms")
    slug = str(highlight.get("episode_slug") or "")
    if not isinstance(start, int) or not isinstance(end, int) or end <= start or not slug:
        return None
    if resolve_slug(root, slug) is None:
        return None
    words = str(highlight.get("quote_text") or "")
    return ReelClip(
        episode_slug=slug,
        moment=Moment(
            insight_id=str(highlight.get("source_insight_id") or highlight.get("id")),
            text=words,
            speaker=highlight.get("speaker"),
            start_ms=start,
            end_ms=end,
            score=0.0,
            components={},
            quote_text=words,
        ),
    )


def board_reel(
    root: Path, board: Board, *, minutes: int = 15, config: Optional[MomentsConfig] = None
) -> list[ReelClip]:
    """The board's own moments back to back, within ``minutes``: every saved moment that has a
    time, in board order, then the strongest clip of each saved episode (the clip a topic's short
    listen would take from it). A clip that does not fit is skipped; a shorter one later may."""
    cfg = config or MomentsConfig.from_env()
    budget_ms = int(minutes * 60 * 1000)
    used = 0
    out: list[ReelClip] = []

    def fits(clip: ReelClip) -> bool:
        nonlocal used
        length = clip.moment.end_ms - clip.moment.start_ms
        if used + length > budget_ms:
            return False
        used += length
        out.append(clip)
        return True

    for item in board.items:
        if item.get("kind") == "highlight":
            h = board.highlights.get(str(item.get("ref")))
            if h is not None and (clip := _saved_moment_clip(root, h)) is not None:
                fits(clip)
    saved_from = {c.episode_slug for c in out}
    for item in board.items:
        if item.get("kind") != "episode":
            continue
        slug = str(item.get("ref"))
        row = resolve_slug(root, slug)
        if row is None:
            continue
        # A moment of this episode is saved: that is what the listener chose to hear from it, and
        # its strongest clip as well would play the same episode twice in a row.
        if slug in saved_from:
            continue
        refs = episode_clips(root, row, cfg)
        if not refs:
            continue
        best = min(refs, key=lambda r: (-r.score, r.order))
        for moment in hydrate(root, row, [best], cfg):
            fits(ReelClip(episode_slug=slug, moment=moment))
    return out


# --- boards in search ----------------------------------------------------------------------


def query_words(q: str) -> list[str]:
    """The words a board must contain, every one of them, in any order (as show titles match)."""
    return re.findall(r"\w+", (q or "").casefold())


def _contains(text: Optional[str], words: list[str]) -> bool:
    haystack = (text or "").casefold()
    return bool(words) and all(w in haystack for w in words)


def _deslug(ns_id: str) -> str:
    return ns_id.split(":", 1)[-1].replace("-", " ").replace("_", " ")


class _Labels:
    """Display text for each saved item, read lazily: most searches match a board by name, and a
    show or subject label costs a catalog or enrichment read."""

    def __init__(self, root: Optional[Path]) -> None:
        self.root = root
        self._shows: Optional[dict[str, str]] = None
        self._subjects: Optional[dict[str, str]] = None

    def show(self, feed_id: str) -> Optional[str]:
        if self.root is None:
            return None
        if self._shows is None:
            self._shows = {
                str(f["feed_id"]): str(f.get("title") or "")
                for f in aggregate_feeds(cached_catalog(self.root))
                if f.get("feed_id")
            }
        return self._shows.get(feed_id) or None

    def subject(self, ref: str) -> str:
        if self.root is not None and self._subjects is None:
            from podcast_scraper.server.app_momentum import _entity_labels

            self._subjects = _entity_labels(self.root)
        return (self._subjects or {}).get(ref) or _deslug(ref)

    def episode(self, slug: str) -> Optional[str]:
        if self.root is None:
            return None
        row = resolve_slug(self.root, slug)
        return row.episode_title if row is not None else None


def _item_label(
    item: dict[str, Any], highlights: dict[str, dict], labels: _Labels
) -> Optional[str]:
    kind, ref = str(item.get("kind")), str(item.get("ref"))
    if kind == "highlight":
        return highlights.get(ref, {}).get("quote_text")
    if kind == "episode":
        return labels.episode(ref)
    if kind == "show":
        return labels.show(ref)
    if kind in _SUBJECT_KINDS:
        return labels.subject(ref)
    if kind == "search":
        return ref
    if kind == "link":
        return str(item.get("title") or ref)
    return None


def match_boards(
    root: Optional[Path], data_dir: Path, user_id: str, q: str
) -> list[dict[str, Any]]:
    """The user's boards matching ``q``: those whose name holds every word first, then those
    holding an item whose title or words do. Each is ``{collection, matched, item_kind,
    item_label}``; within each group the user's own board order stands."""
    words = query_words(q)
    if not words:
        return []
    highlights = {
        str(h["id"]): h for h in app_user_state.get_highlights(data_dir, user_id) if h.get("id")
    }
    rows = app_collections_store.list_collections(data_dir, user_id, live_item_ids=set(highlights))
    labels = _Labels(root)
    by_name: list[dict[str, Any]] = []
    by_item: list[dict[str, Any]] = []
    for row in rows:
        if _contains(str(row.get("name") or ""), words):
            by_name.append({"collection": row, "matched": "name"})
            continue
        for item in app_collections_store.get_items(data_dir, user_id, str(row["id"])):
            if item.get("kind") == "highlight" and str(item.get("ref")) not in highlights:
                continue
            label = _item_label(item, highlights, labels)
            if _contains(label, words):
                by_item.append(
                    {
                        "collection": row,
                        "matched": "item",
                        "item_kind": str(item.get("kind")),
                        "item_label": label,
                    }
                )
                break
    return by_name + by_item
