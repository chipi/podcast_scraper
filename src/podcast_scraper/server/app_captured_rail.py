"""More from what you captured — Discover arc Step 2.2 (ADR-163; competitive scan #3, unique).

No app recommends from what a listener highlighted or noted. This rail does: the listener's own
captures (highlights, notes, saves) seed episodes on shows they do NOT follow, linked by the same
topic, person or storyline, each card saying so — "Because you highlighted '…'" — over a quote
from the recommended episode that grounds the link (``app_reasons``, extractive only).

How a capture links out:

* an **insight** capture (or a highlight of one) links through that insight's own topics (its
  ABOUT edges) and the person who said it;
* any other capture links through its episode's topics and people (the episode KG), at most
  ``MAX_LINKS_PER_CAPTURE`` of them;
* a topic also brings in the other topics of its **storyline**.

Candidates are episodes elsewhere — not on a show the listener follows or captured from, and not
one they heard, captured or saved — each with a grounded quote about the linking topic (topic
perspectives) or by the linking person (their quote in that episode). The rail takes captures
newest first, round-robin, so one highlight does not fill it, and one card per episode. Built from
what is on disk: no LLM, no fetch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from podcast_scraper.server.app_reasons import (
    best_grounded_quote,
    insights_by_episode,
    Reason,
    ReasonCapture,
    ReasonQuote,
    ReasonSubject,
)
from podcast_scraper.server.app_trending_episodes import TrendingEpisodeCard

#: How many of the listener's most recent captures seed the rail.
MAX_CAPTURES = 15
#: Topics/people taken from one capture's episode when it has no insight of its own.
MAX_LINKS_PER_CAPTURE = 4
#: Candidates kept per link, so one popular topic does not fill the rail.
PER_LINK = 2


@dataclass(frozen=True)
class Capture:
    """One thing the listener kept."""

    kind: str  # highlight | note | save
    episode_slug: str
    created_at: int
    text: Optional[str] = None
    start_ms: Optional[int] = None
    insight_id: Optional[str] = None


def recent_captures(data_dir: Path, user_id: str, *, limit: int = MAX_CAPTURES) -> list[Capture]:
    """The listener's captures, newest first: highlights, notes on an episode or a highlight, and
    saved episodes or insights."""
    from podcast_scraper.server import app_user_state

    out: list[Capture] = []
    highlights = app_user_state.get_highlights(data_dir, user_id)
    by_id = {str(h.get("id")): h for h in highlights}
    for h in highlights:
        out.append(
            Capture(
                kind="highlight",
                episode_slug=str(h.get("episode_slug") or ""),
                created_at=int(h.get("created_at") or 0),
                text=(h.get("quote_text") or None),
                start_ms=h.get("start_ms") if isinstance(h.get("start_ms"), int) else None,
                insight_id=h.get("source_insight_id") or None,
            )
        )
    for n in app_user_state.get_notes(data_dir, user_id):
        target, tid = n.get("target"), str(n.get("target_id") or "")
        hl = by_id.get(tid) if target == "highlight" else None
        slug = tid if target == "episode" else str((hl or {}).get("episode_slug") or "")
        if slug:
            out.append(
                Capture(
                    kind="note",
                    episode_slug=slug,
                    created_at=int(n.get("created_at") or 0),
                    text=str(n.get("text") or "") or None,
                    insight_id=(hl or {}).get("source_insight_id") or None,
                )
            )
    for f in app_user_state.get_favorites(data_dir, user_id):
        kind, ref = f.get("kind"), str(f.get("ref") or "")
        if kind == "episode" and ref:
            out.append(Capture("save", ref, int(f.get("added_at") or 0)))
        elif kind == "insight" and "#" in ref:
            slug, iid = ref.split("#", 1)
            out.append(Capture("save", slug, int(f.get("added_at") or 0), insight_id=iid))
    out = [c for c in out if c.episode_slug]
    out.sort(key=lambda c: -c.created_at)
    return out[:limit]


def _gi_and_kg(
    root: Path, slug: str
) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]], Any]:
    from podcast_scraper.server.app_corpus_access import load_json_artifact
    from podcast_scraper.server.app_slugs import resolve_slug

    row = resolve_slug(root, slug)
    if row is None:
        return None, None, None
    gi = load_json_artifact(root, row.gi_relative_path) if row.has_gi else None
    kg = load_json_artifact(root, row.kg_relative_path) if row.has_kg else None
    return gi, kg, row


def _label(entity: Any) -> str:
    """A KG entity's display text: topics carry ``label``, people ``name``."""
    return str(getattr(entity, "label", None) or getattr(entity, "name", None) or entity.id)


def capture_links(root: Path, capture: Capture) -> list[ReasonSubject]:
    """What one capture links out through: topics and people (see the module docstring)."""
    from podcast_scraper.server.app_gi_view import insights_from_gi
    from podcast_scraper.server.app_kg_view import entities_from_kg

    gi, kg, _row = _gi_and_kg(root, capture.episode_slug)
    labels: dict[str, str] = {}
    if kg is not None:
        persons, _orgs, topics = entities_from_kg(kg)
        labels = {e.id: _label(e) for e in [*persons, *topics]}
    links: list[ReasonSubject] = []
    if capture.insight_id and gi is not None:
        about = {
            str(e.get("to"))
            for e in gi.get("edges") or []
            if isinstance(e, dict)
            and e.get("type") == "ABOUT"
            and e.get("from") == capture.insight_id
        }
        for tid in sorted(t for t in about if t.startswith("topic:")):
            links.append(ReasonSubject("topic", tid, labels.get(tid, tid.split(":", 1)[-1])))
        insight = next(
            (
                i
                for i in insights_from_gi(gi, unattributed_fallback=True)
                if i.id == capture.insight_id
            ),
            None,
        )
        speaker = next((q.speaker for q in (insight.quotes if insight else []) if q.speaker), None)
        pid = next((k for k, v in labels.items() if k.startswith("person:") and v == speaker), None)
        if pid and speaker:
            links.append(ReasonSubject("person", pid, speaker))
    if not links and kg is not None:
        persons, _orgs, topics = entities_from_kg(kg)
        for e in [*topics, *persons][:MAX_LINKS_PER_CAPTURE]:
            if e.id.startswith("topic:"):
                links.append(ReasonSubject("topic", e.id, _label(e)))
            else:
                links.append(ReasonSubject("person", e.id, _label(e)))
    return links


def _storyline_topics(root: Path, topic_ids: Iterable[str]) -> dict[str, tuple[str, str]]:
    """``{other member topic: (storyline id, label)}`` for the storylines of ``topic_ids``."""
    from podcast_scraper.search.storylines import storyline_map_by_topic

    topic_ids = set(topic_ids)
    smap = storyline_map_by_topic(root)
    wanted = {smap[t]["storyline_id"]: smap[t] for t in topic_ids if t in smap}
    out: dict[str, tuple[str, str]] = {}
    for tid, info in smap.items():
        sid = info.get("storyline_id")
        if sid in wanted and tid not in topic_ids:
            out[tid] = (str(sid), str(info.get("storyline_label") or sid))
    return out


def topic_quotes(root: Path, topic_id: str) -> list[ReasonQuote]:
    """Grounded quotes about ``topic_id``, one per episode, by speaker take (topic perspectives)."""
    from podcast_scraper.server.app_relational_view import build_topic_perspectives

    resp = build_topic_perspectives(root, topic_id, mine_slugs=None)
    out: list[ReasonQuote] = []
    seen: set[str] = set()
    for person in resp.perspectives if resp else []:
        for slug, insights in insights_by_episode(person.insights):
            if slug in seen:
                continue
            q = best_grounded_quote(insights, slug, speaker=person.person_name)
            if q is not None:
                out.append(q)
                seen.add(slug)
    return out


def person_quotes(root: Path, person_id: str, person_name: str) -> list[ReasonQuote]:
    """A quote ``person_name`` said, in each of their episodes (person card episodes, KG-based)."""
    from podcast_scraper.server.app_gi_view import insights_from_gi
    from podcast_scraper.server.app_relational_view import build_person_card

    card = build_person_card(root, person_id)
    out: list[ReasonQuote] = []
    for ep in card.episodes if card else []:
        gi, _kg, _row = _gi_and_kg(root, ep.slug)
        if gi is None:
            continue
        q = best_grounded_quote(
            insights_from_gi(gi, unattributed_fallback=True),
            ep.slug,
            speaker=person_name,
            said_by=person_name,
        )
        if q is not None:
            out.append(q)
    return out


def captured_rail(
    root: Path, data_dir: Path, user_id: str, *, limit: int = 6
) -> list[TrendingEpisodeCard]:
    """Episodes on shows the listener does not follow, each linked to one of their captures."""
    from podcast_scraper.server.app_slugs import resolve_slug
    from podcast_scraper.server.app_user_corpus import (
        experienced_episode_set,
        followed_show_ids,
        saved_episode_set,
    )

    captures = recent_captures(data_dir, user_id)
    if not captures:
        return []
    # Elsewhere = not a show they follow, and not the show the capture came from: more of the same
    # show is not discovery.
    captured_shows = {
        (row.feed_id or "")
        for row in (resolve_slug(root, c.episode_slug) for c in captures)
        if row is not None
    }
    followed = followed_show_ids(data_dir, user_id) | captured_shows
    excluded = (
        experienced_episode_set(root, data_dir, user_id)
        | saved_episode_set(data_dir, user_id)
        | {c.episode_slug for c in captures}
    )

    def eligible(slug: str) -> bool:
        if slug in excluded:
            return False
        row = resolve_slug(root, slug)
        return row is not None and (row.feed_id or "") not in followed

    # Per capture, its candidates in link order: [(capture, subject, quote), ...].
    per_capture: list[list[tuple[Capture, ReasonSubject, ReasonQuote]]] = []
    for cap in captures:
        links = capture_links(root, cap)
        topics = [s.id for s in links if s.kind == "topic"]
        story = _storyline_topics(root, topics)
        found: list[tuple[Capture, ReasonSubject, ReasonQuote]] = []
        for subj in links:
            quotes = (
                topic_quotes(root, subj.id)
                if subj.kind == "topic"
                else person_quotes(root, subj.id, subj.label)
            )
            found += [(cap, subj, q) for q in quotes if eligible(q.episode_slug)][:PER_LINK]
        for tid, (sid, slabel) in sorted(story.items()):
            subj = ReasonSubject("storyline", sid, slabel)
            found += [(cap, subj, q) for q in topic_quotes(root, tid) if eligible(q.episode_slug)][
                :1
            ]
        per_capture.append(found)

    cards: list[TrendingEpisodeCard] = []
    taken: set[str] = set()
    depth = 0
    while len(cards) < limit and any(depth < len(f) for f in per_capture):
        for found in per_capture:
            if depth < len(found) and len(cards) < limit:
                cap, subj, q = found[depth]
                if q.episode_slug in taken:
                    continue
                taken.add(q.episode_slug)
                capture = ReasonCapture(
                    kind=cap.kind,  # type: ignore[arg-type]
                    episode_slug=cap.episode_slug,
                    text=cap.text,
                    start_ms=cap.start_ms,
                )
                cards.append(
                    TrendingEpisodeCard(
                        slug=q.episode_slug,
                        why=Reason(kind="captured", quote=q, subject=subj, capture=capture),
                    )
                )
        depth += 1
    return cards
