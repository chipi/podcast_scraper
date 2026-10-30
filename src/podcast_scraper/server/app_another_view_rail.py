"""Hear another view — Discover arc Step 3 (ADR-163; competitive scan #7).

Breadth through relevance. For the topics the listener engages with most
(``derived_interest_counts`` over their heard-or-captured episodes): a speaker on that topic they
have NOT heard on it, in an episode they have not heard, captured or saved. Each card reads
"Another view on <topic>" over that speaker's quote about it (topic perspectives,
``app_reasons``; extractive only).

"Another" means a speaker the listener does not already hear — not any host or guest of their
heard-or-captured episodes — and a different speaker, not an opposing one: the corpus records
agreement (``enrichments/topic_consensus.json`` keeps pairs below its contradiction threshold) but
not disagreement, so contrast cannot be claimed. What the agreement data does decide: a speaker
recorded as agreeing, on that topic, with someone the listener already heard on it is not another
view and is skipped.

Umbrella topics carried by nearly every episode (``MAX_TOPIC_SHARE``, #1669) are skipped: "another
view on lifelong learning" says nothing. Topics take turns, one card per episode. No LLM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from podcast_scraper.server.app_rail_turns import take_turns
from podcast_scraper.server.app_reasons import (
    best_grounded_quote,
    insights_by_episode,
    Reason,
    ReasonQuote,
    ReasonSubject,
)
from podcast_scraper.server.app_trending_episodes import TrendingEpisodeCard

#: The listener's top topics considered.
MAX_TOPICS = 6
#: A topic carried by at least this share of the corpus's episodes is an umbrella, not a subject.
MAX_TOPIC_SHARE = 0.9
#: Speakers offered per topic, so one topic does not fill the rail.
PER_TOPIC = 2


def engaged_topics(root: Path, data_dir: Path, user_id: str) -> list[tuple[str, str]]:
    """``[(topic id, label)]`` the listener engages with most, umbrellas dropped."""
    from podcast_scraper.server.app_discover_view import interest_episode_index
    from podcast_scraper.server.app_user_corpus import derived_interest_counts
    from podcast_scraper.server.slugs import _slug_index

    index = interest_episode_index(root)
    total = len(_slug_index(root))
    out: list[tuple[str, str]] = []
    for sig in derived_interest_counts(root, data_dir, user_id):
        token = str(sig.get("token") or "")
        if sig.get("kind") != "topic" or not token.startswith("topic:"):
            continue
        if total and len(index.get(token, ())) >= MAX_TOPIC_SHARE * total:
            continue
        out.append((token, str(sig.get("label") or token)))
        if len(out) >= MAX_TOPICS:
            break
    return out


def agreeing_pairs(root: Path) -> set[tuple[str, frozenset[str]]]:
    """``{(topic id, {person a, person b})}`` recorded as agreeing (topic consensus enricher)."""
    from podcast_scraper.search.operators import _load_consensus_payload

    payload = _load_consensus_payload(root) or {}
    data = payload.get("data")
    pairs = data.get("consensus") if isinstance(data, dict) else None
    out: set[tuple[str, frozenset[str]]] = set()
    for p in pairs if isinstance(pairs, list) else []:
        if (
            isinstance(p, dict)
            and p.get("topic_id")
            and p.get("person_a_id")
            and p.get("person_b_id")
        ):
            out.add((str(p["topic_id"]), frozenset({str(p["person_a_id"]), str(p["person_b_id"])})))
    return out


def speakers_heard(root: Path, episodes: set[str]) -> set[str]:
    """Every host and guest of the listener's heard-or-captured episodes (their KG roles). Someone
    they already hear — the host of a show they listen to — is never "another view", whatever the
    topic (2026-10-11: the topic-only rule offered a heard host once his lines moved topic)."""
    from podcast_scraper.server.corpus_access import load_json_artifact
    from podcast_scraper.server.kg_view import entities_from_kg
    from podcast_scraper.server.slugs import resolve_slug

    out: set[str] = set()
    for slug in episodes:
        row = resolve_slug(root, slug)
        if row is None or not row.has_kg:
            continue
        persons, _orgs, _topics = entities_from_kg(load_json_artifact(root, row.kg_relative_path))
        out |= {p.id for p in persons if p.role in ("host", "guest")}
    return out


def _person_key(person: Any) -> str:
    return str(getattr(person, "person_id", None) or getattr(person, "person_name", "") or "")


def another_view_rail(
    root: Path, data_dir: Path, user_id: str, *, limit: int = 6
) -> list[TrendingEpisodeCard]:
    """A different speaker on each of the listener's topics (see the module docstring)."""
    from podcast_scraper.server.app_user_corpus import experienced_episode_set, saved_episode_set
    from podcast_scraper.server.relational_view import build_topic_perspectives

    topics = engaged_topics(root, data_dir, user_id)
    if not topics:
        return []
    heard = experienced_episode_set(root, data_dir, user_id)
    excluded = heard | saved_episode_set(data_dir, user_id)
    agree = agreeing_pairs(root)
    voices = speakers_heard(root, heard)

    per_topic: list[list[tuple[ReasonSubject, ReasonQuote]]] = []
    for tid, label in topics:
        resp = build_topic_perspectives(root, tid, mine_slugs=None)
        perspectives = list(resp.perspectives) if resp else []
        heard_on_topic = {
            _person_key(p)
            for p in perspectives
            if any(str(i.episode_slug or "") in heard for i in p.insights)
        }
        if not heard_on_topic:
            continue  # nobody heard on it yet: there is no first view to offer another to
        subject = ReasonSubject("topic", tid, label)
        found: list[tuple[ReasonSubject, ReasonQuote]] = []
        for p in perspectives:
            key = _person_key(p)
            if key in heard_on_topic or key in voices:
                continue  # someone they already hear (the host of their show) is not "another"
            if any((tid, frozenset({key, h})) in agree for h in heard_on_topic):
                continue  # recorded as agreeing with someone they heard: not another view
            for slug, insights in insights_by_episode(p.insights):
                if slug in excluded:
                    continue
                q = best_grounded_quote(insights, slug, speaker=p.person_name)
                if q is not None:
                    found.append((subject, q))
                    break  # one episode per speaker
        per_topic.append(found)

    picked = take_turns(per_topic, limit=limit, key=lambda f: f[1].episode_slug, per_list=PER_TOPIC)
    return [
        TrendingEpisodeCard(
            slug=q.episode_slug, why=Reason(kind="another_view", quote=q, subject=subject)
        )
        for subject, q in picked
    ]
