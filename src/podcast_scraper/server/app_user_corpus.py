"""The signed-in user's personal episode set (P3 Consolidation, RFC-101 §1 / #1120).

The "corpus" a user can recall over is **read-time derived** from their per-user files (RFC-098) —
no new artifact, no per-user graph. An episode is in the set when the user has **heard** it
(≥``threshold`` of its duration played, default 30%) **or captured** from it (any highlight, saved
insight, or favourite). This set is the scope filter for ``scope=mine`` recall, connections and
resurfacing — recall cites the user's own experience, never the global corpus.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from podcast_scraper.server import app_user_state
from podcast_scraper.server.app_catalog_cache import cached_catalog
from podcast_scraper.server.app_corpus_access import load_json_artifact
from podcast_scraper.server.app_kg_view import entities_from_kg
from podcast_scraper.server.app_slugs import slug_for_row
from podcast_scraper.server.corpus_catalog import (
    CatalogEpisodeRow,
)

HEARD_THRESHOLD = 0.30


def derive_episode_set(
    playback: Iterable[dict],
    captured_slugs: Iterable[str],
    durations: dict[str, float],
    *,
    threshold: float = HEARD_THRESHOLD,
) -> set[str]:
    """Pure: the user's heard∪captured episode slugs.

    ``playback`` rows are ``{slug, position_seconds}``; an episode counts as *heard* when its saved
    position reaches ``threshold`` of its known duration (episodes with unknown/zero duration need a
    capture to qualify — a bare open is not "heard"). ``captured_slugs`` are slugs the user has a
    highlight / saved insight / favourite on; they always qualify.
    """
    heard: set[str] = set()
    for row in playback:
        slug = str(row.get("slug") or "")
        if not slug:
            continue
        dur = durations.get(slug, 0.0)
        if dur > 0 and float(row.get("position_seconds", 0.0)) >= threshold * dur:
            heard.add(slug)
    captured = {str(s) for s in captured_slugs if s}
    return heard | captured


def slug_durations(root: Path) -> dict[str, float]:
    """Map each episode slug to its duration in seconds (0.0 when unknown), from the catalog."""
    out: dict[str, float] = {}
    for row in cached_catalog(root):
        out[slug_for_row(row)] = float(row.duration_seconds or 0)
    return out


def _captured_slugs(data_dir: Path, user_id: str) -> set[str]:
    """Slugs the user engaged with — highlights, notes, saved-**insights** (RFC-114 §1.1).

    NOTE (RFC-114 correction): a whole-**episode** favorite is *saved-for-later*, not engagement, so
    it is NOT captured here (it belongs to the :func:`saved_episode_set` facet). Saved-*insights*
    stay — bookmarking a grounded insight is engagement with that episode's content.
    """
    slugs: set[str] = set()
    for h in app_user_state.get_highlights(data_dir, user_id):
        if h.get("episode_slug"):
            slugs.add(str(h["episode_slug"]))
    for fav in app_user_state.get_favorites(data_dir, user_id):
        # kind == "episode" → the `saved` facet, excluded here. Non-episode favorites (saved
        # insights) carry their episode slug and count as engagement.
        if fav.get("kind") != "episode" and fav.get("slug"):
            slugs.add(str(fav["slug"]))
    for note in app_user_state.get_notes(data_dir, user_id, target="episode"):
        if note.get("target_id"):
            slugs.add(str(note["target_id"]))
    return slugs


def experienced_episode_set(root: Path, data_dir: Path, user_id: str) -> set[str]:
    """The user's `experienced` corpus (RFC-114): heard ∪ highlights ∪ notes ∪ saved-insights.

    Excludes whole-episode favorites (those are `saved`). This is the set recall / connections /
    `scope=mine` read.
    """
    playback = app_user_state.list_playback(data_dir, user_id)
    captured = _captured_slugs(data_dir, user_id)
    # Durations are only needed to judge "heard"; skip the catalog scan when there's no playback.
    durations = slug_durations(root) if playback else {}
    return derive_episode_set(playback, captured, durations)


#: Back-compat alias — recall/connections/digest call ``user_episode_set``; it now returns
#: the corrected ``experienced`` set (episode-favorites removed per RFC-114).
def user_episode_set(root: Path, data_dir: Path, user_id: str) -> set[str]:
    """Alias for :func:`experienced_episode_set` (RFC-114 concept rename; callers unchanged)."""
    return experienced_episode_set(root, data_dir, user_id)


def saved_episode_set(data_dir: Path, user_id: str) -> set[str]:
    """The user's `saved` facet (RFC-114): whole-episode favorites (may overlap `experienced`).

    Pure per-user read (no catalog needed). Consumers that want "saved but not experienced" subtract
    :func:`experienced_episode_set` themselves.
    """
    out: set[str] = set()
    for fav in app_user_state.get_favorites(data_dir, user_id):
        if fav.get("kind") == "episode" and fav.get("ref"):
            out.add(str(fav["ref"]))
    return out


def followed_show_ids(data_dir: Path, user_id: str) -> set[str]:
    """The shows the user follows — their Library subscriptions, by ``feed_id``."""
    return {
        str(s["feed_id"]) for s in app_user_state.get_library(data_dir, user_id) if s.get("feed_id")
    }


def world_episode_set(root: Path, data_dir: Path, user_id: str) -> set[str]:
    """Every EPISODE in the listener's own world — what search "Mine" filters to (2026-10-09).

    Heard or captured from (:func:`experienced_episode_set`), saved (:func:`saved_episode_set`), or
    an episode of a show they follow. A saved SHOW does not pull its episodes in: a save is a
    bookmark, a follow is a subscription. The one meaning of "mine" per kind of item is
    docs/adr — "Mine means the listener's own world".
    """
    episodes = experienced_episode_set(root, data_dir, user_id) | saved_episode_set(
        data_dir, user_id
    )
    follows = followed_show_ids(data_dir, user_id)
    if follows:
        episodes |= {
            slug_for_row(r) for r in cached_catalog(root) if (r.feed_id or "").strip() in follows
        }
    return episodes


def personal_show_ids(root: Path, data_dir: Path, user_id: str) -> set[str]:
    """Every SHOW in the listener's own world, by ``feed_id`` (2026-10-09).

    Followed, saved, or the show of an episode they heard, captured from or saved. Trending shows
    "Mine" used to rank against a set with no shows in it, so it was always empty.
    """
    shows = followed_show_ids(data_dir, user_id)
    for fav in app_user_state.get_favorites(data_dir, user_id):
        if fav.get("kind") == "show" and fav.get("ref"):
            shows.add(str(fav["ref"]))
    episodes = experienced_episode_set(root, data_dir, user_id) | saved_episode_set(
        data_dir, user_id
    )
    if episodes:
        shows |= {
            (r.feed_id or "").strip()
            for r in cached_catalog(root)
            if (r.feed_id or "").strip() and slug_for_row(r) in episodes
        }
    return shows


#: How many of the user's episodes any derived-interest read may load a KG for. One number, because
#: there is one definition — see :func:`derived_interest_counts`.
DERIVED_MAX_EPISODES = 40
#: Default size of a derived-interest list handed to a caller.
DERIVED_TOP_K = 8

#: Half-life, in days, of an engagement's contribution to the derived-interest profile (#24).
#:
#: MEASURED, not chosen by feel. The case decay exists for is a user whose taste has MOVED: 12
#: episodes of outdoor shows 180-235 days ago, then 4 investing episodes in the last week. Without
#: decay their profile is still outdoor — ``topic:personal-finance`` ties the stale tokens at count
#: 4 and loses the alphabetical tie-break, so it misses the top-8 entirely and only one of the five
#: investing tokens survives. The profile describes who they used to be.
#:
#: Sweeping the half-life over that case and over two steady-taste listeners:
#:
#:      half-life   investing tokens in top-8   oldest engagement's weight, light listener
#:      none                    1                          1.000
#:       30d                    5                          0.002
#:       60d                    5                          0.048
#:       90d                    5                          0.132
#:      180d                    4                          0.364
#:      365d                    2                          0.607
#:
#: 90 days is the LARGEST value that still fully recovers the shift (5/5); past it responsiveness
#: falls away (180d -> 4, 365d -> 2). Going lower buys nothing on the shift case and costs the
#: light listener dearly — at 30 days someone who hears an episode a week has their oldest
#: engagement weighted 0.002, which is deletion, not decay. So: the most forgiving half-life that
#: still does the job.
#:
#: Age is measured from the user's OWN most recent engagement, not wall-clock. Someone returning
#: after six months away should find the profile they left, not a flat one — and it keeps this
#: deterministic for fixtures and tests, the same choice ``_recency_boost`` makes by decaying from
#: the newest episode in the pool rather than from today.
DERIVED_HALF_LIFE_DAYS = 90.0

_SECONDS_PER_DAY = 86400.0


def derived_interest_counts(
    root: Path,
    data_dir: Path,
    user_id: str,
    *,
    max_episodes: int = DERIVED_MAX_EPISODES,
) -> list[dict[str, Any]]:
    """THE definition of "what this user is into", as ranked ``{token, kind, label, count}``.

    Every surface that answers that question reads this — ``/discover`` ranking via
    :func:`derive_interests`, ``GET /corpus``'s top entities, and ``GET /interests/derived``.

    It is one function because it was three, and they disagreed. All three counted person/topic
    occurrences across the user's heard∪captured episodes, but each chose ITS OWN episodes:

        derive_interests            recency-ranked, 40   (only after #18 fixed it)
        /corpus  _top_entities      sorted(slugs)[:40]   the alphabetical freeze #18 fixed
        /interests/derived          every episode        no bound at all

    So the same user could be told they are into three different things depending on which screen
    they opened, the #18 fix reached one of the three, and a heavy listener's ``/interests/derived``
    did an unbounded number of KG loads. They also differed in token FORMAT until d390f7b0 (the
    doubled ``topic:topic:`` prefix) and still differed in shape, bounds and tie-breaks.

    Two sources of truth for one concept is how they drifted apart; the fix is to have one, with
    the callers projecting from it rather than re-deriving it.

    Ranking is by TIME-DECAYED weight descending, token ascending — deterministic, so a tie does
    not reorder between reads. ``count`` stays the raw number of episodes the token occurs in,
    because that is what the UI says out loud ("in 4 of your episodes"); ``weight`` is what the
    order is actually built from. Ranking on raw counts alone made this a pure accumulator with no
    way to ever forget, so a taste the user had moved on from outranked the one they had moved to
    (#24) — see :data:`DERIVED_HALF_LIFE_DAYS` for the measurement.
    """
    slugs = user_episode_set(root, data_dir, user_id)
    if not slugs:
        return []
    rows_by_slug = {slug_for_row(r): r for r in cached_catalog(root)}
    counts: Counter[tuple[str, str]] = Counter()
    weights: dict[tuple[str, str], float] = {}
    labels: dict[tuple[str, str], str] = {}
    engaged = _most_recently_engaged(data_dir, user_id, slugs, max_episodes)
    for slug, decay in _decayed(engaged):
        row = rows_by_slug.get(slug)
        if row is None:
            continue
        for kind, ent_id, label in _episode_entities(root, row):
            key = (kind, ent_id)
            counts[key] += 1
            weights[key] = weights.get(key, 0.0) + decay
            labels.setdefault(key, label or ent_id)
    ranked = sorted(weights.items(), key=lambda kv: (-kv[1], interest_token(*kv[0])))
    return [
        {
            "token": interest_token(kind, ent_id),
            "kind": kind,
            "label": labels[(kind, ent_id)],
            "count": counts[(kind, ent_id)],
            "weight": round(weight, 6),
        }
        for (kind, ent_id), weight in ranked
    ]


def _decayed(
    engaged: list[tuple[str, int]], *, half_life_days: float = DERIVED_HALF_LIFE_DAYS
) -> list[tuple[str, float]]:
    """``(slug, decay)`` for each engagement, newest weighted 1.0 and older ones halving.

    Two degenerate inputs matter, because the timestamps come from user files that may predate the
    fields being written:

    * **no engagement carries a timestamp** — every weight is 1.0, i.e. exactly the old
      count-ranked behaviour. A corpus without engagement metadata degrades to "unranked by time",
      never to "empty";
    * **some do, some do not** — an untimed episode inherits the OLDEST known engagement's decay
      rather than an age of 55 years off the epoch. It sorts last but stays eligible, which is the
      promise :func:`_most_recently_engaged` already makes; treating a missing timestamp as
      infinitely old would silently delete those episodes from the profile instead.
    """
    if not engaged or half_life_days <= 0:
        return [(slug, 1.0) for slug, _ts in engaged]
    known = [ts for _slug, ts in engaged if ts > 0]
    if not known:
        return [(slug, 1.0) for slug, _ts in engaged]
    newest, oldest = max(known), min(known)
    out: list[tuple[str, float]] = []
    for slug, ts in engaged:
        effective = ts if ts > 0 else oldest
        age_days = max(0.0, (newest - effective) / _SECONDS_PER_DAY)
        out.append((slug, float(2.0 ** (-age_days / half_life_days))))
    return out


def derive_interests(
    root: Path,
    data_dir: Path,
    user_id: str,
    *,
    k: int = DERIVED_TOP_K,
    max_episodes: int = DERIVED_MAX_EPISODES,
) -> list[str]:
    """The top-``k`` derived interest tokens — the ``/discover`` ranker's projection of #1139.

    A thin view over :func:`derived_interest_counts`: same episodes, same counts, same order, just
    tokens. These feed discovery ranking exactly as an explicit follow does, so personalisation
    works from behaviour alone — no picker, no follows needed. The ids come from the same
    :func:`entities_from_kg` the ranker reads, so they match its id space exactly.
    """
    counts = derived_interest_counts(root, data_dir, user_id, max_episodes=max_episodes)
    return [row["token"] for row in counts[:k]]


def personal_entity_ids(root: Path, data_dir: Path, user_id: str) -> set[str]:
    """Every entity in this listener's own world — what Trends "Mine" ranks (operator 2026-10-07).

    Topics and people they follow, saved, or met in episodes they heard, captured from
    (:func:`derived_interest_counts`, all of it — not the top-k the ranker uses) or saved, the
    shows of their world (:func:`personal_show_ids`), plus the themes
    (``tc:``) and storylines (``thc:``) they follow or that contain one of those topics. "Mine"
    used to blend the user's engagement into the corpus-wide list, which for a light listener
    left it indistinguishable from "everyone"; this is the set it filters to instead.
    """
    from podcast_scraper.search.storylines import storyline_map_by_topic
    from podcast_scraper.search.topic_clusters import theme_map_by_topic

    ids: set[str] = {str(t) for t in app_user_state.get_interests(data_dir, user_id) if str(t)}
    for fav in app_user_state.get_favorites(data_dir, user_id):
        kind, ref = str(fav.get("kind") or ""), str(fav.get("ref") or "")
        if kind in ("topic", "person", "storyline", "theme") and ref:
            ids.add(ref)
    ids |= {str(row["token"]) for row in derived_interest_counts(root, data_dir, user_id)}
    ids |= _saved_episode_entity_tokens(root, data_dir, user_id)
    ids |= personal_show_ids(root, data_dir, user_id)
    topics = {t for t in ids if t.startswith("topic:")}
    if topics:
        for topic_id, info in theme_map_by_topic(root).items():
            if topic_id in topics and info.get("cluster_id"):
                ids.add(str(info["cluster_id"]))
        for topic_id, info in storyline_map_by_topic(root).items():
            if topic_id in topics and info.get("storyline_id"):
                ids.add(str(info["storyline_id"]))
    return ids


def _saved_episode_entity_tokens(root: Path, data_dir: Path, user_id: str) -> set[str]:
    """Topics and people of the episodes the user SAVED — theirs for Trends "Mine" (2026-10-09).

    Kept out of :func:`derived_interest_counts` on purpose: that is "what this user is into" for
    the discover ranker too, and a bookmark is weaker evidence than a listen. Bounded like it — the
    newest :data:`DERIVED_MAX_EPISODES` saves (favorites are stored oldest first), one KG load each.
    """
    saved = [
        str(f["ref"])
        for f in app_user_state.get_favorites(data_dir, user_id)
        if f.get("kind") == "episode" and f.get("ref")
    ][-DERIVED_MAX_EPISODES:]
    if not saved:
        return set()
    rows_by_slug = {slug_for_row(r): r for r in cached_catalog(root)}
    out: set[str] = set()
    for slug in saved:
        row = rows_by_slug.get(slug)
        if row is not None:
            out |= {
                interest_token(kind, ent_id) for kind, ent_id, _ in _episode_entities(root, row)
            }
    return out


def interest_token(kind: str, ent_id: str) -> str:
    """``kind:id``, without doubling a prefix the id already carries.

    Real ids from :func:`entities_from_kg` already carry ``person:`` / ``topic:``, so prepending
    unconditionally produced ``topic:topic:systems-thinking`` — a token that can never match
    anything the ranker holds (d390f7b0). Conditional, so hand-written ids like ``t:ai`` still get
    a prefix.
    """
    prefix = f"{kind}:"
    return ent_id if ent_id.startswith(prefix) else f"{prefix}{ent_id}"


def _most_recently_engaged(
    data_dir: Path, user_id: str, slugs: set[str], limit: int
) -> list[tuple[str, int]]:
    """The ``limit`` episodes the user engaged with MOST RECENTLY, newest first, with their times.

    Returns ``(slug, engagement_ts)`` — ``0`` when the records carry no usable timestamp. The
    timestamp was always computed here and thrown away at the return; #24 needed it to weight each
    episode's contribution, and re-deriving it in the caller would have been the same two-sources
    -of-truth mistake #28 collapsed.

    Derivation has to be bounded — it costs one KG load per episode — but *which* episodes get
    dropped decides whether the interest profile keeps up with the user. It used to be
    ``sorted(slugs)[:limit]``, i.e. lexicographic, and slugs are ``{feed-slug}-{hash}`` — so the
    sort grouped by SHOW. Past ``limit`` episodes, derivation only ever read the alphabetically
    first shows, and new listening stopped moving the profile at all: it froze, permanently, biased
    by how the feed ids happen to be spelled. Exactly backwards for a signal meant to track what
    someone is into lately.

    Recency comes from the same records the episode set is built from — playback ``updated_at``,
    highlight/note ``created_at``. An episode with no timestamp sorts last but is still eligible,
    so a corpus without engagement metadata degrades to "some bounded subset" rather than none.
    Ties break on the slug so the result stays deterministic.
    """
    recency: dict[str, int] = {}

    def _bump(slug: str, ts: Any) -> None:
        if not slug or slug not in slugs:
            return
        try:
            value = int(ts)
        except (TypeError, ValueError):
            return
        if value > recency.get(slug, 0):
            recency[slug] = value

    for row in app_user_state.list_playback(data_dir, user_id):
        _bump(str(row.get("slug") or ""), row.get("updated_at"))
    for highlight in app_user_state.get_highlights(data_dir, user_id):
        _bump(str(highlight.get("episode_slug") or ""), highlight.get("created_at"))
    for note in app_user_state.get_notes(data_dir, user_id, target="episode"):
        _bump(str(note.get("target_id") or ""), note.get("created_at"))

    ordered = sorted(slugs, key=lambda s: (-recency.get(s, 0), s))[:limit]
    return [(slug, recency.get(slug, 0)) for slug in ordered]


def _episode_entities(root: Path, row: CatalogEpisodeRow) -> list[tuple[str, str, str]]:
    """``(kind, id, label)`` for every topic + person one episode touches; empty without a KG."""
    if not row.has_kg:
        return []
    artifact = load_json_artifact(root, row.kg_relative_path)
    if artifact is None:
        return []
    persons, _orgs, topics = entities_from_kg(artifact)
    return [("topic", t.id, t.label) for t in topics if t.id] + [
        ("person", p.id, p.name) for p in persons if p.id
    ]
