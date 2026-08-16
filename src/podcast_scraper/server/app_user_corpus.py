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
from podcast_scraper.server.app_corpus_access import load_json_artifact
from podcast_scraper.server.app_kg_view import entities_from_kg
from podcast_scraper.server.app_slugs import slug_for_row
from podcast_scraper.server.corpus_catalog import (
    build_catalog_rows_cumulative,
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
    for row in build_catalog_rows_cumulative(root):
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


def derive_interests(
    root: Path,
    data_dir: Path,
    user_id: str,
    *,
    k: int = 8,
    max_episodes: int = 40,
) -> list[str]:
    """Interest tokens inferred from the user's episode set — #1139.

    Aggregates the topics + people across the episodes the user has heard/captured
    (their :func:`user_episode_set`) and returns the top-``k`` by frequency as
    interest tokens (``topic:…`` / ``person:…``). These feed discovery ranking the
    same way an explicit follow does, so personalization works from behaviour alone
    — no picker, no follows needed. Deterministic (frequency desc, id asc as a
    stable tiebreak); bounded to ``max_episodes`` KG loads to keep ``/discover``
    snappy. The ids come from the same :func:`entities_from_kg` the ranker reads,
    so they match its topic/person id space exactly.
    """
    slugs = user_episode_set(root, data_dir, user_id)
    if not slugs:
        return []
    rows_by_slug = {slug_for_row(r): r for r in build_catalog_rows_cumulative(root)}
    counts: Counter[str] = Counter()
    for slug in _most_recently_engaged(data_dir, user_id, slugs, max_episodes):
        row = rows_by_slug.get(slug)
        if row is not None:
            counts.update(_episode_interest_tokens(root, row))
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [token for token, _count in ranked[:k]]


def _most_recently_engaged(
    data_dir: Path, user_id: str, slugs: set[str], limit: int
) -> list[str]:
    """The ``limit`` episodes the user engaged with MOST RECENTLY, newest first.

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

    return sorted(slugs, key=lambda s: (-recency.get(s, 0), s))[:limit]


def _episode_interest_tokens(root: Path, row: CatalogEpisodeRow) -> list[str]:
    """The topic + person ids one episode touches (empty when it has no KG)."""
    if not row.has_kg:
        return []
    artifact = load_json_artifact(root, row.kg_relative_path)
    if artifact is None:
        return []
    persons, _orgs, topics = entities_from_kg(artifact)
    return [t.id for t in topics if t.id] + [p.id for p in persons if p.id]
