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
    """Every slug the user has captured from — highlights, favourited episodes/insights, notes."""
    slugs: set[str] = set()
    for h in app_user_state.get_highlights(data_dir, user_id):
        if h.get("episode_slug"):
            slugs.add(str(h["episode_slug"]))
    for fav in app_user_state.get_favorites(data_dir, user_id):
        kind, ref = fav.get("kind"), fav.get("ref")
        if kind == "episode" and ref:
            slugs.add(str(ref))
        elif fav.get("slug"):  # saved insight carries its episode slug
            slugs.add(str(fav["slug"]))
    for note in app_user_state.get_notes(data_dir, user_id, target="episode"):
        if note.get("target_id"):
            slugs.add(str(note["target_id"]))
    return slugs


def user_episode_set(root: Path, data_dir: Path, user_id: str) -> set[str]:
    """Assemble the user's heard∪captured episode set from their per-user files + the catalog."""
    playback = app_user_state.list_playback(data_dir, user_id)
    captured = _captured_slugs(data_dir, user_id)
    # Durations are only needed to judge "heard"; skip the catalog scan when there's no playback.
    durations = slug_durations(root) if playback else {}
    return derive_episode_set(playback, captured, durations)


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
    for slug in sorted(slugs)[:max_episodes]:
        row = rows_by_slug.get(slug)
        if row is not None:
            counts.update(_episode_interest_tokens(root, row))
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [token for token, _count in ranked[:k]]


def _episode_interest_tokens(root: Path, row: CatalogEpisodeRow) -> list[str]:
    """The topic + person ids one episode touches (empty when it has no KG)."""
    if not row.has_kg:
        return []
    artifact = load_json_artifact(root, row.kg_relative_path)
    if artifact is None:
        return []
    persons, _orgs, topics = entities_from_kg(artifact)
    return [t.id for t in topics if t.id] + [p.id for p in persons if p.id]
