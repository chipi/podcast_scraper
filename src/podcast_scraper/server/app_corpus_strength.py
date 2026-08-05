"""Per-episode corpus **strength** — the ranking layer (RFC-114 Phase 2, #1470).

A transparent, tunable, **no-ML** score in ``[0, 1]`` for how strongly an episode sits in the user's
corpus, from four present signals: how much was heard, how many captures it carries, whether it's
favorited, and how often it was re-listened. Used to rank recall results, order the digest, and pick
"strongest items". Nothing **blocks** on this (Phase 1 membership is the dependency); it only ranks.

Monotonic by construction — every weight is non-negative and every term is non-decreasing in its
signal, so adding any signal (holding the rest) never lowers strength. Comparable **within** a user;
not across users (v1). Recency decay + negative signals (dismissed/skipped) are RFC-114 open
questions, deliberately out of v1.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from podcast_scraper.server import app_user_corpus, app_user_state


@dataclass(frozen=True)
class Weights:
    """Signal weights (sum to 1.0) + saturation caps. Tune here; the formula is otherwise fixed."""

    heard: float = 0.4  # how much of the episode was played (the core signal)
    captures: float = 0.3  # highlights + notes (active engagement)
    favorited: float = 0.1  # a bookmark (weak on its own)
    relistens: float = 0.2  # repeated listens (revisit value)
    captures_cap: int = 5  # captures beyond this add no more signal
    relistens_cap: int = 3  # re-listens beyond this add no more signal


DEFAULT = Weights()


def strength(
    *,
    heard_fraction: float,
    captures: int,
    favorited: bool,
    relistens: int,
    weights: Weights = DEFAULT,
) -> float:
    """The episode's strength in ``[0, 1]`` from its signals (pure; monotonic in each signal)."""
    hf = min(max(heard_fraction, 0.0), 1.0)
    cap = min(max(captures, 0), weights.captures_cap) / weights.captures_cap
    rel = min(max(relistens, 0), weights.relistens_cap) / weights.relistens_cap
    score = (
        weights.heard * hf
        + weights.captures * cap
        + weights.favorited * (1.0 if favorited else 0.0)
        + weights.relistens * rel
    )
    return round(min(max(score, 0.0), 1.0), 4)


def _capture_counts(data_dir: Path, user_id: str) -> dict[str, int]:
    """slug → number of highlights + episode-notes on it (the capture signal)."""
    counts: dict[str, int] = {}
    for h in app_user_state.get_highlights(data_dir, user_id):
        slug = str(h.get("episode_slug") or "")
        if slug:
            counts[slug] = counts.get(slug, 0) + 1
    for note in app_user_state.get_notes(data_dir, user_id, target="episode"):
        slug = str(note.get("target_id") or "")
        if slug:
            counts[slug] = counts.get(slug, 0) + 1
    return counts


def _relisten_counts(data_dir: Path, user_id: str) -> dict[str, int]:
    """slug → re-listens = max(0, opens − 1), from the append-only listen log."""
    opens: dict[str, int] = {}
    for ev in app_user_state.list_listen_events(data_dir, user_id):
        slug = str(ev.get("slug") or "")
        if slug:
            opens[slug] = opens.get(slug, 0) + 1
    return {slug: max(0, n - 1) for slug, n in opens.items()}


def episode_strengths(
    root: Path, data_dir: Path, user_id: str, *, weights: Weights = DEFAULT
) -> dict[str, float]:
    """Strength per **experienced** episode (RFC-114): the ranking map consumers sort by."""
    experienced = app_user_corpus.experienced_episode_set(root, data_dir, user_id)
    if not experienced:
        return {}
    durations = app_user_corpus.slug_durations(root)
    positions = {
        str(p["slug"]): float(p.get("position_seconds") or 0.0)
        for p in app_user_state.list_playback(data_dir, user_id)
    }
    captures = _capture_counts(data_dir, user_id)
    relistens = _relisten_counts(data_dir, user_id)
    favorites = app_user_corpus.saved_episode_set(
        data_dir, user_id
    )  # episode-favorites (kind==episode)
    out: dict[str, float] = {}
    for slug in experienced:
        dur = durations.get(slug, 0.0)
        hf = (positions.get(slug, 0.0) / dur) if dur > 0 else 0.0
        out[slug] = strength(
            heard_fraction=hf,
            captures=captures.get(slug, 0),
            favorited=slug in favorites,
            relistens=relistens.get(slug, 0),
            weights=weights,
        )
    return out
