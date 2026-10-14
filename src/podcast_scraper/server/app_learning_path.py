"""Learn X this week — a seven-day path through a topic, theme or storyline (operator 2026-10-11).

Goal-shaped listening, the slow form of the reel (``app_goal_reel``): one step a day for a week,
each step an episode and its best clip ABOUT the subject. The steps are the subject's strongest
episodes (by their best clip, the reel's ranking), then put in the order they were published, so the
week follows how the conversation developed: oldest first.

The path is frozen when the listener starts it (``learning_paths.json``): a corpus that grows during
the week does not move the steps under them. Step ``k`` (0-based) opens on day ``k`` after the
start, counted in whole days since ``started_at`` (UTC); a step can be done late. Up to
``MAX_PATHS`` at once, one per subject (operator 2026-10-11): a fourth needs one stopped, or
replaces one in the same request. Built from what is on disk: no LLM.

A day a day is the rhythm, not a wall (operator 2026-10-11): once the newest open day is done, the
listener may open the next one now (``ahead`` counts the days opened early), so a keen listener can
do the whole path in one sitting.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from podcast_scraper.server.app_goal_reel import (
    hydrate,
    reel_subject,
    ReelKind,
    subject_episodes,
    SubjectEpisode,
)
from podcast_scraper.server.app_moments import MomentsConfig

#: Days in a path, at most; a subject with fewer good episodes gives a shorter path.
PATH_DAYS = 7
#: Paths a listener may follow at once (operator 2026-10-11: "a habit, not a backlog").
MAX_PATHS = 3
#: Fewer steps than this is not a path worth a week.
MIN_STEPS = 3
#: Days one speaker, or one show, may take (operator 2026-10-11: a week of one host is not a path
#: through a subject, it is that host's show). The next-best episodes fill the gaps.
MAX_PER_SPEAKER = 2
MAX_PER_SHOW = 2
_DAY_S = 86_400


@dataclass(frozen=True)
class PathStep:
    """One day of the path: an episode and the clip that is its reason."""

    episode_slug: str
    insight_id: str
    text: str
    quote_text: str
    speaker: Optional[str]
    start_ms: int
    end_ms: int


def path_plan(
    root: Path, subject_id: str, config: Optional[MomentsConfig] = None
) -> list[SubjectEpisode]:
    """Which episodes the path takes, without reading a transcript: the subject's strongest (at
    most ``PATH_DAYS``), at most ``MAX_PER_SPEAKER`` days per speaker and ``MAX_PER_SHOW`` per
    show, the next-best episodes filling the gaps; oldest first. Empty when there are fewer than
    ``MIN_STEPS``. Enough to say whether a subject has a path, and how long."""
    best: list[SubjectEpisode] = []
    by_speaker: dict[str, int] = {}
    by_show: dict[str, int] = {}
    for ep in subject_episodes(root, subject_id, config):
        if len(best) >= PATH_DAYS:
            break
        speaker = (ep.clips[0].speaker or "").strip().casefold()
        show = str(ep.row.feed_id or "")
        if speaker and by_speaker.get(speaker, 0) >= MAX_PER_SPEAKER:
            continue
        if show and by_show.get(show, 0) >= MAX_PER_SHOW:
            continue
        best.append(ep)
        if speaker:
            by_speaker[speaker] = by_speaker.get(speaker, 0) + 1
        if show:
            by_show[show] = by_show.get(show, 0) + 1
    if len(best) < MIN_STEPS:
        return []
    return sorted(best, key=lambda e: (str(e.row.publish_date or "9999"), e.slug))


def build_path(
    root: Path, subject_id: str, config: Optional[MomentsConfig] = None
) -> list[PathStep]:
    """:func:`path_plan`, each episode with its best clip read in full."""
    steps: list[PathStep] = []
    for ep in path_plan(root, subject_id, config):
        for m in hydrate(root, ep.row, ep.clips[:1], config):
            steps.append(
                PathStep(
                    episode_slug=ep.slug,
                    insight_id=m.insight_id,
                    text=m.text,
                    quote_text=m.quote_text,
                    speaker=m.speaker,
                    start_ms=m.start_ms,
                    end_ms=m.end_ms,
                )
            )
    return steps


def subject_label(root: Path, subject_id: str) -> tuple[ReelKind, str]:
    """``(kind, display label)`` for a path's subject."""
    subject = reel_subject(root, subject_id)
    if subject.label:
        return subject.kind, subject.label
    from podcast_scraper.server.app_momentum import _entity_labels

    label = _entity_labels(root).get(subject_id) or subject_id.split(":", 1)[-1].replace("-", " ")
    return subject.kind, str(label)


def start_path(
    root: Path,
    data_dir: Path,
    user_id: str,
    subject_id: str,
    *,
    config: Optional[MomentsConfig] = None,
    now: Optional[float] = None,
    replace: Optional[str] = None,
) -> tuple[str, Optional[dict[str, Any]]]:
    """Freeze a new path for the listener and return ``("ok", path)``. ``("too_few", None)`` when
    the subject has too few good episodes; ``("at_cap", None)`` with ``MAX_PATHS`` already and no
    ``replace``. A subject already being learned returns its path unchanged."""
    from podcast_scraper.server import app_user_state

    existing = app_user_state.get_learning_path(data_dir, user_id, subject_id)
    # One finished is started afresh (operator 2026-10-14); it is in the learning history.
    if existing is not None and app_user_state.learning_path_active(existing):
        return "ok", existing
    steps = build_path(root, subject_id, config)
    if not steps:
        return "too_few", None
    kind, label = subject_label(root, subject_id)
    path = {
        "subject_id": subject_id,
        "subject_kind": kind,
        "subject_label": label,
        "started_at": int(now if now is not None else time.time()),
        "steps": [vars(s) for s in steps],
        "done": [],
    }
    kept = app_user_state.add_learning_path(data_dir, user_id, path, cap=MAX_PATHS, replace=replace)
    return ("ok", kept) if kept is not None else ("at_cap", None)


def day_of(path: dict[str, Any], now: Optional[float] = None) -> int:
    """Whole days since the path started (0 on the first day)."""
    elapsed = (now if now is not None else time.time()) - int(path.get("started_at") or 0)
    return max(0, int(elapsed // _DAY_S))


def open_days(path: dict[str, Any], now: Optional[float] = None) -> int:
    """How many days are open: one per day since the start, plus those opened early."""
    total = len(path.get("steps") or [])
    return min(total, day_of(path, now) + 1 + int(path.get("ahead") or 0))


def unlocked(path: dict[str, Any], index: int, now: Optional[float] = None) -> bool:
    """Step ``index`` is open: its day has come, or the listener opened it early."""
    return 0 <= index < open_days(path, now)


def can_open_next(
    path: dict[str, Any], now: Optional[float] = None, gone: frozenset[int] = frozenset()
) -> bool:
    """The next day may be opened now: the newest open day is done (or its episode has left the
    corpus — ``gone``, step indexes — so it can never be done) and a later one is left."""
    n = open_days(path, now)
    done = {int(i) for i in path.get("done") or []} | gone
    return n < len(path.get("steps") or []) and (n - 1) in done


def next_opens_at(path: dict[str, Any], now: Optional[float] = None) -> Optional[int]:
    """When the next day opens by itself (epoch seconds), or ``None`` with every day open. Days
    count whole 24 h periods from the start, not calendar days."""
    if open_days(path, now) >= len(path.get("steps") or []):
        return None
    return int(path.get("started_at") or 0) + (day_of(path, now) + 1) * _DAY_S
