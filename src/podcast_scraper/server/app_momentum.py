"""Momentum capability (RFC-103 Phase 3) — read-time trending across saveable entities.

Derives **velocity** (rising) + **volume** (recent level) per entity from two durable weekly series
— the enricher's ``content_series`` (mentions/appearances) and the engagement aggregator
(``app_engagement_series``, saves/plays/opens/follows) — via one EWMA oscillator anchored to a
reference week (``today`` in prod, pinned via ``APP_TRENDING_NOW`` in tests). Groups (``tc:`` /
``thc:``) aggregate their members' series; content and engagement are blended per-kind (renormalized
so an entity with only one source still scores on it). Serves the consumer + operator trending
endpoints and the discover ranker — one source of "hot" everywhere.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from podcast_scraper.server.app_corpus_access import load_json_artifact
from podcast_scraper.server.app_engagement_series import engagement_series

_CONTENT_REL = "enrichments/temporal_velocity.json"
_TOPIC_CLUSTERS_REL = "search/topic_clusters.json"
_THEME_CLUSTERS_REL = "enrichments/topic_theme_clusters.json"
_LOOKBACK_WEEKS = 52  # history the EWMA integrates (older weeks are negligible after decay)


# --------------------------------------------------------------------------- #
# Config (RFC-103 §10) — global defaults + per-kind blend overrides.
# --------------------------------------------------------------------------- #
_DEFAULT_BLEND: dict[str, tuple[float, float]] = {  # kind → (w_content, w_engagement)
    "topic": (0.85, 0.15),
    "cluster": (0.85, 0.15),
    "storyline": (0.85, 0.15),
    "person": (0.80, 0.20),
    "episode": (0.50, 0.50),
    "show": (0.60, 0.40),
    "insight": (0.60, 0.40),
}


@dataclass(frozen=True)
class MomentumConfig:
    """Tunable momentum knobs; all overridable via the ``momentum`` config block."""

    fast_half_life_weeks: float = 3.0
    slow_half_life_weeks: float = 12.0
    velocity_threshold: float = 1.5  # τ — velocity ≥ τ ⇒ heating_up
    min_total: int = 3  # sample-noise floor
    min_events_corpus: int = 5  # engagement identifiability floor (corpus scope only)
    blend_default: tuple[float, float] = (0.70, 0.30)
    blend_per_kind: dict[str, tuple[float, float]] = field(
        default_factory=lambda: dict(_DEFAULT_BLEND)
    )

    def blend_for(self, kind: str) -> tuple[float, float]:
        return self.blend_per_kind.get(kind, self.blend_default)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "MomentumConfig":
        d = raw or {}
        ewma = d.get("ewma") or {}
        heat = d.get("heating_up") or {}
        eng = d.get("engagement") or {}
        blend = d.get("blend") or {}
        per_kind = dict(_DEFAULT_BLEND)
        for k, v in (blend.get("per_kind") or {}).items():
            if isinstance(v, dict) and "content" in v and "engagement" in v:
                per_kind[str(k)] = (float(v["content"]), float(v["engagement"]))
        default = blend.get("default") or {}
        return cls(
            fast_half_life_weeks=float(ewma.get("fast_half_life_weeks", 3.0)),
            slow_half_life_weeks=float(ewma.get("slow_half_life_weeks", 12.0)),
            velocity_threshold=float(heat.get("velocity_threshold", 1.5)),
            min_total=int(heat.get("min_total", 3)),
            min_events_corpus=int(eng.get("min_events_corpus", 5)),
            blend_default=(
                float(default.get("content", 0.70)),
                float(default.get("engagement", 0.30)),
            ),
            blend_per_kind=per_kind,
        )


# --------------------------------------------------------------------------- #
# The EWMA momentum primitive.
# --------------------------------------------------------------------------- #
def _alpha(half_life_weeks: float) -> float:
    return float(1.0 - 0.5 ** (1.0 / half_life_weeks))


def _ewma_last(series: list[int], alpha: float) -> float:
    # Warm-start at the first value so a *steady* series reads flat (~1) instead of ramping from
    # zero; a series with leading zeros (a genuinely new/growing entity) still ramps up.
    if not series:
        return 0.0
    prev = float(series[0])
    for x in series[1:]:
        prev = alpha * x + (1.0 - alpha) * prev
    return prev


def momentum(series: list[int], cfg: MomentumConfig) -> tuple[float, float]:
    """(velocity, volume) for a weekly series — velocity = fast÷slow EWMA, volume = fast level."""
    fast = _ewma_last(series, _alpha(cfg.fast_half_life_weeks))
    slow = _ewma_last(series, _alpha(cfg.slow_half_life_weeks))
    velocity = round(fast / slow, 4) if slow > 0 else 0.0
    return velocity, round(fast, 4)


# --------------------------------------------------------------------------- #
# Reference week + series shaping.
# --------------------------------------------------------------------------- #
def _iso_week(dt: datetime) -> str:
    iso = dt.isocalendar()
    return f"{iso.year:04d}-W{iso.week:02d}"


def resolve_as_of_week(now_override: str | None = None) -> str:
    """The reference ISO week: ``APP_TRENDING_NOW`` (ISO date/datetime) → else real today (UTC)."""
    raw = now_override if now_override is not None else os.environ.get("APP_TRENDING_NOW")
    if raw:
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            return _iso_week(dt.astimezone(timezone.utc))
        except (ValueError, TypeError):
            pass
    return _iso_week(datetime.now(timezone.utc))


def _weeks_ending(as_of_week: str, lookback: int = _LOOKBACK_WEEKS) -> list[str]:
    """The ``lookback`` contiguous ISO weeks ending at ``as_of_week`` (inclusive), oldest first."""
    try:
        year, week = as_of_week.split("-W")
        cur = datetime.fromisocalendar(int(year), int(week), 1).replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        cur = datetime.now(timezone.utc)
    weeks = [_iso_week(cur - timedelta(weeks=i)) for i in range(lookback)]
    weeks.reverse()
    return weeks


def _series(weekly_counts: dict[str, int], weeks: list[str]) -> list[int]:
    """Zero-filled contiguous series over ``weeks`` from a sparse ``{week: count}`` map."""
    return [int(weekly_counts.get(w, 0)) for w in weeks]


def _sum_weekly(maps: list[dict[str, int]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for m in maps:
        for w, c in m.items():
            out[w] = out.get(w, 0) + int(c)
    return out


# --------------------------------------------------------------------------- #
# Content momentum — from the enricher's content_series (+ cluster/storyline aggregation).
# --------------------------------------------------------------------------- #
def _content_weekly_by_entity(root: Path) -> dict[tuple[str, str], dict[str, int]]:
    """``(kind, id)`` → weekly_counts for content entities (topic/person/cluster/storyline)."""
    env = load_json_artifact(root, _CONTENT_REL)
    data = (env.get("data", env) if isinstance(env, dict) else {}) or {}
    cs = data.get("content_series") or {}
    out: dict[tuple[str, str], dict[str, int]] = {}
    by_topic: dict[str, dict[str, int]] = {}
    for row in cs.get("topics") or []:
        tid, wc = str(row.get("topic_id") or ""), dict(row.get("weekly_counts") or {})
        if tid:
            out[("topic", tid)] = wc
            by_topic[tid] = wc
    for row in cs.get("persons") or []:
        pid = str(row.get("person_id") or "")
        if pid:
            out[("person", pid)] = dict(row.get("weekly_counts") or {})
    _add_cluster_series(out, by_topic, root, _TOPIC_CLUSTERS_REL, "cluster")
    _add_cluster_series(out, by_topic, root, _THEME_CLUSTERS_REL, "storyline")
    return out


def _add_cluster_series(
    out: dict[tuple[str, str], dict[str, int]],
    by_topic: dict[str, dict[str, int]],
    root: Path,
    rel: str,
    kind: str,
) -> None:
    """Aggregate member topics' weekly series into each cluster/storyline (Σ members)."""
    env = load_json_artifact(root, rel)
    data = (env.get("data", env) if isinstance(env, dict) else {}) or {}
    for cl in data.get("clusters") or []:
        cid = str(cl.get("graph_compound_parent_id") or "")
        if not cid:
            continue
        member_series = [
            by_topic[mid]
            for m in (cl.get("members") or [])
            if (mid := str(m.get("topic_id") or "")) in by_topic
        ]
        if member_series:
            out[(kind, cid)] = _sum_weekly(member_series)


def _engagement_weekly_by_entity(
    data_dir: Path | None, user_id: str | None
) -> dict[tuple[str, str], dict[str, int]]:
    """``(kind, id)`` → weekly engagement counts from the engagement aggregator."""
    if data_dir is None:
        return {}
    data = engagement_series(data_dir, user_id=user_id)
    return {
        (str(e["kind"]), str(e["entity_id"])): dict(e.get("weekly_counts") or {})
        for e in data.get("entities") or []
    }


# --------------------------------------------------------------------------- #
# Blended, ranked trending.
# --------------------------------------------------------------------------- #
@dataclass
class TrendingEntity:
    entity_id: str
    kind: str
    velocity: float
    volume: float
    heating_up: bool
    total: int
    series: list[int]


def _blend(
    kind: str, content: float | None, engagement: float | None, cfg: MomentumConfig
) -> float:
    """Renormalized content⊕engagement blend — an entity with one source scores fully on it."""
    w_c, w_e = cfg.blend_for(kind)
    num = 0.0
    den = 0.0
    if content is not None:
        num += w_c * content
        den += w_c
    if engagement is not None:
        num += w_e * engagement
        den += w_e
    return round(num / den, 4) if den > 0 else 0.0


def trending(
    root: Path,
    data_dir: Path | None,
    *,
    kind: str,
    scope: str = "corpus",
    user_id: str | None = None,
    now: str | None = None,
    limit: int = 12,
    config: MomentumConfig | None = None,
) -> list[TrendingEntity]:
    """Ranked trending entities of ``kind`` (velocity desc), blended over content + engagement."""
    cfg = config or MomentumConfig()
    as_of = resolve_as_of_week(now)
    weeks = _weeks_ending(as_of)
    content = _content_weekly_by_entity(root)
    eng_user = user_id if scope == "mine" else None
    engagement = _engagement_weekly_by_entity(data_dir, eng_user)

    ids = {eid for (k, eid) in content if k == kind} | {eid for (k, eid) in engagement if k == kind}
    out: list[TrendingEntity] = []
    for eid in ids:
        c_wc = content.get((kind, eid))
        e_wc = engagement.get((kind, eid))
        # Corpus-scope engagement identifiability floor (no floor for scope=mine).
        if scope == "corpus" and e_wc is not None and sum(e_wc.values()) < cfg.min_events_corpus:
            e_wc = None
        c_vel, c_vol = momentum(_series(c_wc, weeks), cfg) if c_wc is not None else (None, None)
        e_series = _series(e_wc, weeks) if e_wc is not None else None
        e_vel, e_vol = momentum(e_series, cfg) if e_series is not None else (None, None)
        if c_vel is None and e_vel is None:
            continue
        velocity = _blend(kind, c_vel, e_vel, cfg)
        volume = _blend(kind, c_vol, e_vol, cfg)
        total = sum((c_wc or {}).values()) + sum((e_wc or {}).values())
        series = _series(_sum_weekly([m for m in (c_wc, e_wc) if m is not None]), weeks)
        heating = velocity >= cfg.velocity_threshold and total >= cfg.min_total
        out.append(TrendingEntity(eid, kind, velocity, round(volume, 4), heating, total, series))
    out.sort(key=lambda t: (-t.velocity, -t.volume, t.entity_id))
    return out[: max(limit, 0)]


def content_topic_velocities(
    root: Path, config: MomentumConfig | None = None, now: str | None = None
) -> dict[str, float] | None:
    """Topic content-momentum ``{topic_id: velocity}`` for the discover ranker's trend boost.

    Returns ``None`` when the corpus has no ``content_series`` yet, so the ranker falls back to the
    pre-baked ``velocity_last_over_6mo`` (RFC-103 migration). Content-only (data_dir=None) — the
    ranker's trend signal is about corpus content heating up, not per-user engagement.
    """
    env = load_json_artifact(root, _CONTENT_REL)
    data = (env.get("data", env) if isinstance(env, dict) else {}) or {}
    if "content_series" not in data:
        return None
    rows = trending(root, None, kind="topic", now=now, limit=100_000, config=config)
    return {t.entity_id: t.velocity for t in rows}


__all__ = [
    "MomentumConfig",
    "TrendingEntity",
    "content_topic_velocities",
    "momentum",
    "resolve_as_of_week",
    "trending",
]
