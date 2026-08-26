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

import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from podcast_scraper import perf_cache
from podcast_scraper.server.app_catalog_cache import cached_catalog
from podcast_scraper.server.app_corpus_access import cached_json_artifact
from podcast_scraper.server.app_engagement_series import engagement_series
from podcast_scraper.server.app_kg_index import get_kg_index
from podcast_scraper.server.corpus_catalog import aggregate_feeds

_CONTENT_REL = "enrichments/temporal_velocity.json"
_TOPIC_CLUSTERS_REL = "search/topic_clusters.json"
_THEME_CLUSTERS_REL = "enrichments/topic_theme_clusters.json"

_PERSON_ROLES_NS = "app_momentum.person_roles"
# Strongest speaker role wins as a person's headline (host outranks guest outranks mentioned).
_SPEAKER_ROLE_RANK = {"host": 3, "guest": 2, "mentioned": 1}
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
    min_total: int = 3  # sample-noise floor — R2: also the list-INCLUSION floor, not just the badge
    min_events_corpus: int = 5  # engagement identifiability floor (corpus scope only)
    # RFC-103 R2 — monthly window trending.
    default_window: str = "3m"  # 1m | 3m | 6m | 1y — the browse/catch-up cadence
    new_entity_velocity: float = (
        6.0  # velocity for an entity with no pre-window history (new/rising)
    )
    blend_default: tuple[float, float] = (0.70, 0.30)
    blend_per_kind: dict[str, tuple[float, float]] = field(
        default_factory=lambda: dict(_DEFAULT_BLEND)
    )

    def blend_for(self, kind: str) -> tuple[float, float]:
        """The ``(content, engagement)`` blend weights for a kind (per-kind override or default)."""
        return self.blend_per_kind.get(kind, self.blend_default)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "MomentumConfig":
        """Build a MomentumConfig from an operator ``momentum:`` block; missing keys → defaults."""
        d = raw or {}
        ewma = d.get("ewma") or {}
        heat = d.get("heating_up") or {}
        eng = d.get("engagement") or {}
        trend = d.get("trend") or {}
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
            # R2: min_total is the list-inclusion floor; `trend.min_total` overrides the default.
            min_total=int(trend.get("min_total", heat.get("min_total", 3))),
            min_events_corpus=int(eng.get("min_events_corpus", 5)),
            default_window=str(trend.get("default_window", "3m")),
            new_entity_velocity=float(trend.get("new_entity_velocity", 6.0)),
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
# RFC-103 Revision 2 — monthly, corpus-anchored, window-selectable trending.
#
# The weekly fast/slow-EWMA ratio above degenerates on a real back-catalog (weekly buckets are too
# sparse — a single recent mention maxes velocity and every singleton ties). R2 computes the trend
# over MONTHLY counts, a selectable window, anchored to the corpus's latest content month instead of
# wall-clock now. The weekly `series` is retained purely as the full-history sparkline.
# --------------------------------------------------------------------------- #

# UI window presets → number of trailing months that form the "recent" bucket.
WINDOW_MONTHS: dict[str, int] = {"1m": 1, "3m": 3, "6m": 6, "1y": 12}
_LOOKBACK_MONTHS = 24  # history the monthly axis spans (older than the 18-mo corpus is negligible)


def _iso_month(week_key: str) -> str:
    """``2026-W33`` → ``2026-08`` (the calendar month of that ISO week's Monday)."""
    try:
        year, week = week_key.split("-W")
        dt = datetime.fromisocalendar(int(year), int(week), 1)
    except (ValueError, TypeError):
        return ""
    return f"{dt.year:04d}-{dt.month:02d}"


def _monthly_from_weekly(weekly_counts: dict[str, int]) -> dict[str, int]:
    """Roll a sparse ``{iso_week: count}`` map up into ``{YYYY-MM: count}``."""
    out: dict[str, int] = {}
    for wk, c in weekly_counts.items():
        m = _iso_month(wk)
        if m:
            out[m] = out.get(m, 0) + int(c)
    return out


def _latest_content_month(content: dict[tuple[str, str], dict[str, int]]) -> str | None:
    """The most recent calendar month with ANY mention across every entity — the corpus anchor."""
    months = {m for wc in content.values() for wk in wc if (m := _iso_month(wk))}
    return max(months) if months else None


def _latest_content_week(content: dict[tuple[str, str], dict[str, int]]) -> str | None:
    """The most recent ISO week with ANY mention — anchors the full-history weekly sparkline."""
    weeks = {wk for wc in content.values() for wk in wc}
    return max(weeks) if weeks else None


def _months_ending(as_of_month: str, count: int) -> list[str]:
    """The ``count`` contiguous months ending at ``as_of_month`` (inclusive), oldest first."""
    try:
        year, mon = (int(x) for x in as_of_month.split("-"))
    except (ValueError, TypeError):
        return []
    months: list[str] = []
    y, m = year, mon
    for _ in range(count):
        months.append(f"{y:04d}-{m:02d}")
        m -= 1
        if m == 0:
            m, y = 12, y - 1
    months.reverse()
    return months


def _monthly_series(monthly_counts: dict[str, int], months: list[str]) -> list[int]:
    """Zero-filled contiguous monthly series over ``months`` (oldest→newest)."""
    return [int(monthly_counts.get(m, 0)) for m in months]


def window_momentum(
    monthly: list[int], window_months: int, cfg: MomentumConfig
) -> tuple[float, float, int]:
    """(velocity, volume, window_total) for a monthly series anchored to the corpus's latest month.

    ``velocity`` = recent per-month rate ÷ the entity's prior-history per-month rate — >1 rising, <1
    cooling, ~1 flat — measured over the last ``window_months``. Leading zeros (months before the
    entity first appears) are trimmed so velocity reflects the entity's OWN trajectory, not how long
    the corpus predates it. A brand-new entity with no prior history reads as rising (capped).
    ``volume`` = ``window_total`` = mentions in the recent window (also the ``min_total`` floor).
    """
    first = next((i for i, v in enumerate(monthly) if v > 0), len(monthly))
    series = monthly[first:]
    if not series:
        return 0.0, 0.0, 0
    w = max(1, window_months)
    recent, prior = series[-w:], series[:-w]
    window_total = sum(recent)
    recent_rate = window_total / len(recent)
    if prior:
        prior_rate = sum(prior) / len(prior)
        velocity = recent_rate / prior_rate if prior_rate > 0 else float(cfg.new_entity_velocity)
    else:
        # No history before the window → genuinely new; rising by definition, but only a real signal
        # once it clears the min_total floor (a one-off mention is filtered downstream).
        velocity = float(cfg.new_entity_velocity) if window_total > 0 else 0.0
    return round(velocity, 4), float(window_total), window_total


# --------------------------------------------------------------------------- #
# Content momentum — from the enricher's content_series (+ cluster/storyline aggregation).
# --------------------------------------------------------------------------- #
def _show_content_series(root: Path) -> dict[tuple[str, str], dict[str, int]]:
    """``("show", feed_id) → weekly episode-publish counts`` — a show's publishing cadence.

    A whole show has no per-week "mention count" the way a topic does; its *content* momentum
    is how often it ships episodes (RFC-103 §show). Blended with engagement (opens/subscribes)
    this gives "trending show" = publishing actively + people engaging — not merely covering a
    hot topic. Built from catalog publish dates.
    """
    out: dict[tuple[str, str], dict[str, int]] = {}
    for row in cached_catalog(root):
        fid = (row.feed_id or "").strip()
        pub = (row.publish_date or "").strip()
        if not fid or len(pub) < 10:
            continue
        try:
            wk = _iso_week(datetime.fromisoformat(pub[:10]).replace(tzinfo=timezone.utc))
        except ValueError:
            continue
        bucket = out.setdefault(("show", fid), {})
        bucket[wk] = bucket.get(wk, 0) + 1
    return out


def _content_weekly_by_entity(root: Path) -> dict[tuple[str, str], dict[str, int]]:
    """``(kind, id)`` → weekly_counts for content entities (topic/person/cluster/storyline/show)."""
    env = cached_json_artifact(root, _CONTENT_REL)
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
    out.update(_show_content_series(root))  # shows: publishing cadence (RFC-103 §show)
    return out


def _add_cluster_series(
    out: dict[tuple[str, str], dict[str, int]],
    by_topic: dict[str, dict[str, int]],
    root: Path,
    rel: str,
    kind: str,
) -> None:
    """Aggregate member topics' weekly series into each cluster/storyline (Σ members)."""
    env = cached_json_artifact(root, rel)
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
    """One ranked trending entity: its blended momentum score plus the component signals."""

    entity_id: str
    kind: str
    label: str
    velocity: float
    volume: float
    heating_up: bool
    total: int
    series: list[int]
    # Headline speaker role (host/guest/mentioned) for person entities — lets a trending-people list
    # say WHY a person is trending (a busy host vs a recurring guest vs a much-mentioned figure).
    # None for non-person kinds and for people whose KG nodes carry no role.
    role: str | None = None
    # RFC-103 R2 — the trend window this row was ranked under (1m|3m|6m|1y).
    window: str = "3m"


def _readable_id(entity_id: str) -> str:
    """Fallback label from a namespaced id (``topic:risk-management`` → ``risk management``)."""
    return entity_id.split(":", 1)[-1].replace("-", " ").replace("_", " ")


def _labels_from_content(root: Path) -> dict[str, str]:
    env = cached_json_artifact(root, _CONTENT_REL)
    data = (env.get("data", env) if isinstance(env, dict) else {}) or {}
    cs = data.get("content_series") or {}
    out: dict[str, str] = {}
    for row in cs.get("topics") or []:
        out[str(row.get("topic_id") or "")] = str(row.get("topic_label") or "")
    for row in cs.get("persons") or []:
        out[str(row.get("person_id") or "")] = str(row.get("person_label") or "")
    return out


def _labels_from_clusters(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for rel in (_TOPIC_CLUSTERS_REL, _THEME_CLUSTERS_REL):
        env = cached_json_artifact(root, rel)
        data = (env.get("data", env) if isinstance(env, dict) else {}) or {}
        for cl in data.get("clusters") or []:
            cid = str(cl.get("graph_compound_parent_id") or "")
            if cid:
                out[cid] = str(cl.get("canonical_label") or "")
    return out


def _show_labels(root: Path) -> dict[str, str]:
    """``feed_id`` → show display title (so trending shows read as titles, not sha256 ids)."""
    out: dict[str, str] = {}
    for f in aggregate_feeds(cached_catalog(root)):
        fid = str(f.get("feed_id") or "")
        title = str(f.get("display_title") or "")
        if fid and title:
            out[fid] = title
    return out


def _entity_labels(root: Path) -> dict[str, str]:
    """``entity_id`` → display label from content_series + cluster + show titles."""
    labels = {**_labels_from_content(root), **_labels_from_clusters(root), **_show_labels(root)}
    return {k: v for k, v in labels.items() if k and v}


def _person_roles(root: Path) -> dict[str, str]:
    """``person_id`` → strongest speaker role (host > guest > mentioned), cached by corpus mtime.

    Reduces each person's per-episode KG roles to one headline role so a trending-people list can
    say WHY someone is trending (a busy host vs a recurring guest vs a much-mentioned figure).
    Reads the shared KG entity index (already built once per ingest) — a cheap reduce, no re-parse.
    """

    def build() -> dict[str, str]:
        best: dict[str, str] = {}
        for ep in get_kg_index(root).episodes:
            for p in ep.persons:
                role = (p.role or "").strip().lower()
                rank = _SPEAKER_ROLE_RANK.get(role)
                if rank is None:
                    continue
                cur = best.get(p.id)
                if cur is None or rank > _SPEAKER_ROLE_RANK[cur]:
                    best[p.id] = role
        return best

    roles: dict[str, str] = perf_cache.get_or_compute(
        _PERSON_ROLES_NS, str(Path(root).resolve()), perf_cache.corpus_mtime(root), build
    )
    return roles


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
    window: str | None = None,
    config: MomentumConfig | None = None,
) -> list[TrendingEntity]:
    """Ranked trending entities of ``kind`` over the selected ``window`` (RFC-103 R2).

    Velocity is a MONTHLY signal (recent-window rate ÷ the entity's prior-history rate) anchored to
    the corpus's latest content month, not wall-clock now; the list is floored at ``min_total`` and
    ranked by ``velocity × log1p(volume)`` so a genuinely big-and-rising entity beats a tiny recent
    spike. The weekly ``series`` is retained as the full-history sparkline.
    """
    cfg = config or MomentumConfig()
    win_key = window if window in WINDOW_MONTHS else cfg.default_window
    win_months = WINDOW_MONTHS.get(win_key, 3)
    content = _content_weekly_by_entity(root)
    eng_user = user_id if scope == "mine" else None
    engagement = _engagement_weekly_by_entity(data_dir, eng_user)
    labels = _entity_labels(root)
    roles = _person_roles(root) if kind == "person" else {}

    # Anchor to the corpus's latest content month/week — unless a test pins the reference via
    # ``now`` / ``APP_TRENDING_NOW`` (then honour that so fixtures stay deterministic).
    override = now if now is not None else os.environ.get("APP_TRENDING_NOW")
    if override:
        anchor_week = resolve_as_of_week(now)
        anchor_month = _iso_month(anchor_week)
    else:
        anchor_month = _latest_content_month(content) or _iso_month(resolve_as_of_week(now))
        anchor_week = _latest_content_week(content) or resolve_as_of_week(now)
    months = _months_ending(anchor_month, _LOOKBACK_MONTHS)
    weeks = _weeks_ending(anchor_week)

    ids = {eid for (k, eid) in content if k == kind} | {eid for (k, eid) in engagement if k == kind}
    out: list[TrendingEntity] = []
    for eid in ids:
        c_wc = content.get((kind, eid))
        e_wc = engagement.get((kind, eid))
        # Corpus-scope engagement identifiability floor (no floor for scope=mine).
        if scope == "corpus" and e_wc is not None and sum(e_wc.values()) < cfg.min_events_corpus:
            e_wc = None
        c_vel: float | None = None
        c_vol: float | None = None
        c_total = 0
        if c_wc is not None:
            c_vel, c_vol, c_total = window_momentum(
                _monthly_series(_monthly_from_weekly(c_wc), months), win_months, cfg
            )
        e_vel: float | None = None
        e_vol: float | None = None
        e_total = 0
        if e_wc is not None:
            e_vel, e_vol, e_total = window_momentum(
                _monthly_series(_monthly_from_weekly(e_wc), months), win_months, cfg
            )
        if c_vel is None and e_vel is None:
            continue
        velocity = _blend(kind, c_vel, e_vel, cfg)
        volume = _blend(kind, c_vol, e_vol, cfg)
        window_total = c_total + e_total
        # R2: min_total is the list-INCLUSION floor — a trend needs a minimum sample in the window
        # (a one-off mention is an anecdote, not a trend). Not applied to a user's own (mine) data.
        if scope == "corpus" and window_total < cfg.min_total:
            continue
        series = _series(_sum_weekly([m for m in (c_wc, e_wc) if m is not None]), weeks)
        heating = velocity >= cfg.velocity_threshold and window_total >= cfg.min_total
        label = labels.get(eid) or _readable_id(eid)
        out.append(
            TrendingEntity(
                eid,
                kind,
                label,
                velocity,
                round(volume, 4),
                heating,
                window_total,
                series,
                role=roles.get(eid),
                window=win_key,
            )
        )
    # R2: rank by velocity × volume (dampened) so big-and-rising outranks a tiny recent spike.
    out.sort(key=lambda t: (-(t.velocity * math.log1p(t.volume)), -t.velocity, t.entity_id))
    return out[: max(limit, 0)]


def content_topic_velocities(
    root: Path, config: MomentumConfig | None = None, now: str | None = None
) -> dict[str, float] | None:
    """Topic content-momentum ``{topic_id: velocity}`` for the discover ranker's trend boost.

    Returns ``None`` when the corpus has no ``content_series`` yet, so the ranker falls back to the
    pre-baked ``velocity_last_over_6mo`` (RFC-103 migration). Content-only (data_dir=None) — the
    ranker's trend signal is about corpus content heating up, not per-user engagement.
    """
    env = cached_json_artifact(root, _CONTENT_REL)
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
