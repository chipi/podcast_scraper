"""Moments — the few strongest points of an episode, played back to back (operator 2026-10-10).

A quick-listening mode pulls the best moments instead of synthesising: each moment is one insight
and a clip of the person making that point.

WHICH insights (``ranking``). The default is ``player``: the order the player already shows
(``insights_from_gi`` — salience, which ties for most of an episode, then extraction order). A
blind check on 100 prod episodes (2026-10-10, ``scripts/eval/moments_eval.py``) found it beats a
score built from depth, topic centrality, insight kind and speaker: the judge scored the player
order 3.02 vs 2.90 and preferred it in 54 episodes to 25. Per signal, depth, centrality and kind
were noise (Spearman −0.05, +0.01, −0.01); extraction order was not (−0.17: the extractor writes
its strongest insights first). ``ranking: "score"`` keeps the score available to retry.

WHAT plays (``clip_rule``). The strongest signal in that check was the clip itself (quote length
+0.42 against the judge's score; model judges also favour longer text, so part of that may be the
judge). The ``segments`` rule starts at the transcript segment the first quote falls in, covers the
insight's quotes that follow closely, and extends by whole segments to at least
``clip_min_target_seconds``, never past ``clip_max_seconds``. ``first_quote`` is the first rule:
the earliest quote alone, cut at the cap.

HOW MANY: one per ``minutes_per_moment`` of episode, clamped, at least ``min_gap_seconds`` apart,
returned in timeline order. Every number is a knob in ``MomentsConfig``, overridable from the
``APP_MOMENTS_CONFIG`` environment variable (JSON), so tuning needs no app release.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields, replace
from typing import Any, Sequence

from podcast_scraper.server.app_gi_view import insights_from_gi
from podcast_scraper.server.schemas import AppInsight, AppQuote

_DEFAULT_WEIGHTS = {"depth": 0.30, "centrality": 0.25, "kind": 0.20, "speaker": 0.10, "clip": 0.15}
_DEFAULT_KIND_SCORES = {"claim": 1.0, "recommendation": 1.0, "question": 0.6, "observation": 0.4}

#: A transcript segment: ``(start_ms, end_ms, text)``.
Segment = tuple[int, int, str]


@dataclass(frozen=True)
class MomentsConfig:
    """Every number that decides which moments play, what plays, and how many."""

    ranking: str = "player"  # "player" | "score"
    clip_rule: str = "segments"  # "segments" | "first_quote"
    minutes_per_moment: float = 6.0
    min_count: int = 5
    max_count: int = 15
    clip_max_seconds: float = 30.0
    min_gap_seconds: float = 180.0
    #: The segments rule extends a clip by whole segments until it is at least this long.
    clip_min_target_seconds: float = 12.0
    #: ...and starts at the segment the first quote falls in, if that is at most this far back.
    clip_lead_max_seconds: float = 6.0
    #: Another quote of the same insight starting within this of the clip's end joins the clip.
    clip_merge_gap_seconds: float = 10.0
    # --- ranking: "score" only ---
    clip_min_seconds: float = 8.0
    clip_ideal_max_seconds: float = 35.0
    depth_cap: int = 3
    weights: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_WEIGHTS))
    kind_scores: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_KIND_SCORES))
    kind_default: float = 0.5

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "MomentsConfig":
        """Defaults, overridden by whatever keys ``raw`` names; unknown keys are ignored."""
        base = cls()
        if not raw:
            return base
        known = {f.name for f in fields(cls)}
        updates: dict[str, Any] = {}
        for key, value in raw.items():
            if key not in known:
                continue
            if key in ("weights", "kind_scores"):
                merged = dict(getattr(base, key))
                merged.update({str(k): float(v) for k, v in dict(value).items()})
                updates[key] = merged
            else:
                updates[key] = type(getattr(base, key))(value)
        return replace(base, **updates)

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "MomentsConfig":
        """``APP_MOMENTS_CONFIG`` as JSON; unset or unparsable means the defaults."""
        raw = (env if env is not None else os.environ).get("APP_MOMENTS_CONFIG", "").strip()
        if not raw:
            return cls()
        try:
            parsed = json.loads(raw)
        except ValueError:
            return cls()
        return cls.from_dict(parsed if isinstance(parsed, dict) else None)


@dataclass(frozen=True)
class Moment:
    """One moment: the insight, the clip that plays for it, and why it was picked."""

    insight_id: str
    text: str
    speaker: str | None
    start_ms: int
    end_ms: int
    score: float
    components: dict[str, float]
    insight_type: str | None = None
    #: The first supporting quote, verbatim.
    quote_text: str = ""
    #: What the listener hears: the transcript of [start_ms, end_ms), or the quotes it covers.
    clip_text: str = ""
    #: Position in the player's order (0 = first), the default ranking.
    player_rank: int = 0


def moment_count(duration_seconds: float, available: int, config: MomentsConfig) -> int:
    """How many moments an episode gets: one per ``minutes_per_moment``, clamped, never more
    than the episode has candidates for."""
    if available <= 0:
        return 0
    per = max(config.minutes_per_moment, 0.1)
    wanted = round(max(duration_seconds, 0.0) / 60.0 / per)
    wanted = max(config.min_count, min(config.max_count, wanted))
    return min(wanted, available)


def _timed(quotes: Sequence[AppQuote]) -> list[AppQuote]:
    ok = [
        q
        for q in quotes
        if q.start_ms is not None and q.end_ms is not None and 0 <= q.start_ms < q.end_ms
    ]
    return sorted(ok, key=lambda q: q.start_ms or 0)


def _segments_text(segments: Sequence[Segment], start: int, end: int) -> str:
    return " ".join(t.strip() for s, e, t in segments if e > start and s < end and t.strip())


def build_clip(
    quotes: Sequence[AppQuote], segments: Sequence[Segment] | None, config: MomentsConfig
) -> tuple[int, int, str] | None:
    """The clip for one insight: ``(start_ms, end_ms, text)``, or None with no timed quote."""
    timed = _timed(quotes)
    if not timed:
        return None
    cap = int(config.clip_max_seconds * 1000)
    first = timed[0]
    start, end = int(first.start_ms or 0), int(first.end_ms or 0)
    if config.clip_rule == "first_quote":
        end = min(end, start + cap)
        return start, end, first.text

    merge_gap = int(config.clip_merge_gap_seconds * 1000)
    covered = [first]
    for q in timed[1:]:
        q_start, q_end = int(q.start_ms or 0), int(q.end_ms or 0)
        if q_start - end <= merge_gap and q_end - start <= cap:
            end = max(end, q_end)
            covered.append(q)

    segs = sorted(segments or [], key=lambda s: s[0])
    if not segs:
        end = min(end, start + cap)
        return start, end, " ".join(q.text for q in covered)

    lead = int(config.clip_lead_max_seconds * 1000)
    holder = next((s for s in reversed(segs) if s[0] <= start), None)
    if holder is not None and start - holder[0] <= lead:
        start = holder[0]
    tail = next((s for s in segs if s[0] < end <= s[1]), None)
    if tail is not None and tail[1] - start <= cap:
        end = tail[1]
    target = int(config.clip_min_target_seconds * 1000)
    for s in segs:
        if end - start >= target:
            break
        if s[0] >= end - 1 and s[1] - start <= cap:
            end = s[1]
    end = min(end, start + cap)
    return start, end, _segments_text(segs, start, end)


def _clip_score(seconds: float, config: MomentsConfig) -> float:
    if seconds <= 0:
        return 0.0
    if seconds < config.clip_min_seconds:
        return seconds / config.clip_min_seconds
    if seconds <= config.clip_ideal_max_seconds:
        return 1.0
    return max(0.3, config.clip_ideal_max_seconds / seconds)


def _topic_links(artifact: dict[str, Any]) -> dict[str, set[str]]:
    """insight id -> the topic ids it is ABOUT."""
    out: dict[str, set[str]] = {}
    for edge in artifact.get("edges") or []:
        if isinstance(edge, dict) and edge.get("type") == "ABOUT":
            out.setdefault(str(edge.get("from")), set()).add(str(edge.get("to")))
    return out


def score_candidates(
    artifact: Any, config: MomentsConfig, segments: Sequence[Segment] | None = None
) -> list[Moment]:
    """Every insight that could be a moment, in the player's order, with its clip and score.
    Candidates are the insights the player shows (``insights_from_gi``: named speakers, never
    ``drop``) that have a timed quote."""
    if not isinstance(artifact, dict):
        return []
    insights: list[AppInsight] = insights_from_gi(artifact)
    links = _topic_links(artifact)
    clips = {ins.id: build_clip(ins.quotes, segments, config) for ins in insights}
    candidates = [ins for ins in insights if clips[ins.id] is not None]

    # Centrality: how many CANDIDATES share each topic, normalised by the most-shared topic.
    topic_weight: dict[str, int] = {}
    for ins in candidates:
        for topic in links.get(ins.id, ()):
            topic_weight[topic] = topic_weight.get(topic, 0) + 1
    top_topic = max(topic_weight.values(), default=0)

    w = config.weights
    total_w = sum(max(v, 0.0) for v in w.values()) or 1.0
    out: list[Moment] = []
    for rank, ins in enumerate(candidates):
        start, end, clip_text = clips[ins.id]  # type: ignore[misc]
        timed = _timed(ins.quotes)
        first = timed[0]
        depth = min(len(timed), config.depth_cap) / max(config.depth_cap, 1)
        topics = links.get(ins.id, ())
        centrality = (
            max(topic_weight[t] for t in topics) / top_topic if topics and top_topic else 0.0
        )
        kind = config.kind_scores.get(ins.insight_type or "", config.kind_default)
        speaker_named = 1.0 if (first.speaker or ins.attributed) else 0.5
        clip = _clip_score((end - start) / 1000.0, config)
        components = {
            "depth": depth,
            "centrality": centrality,
            "kind": kind,
            "speaker": speaker_named,
            "clip": clip,
        }
        score = sum(max(w.get(k, 0.0), 0.0) * v for k, v in components.items()) / total_w
        out.append(
            Moment(
                insight_id=ins.id,
                text=ins.text,
                speaker=first.speaker,
                start_ms=start,
                end_ms=end,
                score=round(score, 4),
                components={k: round(v, 4) for k, v in components.items()},
                insight_type=ins.insight_type,
                quote_text=first.text,
                clip_text=clip_text,
                player_rank=rank,
            )
        )
    return out


def spread_pick(ranked: list[Moment], count: int, min_gap_ms: int) -> list[Moment]:
    """Take moments in ``ranked`` order, skipping any closer than ``min_gap_ms`` to one already
    taken; if spacing leaves the reel short, a second pass at half the gap fills it. Returned in
    timeline order."""
    picked: list[Moment] = []
    for gap in (min_gap_ms, min_gap_ms // 2):
        for m in ranked:
            if len(picked) >= count:
                break
            if m in picked:
                continue
            if all(abs(m.start_ms - p.start_ms) >= gap for p in picked):
                picked.append(m)
        if len(picked) >= count:
            break
    return sorted(picked, key=lambda m: m.start_ms)


def rank_moments(scored: list[Moment], config: MomentsConfig) -> list[Moment]:
    """Order candidates for picking: the player's order, or the score (``ranking``)."""
    if config.ranking == "score":
        return sorted(scored, key=lambda m: (-m.score, m.start_ms))
    return sorted(scored, key=lambda m: m.player_rank)


def pick_moments(
    artifact: Any,
    duration_seconds: float | None,
    config: MomentsConfig | None = None,
    segments: Sequence[Segment] | None = None,
) -> list[Moment]:
    """The episode's moments, in timeline order. ``duration_seconds`` unknown (None or 0) falls
    back to the end of the last clip, which under-counts a long tail but never invents length."""
    cfg = config or MomentsConfig()
    scored = score_candidates(artifact, cfg, segments)
    if not scored:
        return []
    duration = duration_seconds or max(m.end_ms for m in scored) / 1000.0
    count = moment_count(duration, len(scored), cfg)
    return spread_pick(rank_moments(scored, cfg), count, int(cfg.min_gap_seconds * 1000))
