"""``temporal_velocity`` — monthly Topic mention counts + EWMA trend (deterministic).

For each Topic mentioned in any episode, compute:

* ``monthly_counts`` — episode mentions bucketed by ``YYYY-MM`` over
  the last 12 calendar months (zero-filled).
* ``ewma`` — 3-period exponentially-weighted moving average over the
  monthly series (α=0.5), aligned to the latest month.
* ``velocity`` — last-month count divided by the 6-month rolling
  average (gives a "rising vs falling" signal). "Last month" is the
  most recent calendar month with ANY topic activity across the
  corpus, so a stale / partial current month doesn't collapse every
  topic's velocity to zero.
* ``weekly_counts`` — episode mentions bucketed by ISO week
  (``YYYY-Www``) over a trailing weekly window (zero-filled).
* ``weekly_velocity`` — the velocity signal computed at **every** week
  (each week's count over its trailing-average), so callers can plot
  how a topic's momentum actually moved instead of reading one scalar.
* ``content_series`` (RFC-103 Phase 1) — the durable, ``now``-free atom
  the momentum layer reads: full-history per-**topic** and per-**person**
  weekly mention counts (sparse ``weekly_counts`` + a contiguous
  ``window_weeks`` axis). The read-time momentum capability derives
  velocity/volume from this against its own reference week, so it does
  not depend on when this enricher ran.

The monthly/weekly window fields above are computed relative to ``now``
(UTC) and stay as a fallback (``effective_last_month`` flags when "now"
lags the data); ``content_series`` is corpus-anchored and ``now``-free.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)

from podcast_scraper.enrichment.enrichers._loaders import (
    is_unresolved_speaker_placeholder,
    load_kg,
    node_label,
    nodes_of_type,
    publish_date,
)
from podcast_scraper.enrichment.protocol import (
    EnricherManifest,
    EnricherResult,
    EnricherScope,
    EnricherTier,
    EpisodeArtifactBundle,
    RunContext,
    sync_enricher,
)

_DEFAULT_ALPHA = 0.5
_DEFAULT_WINDOW_MONTHS = 12
_DEFAULT_WEEKLY_WINDOW = 26
# Trailing weeks the per-week velocity averages over (the weekly analogue of the
# monthly signal's 6-month denominator; ~2 months keeps it responsive).
_VELOCITY_AVG_WEEKS = 8


def _month_key(date_str: str) -> str | None:
    """Parse an ISO date and return ``YYYY-MM`` (or ``None`` on failure)."""
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00")).strftime("%Y-%m")
    except (ValueError, TypeError):
        return None


def _window_months(now: datetime, window: int) -> list[str]:
    """Return the *window* most-recent month-keys ending at *now*, oldest first."""
    months: list[str] = []
    year, month = now.year, now.month
    for _ in range(window):
        months.append(f"{year:04d}-{month:02d}")
        month -= 1
        if month == 0:
            month = 12
            year -= 1
    months.reverse()
    return months


def _week_key(date_str: str) -> str | None:
    """Parse an ISO date and return an ISO year-week ``YYYY-Www`` (or ``None`` on failure)."""
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    iso = dt.isocalendar()
    return f"{iso.year:04d}-W{iso.week:02d}"


def _window_weeks(now: datetime, window: int) -> list[str]:
    """Return the *window* most-recent ISO week-keys ending at *now*, oldest first.

    Walks back one week at a time so year boundaries (52- vs 53-week years) are
    handled by ``isocalendar`` rather than manual arithmetic.
    """
    weeks: list[str] = []
    cur = now
    for _ in range(window):
        iso = cur.isocalendar()
        weeks.append(f"{iso.year:04d}-W{iso.week:02d}")
        cur -= timedelta(weeks=1)
    weeks.reverse()
    return weeks


def _ewma(series: list[int], alpha: float) -> list[float]:
    """Compute the EWMA over *series* with smoothing factor *alpha*."""
    out: list[float] = []
    prev = 0.0
    for x in series:
        prev = alpha * x + (1 - alpha) * prev
        out.append(round(prev, 4))
    return out


def _velocity(series: list[int], last_idx: int | None = None) -> float:
    """Last-month count over the 6-month trailing average (1.0 = flat).

    *last_idx* lets the caller pick which bucket is the "last" month.
    Defaults to the final element. The 6-month window ends at *last_idx*
    inclusive. Use a non-final ``last_idx`` to skip a partial / stale
    current-month bucket whose count is artificially low.
    """
    if not series:
        return 0.0
    if last_idx is None:
        last_idx = len(series) - 1
    if not 0 <= last_idx < len(series):
        return 0.0
    last = series[last_idx]
    lo = max(0, last_idx - 5)
    six = series[lo : last_idx + 1]
    avg = sum(six) / len(six) if six else 0.0
    if avg == 0:
        return 0.0
    return round(last / avg, 4)


def _velocity_series(series: list[int], avg_weeks: int = _VELOCITY_AVG_WEEKS) -> list[float]:
    """Per-bucket velocity: each bucket's count over its trailing-*avg_weeks* average.

    ``velocity_last_over_6mo`` collapses the trend to a single number; this exposes
    the **actual velocity at every week** (1.0 = flat, >1 rising, <1 cooling) so the
    momentum can be plotted over time — the weekly analogue of the monthly signal.
    Numerator and denominator share the weekly granularity, so 1.0 stays "flat".
    """
    out: list[float] = []
    for i in range(len(series)):
        lo = max(0, i - (avg_weeks - 1))
        window = series[lo : i + 1]
        avg = sum(window) / len(window) if window else 0.0
        out.append(round(series[i] / avg, 4) if avg > 0 else 0.0)
    return out


def _effective_last_idx(counts_by_topic: dict[str, dict[str, int]], months: list[str]) -> int:
    """Find the most recent month with ANY topic activity across the corpus.

    The window's final bucket is the current calendar month. On
    laggy / partial corpora that month has zero data and every topic's
    velocity collapses to ``0 / avg``. Walking back to the most recent
    month with at least one mention anywhere in the corpus gives a
    stable "effective now" that handles both stale data and start-of-
    calendar-month invocations. Falls back to the last index when the
    whole window is empty (vacuously consistent with the old
    behaviour).
    """
    monthly_totals: dict[str, int] = {m: 0 for m in months}
    for monthly in counts_by_topic.values():
        for m, c in monthly.items():
            if m in monthly_totals:
                monthly_totals[m] += c
    for idx in range(len(months) - 1, -1, -1):
        if monthly_totals[months[idx]] > 0:
            return idx
    return len(months) - 1


def _now_utc(config: dict[str, Any]) -> datetime:
    """Use config-provided 'now' for testability (defaults to current UTC)."""
    raw = config.get("now")
    if isinstance(raw, str):
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def _read_alpha(config: dict[str, Any]) -> float:
    raw = config.get("alpha", _DEFAULT_ALPHA)
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_ALPHA
    if not 0.0 < v <= 1.0:
        return _DEFAULT_ALPHA
    return v


def _read_window_months(config: dict[str, Any]) -> int:
    raw = config.get("window_months", _DEFAULT_WINDOW_MONTHS)
    try:
        v = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_WINDOW_MONTHS
    if v < 1 or v > 36:
        return _DEFAULT_WINDOW_MONTHS
    return v


def _read_weekly_window(config: dict[str, Any]) -> int:
    raw = config.get("weekly_window", _DEFAULT_WEEKLY_WINDOW)
    try:
        v = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_WEEKLY_WINDOW
    if v < 4 or v > 104:
        return _DEFAULT_WEEKLY_WINDOW
    return v


def _tally_bundle(
    b: EpisodeArtifactBundle,
    months: list[str],
    weeks_set: set[str],
    monthly: dict[str, dict[str, int]],
    weekly: dict[str, dict[str, int]],
    labels: dict[str, str],
) -> None:
    """Fold one episode's Topic mentions into the monthly + weekly tallies (in place)."""
    kg = load_kg(b)
    date = publish_date(kg)
    if not date:
        return
    raw_month = _month_key(date)
    raw_week = _week_key(date)
    month_key = raw_month if raw_month and raw_month in months else None
    week_key = raw_week if raw_week and raw_week in weeks_set else None
    if month_key is None and week_key is None:
        return
    for t in nodes_of_type(kg, "Topic"):
        tid = str(t.get("id") or "")
        if not tid:
            continue
        labels[tid] = node_label(t)
        if month_key is not None:
            monthly[tid][month_key] += 1
        if week_key is not None:
            weekly[tid][week_key] += 1


def _count_topic_mentions(
    bundles: list[EpisodeArtifactBundle],
    months: list[str],
    weeks_set: set[str],
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]], dict[str, str]]:
    """Bucket every Topic mention into monthly + weekly counts across all episodes.

    Returns ``(monthly_by_topic, weekly_by_topic, labels)``.
    """
    monthly: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    weekly: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    labels: dict[str, str] = {}
    for b in bundles:
        _tally_bundle(b, months, weeks_set, monthly, weekly, labels)
    return monthly, weekly, labels


def _full_week_axis(dates: list[str]) -> list[str]:
    """Contiguous ISO-week axis spanning the corpus's own publish dates (``now``-independent).

    Unlike ``_window_weeks`` (a trailing window ending at ``now``), this is anchored to the corpus,
    so the durable content series is deterministic regardless of when the enricher runs. The
    read-time momentum layer (RFC-103) zero-fills against this axis up to its own reference week.
    """
    parsed: list[datetime] = []
    for d in dates:
        try:
            dt = datetime.fromisoformat(d.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            continue
        # Publish dates in the corpus mix ISO date-only strings
        # (``2026-06-27`` → naive) with ISO datetimes carrying ``Z`` or
        # ``+00:00`` (aware). ``min`` / ``max`` on the mixed list raises
        # ``TypeError: can't compare offset-naive and offset-aware
        # datetimes`` (v1.2.0 prod-v2 regression, 2026-07-17). Coerce
        # naive → UTC so the axis is uniformly aware.
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        parsed.append(dt)
    if not parsed:
        return []
    lo, hi = min(parsed), max(parsed)
    weeks: list[str] = []
    seen: set[str] = set()
    cur = lo
    while cur <= hi:
        wk = _week_key(cur.isoformat())
        if wk and wk not in seen:
            seen.add(wk)
            weeks.append(wk)
        cur += timedelta(days=7)
    hk = _week_key(hi.isoformat())  # loop can stop one step short of hi's week
    if hk and hk not in seen:
        weeks.append(hk)
    return weeks


def _tally_content_week(
    kg: dict[str, Any],
    node_type: str,
    week: str,
    weekly: dict[str, dict[str, int]],
    labels: dict[str, str],
) -> None:
    """Fold one episode's nodes of ``node_type`` into the full-history weekly tally (in place)."""
    for n in nodes_of_type(kg, node_type):
        nid = str(n.get("id") or "")
        if not nid:
            continue
        # Unresolved diarization voices are not real people — keep them out of the
        # trending person series (#1167). Topic ids are never placeholders.
        if node_type == "Person" and is_unresolved_speaker_placeholder(nid, node_label(n)):
            continue
        labels[nid] = node_label(n)
        weekly[nid][week] += 1


def _content_series(bundles: list[EpisodeArtifactBundle]) -> dict[str, Any]:
    """Full-history, ``now``-free per-topic and per-person weekly mention series (RFC-103 Phase 1).

    The durable "content event" atom the momentum layer reads: for every Topic and Person in the
    corpus KG, mentions/appearances bucketed by ISO week over ALL history. ``weekly_counts`` is
    sparse (only weeks with activity); ``window_weeks`` is the contiguous axis for zero-filling at
    read. Emitted alongside the (``now``-anchored) monthly/weekly windows, which stay as fallback.
    """
    weekly_topic: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    weekly_person: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    labels: dict[str, str] = {}
    dates: list[str] = []
    for b in bundles:
        kg = load_kg(b)
        date = publish_date(kg)
        wk = _week_key(date) if date else None
        if not date or not wk:
            continue
        dates.append(date)
        _tally_content_week(kg, "Topic", wk, weekly_topic, labels)
        _tally_content_week(kg, "Person", wk, weekly_person, labels)

    def _rows(weekly: dict[str, dict[str, int]], id_key: str, lab: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = [
            {
                id_key: eid,
                lab: labels.get(eid, eid),
                "weekly_counts": dict(sorted(counts.items())),
                "total": sum(counts.values()),
            }
            for eid, counts in weekly.items()
        ]
        rows.sort(key=lambda r: (-int(r["total"]), r[id_key]))
        return rows

    return {
        "window_weeks": _full_week_axis(dates),
        "topics": _rows(weekly_topic, "topic_id", "topic_label"),
        "persons": _rows(weekly_person, "person_id", "person_label"),
    }


def _topic_row(
    tid: str,
    monthly: dict[str, int],
    weekly: dict[str, int],
    label: str,
    months: list[str],
    weeks: list[str],
    alpha: float,
    effective_idx: int,
) -> dict[str, Any]:
    """One topic's envelope row: monthly counts + EWMA + scalar velocity + weekly series."""
    series = [monthly.get(m, 0) for m in months]
    weekly_series = [weekly.get(w, 0) for w in weeks]
    return {
        "topic_id": tid,
        "topic_label": label,
        "monthly_counts": dict(zip(months, series)),
        "ewma": dict(zip(months, _ewma(series, alpha))),
        "velocity_last_over_6mo": _velocity(series, last_idx=effective_idx),
        "weekly_counts": dict(zip(weeks, weekly_series)),
        "weekly_velocity": dict(zip(weeks, _velocity_series(weekly_series))),
        "total": sum(series),
    }


def _compute(
    bundle: EpisodeArtifactBundle | None,
    corpus_root: Path,
    all_bundles: list[EpisodeArtifactBundle] | None,
    config: dict[str, Any],
    ctx: RunContext,
) -> dict[str, Any]:
    alpha = _read_alpha(config)
    now = _now_utc(config)
    months = _window_months(now, _read_window_months(config))
    weeks = _window_weeks(now, _read_weekly_window(config))
    monthly, weekly, labels = _count_topic_mentions(all_bundles or [], months, set(weeks))
    effective_idx = _effective_last_idx(monthly, months)
    topics_out = [
        _topic_row(
            tid,
            monthly.get(tid, {}),
            weekly.get(tid, {}),
            labels.get(tid, tid),
            months,
            weeks,
            alpha,
            effective_idx,
        )
        for tid in set(monthly) | set(weekly)
    ]
    topics_out.sort(key=lambda r: (-r["velocity_last_over_6mo"], -r["total"], r["topic_id"]))

    # #1208 — no-silent-fail contract. When input is empty (no bundles) or
    # produces an empty output (all bundles carried Topics with no dates or
    # no in-window activity), emit an explicit ``partial_reason`` field so
    # downstream consumers (viewer velocity halo lens, momentum layer) can
    # distinguish "enricher ran cleanly, no data to report" from "enricher
    # never had usable input". Consumers key on ``partial_reason is not None``.
    partial_reason: str | None = None
    bundle_count = len(all_bundles or [])
    if bundle_count == 0:
        partial_reason = "no_bundles"
    elif not topics_out:
        partial_reason = "no_topics_in_window"
    if partial_reason is not None:
        _logger.warning(
            "temporal_velocity empty output run_id=%s enricher=%s "
            "reason=%s bundles=%d months=%d weeks=%d",
            ctx.run_id,
            ctx.enricher_id,
            partial_reason,
            bundle_count,
            len(months),
            len(weeks),
        )

    return {
        "window_months": months,
        "window_weeks": weeks,
        "now": now.isoformat(),
        "alpha": alpha,
        "effective_last_month": months[effective_idx] if months else None,
        "topics": topics_out,
        # #1208 — no-silent-fail marker. See _compute docstring / issue.
        "partial_reason": partial_reason,
        # RFC-103 Phase 1: the durable, now-free content atom the momentum layer reads. The fields
        # above stay as the now-anchored fallback until the read-time capability supersedes them.
        "content_series": _content_series(all_bundles or []),
    }


_enrich_async = sync_enricher(_compute)


class TemporalVelocityEnricher:
    """Corpus-scope monthly Topic mention counts + EWMA + velocity."""

    manifest = EnricherManifest(
        id="temporal_velocity",
        version="1.2.0",
        scope=EnricherScope.CORPUS,
        tier=EnricherTier.DETERMINISTIC,
        reads=[".kg.json"],
        writes="temporal_velocity.json",
        description=(
            "Monthly/weekly Topic mention counts + EWMA + velocity, plus a full-history "
            "now-free content_series (per-topic/person weekly counts) for the momentum layer."
        ),
        expected_duration_s=30,
        config_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "alpha": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "maximum": 1,
                    "default": _DEFAULT_ALPHA,
                    "description": (
                        "EWMA smoothing factor (0 < α ≤ 1). "
                        "Higher = more weight on recent months."
                    ),
                },
                "window_months": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 36,
                    "default": _DEFAULT_WINDOW_MONTHS,
                    "description": "Trailing window size in months for monthly counts + EWMA.",
                },
                "weekly_window": {
                    "type": "integer",
                    "minimum": 4,
                    "maximum": 104,
                    "default": _DEFAULT_WEEKLY_WINDOW,
                    "description": (
                        "Trailing window size in ISO weeks for the weekly counts + "
                        "velocity series."
                    ),
                },
            },
        },
    )

    async def enrich(
        self,
        *,
        bundle: EpisodeArtifactBundle | None,
        corpus_root: Path,
        all_bundles: list[EpisodeArtifactBundle] | None,
        config: dict[str, Any],
        ctx: RunContext,
    ) -> EnricherResult:
        """Enricher.enrich impl — delegates to the sync body via @sync_enricher."""
        return await _enrich_async(bundle, corpus_root, all_bundles, config, ctx)


__all__ = ["TemporalVelocityEnricher"]
