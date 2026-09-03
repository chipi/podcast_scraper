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
import math
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

#: Minimum total in-window mentions before a topic gets a velocity series (#1930/#1931).
#:
#: A topic mentioned once has no trend to measure, and the corpus is overwhelmingly made of those:
#: 8,743 of 9,345 topics (93.6%) have ``total == 1`` on the 1,066-episode corpus. Emitting a
#: 128-bucket series for each produced a **65 MB artifact that is ~94% zeros**, costing 2.2s of
#: JSON parse on the first read after every ingest so the trending rail can render ~12 rows.
#:
#: 2 is chosen to match the guard its sibling enrichers already use (``topic_theme_clusters``
#: ``min_pair_episode_count``, ``guest_coappearance`` ``community_min_pair``) — the two corpus-scope
#: enrichers whose output is usable. Measured effect: 9,345 -> 602 topics, 65 MB -> ~4 MB, and
#: **no loss to any consumer**, because ``/api/app/corpus/trending-topics`` already defaults to
#: ``min_total=3`` and therefore never surfaced a single-mention topic in the first place.
#:
#: Deliberately a floor on *emission*, not on counting: mentions still accumulate normally, so a
#: topic crossing the floor next run gets its full window with no backfill.
#: Pseudo-count controlling how hard a sparse velocity estimate is pulled toward flat (#1931).
#:
#: Acts as a prior of "this many mentions of evidence for 1.0". Larger = more conservative.
#: 3 is chosen so a single mention retains 1/4 of its apparent movement (enough to appear, not
#: enough to top the ranking) while a topic with a dozen mentions retains 4/5 and ranks on its
#: own merit. Measured effect on the 1,066-episode corpus: 4 distinct velocity values -> a
#: continuous distribution, and the top of the ranking stops being single-mention topics.
_VELOCITY_PRIOR_MENTIONS = 3.0

#: E-folding constant (in weeks) for the recency decay in ``trend_score`` (#1931).
#:
#: NOT a half-life: the weight is ``exp(-age_weeks / 12)``, so 12 weeks is where a mention
#: decays to 1/e (~37%), not to 50%. The half-life is ``12 * ln 2`` ~= 8.3 weeks.
#:
#: ``velocity_last_over_6mo`` answers "is this accelerating?" — a RATIO, and on a sparse corpus a
#: ratio cannot separate "discussed once, recently" from "discussed all year". Every sustained
#: topic sits at raw ~1.0 (flat) while a single recent mention scores the maximum, so no amount of
#: shrinkage reorders them: measured, the top 10 stayed 7/10 single-mention topics at every prior
#: from 3 to 10.
#:
#: Discoverability needs a different question — "what is being talked about, lately, repeatedly?" —
#: which is volume, not acceleration. ``trend_score`` answers that: mentions decayed by recency,
#: scaled by how many distinct weeks they span, so one big week cannot beat sustained presence and
#: an old burst fades. On the 1,066-episode corpus it surfaces ``open source ai models``,
#: ``ai regulation``, ``ai in education``, ``federal reserve policy``, ``us-china ai competition``.
#:
#: CORRECTION (2026-09-03). An earlier version of this note claimed those results carried "ZERO
#: single-mention topics in the top 12, against 7 of 10 before", implying ``trend_score`` earned
#: it. It did not, and the two numbers are not comparable:
#:
#: * Single-mention topics cannot appear in ANY top-N now, whatever the ranking, because
#:   ``_DEFAULT_MIN_TOTAL_MENTIONS`` (2) drops them from the artifact before ranking — see the
#:   ``min_total`` filter in ``_build_payload``. Zero is guaranteed by the floor, not achieved.
#: * The "7 of 10" was measured with no floor and the old velocity ordering, so it varies both
#:   knobs at once.
#:
#: What ``trend_score`` is independently responsible for is the ORDER among topics that clear the
#: floor: it ranks by decayed volume x week-spread instead of by a ratio, so a topic mentioned
#: twice in one recent week no longer outranks one discussed across a quarter.
#:
#: 12 weeks: long enough that a topic discussed monthly still registers, short enough that last
#: quarter's story yields to this one.
_TREND_DECAY_WEEKS = 12.0

#: Corpus mentions a topic needs before it is emitted at all.
#:
#: A topic mentioned once has no trend to measure — see the long note above, which this constant
#: is the subject of (it documented the min-total rationale while sitting attached to
#: ``_TREND_DECAY_WEEKS``). 8,743 of ~9.3k topics have ``total == 1``; dropping them took the
#: artifact from 65 MB of ~94% zeros to ~4 MB. Rows removed here are counted into
#: ``topics_below_min_total`` rather than vanishing, so the old total is reconstructible.
_DEFAULT_MIN_TOTAL_MENTIONS = 2

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


def _trend_score(weekly: dict[str, int], weeks: list[str]) -> float:
    """Recency-decayed mention volume scaled by weekly spread (#1931).

    ``sum(count * exp(-age_weeks / _TREND_DECAY_WEEKS)) * log1p(distinct_weeks)``.

    The decay makes recent mentions worth more; the ``log1p(spread)`` factor means a topic
    mentioned in eight different weeks beats one mentioned eight times in a single week, which is
    the difference between a running story and a one-off. Both halves matter: volume alone ranks
    evergreen topics forever, spread alone ranks anything long-running regardless of whether it is
    live now.

    Returns 0.0 when there is nothing in the window, so it is safe to sort on directly.
    """
    if not weekly or not weeks:
        return 0.0
    index = {w: i for i, w in enumerate(weeks)}
    newest = len(weeks) - 1
    decayed = 0.0
    spread = 0
    for week, count in weekly.items():
        if not count:
            continue
        i = index.get(week)
        if i is None:
            continue
        spread += 1
        decayed += count * math.exp(-(newest - i) / _TREND_DECAY_WEEKS)
    if not spread:
        return 0.0
    return round(decayed * math.log1p(spread), 4)


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
    total = sum(six)
    avg = total / len(six) if six else 0.0
    if avg == 0:
        return 0.0
    # #1931 — SHRINK toward flat when the window carries almost no evidence.
    #
    # The raw ratio is degenerate on a sparse corpus: one mention in the last month over a
    # six-month mean of 1/6 gives exactly 6.0, the maximum, for a topic discussed once. Measured
    # on the 1,066-episode corpus BEFORE this: 9,335 of 9,345 topics scored 0.0, 7 scored 6.0,
    # and the whole corpus held FOUR distinct values — so the field ranked "mentioned recently,
    # once" above "mentioned sixteen times across the year". #1650 documented the same collapse
    # at 678 episodes and the docstring below still warns not to read it as trending.
    #
    # James-Stein-style shrinkage fixes it without changing the meaning: the estimate is pulled
    # toward 1.0 (flat) in proportion to how little data supports it, so a single observation can
    # no longer outrank a sustained trend. With _VELOCITY_PRIOR_MENTIONS = 3, one mention keeps
    # a quarter of its excess over flat, twelve mentions keep four fifths, and a genuinely busy
    # topic is essentially unshrunk. The field stays "1.0 = flat, >1 rising, <1 cooling"; only
    # its confidence changes, so existing consumers and thresholds keep working.
    raw = last / avg
    weight = total / (total + _VELOCITY_PRIOR_MENTIONS)
    return round(1.0 + weight * (raw - 1.0), 4)


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


def _read_min_total(config: dict[str, Any]) -> int:
    """Minimum in-window mentions for a topic to get a series (see the constant)."""
    raw = config.get("min_total_mentions", _DEFAULT_MIN_TOTAL_MENTIONS)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_MIN_TOTAL_MENTIONS
    return value if value >= 1 else 1


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
    """One topic's envelope row: monthly counts + EWMA + scalar velocity + weekly series.

    ``velocity_last_over_6mo`` was previously unusable for ranking (#1650): on the 678-episode
    corpus 5,632 of 5,918 topics scored 0.0 and 256 scored exactly 6.0, and by 1,066 episodes it
    had degenerated further to **four distinct values across 9,345 topics** — a single recent
    mention scored the maximum and outranked a topic mentioned sixteen times.

    Fixed in #1931 by shrinking sparse estimates toward flat (see ``_velocity``), so the field is
    now a continuous, confidence-weighted signal that is safe to rank on. ``weekly_counts`` — and
    ``content_series`` at the envelope level — still carry the finer-grained signal and remain the
    better input for a recency-and-spread ranking.
    """
    series = [monthly.get(m, 0) for m in months]
    weekly_series = [weekly.get(w, 0) for w in weeks]
    velocity = _velocity(series, last_idx=effective_idx)
    return {
        "topic_id": tid,
        "topic_label": label,
        "monthly_counts": dict(zip(months, series)),
        "ewma": dict(zip(months, _ewma(series, alpha))),
        "velocity_last_over_6mo": velocity,
        # #1650 shipped this flag because the field was degenerate and a consumer reaching for
        # something called "velocity" had no way to know. #1931 made the value honest (shrinkage
        # toward flat on thin evidence), but the flag STAYS: velocity is an acceleration ratio,
        # and on a sparse corpus a ratio still cannot separate "discussed once, recently" from
        # "discussed all year" — every sustained topic sits at ~1.0. Rank on ``trend_score``.
        "velocity_is_indicative_only": True,
        # #1931 — the field to RANK on for discoverability. Recency-decayed mention volume scaled
        # by weekly spread: "what is being talked about, lately, repeatedly". Unlike velocity this
        # is a magnitude, not a ratio, so sustained topics outrank one-off spikes by construction.
        "trend_score": _trend_score(weekly, weeks),
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

    # #1930/#1931 — drop topics with too few mentions to carry a trend. See the constant.
    #
    # ``had_topics`` is captured BEFORE the floor on purpose: "the corpus produced no topics" and
    # "we withheld the topics it produced" are different states and must not collapse into the
    # same ``partial_reason``. A first cut of this filter reported ``no_topics_in_window`` for a
    # single-episode corpus, which reads as an input failure when it is a deliberate policy.
    min_total = _read_min_total(config)
    had_topics = bool(topics_out)
    below_floor = 0
    if min_total > 1:
        kept = [r for r in topics_out if (r.get("total") or 0) >= min_total]
        below_floor = len(topics_out) - len(kept)
        topics_out = kept

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
        # Distinguish "nothing to report" from "everything was below the floor" (#1930).
        partial_reason = "all_topics_below_min_total" if had_topics else "no_topics_in_window"
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
        # #1930/#1931 — say what was withheld, so a small artifact reads as a policy rather than
        # as missing data. ``topics_below_min_total`` + ``len(topics)`` reconstructs the old count.
        "min_total_mentions": min_total,
        "topics_below_min_total": below_floor,
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
        version="1.3.0",  # +trend_score, min_total floor drops rows (#1930/#1931)
        scope=EnricherScope.CORPUS,
        tier=EnricherTier.DETERMINISTIC,
        reads=[".kg.json"],
        writes="temporal_velocity.json",
        description=(
            "Monthly/weekly Topic mention counts + EWMA + velocity + trend_score, plus a "
            "full-history now-free content_series (per-topic/person weekly counts) for the "
            "momentum layer. Rank on trend_score, not velocity: velocity is an acceleration "
            "ratio and cannot separate 'discussed once, recently' from 'discussed all year' "
            "(#1931). Topics below min_total_mentions get no series (#1930)."
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
                "min_total_mentions": {
                    "type": "integer",
                    "minimum": 1,
                    "default": _DEFAULT_MIN_TOTAL_MENTIONS,
                    "description": (
                        "Minimum in-window mentions before a topic gets a velocity series "
                        "(#1930). A topic mentioned once has no trend; 93.6% of topics in the "
                        "measured corpus are in that state, and emitting a 128-bucket flat line "
                        "for each produced a 65 MB artifact that was 94% zeros. 1 disables."
                    ),
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
