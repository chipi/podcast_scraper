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

Output drives dashboard "trending topics" and autoresearch follow-on
hypotheses. The window is computed relative to ``now`` (UTC); the
envelope also surfaces ``effective_last_month`` so callers can tell
when "now" lags the data.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from podcast_scraper.enrichment.enrichers._loaders import (
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


def _compute(
    bundle: EpisodeArtifactBundle | None,
    corpus_root: Path,
    all_bundles: list[EpisodeArtifactBundle] | None,
    config: dict[str, Any],
    ctx: RunContext,
) -> dict[str, Any]:
    alpha = _read_alpha(config)
    window = _read_window_months(config)
    now = _now_utc(config)
    months = _window_months(now, window)
    counts_by_topic: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    labels: dict[str, str] = {}
    bundles = all_bundles or []
    for b in bundles:
        kg = load_kg(b)
        date = publish_date(kg)
        if not date:
            continue
        mkey = _month_key(date)
        if not mkey or mkey not in months:
            continue
        for t in nodes_of_type(kg, "Topic"):
            tid = str(t.get("id") or "")
            if not tid:
                continue
            labels[tid] = node_label(t)
            counts_by_topic[tid][mkey] += 1

    effective_idx = _effective_last_idx(counts_by_topic, months)
    topics_out: list[dict[str, Any]] = []
    for tid, monthly in counts_by_topic.items():
        series = [monthly.get(m, 0) for m in months]
        topics_out.append(
            {
                "topic_id": tid,
                "topic_label": labels.get(tid, tid),
                "monthly_counts": dict(zip(months, series)),
                "ewma": dict(zip(months, _ewma(series, alpha))),
                "velocity_last_over_6mo": _velocity(series, last_idx=effective_idx),
                "total": sum(series),
            }
        )
    topics_out.sort(key=lambda r: (-r["velocity_last_over_6mo"], -r["total"], r["topic_id"]))
    return {
        "window_months": months,
        "now": now.isoformat(),
        "alpha": alpha,
        "effective_last_month": months[effective_idx] if months else None,
        "topics": topics_out,
    }


_enrich_async = sync_enricher(_compute)


class TemporalVelocityEnricher:
    """Corpus-scope monthly Topic mention counts + EWMA + velocity."""

    manifest = EnricherManifest(
        id="temporal_velocity",
        version="1.0.0",
        scope=EnricherScope.CORPUS,
        tier=EnricherTier.DETERMINISTIC,
        reads=[".kg.json"],
        writes="temporal_velocity.json",
        description="Monthly Topic mention counts over a rolling window + EWMA trend.",
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
