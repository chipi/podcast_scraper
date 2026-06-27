"""``temporal_velocity`` — monthly Topic mention counts + EWMA trend (deterministic).

For each Topic mentioned in any episode, compute:

* ``monthly_counts`` — episode mentions bucketed by ``YYYY-MM`` over
  the last 12 calendar months (zero-filled).
* ``ewma`` — 3-period exponentially-weighted moving average over the
  monthly series (α=0.5), aligned to the latest month.
* ``velocity`` — last-month count divided by the 6-month rolling
  average (gives a "rising vs falling" signal).

Output drives dashboard "trending topics" and autoresearch follow-on
hypotheses. The window is computed relative to ``now`` (UTC).
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

_ALPHA = 0.5
_WINDOW_MONTHS = 12


def _month_key(date_str: str) -> str | None:
    """Parse an ISO date and return ``YYYY-MM`` (or ``None`` on failure)."""
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00")).strftime("%Y-%m")
    except (ValueError, TypeError):
        return None


def _twelve_month_window(now: datetime) -> list[str]:
    """Return the 12 month-keys ending at *now*, oldest first."""
    months: list[str] = []
    year, month = now.year, now.month
    for _ in range(_WINDOW_MONTHS):
        months.append(f"{year:04d}-{month:02d}")
        month -= 1
        if month == 0:
            month = 12
            year -= 1
    months.reverse()
    return months


def _ewma(series: list[int]) -> list[float]:
    """Compute α=0.5 EWMA over the monthly series."""
    out: list[float] = []
    prev = 0.0
    for x in series:
        prev = _ALPHA * x + (1 - _ALPHA) * prev
        out.append(round(prev, 4))
    return out


def _velocity(series: list[int]) -> float:
    """Last-month count over the 6-month trailing average (1.0 = flat)."""
    if not series:
        return 0.0
    last = series[-1]
    six = series[-6:] if len(series) >= 6 else series
    avg = sum(six) / len(six)
    if avg == 0:
        return 0.0
    return round(last / avg, 4)


def _now_utc(config: dict[str, Any]) -> datetime:
    """Use config-provided 'now' for testability (defaults to current UTC)."""
    raw = config.get("now")
    if isinstance(raw, str):
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def _compute(
    bundle: EpisodeArtifactBundle | None,
    corpus_root: Path,
    all_bundles: list[EpisodeArtifactBundle] | None,
    config: dict[str, Any],
    ctx: RunContext,
) -> dict[str, Any]:
    now = _now_utc(config)
    months = _twelve_month_window(now)
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

    topics_out: list[dict[str, Any]] = []
    for tid, monthly in counts_by_topic.items():
        series = [monthly.get(m, 0) for m in months]
        topics_out.append(
            {
                "topic_id": tid,
                "topic_label": labels.get(tid, tid),
                "monthly_counts": dict(zip(months, series)),
                "ewma": dict(zip(months, _ewma(series))),
                "velocity_last_over_6mo": _velocity(series),
                "total": sum(series),
            }
        )
    topics_out.sort(key=lambda r: (-r["velocity_last_over_6mo"], -r["total"], r["topic_id"]))
    return {
        "window_months": months,
        "now": now.isoformat(),
        "alpha": _ALPHA,
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
        description="Monthly Topic mention counts over a 12-month window + EWMA trend.",
        expected_duration_s=30,
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
