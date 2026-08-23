"""Aggregate per-episode stage outcomes and attribution health into a run-scale report (#1647).

The acceptance test this exists to satisfy:

    "I run 1 episode, then 10, then 50, then 5000. Tell me how it went."

Before #1647 the honest answer was that nobody could. Every available signal counted
*artifacts* — episodes indexed, GI present, KG present — and every one of them stays green on
an episode whose speakers were never named and whose insights therefore reach no surface. A
corpus could report ``with_gi=678, with_kg=678, with_neither=0`` while 23 % of its insights
were unusable, which is precisely what #1646 did for two months.

Two properties this module is built around:

**Silence is reported, not omitted.** ``EpisodeQuality`` carries ``notes`` for what could not
be determined, and the report has a ``not_measured`` block. A dimension that was not measured
must say so with the same prominence as one that passed — otherwise absence reads as health,
which is the original defect in a new costume.

**Ratios are ``None``, never ``0.0``, when there is no denominator.** ``0.0`` claims total
failure; ``None`` says "no data". Confusing the two makes a post-repair diff lie.

Pure functions over plain records: the transport (operator API, on-disk corpus, a live run's
metrics) is the caller's problem, so the same aggregation serves all three and cannot drift
between them.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

# An insight is excluded from every surface when GI marks it unsurfaceable — see
# ``enrichment/enrichers/_loaders.py::is_surfaceable_insight``. Absent means surfaceable;
# only an explicit False excludes, so the check must be ``is not False`` and never truthiness.
SURFACEABLE_ABSENT_MEANS_YES = True


@dataclass
class EpisodeQuality:
    """One episode's stage outcomes and attribution counts.

    Every count is ``Optional``: ``None`` means "not determined" (the artifact was missing or
    unreadable), which is a different fact from ``0`` and must survive into the report.
    """

    episode_id: Optional[str] = None
    feed: Optional[str] = None
    duration_seconds: Optional[float] = None
    # stage name -> outcome record ({"outcome", "reason", "detail", "duration_seconds"})
    stage_ledger: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    insights_total: Optional[int] = None
    insights_surfaceable: Optional[int] = None
    voices_total: Optional[int] = None
    voices_named: Optional[int] = None
    # Why a dimension could not be determined for this episode (never silently dropped).
    notes: List[str] = field(default_factory=list)

    @property
    def has_ledger(self) -> bool:
        return bool(self.stage_ledger)

    @property
    def attribution_known(self) -> bool:
        return self.insights_total is not None and self.insights_surfaceable is not None

    @property
    def fully_zeroed(self) -> bool:
        """Had insights, none of them usable — the episode contributes nothing downstream.

        Requires ``insights_total > 0``: an episode that produced no insights at all is a
        different problem and counting it here would inflate the damage.
        """
        return bool(
            self.attribution_known
            and (self.insights_total or 0) > 0
            and (self.insights_surfaceable or 0) == 0
        )


def ratio(numerator: Optional[int], denominator: Optional[int]) -> Optional[float]:
    """Rounded ratio, or None when there is no denominator to divide by.

    Deliberately not ``0.0`` on an empty denominator: that would assert total failure where
    there is simply no data, and a baseline diff cannot tell the two apart afterwards.
    """
    if not denominator:
        return None
    return round((numerator or 0) / denominator, 6)


def summarise_stage(episodes: Iterable[EpisodeQuality], stage: str) -> Dict[str, Any]:
    """Outcome counts for one stage, with skips/failures broken down by reason.

    ``reason`` is what makes this actionable — "412 skipped" is a number, while
    "412 skipped: media_over_size_limit_no_transcript_urls" is a bug report.
    """
    episodes = list(episodes)
    outcomes: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    unknown = 0
    for ep in episodes:
        record = ep.stage_ledger.get(stage)
        if not record:
            # No ledger entry: this run predates #1647, or the stage never reported. Counted
            # separately rather than assumed to have run — assuming is the original sin here.
            unknown += 1
            continue
        outcome = str(record.get("outcome", "unknown"))
        outcomes[outcome] += 1
        if outcome in ("skipped", "failed", "degraded"):
            reasons[str(record.get("reason") or "reason_not_recorded")] += 1
    return {
        "stage": stage,
        "episodes": len(episodes),
        "outcomes": dict(outcomes),
        "reasons": dict(reasons),
        "no_ledger_entry": unknown,
        "ran_ratio": ratio(outcomes.get("ran", 0) + outcomes.get("degraded", 0), len(episodes)),
    }


def _feed_key(ep: EpisodeQuality) -> str:
    return ep.feed or "(unknown feed)"


def build_report(
    episodes: Iterable[EpisodeQuality],
    *,
    stages: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Aggregate episodes into the run-scale quality report.

    Works unchanged for 1 episode or 5 000 — the shape does not depend on scale, so the same
    output answers "how did this one go?" and "how is the corpus?".
    """
    episodes = list(episodes)
    stages = stages or sorted({stage for ep in episodes for stage in ep.stage_ledger})

    attributed = [ep for ep in episodes if ep.attribution_known]
    insights_total = sum(ep.insights_total or 0 for ep in attributed)
    insights_surfaceable = sum(ep.insights_surfaceable or 0 for ep in attributed)
    voiced = [ep for ep in episodes if ep.voices_total is not None]
    voices_total = sum(ep.voices_total or 0 for ep in voiced)
    voices_named = sum(ep.voices_named or 0 for ep in voiced)
    zeroed = [ep for ep in attributed if ep.fully_zeroed]

    per_feed: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"episodes": 0, "insights": 0, "surfaceable": 0, "fully_zeroed": 0}
    )
    for ep in episodes:
        cell = per_feed[_feed_key(ep)]
        cell["episodes"] += 1
        if ep.attribution_known:
            cell["insights"] += ep.insights_total or 0
            cell["surfaceable"] += ep.insights_surfaceable or 0
            if ep.fully_zeroed:
                cell["fully_zeroed"] += 1
    for cell in per_feed.values():
        cell["attribution_ratio"] = ratio(cell["surfaceable"], cell["insights"])

    without_ledger = [ep for ep in episodes if not ep.has_ledger]
    without_attribution = [ep for ep in episodes if not ep.attribution_known]

    return {
        "episodes": len(episodes),
        "stages": [summarise_stage(episodes, stage) for stage in stages],
        "attribution": {
            "insights_total": insights_total,
            "insights_surfaceable": insights_surfaceable,
            "insights_unsurfaceable": insights_total - insights_surfaceable,
            "attribution_ratio": ratio(insights_surfaceable, insights_total),
            "voices_total": voices_total,
            "voices_named": voices_named,
            "voices_named_ratio": ratio(voices_named, voices_total),
            "episodes_fully_zeroed": len(zeroed),
            "episodes_fully_zeroed_ratio": ratio(len(zeroed), len(attributed)),
        },
        "per_feed": dict(sorted(per_feed.items())),
        # Equal prominence, by design. A report that lists only what it checked lets silence
        # read as health — the exact failure mode this epic exists to close.
        "not_measured": {
            "episodes_without_stage_ledger": len(without_ledger),
            "episodes_without_attribution_data": len(without_attribution),
            "notes": sorted({note for ep in episodes for note in ep.notes}),
            "semantic_correctness": (
                "NOT MEASURED — nothing here checks whether an insight is true or faithful "
                "to its transcript; this report is structural only (see #1660)."
            ),
        },
    }


def format_report(report: Dict[str, Any]) -> str:
    """Render the report for a terminal, leading with what is wrong."""
    lines: List[str] = []
    attribution = report["attribution"]
    lines.append(f"episodes: {report['episodes']}")
    lines.append("")
    lines.append("stages")
    for stage in report["stages"]:
        outcomes = ", ".join(f"{k}={v}" for k, v in sorted(stage["outcomes"].items())) or "-"
        lines.append(
            f"  {stage['stage']}: {outcomes} (no ledger entry: {stage['no_ledger_entry']})"
        )
        for reason, count in sorted(stage["reasons"].items(), key=lambda kv: -kv[1]):
            lines.append(f"      {count:>5}  {reason}")
    lines.append("")
    lines.append("attribution")
    lines.append(
        f"  insights surfaceable : {attribution['insights_surfaceable']}"
        f"/{attribution['insights_total']} = {attribution['attribution_ratio']}"
    )
    lines.append(
        f"  voices named         : {attribution['voices_named']}"
        f"/{attribution['voices_total']} = {attribution['voices_named_ratio']}"
    )
    lines.append(
        f"  episodes fully zeroed: {attribution['episodes_fully_zeroed']} "
        f"({attribution['episodes_fully_zeroed_ratio']})"
    )
    lines.append("")
    lines.append("NOT MEASURED")
    not_measured = report["not_measured"]
    lines.append(
        f"  episodes without a stage ledger : {not_measured['episodes_without_stage_ledger']}"
    )
    lines.append(
        f"  episodes without attribution data: {not_measured['episodes_without_attribution_data']}"
    )
    for note in not_measured["notes"]:
        lines.append(f"  note: {note}")
    lines.append(f"  {not_measured['semantic_correctness']}")
    return "\n".join(lines)
