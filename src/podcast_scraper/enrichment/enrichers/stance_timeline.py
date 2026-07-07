"""``stance_timeline`` — per-(Person, Topic) *stance over time* (ml tier, ADR-108).

The reimagining of ``stance_disagreement`` (ADR-108). The disagreement enrichers hit 0% precision
because sentence-pair NLI can't tell "same contested *proposition*" from "same *topic*" (the
**shared-question gate**), which needs an LLM the operator ruled out. This enricher **deletes** that
problem by tracking **one speaker's stance on one topic across time** — same person + same topic, so
"same proposition" holds by construction.

**Stance (no LLM):** NLI-**entailment against fixed anchors** — for each Insight, score it against
``H+ = "{topic} is good/promising"`` and ``H− = "{topic} is bad/overhyped"``; stance =
``entail(H+) − entail(H−) ∈ [−1, +1]``. This is the *canonical* NLI setup (premise = insight,
hypothesis = anchor) — the half DeBERTa is genuinely good at. **Deviation** (did the view move?) is
fully deterministic: range, sign-flips, and a least-squares slope over the time-ordered series.

Output: one record per ``(person, topic)`` with a ``points`` series (episode publish date → stance)
and a ``deviation`` block. A ``(person × topic)`` stance series is a read-time time series — its
derivative is a "trending opinion shifts" signal for the RFC-103 momentum layer.

Reuses the CPU DeBERTa ``NliScorer`` the project already loads (no marginal cost). Gated by the
data-driven accuracy gate until an eval clears ``precision ≥ 0.5`` against the v3 position-arc gold.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from podcast_scraper.enrichment.enrichers._loaders import (
    edges_of_type,
    load_gi,
    node_label,
    nodes_of_type,
    publish_date,
)
from podcast_scraper.enrichment.protocol import (
    AccuracyGateRule,
    AccuracyGateSpec,
    EnricherManifest,
    EnricherResult,
    EnricherScope,
    EnricherTier,
    EpisodeArtifactBundle,
    ProviderRequirement,
    RunContext,
    STATUS_CANCELLED,
    STATUS_OK,
)
from podcast_scraper.enrichment.scorers.protocol import NliScorer

# (person_id, topic_id) → [(publish_date, insight_text), ...]
_Series = dict[tuple[str, str], list[tuple[str, str]]]


def _person_topic_time_index(
    bundles: list[EpisodeArtifactBundle],
) -> tuple[_Series, dict[str, str], dict[str, str]]:
    """``(person, topic) → [(date, text)]`` across the corpus, plus topic + person label maps."""
    series: _Series = {}
    topic_label: dict[str, str] = {}
    person_label: dict[str, str] = {}
    for b in bundles:
        gi = load_gi(b)
        date = publish_date(gi) or ""
        insight_text = {
            str(n.get("id") or ""): str((n.get("properties") or {}).get("text") or "")
            for n in nodes_of_type(gi, "Insight")
            if n.get("id")
        }
        for n in nodes_of_type(gi, "Person"):
            pid = str(n.get("id") or "")
            if pid:
                person_label[pid] = str((n.get("properties") or {}).get("name") or pid)
        for n in nodes_of_type(gi, "Topic"):
            tid = str(n.get("id") or "")
            if tid:
                topic_label[tid] = node_label(n)
        quote_speaker = {
            str(e.get("from") or ""): str(e.get("to") or "") for e in edges_of_type(gi, "SPOKEN_BY")
        }
        insight_speaker: dict[str, str] = {}
        for e in edges_of_type(gi, "SUPPORTED_BY"):
            spk = quote_speaker.get(str(e.get("to") or ""))
            if str(e.get("from") or "") and spk:
                insight_speaker.setdefault(str(e.get("from")), spk)
        for e in edges_of_type(gi, "ABOUT"):
            iid, tid = str(e.get("from") or ""), str(e.get("to") or "")
            spk = insight_speaker.get(iid)
            text = insight_text.get(iid, "")
            if iid and tid and spk and text.strip():
                series.setdefault((spk, tid), []).append((date, text.strip()))
    return series, topic_label, person_label


def _slope(values: list[float]) -> float:
    """Least-squares slope of ``values`` over their index (0..n-1); 0 for < 2 points."""
    n = len(values)
    if n < 2:
        return 0.0
    xbar = (n - 1) / 2.0
    ybar = sum(values) / n
    num = sum((i - xbar) * (v - ybar) for i, v in enumerate(values))
    den = sum((i - xbar) ** 2 for i in range(n))
    return round(num / den, 6) if den else 0.0


def _deviation(stances: list[float], move_threshold: float) -> dict[str, Any]:
    """Deterministic 'did the view move?' — range, sign-flips, slope, and a shifted flag.

    Sign-flips use a **deadzone** (``move_threshold / 2``): a stance within the deadzone of 0 is
    "neutral", so microscopic NLI noise jittering around 0 (``+0.001`` / ``-0.001``) can't
    masquerade as a pro↔anti reversal. A flip counts only between stances that clearly land on
    opposite sides — a genuine change of view. (Real-corpus eval showed the naive
    straddle-zero rule flagged 11 of 12 essentially-flat trajectories as "shifted".)
    """
    lo, hi = min(stances), max(stances)
    deadzone = move_threshold / 2.0
    signs = [1 if s >= deadzone else -1 if s <= -deadzone else 0 for s in stances]
    non_neutral = [s for s in signs if s != 0]
    sign_flips = sum(1 for a, b in zip(non_neutral, non_neutral[1:]) if a != b)
    span = round(hi - lo, 6)
    slope = _slope(stances)
    return {
        "range": span,
        "min": round(lo, 6),
        "max": round(hi, 6),
        "sign_flips": sign_flips,
        "slope": slope,
        "shifted": span >= move_threshold or sign_flips > 0,
    }


class StanceTimelineEnricher:
    """Per-(Person, Topic) stance trajectory over time via NLI-entailment vs stance anchors."""

    manifest = EnricherManifest(
        id="stance_timeline",
        version="1.0.0",
        scope=EnricherScope.CORPUS,
        tier=EnricherTier.ML,
        reads=[".gi.json"],
        writes="stance_timeline.json",
        description=(
            "Per-(Person, Topic) stance over time via NLI-entailment vs stance anchors; "
            "deviation = flip/drift (ADR-108, RFC-103 tie-in). No LLM."
        ),
        expected_duration_s=120,
        config_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "min_points": {
                    "type": "integer",
                    "minimum": 2,
                    "default": 2,
                    "description": "Min episodes a (person, topic) needs to form a timeline.",
                },
                "move_threshold": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 2,
                    "default": 0.4,
                    "description": "Min stance range to flag a shift (also flags on a sign flip).",
                },
                "positive_anchor": {
                    "type": "string",
                    "default": "{topic} is good and promising.",
                    "description": "Positive stance hypothesis ({topic} substituted).",
                },
                "negative_anchor": {
                    "type": "string",
                    "default": "{topic} is bad and overhyped.",
                    "description": "Negative stance hypothesis ({topic} substituted).",
                },
            },
        },
        provider_requirement=ProviderRequirement(
            protocol="NliScorer",
            description="NLI scorer (DeBERTa local, scripted fixture, or future hosted NLI).",
        ),
        accuracy_gate=AccuracyGateSpec(
            rules=(AccuracyGateRule(metric_name="precision", min_value=0.5),),
            on_missing_data="reject",
        ),
    )

    def __init__(
        self,
        scorer: NliScorer,
        *,
        model_id: str = "cross-encoder/nli-deberta-v3-small",
        model_version: str = "v1",
        min_points: int = 2,
        move_threshold: float = 0.4,
        positive_anchor: str = "{topic} is good and promising.",
        negative_anchor: str = "{topic} is bad and overhyped.",
    ) -> None:
        self._scorer = scorer
        self._model_id = model_id
        self._model_version = model_version
        self._min_points = min_points
        self._move_threshold = move_threshold
        self._positive = positive_anchor
        self._negative = negative_anchor

    async def _stance(self, text: str, topic_human: str) -> float:
        """entail(text, H+) − entail(text, H−) ∈ [−1, +1] — the speaker's stance on the topic."""
        pos = await self._scorer.score(text, self._positive.format(topic=topic_human))
        neg = await self._scorer.score(text, self._negative.format(topic=topic_human))
        return pos.entailment - neg.entailment

    async def enrich(
        self,
        *,
        bundle: EpisodeArtifactBundle | None,
        corpus_root: Path,
        all_bundles: list[EpisodeArtifactBundle] | None,
        config: dict[str, Any],
        ctx: RunContext,
    ) -> EnricherResult:
        """Build each (person, topic) stance series + deviation over time."""
        min_points = int(config.get("min_points", self._min_points))
        move_threshold = float(config.get("move_threshold", self._move_threshold))
        series, topic_label, person_label = _person_topic_time_index(all_bundles or [])

        timelines: list[dict[str, Any]] = []
        for (pid, tid), entries in sorted(series.items()):
            if len({d for d, _ in entries if d}) < min_points:
                continue
            topic_human = topic_label.get(tid, tid.split(":", 1)[-1]).replace("-", " ")
            points: list[dict[str, Any]] = []
            for date, text in sorted(entries, key=lambda e: e[0]):
                if ctx.cancel_event.is_set():
                    return EnricherResult(status=STATUS_CANCELLED, error="cancel_requested")
                stance = await self._stance(text, topic_human)
                points.append({"date": date, "stance": round(stance, 6)})
            stances = [p["stance"] for p in points]
            timelines.append(
                {
                    "person_id": pid,
                    "person_name": person_label.get(pid, pid),
                    "topic_id": tid,
                    "topic_label": topic_label.get(tid, tid),
                    "points": points,
                    "deviation": _deviation(stances, move_threshold),
                    "model_id": self._model_id,
                    "model_version": self._model_version,
                }
            )

        # Biggest movers first (shifted, then range) — the "opinion shift" signal.
        timelines.sort(
            key=lambda r: (not r["deviation"]["shifted"], -r["deviation"]["range"], r["person_id"])
        )
        return EnricherResult(
            status=STATUS_OK,
            data={
                "model_id": self._model_id,
                "model_version": self._model_version,
                "min_points": min_points,
                "move_threshold": move_threshold,
                "timelines": timelines,
            },
            records_written=len(timelines),
        )


__all__ = ["StanceTimelineEnricher"]
