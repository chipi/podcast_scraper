"""Regression test for the OBSERVABILITY, not the bug (#1647).

#1646 was not hard to find. It was impossible to *notice*: every signal the estate produced
stayed green while 72 % of episodes had speaker detection skipped and 23 % of insights became
unreachable. Fixing the size gate closes that bug; it does nothing about the blindness that
let it run from 2026-06-05 to 2026-08-14 unremarked.

So this test does not assert anything about the size gate. It replays the **real pre-fix
corpus** — all 678 episodes, captured in ``data/baselines/corpus-integrity-2026-08-14.json``
by ``scripts/baselines/capture_corpus_integrity_baseline.py`` — through the quality report and
asserts the report *says something is wrong*, using only what a reader would see.

If someone later "simplifies" the report and this test still passes, the report still works.
If it fails, we have re-created the blindness, whatever the size gate is doing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from podcast_scraper.quality.attribution import build_report, EpisodeQuality, format_report

pytestmark = [pytest.mark.unit]

BASELINE = (
    Path(__file__).resolve().parents[4] / "data" / "baselines" / "corpus-integrity-2026-08-14.json"
)


def _load_pre_fix_corpus() -> List[EpisodeQuality]:
    """Turn the captured baseline into quality records.

    The baseline predates the stage ledger, so it carries ``speaker_detection_ran`` rather
    than a ledger entry. Mapping it to ``ran``/``skipped`` here is exactly what the ledger
    records natively going forward — the report cannot tell the difference, which is the
    point: it is judging the corpus, not the recording mechanism.
    """
    raw: Dict[str, Any] = json.loads(BASELINE.read_text(encoding="utf-8"))
    episodes: List[EpisodeQuality] = []
    for row in raw["episodes"]:
        ledger: Dict[str, Dict[str, Any]] = {}
        ran = row.get("speaker_detection_ran")
        if ran is not None:
            ledger["speaker_detection"] = (
                {"outcome": "ran"}
                if ran
                else {"outcome": "skipped", "reason": "no_timing_recorded_pre_ledger"}
            )
        episodes.append(
            EpisodeQuality(
                episode_id=row.get("episode_id"),
                feed=row.get("feed"),
                duration_seconds=row.get("duration_seconds"),
                stage_ledger=ledger,
                insights_total=row.get("insights_total"),
                insights_surfaceable=row.get("insights_surfaceable"),
                notes=list(row.get("errors") or []),
            )
        )
    return episodes


@pytest.fixture(scope="module")
def report() -> Dict[str, Any]:
    """Built once for the module: 678 episodes parsed per test would be pure waste."""
    return build_report(_load_pre_fix_corpus())


@pytest.mark.skipif(not BASELINE.exists(), reason="pre-fix baseline artifact not present")
class TestReportSurfacesTheDamageWithoutBeingToldAboutIt:
    def test_the_corpus_under_test_is_the_real_one(self, report: Dict[str, Any]) -> None:
        assert report["episodes"] == 678

    def test_a_reader_sees_that_a_stage_did_not_run(self, report: Dict[str, Any]) -> None:
        """The single fact that was invisible for two months."""
        stage = next(s for s in report["stages"] if s["stage"] == "speaker_detection")
        assert stage["outcomes"].get("skipped") == 488
        assert stage["ran_ratio"] is not None and stage["ran_ratio"] < 0.3

    def test_a_reader_sees_that_insights_are_unreachable(self, report: Dict[str, Any]) -> None:
        attribution = report["attribution"]
        assert attribution["insights_total"] == 8952
        assert attribution["insights_surfaceable"] == 6840
        assert attribution["insights_unsurfaceable"] == 2112
        # ~76 % — the number every coverage endpoint reported as 100 %.
        assert attribution["attribution_ratio"] == pytest.approx(0.764, abs=0.001)

    def test_a_reader_sees_which_episodes_lost_everything(self, report: Dict[str, Any]) -> None:
        assert report["attribution"]["episodes_fully_zeroed"] == 82

    def test_the_worst_feeds_are_identifiable_without_prior_knowledge(
        self, report: Dict[str, Any]
    ) -> None:
        """An operator must be able to find the problem feeds by reading, not by being told."""
        ranked = sorted(
            (f for f, cell in report["per_feed"].items() if cell["attribution_ratio"] is not None),
            key=lambda f: report["per_feed"][f]["attribution_ratio"],
        )
        # The a16z Show (0.435) and NVIDIA AI Podcast (0.554) are the two worst real feeds.
        assert "The a16z Show" in ranked[:3]
        assert report["per_feed"]["The a16z Show"]["attribution_ratio"] < 0.5
        # Unhedged is the control: no episode over the size limit, attribution stays high.
        assert report["per_feed"]["Unhedged"]["attribution_ratio"] > 0.85

    def test_the_rendered_report_leads_with_the_problem(self, report: Dict[str, Any]) -> None:
        """The failure mode is a report that is technically complete and reads as fine."""
        text = format_report(report)
        assert "skipped=488" in text
        assert "6840/8952" in text
        assert "NOT MEASURED" in text

    def test_it_does_not_claim_semantic_correctness(self, report: Dict[str, Any]) -> None:
        assert "NOT MEASURED" in report["not_measured"]["semantic_correctness"]
