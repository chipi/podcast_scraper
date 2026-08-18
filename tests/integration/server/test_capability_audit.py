"""The capability audit measures what it claims, on a corpus whose answers we already know (#1683).

This harness exists to be pointed at PRODUCTION, where nobody can eyeball the result — so the only
thing standing between "it printed numbers" and "the numbers mean something" is this file.

``app-validation-corpus/v3`` is the right fixture for that precisely because it is degenerate. Its
answers were established by hand while investigating #1669, and every one of them is a value the
audit must reproduce:

* 36 episodes across 9 feeds, 4 episodes each;
* exactly 2 topic clusters, BOTH covering all 36 episodes;
* the picker's offered options therefore produce **1** distinct feed — decorative;
* a discriminating band (2 <= n <= 60% coverage) holds **27** tokens whose top 12 produce **8**
  distinct feeds;
* the discover pool window (4 * 12 = 48) EXCEEDS the corpus, so the pool is everything and the
  relevance leg never runs — the blind spot that motivated the epic.

That last one is why these assertions are worth having. A tool that reported "pool reaches 100%"
without also reporting "because the window is larger than the corpus" would be actively
misleading when the same code is aimed at 700 episodes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper.capability_audit import (
    BAND_MAX_SHARE,
    DEFAULT_FEED_LIMIT,
    format_report,
    measure,
)

pytestmark = [pytest.mark.integration]

CORPUS = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "app-validation-corpus" / "v3"


@pytest.fixture(scope="module")
def report():
    if not CORPUS.is_dir():
        pytest.skip(f"corpus missing: {CORPUS}")
    return measure(CORPUS)


class TestTheWalkItself:
    def test_every_area_completed(self, report) -> None:
        """A silently-skipped area would read as 'nothing to report' in the run summary."""
        assert report.errors == {}, report.errors
        assert set(report.sections) == {
            "cluster_structure",
            "picker_discrimination",
            "pool_reachability",
            "corpus_shape",
        }

    def test_it_found_the_corpus(self, report) -> None:
        assert report.episodes == 36
        assert report.feeds == 9


class TestPoolReachability:
    def test_it_reports_that_the_window_swallows_this_corpus(self, report) -> None:
        """The finding that motivated #1682: 4*12=48 > 36, so the relevance leg never runs here.

        This must be REPORTED, not merely true — a bare "100% reachable" would be read as health.
        """
        pool = report.sections["pool_reachability"]
        assert pool["recency_window"] == DEFAULT_FEED_LIMIT * 4
        assert pool["pool_is_whole_corpus"] is True
        assert pool["recency_reach"] == 36
        assert pool["unreachable_without_a_match"] == 0

    def test_the_warning_reaches_the_summary(self, report) -> None:
        assert "NOT exercised" in format_report(report)


class TestPickerDiscrimination:
    def test_the_offered_options_are_decorative_here(self, report) -> None:
        """2 options, both corpus-wide, one feed — the measured claim behind #1669."""
        picker = report.sections["picker_discrimination"]
        assert len(picker["offered"]) == 2
        assert picker["offered_covering_every_episode"] == 2
        assert picker["offered_distinct_feeds"] == 1
        assert picker["decorative"] is True

    def test_a_discriminating_band_exists_and_separates_the_corpus(self, report) -> None:
        """The contrast that makes the verdict actionable rather than just negative."""
        picker = report.sections["picker_discrimination"]
        assert picker["band_candidates"] == 27
        assert picker["band_distinct_feeds"] == 8

    def test_no_band_token_covers_more_than_the_ceiling(self, report) -> None:
        """The band is only meaningful if its own bound holds."""
        picker = report.sections["picker_discrimination"]
        assert picker["band_top"], "band is empty, so the comparison above proves nothing"
        for entry in picker["band_top"]:
            assert entry["share"] <= BAND_MAX_SHARE, entry


class TestClusterStructure:
    def test_both_clusters_are_universal(self, report) -> None:
        """What makes the picker verdict undecidable on v3 — and why prod is needed to settle it."""
        clusters = report.sections["cluster_structure"]
        assert clusters["clusters_total"] == 2
        assert clusters["clusters_covering_every_episode"] == 2


class TestCorpusShape:
    def test_it_reports_the_feed_imbalance_that_makes_normalisation_unmeasurable(
        self, report
    ) -> None:
        """9 feeds x 4 episodes: a per-feed significance mean over 4 samples is noise (#1684)."""
        shape = report.sections["corpus_shape"]
        assert shape["feeds"] == 9
        assert shape["episodes_per_feed_min"] == 4
        assert shape["episodes_per_feed_max"] == 4
        assert shape["feeds_with_fewer_than_5"] == 9

    def test_every_episode_has_a_publish_date(self, report) -> None:
        """Recency decays from publish dates; an undated episode would silently skew the spread."""
        shape = report.sections["corpus_shape"]
        assert shape["episodes_without_publish_date"] == 0
        assert shape["publish_date_earliest"] < shape["publish_date_latest"]


class TestItIsReadOnly:
    def test_the_walk_writes_nothing_to_the_corpus(self, tmp_path) -> None:
        """The contract that lets this run against production without a backup first.

        Copies the fixture, measures the copy, and compares every path + mtime + size before and
        after. The operator plane creates files on read (``viewer_operator.yaml``, a jobs lock), so
        "it only reads" is a claim worth checking rather than assuming.
        """
        import shutil

        work = tmp_path / "v3"
        shutil.copytree(CORPUS, work)

        def snapshot() -> dict:
            return {
                str(p.relative_to(work)): (p.stat().st_size, p.stat().st_mtime_ns)
                for p in sorted(work.rglob("*"))
                if p.is_file()
            }

        before = snapshot()
        measure(work)
        after = snapshot()

        assert set(after) - set(before) == set(), "the audit CREATED files in the corpus"
        assert set(before) - set(after) == set(), "the audit DELETED files from the corpus"
        changed = [k for k in before if before[k] != after[k]]
        assert changed == [], f"the audit MODIFIED corpus files: {changed}"
