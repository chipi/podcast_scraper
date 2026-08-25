"""The capability audit measures what it claims, on a corpus whose answers we already know (#1683).

This harness exists to be pointed at PRODUCTION, where nobody can eyeball the result — so the only
thing standing between "it printed numbers" and "the numbers mean something" is this file.

``app-validation-corpus/v3`` is the right fixture for that precisely because it is degenerate. Its
answers were established by hand while investigating #1669, and every one of them is a value the
audit must reproduce:

* 36 episodes across 9 feeds, 4 episodes each;
* exactly 2 topic clusters, BOTH covering all 36 episodes;
* the picker's offered options therefore produce **1** distinct feed — decorative;
* a discriminating band (2 <= n <= 60% coverage) holds **27** tokens whose top 12 produce **10**
  distinct feeds. NOTE: this was quoted as 8 until 2026-08-19. That figure was measured while tie
  ordering was hash-dependent, so it was never stable — with ties broken by token it is 10. The
  conclusion it supports (1 distinct feed from the picker's own options) is unchanged and in fact
  a wider gap;
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

# critical_path: the audit is what runs against production; its coverage lands on PRs.
pytestmark = [pytest.mark.integration, pytest.mark.critical_path]

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
            "cluster_reach",
            "picker_discrimination",
            "pool_reachability",
            "corpus_shape",
            "graph_coverage",
            "entity_identity",
            "content_quality",
            "bare_name_resolvability",
            "topic_momentum",
            "ranking_calibration",
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
        # 10, not the 8 quoted before 2026-08-19 — see the module docstring. The old figure came
        # from a hash-dependent top-12, so it was never reproducible.
        assert picker["band_distinct_feeds"] == 10

    def test_no_band_token_covers_more_than_the_ceiling(self, report) -> None:
        """The band is only meaningful if its own bound holds."""
        picker = report.sections["picker_discrimination"]
        assert picker["band_top"], "band is empty, so the comparison above proves nothing"
        for entry in picker["band_top"]:
            assert entry["share"] <= BAND_MAX_SHARE, entry


class TestTheReportIsReproducible:
    """The same corpus must produce the same report, every run.

    It did not. `Counter.most_common()` breaks ties in INSERTION order, and insertion came from
    iterating a set per episode — so with eleven tokens tied at 4 episodes, which ones appeared in
    the "top 12" changed with PYTHONHASHSEED. Measured 2026-08-19: three seeds gave three
    different reports (md5 a2dfabff…, 5671b02a…, e566ffef…); after ordering ties by token, all
    three give 92fe303e….

    This matters more at scale, not less: 700 episodes means far more ties, so a run-to-run diff
    would look like the corpus changed when only the hash seed did.
    """

    def test_ties_are_broken_deterministically(self, report) -> None:
        band = report.sections["picker_discrimination"]["band_top"]
        tied = [e for e in band if e["episodes"] == band[-1]["episodes"]]
        assert len(tied) > 1, "no ties present, so this test would prove nothing on this corpus"
        tokens = [e["token"] for e in tied]
        assert tokens == sorted(tokens), f"tied tokens are not in a stable order: {tokens}"

    def test_two_measurements_agree(self) -> None:
        """Belt and braces: measure twice in-process and compare the rendered report."""
        assert format_report(measure(CORPUS)) == format_report(measure(CORPUS))


class TestUniversalTokensAreNotOnlyClusters:
    """The finding that changed #1669's suggested fix, pinned so it cannot be lost again.

    #1669 originally proposed offering individual TOPICS alongside clusters, on the assumption
    that topics are narrow. Measured on v3 they are not: five tokens cover 36/36 and only two are
    `tc:` clusters. `topic:lifelong-learning` and `topic:expert-interviews` are equally
    decorative, and `thc:managing-risk` means the STORYLINES rail shares the defect.

    So the rule is coverage, not kind. If a future change filters only `tc:` and this test still
    passes, the fix is incomplete and this is what says so.
    """

    def test_five_tokens_cover_the_whole_corpus(self, report) -> None:
        universal = report.sections["picker_discrimination"]["universal_tokens_any_kind"]
        assert len(universal) == 5, universal

    def test_they_are_not_all_clusters(self, report) -> None:
        universal = set(report.sections["picker_discrimination"]["universal_tokens_any_kind"])
        assert {"topic:lifelong-learning", "topic:expert-interviews"} <= universal
        assert "thc:managing-risk" in universal, "the storylines rail shares this defect"
        assert sum(1 for t in universal if t.startswith("tc:")) == 2

    def test_the_warning_names_them_in_the_report(self, report) -> None:
        """A number in a dict nobody reads is not a finding; it has to reach the run summary."""
        text = format_report(report)
        assert "corpus-wide tokens of ANY kind" in text
        assert "only filters `tc:` clusters" in text


class TestTheBandIsNotAutomaticallyOfferable:
    """Discriminating power is necessary but NOT sufficient — the other half of the #1669 fix.

    The band separates the corpus beautifully and contains `person:a-correspondent`, whose KG name
    is literally "A. correspondent", plus first-name-only entities. Offering those would be worse
    than offering a decorative cluster. The report must show WHAT is in the band, not just how many
    distinct feeds it produces, or the number reads as a recommendation.
    """

    def test_the_band_contents_reach_the_report(self, report) -> None:
        text = format_report(report)
        assert (
            "person:a-correspondent" in text
        ), "the band's actual contents must be visible; a bare count reads as 'offer these'"


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


class TestClusterReach:
    """Do clusters span SHOWS, or merge synonyms inside one? (#1682)

    A cluster is meant to be a theme crossing shows — the whole reason the picker offers clusters
    rather than raw topics. Production measured 278 clusters at median size 2 from 870 candidate
    tokens, and size alone cannot distinguish "two names for one idea inside one podcast" from "a
    genuine small theme across two". Feed span can, and it is what decides whether
    `topic_cluster_threshold` (0.75, tuned on v2 fixtures in June, never re-measured on real data)
    is doing its job.
    """

    def test_the_fixture_clusters_do_span_feeds(self, report) -> None:
        reach = report.sections["cluster_reach"]
        assert reach["clusters"] == 2
        assert reach["cross_feed"] == 2
        assert reach["single_feed"] == 0

    def test_member_topics_are_actually_read(self, report) -> None:
        """The bug this catches: `top_clusters_by_member_count` DROPS `members`.

        The first version of this measurement called it and asked for member topic ids, got
        nothing, and reported "0 topics across 0 feeds" for every cluster — which reads like a
        finding ("clusters are empty!") and was a bug in the instrument. Asserting a non-zero
        topic count is what makes the difference visible.
        """
        reach = report.sections["cluster_reach"]
        assert reach["no_feed_data"] == 0, "a cluster resolved to no feeds at all"
        widest = reach["widest"][0]
        assert widest["topics"] > 0, "cluster members were not read — see the docstring"
        assert widest["feeds"] > 0

    def test_the_warning_only_fires_when_most_clusters_are_single_show(self, report) -> None:
        """On this fixture every cluster spans feeds, so the warning must be ABSENT.

        A warning that always prints is not a warning.
        """
        assert report.sections["cluster_reach"]["cross_feed_share"] == 1.0
        assert "synonym merge" not in format_report(report)


class TestTheReportedWindowIsTheRealOne:
    """The audit must ASK the pool for its window, never restate the policy.

    It used to compute `limit * DISCOVER_POOL_MULTIPLE` itself — a second implementation of the
    admission policy living inside the tool that measures it. The moment the real one started
    scaling with corpus size, the report drifted: production 2026-08-19 printed

        recency leg reaches 101/678 (14.9%); window is 48

    which is self-contradictory on its own line. The measurement was right; the number beside it
    was stale. A tool that restates a policy will always drift from it, and the drift shows up as
    a confident wrong number rather than an error.
    """

    def test_the_window_matches_what_the_pool_would_use(self, report) -> None:
        from podcast_scraper.server.app_discover_view import _pool_window

        pool = report.sections["pool_reachability"]
        assert pool["recency_window"] == _pool_window(DEFAULT_FEED_LIMIT, report.episodes)

    def test_at_production_scale_the_window_is_not_the_old_fixed_number(self) -> None:
        """The regression, stated in the shape that actually broke.

        On the 36-episode fixture the answer is legitimately 48 (below the crossover), so the
        fixture alone cannot catch this — the same blind spot that hid the fixed pool for months.
        Assert against production's size instead.
        """
        from podcast_scraper.server.app_discover_view import (
            _pool_window,
            DISCOVER_POOL_MULTIPLE,
        )

        prod_window = _pool_window(DEFAULT_FEED_LIMIT, 678)
        assert prod_window != DEFAULT_FEED_LIMIT * DISCOVER_POOL_MULTIPLE
        assert prod_window == 101, "the production reading this was verified against"


class TestGraphCoverage:
    """How often the graph Your Week promises is actually there (#1685).

    v3 is at 100% — which is itself the finding: this corpus CANNOT exercise the coverage
    question, the same blind-spot class as the discover pool. The assertions here protect the
    measurement's mechanics so the production number can be trusted, not the number itself.
    """

    def test_the_fixture_is_fully_covered(self, report) -> None:
        cov = report.sections["graph_coverage"]
        assert cov["episodes"] == 36
        assert cov["with_kg"] == 36
        assert cov["with_gi"] == 36
        assert cov["kg_share"] == 1.0

    def test_the_warning_is_absent_when_coverage_is_complete(self, report) -> None:
        """A warning that always prints is not a warning."""
        assert report.sections["graph_coverage"]["feeds_below_half_kg"] == 0
        assert "below 50% KG coverage" not in format_report(report)

    def test_it_breaks_down_by_feed(self, report) -> None:
        """A gap concentrated in one show is a different problem from one spread evenly, and a
        corpus-wide percentage cannot tell them apart."""
        worst = report.sections["graph_coverage"]["worst_feeds"]
        assert len(worst) >= 5
        assert {"feed", "episodes", "kg_share", "gi_share"} <= set(worst[0])


class TestEntityIdentity:
    """Near-duplicate person entities split the affinity signal (#1685).

    Affinity keys on `person:<slug>`, so one human under two ids means a follow matches half their
    episodes. The fixture carries the shapes this looks for, which is what makes it a usable
    regression corpus: `person:sam` (first-name-only, cannot be disambiguated) and
    `person:renee-montagne-park` (a name with something welded on).
    """

    def test_it_finds_the_single_word_names(self, report) -> None:
        ident = report.sections["entity_identity"]
        assert ident["person_entities"] == 26
        assert ident["single_word_names"] == 7
        assert "person:sam" in ident["single_word_examples"]

    def test_it_finds_the_shared_surname(self, report) -> None:
        """`jordan-park` and `renee-montagne-park` — spotted by eye first, now caught mechanically."""
        ident = report.sections["entity_identity"]
        groups = {g["surname"]: g["ids"] for g in ident["shared_surname_examples"]}
        assert "park" in groups
        assert set(groups["park"]) == {"person:jordan-park", "person:renee-montagne-park"}

    def test_it_reports_them_rather_than_merging_them(self, report) -> None:
        """Deliberately a FLAG, not a fix. Merging two people who merely share a surname is worse
        than leaving a duplicate, so this surfaces candidates for a human and stops there."""
        text = format_report(report)
        assert "split the affinity signal" in text
        assert "person entities" in text


class TestContentQuality:
    """A defect rate for the most-read text in the product (#1686).

    The fixture carries exactly one survivor of the greeting bug (#14) — `p01_e02`, whose summary
    still opens "welcome back to singletrack sessions" — which makes it a usable regression corpus:
    the expected answer is a specific non-zero number, not "clean".
    """

    def test_it_finds_the_one_known_echo(self, report) -> None:
        cq = report.sections["content_quality"]
        assert cq["episodes"] == 36
        assert cq["transcript_echo"] == 1
        assert cq["defects"] == 1
        assert cq["echo_examples"][0]["episode"].startswith("p01_e02")

    def test_the_rate_reaches_the_report(self, report) -> None:
        assert "defect rate **2.8%**" in format_report(report)


class TestTheEchoCheckActuallyReadsTranscripts:
    """The assertion that separates "measured clean" from "measured nothing".

    First version of `_transcript_opening` appended the suffix to the METADATA path, but
    transcripts live in a sibling `transcripts/` directory. It resolved 0/36 openings and the
    report printed **defect rate 0.0%** — which reads as a healthy corpus and was an instrument
    reading nothing at all.

    Both failure modes produce the same "0". Only asserting that openings RESOLVE tells them
    apart, so that is what these do.
    """

    def test_every_episode_resolves_an_opening(self) -> None:
        from podcast_scraper.capability_audit import _transcript_opening
        from podcast_scraper.server.corpus_catalog import build_catalog_rows_cumulative

        rows = build_catalog_rows_cumulative(CORPUS)
        resolved = [r for r in rows if _transcript_opening(CORPUS, r.metadata_relative_path)]
        assert len(resolved) == len(rows), (
            f"only {len(resolved)}/{len(rows)} transcript openings resolved — the echo check is "
            "reporting on episodes it cannot actually see"
        )

    def test_an_opening_is_real_text_not_a_fragment(self) -> None:
        """A regex that matched punctuation would 'resolve' while measuring nothing."""
        from podcast_scraper.capability_audit import _transcript_opening, ECHO_PREFIX_CHARS
        from podcast_scraper.server.corpus_catalog import build_catalog_rows_cumulative

        rows = build_catalog_rows_cumulative(CORPUS)
        opening = _transcript_opening(CORPUS, rows[0].metadata_relative_path)
        assert len(opening) >= ECHO_PREFIX_CHARS, opening
        assert len(opening.split()) >= 5, opening

    def test_a_missing_transcript_is_not_a_defect(self) -> None:
        """ "Cannot judge" must never be scored as "junk summary"."""
        from podcast_scraper.capability_audit import _transcript_opening

        assert _transcript_opening(CORPUS, "feeds/nope/run_x/metadata/nothing.metadata.json") == ""


class TestTopicMomentum:
    """Does the Trending rail ever have anything to show? (#1668)

    `TrendingTopics.vue` gates on `velocity_last_over_6mo >= 1.5` with `total >= 3`. The fixture's
    MAXIMUM reading is 0.857, so the rail is fully built, mounted, fetching — and has never once
    rendered its own content. It concludes "nothing qualifies" every single time, which is
    indistinguishable on screen from "nothing is trending this week".

    That is what makes the two Home rails contradict each other: the momentum rail called
    `systems thinking` 1.78x and "heating up" while this one computed 0.86x and showed nothing.
    """

    def test_nothing_clears_the_gate_on_this_corpus(self, report) -> None:
        mom = report.sections["topic_momentum"]
        assert mom["would_render"] == [], "the rail would show rows the gate should have hidden"
        assert mom["available"] is True
        assert mom["topics"] == 10
        assert mom["qualifying"] == 0
        assert mom["rail_is_always_empty"] is True

    def test_the_headroom_is_reported_not_just_the_verdict(self, report) -> None:
        """ "Nothing qualifies" is not actionable; "short by 0.64" says how far off the gate is."""
        mom = report.sections["topic_momentum"]
        assert mom["max_velocity"] == pytest.approx(0.857, abs=0.01)
        assert mom["headroom_to_gate"] == pytest.approx(0.643, abs=0.01)
        assert "short by 0.64" in format_report(report)

    def test_the_gate_matches_the_component(self, report) -> None:
        """The audit reports what the RAIL would show. If the gate constant moves and this does not,
        the audit measures a gate nothing applies.

        The rail's filtering moved SERVER-SIDE to the lean ``/corpus/trending-topics`` endpoint
        (the client stopped downloading the whole velocity envelope to filter ~12 rows), so the gate
        now lives in ``app_enrichment.py`` — ``TrendingTopics.vue`` is a dumb renderer of what the
        server already filtered/sorted/trimmed. Assert the audit's gate + min-total against THAT
        source of truth.
        """
        backend = (
            Path(__file__).resolve().parents[3]
            / "src"
            / "podcast_scraper"
            / "server"
            / "routes"
            / "app_enrichment.py"
        )
        if not backend.is_file():
            pytest.skip(f"backend route missing: {backend}")
        source = backend.read_text(encoding="utf-8")
        assert f"_RISING_DEFAULT = {report.sections['topic_momentum']['gate']}" in source
        assert "_MIN_TOTAL_DEFAULT = 3" in source


class TestTheMomentumReaderHandlesTheEnvelope:
    """`topics` lives under `data`, not at the top level.

    Reading the top level yields zero topics and would report "nothing is trending" — a plausible
    finding — when the truth is the audit read the wrong key. Same class as the transcript-path
    bug above, so it gets the same treatment: assert the data was actually FOUND.
    """

    def test_topics_were_actually_read(self, report) -> None:
        """Assert the DATA was found, not that anything qualified.

        On this fixture nothing clears the gate, so `would_render` is legitimately empty — an
        assertion on it would conflate "read nothing" with "nothing is trending", which is the
        exact confusion this class exists to prevent. `max_velocity > 0` proves real numbers
        were parsed.
        """
        mom = report.sections["topic_momentum"]
        assert mom["topics"] > 0, "no topics parsed — the envelope shape changed"
        assert mom["max_velocity"] > 0, "every velocity is zero, which suggests a bad read"

    def test_a_missing_artifact_says_so_rather_than_reporting_calm(self, tmp_path) -> None:
        """Absent enrichment must be `available: False`, never "0 topics qualify"."""
        from podcast_scraper.capability_audit import measure_topic_momentum

        out = measure_topic_momentum(tmp_path)
        assert out["available"] is False
        assert "reason" in out
        assert "qualifying" not in out


class TestFeedsAreNamedForHumans:
    """Which SHOW is broken is the actionable half of a coverage or defect report.

    Production `feed_id` is a sha256, so the first real run printed
    `sha256:0c54c0cf2a4f95044a1b4e2f9cd1f9632497e960e6dd9a235a4f64b0f8b5bfbb` as the worst feed —
    correct, and useless. The fixture hid it by using `p01`-style ids, which look fine either way:
    a formatting defect only production could show.
    """

    def test_coverage_names_the_show(self, report) -> None:
        feeds = [f["feed"] for f in report.sections["graph_coverage"]["worst_feeds"]]
        assert feeds, "no feeds reported"
        assert not any(f.startswith("sha256:") for f in feeds), feeds
        assert "Below the Surface" in feeds, feeds

    def test_a_hashed_id_is_truncated_rather_than_dumped(self) -> None:
        """When there is no title the id still has to fit on a line."""
        from podcast_scraper.capability_audit import _feed_label

        class _Row:
            feed_title = ""
            feed_id = "sha256:" + "0" * 64

        label = _feed_label(_Row())
        assert len(label) <= 20, label
        assert label.endswith("…")

    def test_a_title_wins_over_the_id(self) -> None:
        from podcast_scraper.capability_audit import _feed_label

        class _Row:
            feed_title = "Hard Fork"
            feed_id = "sha256:" + "a" * 64

        assert _feed_label(_Row()) == "Hard Fork"


class TestSingleWordEntitiesAreJudgedByFeedSpan:
    """A count of single-word ids is a number without a verdict. Feed span is the verdict.

    `person:sam` confined to ONE show is very likely a recurring first-name reference to a single
    person the extractor never got a surname for — untidy, harmless. `person:alex` across SIX
    shows is almost certainly six different Alexes pooled under one followable token, and that is
    a precision failure the user cannot undo: nothing lets them say which Alex they meant.

    Production measured 155 single-word ids and 420 prefix pairs (~22% of 1931 entities), and
    without span there was no way to say how many of those actually matter.
    """

    def test_the_fixture_case_is_the_benign_one(self, report) -> None:
        ident = report.sections["entity_identity"]
        assert ident["single_word_names"] == 7
        assert (
            ident["single_word_spanning_feeds"] == 0
        ), "every single-word id here is confined to one show, so none is a pooled-people risk"

    def test_the_warning_stays_silent_when_nothing_spans(self, report) -> None:
        """The corollary of the above: no spanning ids, no warning. A warning that always prints
        is not a warning, and this is the fixture that proves it can be quiet."""
        assert "pooled under one followable token" not in format_report(report)

    def test_span_is_reported_per_token(self, report) -> None:
        worst = report.sections["entity_identity"]["single_word_worst"]
        assert worst, "no single-word rows reported"
        assert {"token", "episodes", "feeds"} <= set(worst[0])
        assert all(r["feeds"] >= 1 for r in worst)

    def test_the_worst_are_listed_first(self, report) -> None:
        """Ordered by span then volume, so a production report leads with the real offenders."""
        worst = report.sections["entity_identity"]["single_word_worst"]
        keys = [(-r["feeds"], -r["episodes"]) for r in worst]
        assert keys == sorted(keys), worst


class TestAbsentAndBlankSummariesAreCountedApart:
    """ "8 empty summaries" was two different findings added together (#1686).

    `_summary_text` returns "" for a `summary: null` and for a summary OBJECT that holds no
    readable text, so the counter could not tell them apart — and the two demand opposite
    responses:

      ABSENT (`summary: null`) is the DESIGNED #1496 degradation. Summarisation failed, the
      episode was deliberately kept rather than dropped, and a Sentry warning was filed at the
      time. Nothing was hidden; there is nothing to gate.

      BLANK (an object with no readable text) is the pipeline recording success and producing
      nothing. That is the class of thing that should never reach the corpus.

    The verdict on whether this needs an ingestion gate rests entirely on which of the two the
    production count actually was, and the instrument could not say. Now it can.

    The fixture has ZERO of both, which is why these build their own corpora — the same reason
    `test_discover_pool_scales.py` does.
    """

    @staticmethod
    def _corpus(tmp_path, summaries):
        """Write one metadata artifact per entry and return (root, rows).

        `summaries` maps a name to the value of the `summary` key — `None` for the absent case,
        a dict for present ones.
        """
        import json
        from types import SimpleNamespace

        root = tmp_path / "corpus"
        (root / "feed" / "metadata").mkdir(parents=True)
        rows = []
        for name, summary in summaries.items():
            rel = f"feed/metadata/{name}.metadata.json"
            (root / rel).write_text(json.dumps({"summary": summary}), encoding="utf-8")
            rows.append(SimpleNamespace(metadata_relative_path=rel, feed_title="A Feed"))
        return root, rows

    def test_a_null_summary_counts_as_absent_not_blank(self, tmp_path) -> None:
        from podcast_scraper.capability_audit import measure_content_quality

        root, rows = self._corpus(tmp_path, {"e1": None})
        cq = measure_content_quality(root, rows)
        assert cq["absent_summary"] == 1
        assert cq["blank_summary"] == 0

    def test_a_missing_summary_key_also_counts_as_absent(self, tmp_path) -> None:
        """An artifact written before the field existed is the same finding as `null`."""
        import json
        from types import SimpleNamespace

        from podcast_scraper.capability_audit import measure_content_quality

        root = tmp_path / "corpus"
        (root / "feed" / "metadata").mkdir(parents=True)
        rel = "feed/metadata/e1.metadata.json"
        (root / rel).write_text(json.dumps({"title": "no summary key at all"}), encoding="utf-8")
        rows = [SimpleNamespace(metadata_relative_path=rel, feed_title="A Feed")]
        cq = measure_content_quality(root, rows)
        assert cq["absent_summary"] == 1
        assert cq["blank_summary"] == 0

    @pytest.mark.parametrize(
        "summary",
        [
            {},
            {"bullets": []},
            {"bullets": ["", "   "]},
            {"raw_text": "   ", "bullets": []},
            {"status": "valid", "bullets": []},
        ],
        ids=["empty-object", "no-bullets", "blank-bullets", "blank-raw-text", "valid-but-empty"],
    )
    def test_an_object_with_no_readable_text_counts_as_blank(self, tmp_path, summary) -> None:
        from podcast_scraper.capability_audit import measure_content_quality

        root, rows = self._corpus(tmp_path, {"e1": summary})
        cq = measure_content_quality(root, rows)
        assert cq["blank_summary"] == 1, summary
        assert cq["absent_summary"] == 0, summary

    def test_the_two_states_are_separated_within_one_corpus(self, tmp_path) -> None:
        """The case the production number was: a mix, reported as one figure."""
        from podcast_scraper.capability_audit import measure_content_quality

        root, rows = self._corpus(
            tmp_path,
            {
                "absent1": None,
                "absent2": None,
                "blank1": {"bullets": []},
                "ok": {"bullets": ["A real bullet with plenty of words in it to pass the check"]},
            },
        )
        cq = measure_content_quality(root, rows)
        assert cq["absent_summary"] == 2
        assert cq["blank_summary"] == 1
        assert cq["empty_summary"] == 3, "the combined figure must still be the sum"
        assert cq["defects"] == 3

    def test_a_blank_summary_is_called_out_in_the_report(self, tmp_path) -> None:
        """Absent is by-design; blank is not. The rendered line must not flatten that back."""
        from podcast_scraper.capability_audit import _render_content_quality, AuditReport

        root, rows = self._corpus(tmp_path, {"blank": {"bullets": []}, "absent": None})
        from podcast_scraper.capability_audit import measure_content_quality

        report = AuditReport(corpus_root=root, episodes=len(rows), feeds=1)
        report.sections["content_quality"] = measure_content_quality(root, rows)
        out: list[str] = []
        _render_content_quality(report, out)
        text = "\n".join(out)
        assert "1 absent (no summary written)" in text
        assert "1 blank (summary written, no text)" in text
        assert "recorded success and produced nothing" in text

    def test_no_warning_when_nothing_is_blank(self, tmp_path) -> None:
        """A warning that always prints is not a warning — the absent-only case stays quiet."""
        from podcast_scraper.capability_audit import (
            _render_content_quality,
            AuditReport,
            measure_content_quality,
        )

        root, rows = self._corpus(tmp_path, {"absent": None})
        report = AuditReport(corpus_root=root, episodes=len(rows), feeds=1)
        report.sections["content_quality"] = measure_content_quality(root, rows)
        out: list[str] = []
        _render_content_quality(report, out)
        assert "recorded success and produced nothing" not in "\n".join(out)


class TestBareNameResolutionUsesTheRuleProductionAlreadyTrusts:
    """Could a mint-time rule FIX the single-word person ids, or only quarantine them? (#1685)

    Marko's model, 2026-08-20: people introduce a full name and then use the first name — "did
    you hear about Jensen Huang... and then Jensen". So the question for each bare id is whether
    the SAME EPISODE also carries the full name.

    This is deliberately the same token-subset rule production already uses for insight mentions
    (`gi/relational_edges.py::_resolve_span_to_entities`, #1076 chunk 4-A), including its refusal
    to guess between two candidates. Reusing the rule rather than inventing one is the point: it
    is already tested, already shipped, and already conservative in the right direction.
    """

    @staticmethod
    def _classify(bare, persons):
        from podcast_scraper.capability_audit import classify_bare_name

        return classify_bare_name(bare, persons)

    def test_one_candidate_resolves(self) -> None:
        """The production case: `person:alex` with `alex-mayassi` a co-speaker in that episode."""
        verdict, cands = self._classify("alex", ["alex", "alex-mayassi", "darian-woods"])
        assert verdict == "resolvable"
        assert cands == ["alex-mayassi"]

    def test_two_candidates_refuse(self) -> None:
        """Donald vs Eric. Emitting either would be arbitrary; emitting both scatters."""
        verdict, cands = self._classify("trump", ["trump", "donald-trump", "eric-trump"])
        assert verdict == "ambiguous"
        assert cands == ["donald-trump", "eric-trump"]

    def test_no_candidate_is_an_orphan(self) -> None:
        """`person:jensen` in production: hollow, and `jensen-huang` is NOT in that episode."""
        verdict, cands = self._classify("jensen", ["jensen", "kevin-roose"])
        assert verdict == "orphan"
        assert cands == []

    def test_a_surname_only_reference_resolves_too(self) -> None:
        """Token-subset, not prefix. `musk` is a token of `elon-musk`.

        Prefix matching would catch the first-name half and silently miss this one — and
        surname-only reference is at least as common in interview shows.
        """
        verdict, cands = self._classify("musk", ["musk", "elon-musk"])
        assert verdict == "resolvable"
        assert cands == ["elon-musk"]

    def test_a_full_name_is_not_a_bare_name(self) -> None:
        verdict, _ = self._classify("elon-musk", ["elon-musk", "musk"])
        assert verdict == "not_bare"

    def test_a_coincidental_substring_is_not_a_candidate(self) -> None:
        """`al` must not resolve to `alex-mayassi`. Tokens, not characters."""
        verdict, cands = self._classify("al", ["al", "alex-mayassi"])
        assert (verdict, cands) == ("orphan", [])

    def test_resolution_is_scoped_to_the_episode(self) -> None:
        """`person:alex` spans two feeds and means someone else in the other one.

        A corpus-wide merge would pick one and be wrong for the other; per-episode gives the
        right answer in each. This is the safety property of the whole approach.
        """
        ep_a = ["alex", "alex-mayassi"]
        ep_b = ["alex", "alex-karp", "alex-rampell"]
        assert self._classify("alex", ep_a)[0] == "resolvable"
        assert self._classify("alex", ep_b)[0] == "ambiguous"


class TestTheResolvabilityRatioIsMeasuredNotGuessed:
    """The ratio decides whether the mint-time rule HEALS the graph or merely hides tokens."""

    def test_the_fixture_is_measured_end_to_end(self, report) -> None:
        from podcast_scraper.capability_audit import measure_bare_name_resolvability
        from podcast_scraper.server.corpus_catalog import build_catalog_rows_cumulative

        rows = build_catalog_rows_cumulative(CORPUS)
        out = measure_bare_name_resolvability(CORPUS, rows)
        # The fixture HAS single-word person ids — the audit reports 7 — so a zero here would
        # mean the walk resolved nothing rather than that the corpus is clean.
        assert out["occurrences"] > 0, "measured no bare names at all — the walk saw nothing"
        assert out["distinct_tokens"] > 0
        assert (
            out["resolvable"] + out["ambiguous"] + out["orphan"] == out["occurrences"]
        ), "every occurrence must land in exactly one bucket"

    def test_the_share_is_a_fraction_of_occurrences(self, report) -> None:
        from podcast_scraper.capability_audit import measure_bare_name_resolvability
        from podcast_scraper.server.corpus_catalog import build_catalog_rows_cumulative

        rows = build_catalog_rows_cumulative(CORPUS)
        out = measure_bare_name_resolvability(CORPUS, rows)
        assert 0.0 <= out["resolvable_share"] <= 1.0
        if out["occurrences"]:
            assert out["resolvable_share"] == out["resolvable"] / out["occurrences"]

    def test_a_token_seen_in_two_episodes_can_carry_two_verdicts(self, tmp_path) -> None:
        """Per-occurrence, not per-token — because `person:alex` really is both."""
        from podcast_scraper.capability_audit import classify_bare_name

        assert classify_bare_name("alex", ["alex", "alex-mayassi"])[0] == "resolvable"
        assert classify_bare_name("alex", ["alex"])[0] == "orphan"


class TestBothGraphLayersAreConsulted:
    """The candidate set is the EPISODE's people, not one artifact's (#1685).

    The first production run of this measure reported `person:alex` as an ORPHAN. The corpus
    disagreed: `person:alex` and `person:alex-mayassi` list each other as co-speakers in the very
    same episode, confirmed from both dossiers. The co-speaker relation lives in the GI layer,
    while the measure was reading only the KG entity set — so it answered "is the full name among
    this episode's KG entities?" while reporting it as "is the full name in this episode?".
    Adjacent, not identical, and it undercounted resolvability.

    The 36-episode fixture cannot catch that: its KG and GI person sets are IDENTICAL by
    construction (verified — 73 KG ids, 0 added by GI), so on that corpus "the GI read adds
    nothing" and "the GI read is broken" produce the same number. These build the case the
    fixture cannot.
    """

    @staticmethod
    def _row(tmp_path, kg_ids, gi_ids):
        import json
        from types import SimpleNamespace

        (tmp_path / "meta").mkdir(parents=True, exist_ok=True)
        gi_rel = "meta/ep.gi.json"
        (tmp_path / gi_rel).write_text(
            json.dumps({"nodes": [{"id": i, "type": "Person"} for i in gi_ids]}), encoding="utf-8"
        )
        return SimpleNamespace(has_gi=True, gi_relative_path=gi_rel), list(kg_ids)

    def test_a_person_only_in_gi_is_still_a_candidate(self, tmp_path) -> None:
        """The `alex` / `alex-mayassi` shape: the full name lives in GI, not KG."""
        from podcast_scraper.capability_audit import _episode_person_ids

        row, kg = self._row(tmp_path, ["person:alex"], ["person:alex", "person:alex-mayassi"])
        ids = _episode_person_ids(tmp_path, row, kg)
        assert "person:alex-mayassi" in ids, "the GI-only person was not picked up"

    def test_that_person_makes_the_bare_name_resolvable(self, tmp_path) -> None:
        """The consequence that matters: the verdict flips from orphan to resolvable."""
        from podcast_scraper.capability_audit import _episode_person_ids, classify_bare_name

        row, kg = self._row(tmp_path, ["person:alex"], ["person:alex", "person:alex-mayassi"])
        kg_only = classify_bare_name("alex", kg)
        both = classify_bare_name("alex", _episode_person_ids(tmp_path, row, kg))
        assert kg_only[0] == "orphan", "KG alone is what produced the wrong answer"
        assert both == ("resolvable", ["alex-mayassi"])

    def test_kg_persons_are_never_dropped(self, tmp_path) -> None:
        """Union, not replacement — a KG-only person must survive."""
        from podcast_scraper.capability_audit import _episode_person_ids

        row, kg = self._row(tmp_path, ["person:kg-only"], ["person:gi-only"])
        ids = _episode_person_ids(tmp_path, row, kg)
        assert {"person:kg-only", "person:gi-only"} <= ids

    def test_an_episode_without_gi_still_works(self, tmp_path) -> None:
        from types import SimpleNamespace

        from podcast_scraper.capability_audit import _episode_person_ids

        row = SimpleNamespace(has_gi=False, gi_relative_path="")
        assert _episode_person_ids(tmp_path, row, ["person:a"]) == {"person:a"}

    def test_an_unreadable_gi_artifact_does_not_lose_the_kg_ids(self, tmp_path) -> None:
        """A broken artifact must degrade to the KG set, never to nothing."""
        from types import SimpleNamespace

        from podcast_scraper.capability_audit import _episode_person_ids

        (tmp_path / "meta").mkdir(parents=True, exist_ok=True)
        (tmp_path / "meta" / "ep.gi.json").write_text("{ not json", encoding="utf-8")
        row = SimpleNamespace(has_gi=True, gi_relative_path="meta/ep.gi.json")
        assert _episode_person_ids(tmp_path, row, ["person:a"]) == {"person:a"}

    def test_the_fixture_really_does_read_its_gi_artifacts(self) -> None:
        """Guards the "adds nothing" vs "reads nothing" ambiguity on the real fixture.

        If the GI read silently failed, this and the production measure would both report the
        KG-only answer and look healthy — the same trap as `_transcript_opening` resolving 0/36
        openings and printing a 0.0% defect rate.
        """
        from podcast_scraper.server.app_corpus_access import load_json_artifact
        from podcast_scraper.server.corpus_catalog import build_catalog_rows_cumulative

        rows = build_catalog_rows_cumulative(CORPUS)
        read = 0
        for row in rows:
            if not row.has_gi:
                continue
            art = load_json_artifact(CORPUS, row.gi_relative_path)
            if isinstance(art, dict) and any(
                isinstance(n, dict) and str(n.get("id", "")).startswith("person:")
                for n in (art.get("nodes") or [])
            ):
                read += 1
        assert read == len(rows), f"only {read}/{len(rows)} GI artifacts yielded person nodes"


class TestTheMeasureActuallyUsesBothLayers:
    """Pins the WIRING, not just the helper (#1685).

    Sabotaging `measure_bare_name_resolvability` back to `persons = set(kg_persons)` — literally
    the bug this change fixes — failed ZERO tests. Every test above exercises
    `_episode_person_ids` directly, so nothing noticed that the measure had stopped calling it.
    A helper that is correct and unused is the same as no helper, and the regression would have
    been silent and permanent.
    """

    def test_a_gi_only_full_name_flips_the_measure_to_resolvable(self, tmp_path, monkeypatch):
        import json
        from types import SimpleNamespace

        from podcast_scraper import capability_audit as ca

        (tmp_path / "meta").mkdir(parents=True)
        (tmp_path / "meta" / "ep.gi.json").write_text(
            json.dumps(
                {
                    "nodes": [
                        {"id": "person:alex", "type": "Person"},
                        {"id": "person:alex-mayassi", "type": "Person"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        row = SimpleNamespace(
            has_gi=True,
            gi_relative_path="meta/ep.gi.json",
            feed_title="A Feed",
            feed_id="f1",
        )
        # KG knows ONLY the bare name — the production shape that produced the wrong answer.
        monkeypatch.setattr(
            ca, "_episode_features", lambda *a, **k: (set(), set(), {"person:alex"})
        )
        monkeypatch.setattr(ca, "consumer_topic_cluster_map", lambda root: {}, raising=False)
        monkeypatch.setattr(ca, "consumer_theme_cluster_map", lambda root: {}, raising=False)

        out = ca.measure_bare_name_resolvability(tmp_path, [row])
        assert out["occurrences"] == 1
        assert (
            out["resolvable"] == 1
        ), "the measure ignored the GI layer — it is not calling _episode_person_ids"
        assert out["orphan"] == 0


class TestRankingCalibration:
    """#1684: the two `app_ranking_config` numbers whose tuning is unverifiable at 36 episodes.

    v3 is 9 feeds x 4 episodes — EVERY feed is sparse by the <5 rule, which is exactly why the
    fixture cannot answer the calibration question and production has to. The structural
    assertions here pin that the measurement produces the numbers the decision needs.
    """

    def test_significance_normalisation_numbers(self, report) -> None:
        sig = report.sections["ranking_calibration"]["significance"]
        assert sig["feeds"] == 9
        assert sig["sparse_feeds"] == 9, "all v3 feeds have 4 episodes — a mean over 4 is noise"
        assert sig["feed_mean_min"] <= sig["feed_mean_median"] <= sig["feed_mean_max"]
        # The over-reward question is answered by comparing where sparse feeds land vs their size.
        assert 0.0 <= sig["sparse_top_share"] <= 1.0
        assert 0.0 < sig["sparse_corpus_share"] <= 1.0
        assert sig["global_median_normalized"] > 0

    def test_recency_decay_numbers(self, report) -> None:
        rec = report.sections["ranking_calibration"]["recency"]
        assert rec["half_life_days"] == 730.0, "must read the SHIPPED config, not a copy of it"
        # The newest episode decays from itself: multiplier exactly 1.0 at the top of the range.
        assert rec["multiplier_max"] == 1.0
        assert 0.0 < rec["multiplier_min"] <= rec["multiplier_median"] <= 1.0
        assert 0.0 < rec["share_within_one_half_life"] <= 1.0
        assert rec["share_within_one_half_life"] <= rec["share_within_two_half_lives"] <= 1.0
        assert rec["span_days"] > 0

    def test_the_report_renders_the_section(self, report) -> None:
        from podcast_scraper.capability_audit import format_report as _fmt

        text = _fmt(report)
        assert "Ranking calibration" in text
