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
            "cluster_reach",
            "picker_discrimination",
            "pool_reachability",
            "corpus_shape",
            "graph_coverage",
            "entity_identity",
            "content_quality",
            "topic_momentum",
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
        """The audit reports what the RAIL would show. If the UI constant moves and this does not,
        the audit measures a gate nothing applies."""
        component = (
            Path(__file__).resolve().parents[3]
            / "web"
            / "learning-player"
            / "src"
            / "components"
            / "TrendingTopics.vue"
        )
        if not component.is_file():
            pytest.skip(f"component missing: {component}")
        source = component.read_text(encoding="utf-8")
        assert f"const RISING = {report.sections['topic_momentum']['gate']}" in source
        assert "const MIN_TOTAL = 3" in source


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
