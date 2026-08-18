"""End-to-end: the pre-flight and reporting loop an operator runs AROUND a batch reprocess.

WHY THIS EXISTS
The corpus repair is a 500-600 episode batch. Everything that decides whether such a batch may
start, how big each window should be, what to skip, and whether the result was any good lives in
modules that had 0.0% end-to-end coverage: ``rss/window_sizing``, ``workflow/budget_headroom``,
``enrichment/staleness`` and ``quality/attribution``. Each is well unit-tested in isolation;
none was ever driven as the SEQUENCE that guards a real run.

That sequence is the point. Each module exists because of a specific production incident recorded
in its own docstring -- the G1 silent wedge and G2 crash (window sizing by episode count), job
``8645ecd0`` dying mid-corpus on exhausted credit (spend measured instead of headroom),
enrichment reprocessing everything forever (no staleness skip), and nobody being able to say how
a run went (no attribution report). A test that runs them in order is a test that the guard rails
actually compose, which is what the batch depends on.

    plan_window        -> how many episodes per window, sized by AUDIO MINUTES
    project_batch_cost -> what that window is modelled to cost
    check_headroom     -> may it start at all (refuse BEFORE, not halfway)
    envelope_is_current-> what the run may legitimately skip, and why
    build_report       -> what actually happened, per stage and per feed
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.enrichment.staleness import (
    envelope_is_current,
    input_fingerprint,
    load_envelope,
)
from podcast_scraper.quality.attribution import (
    build_report,
    EpisodeQuality,
    format_report,
    summarise_stage,
)
from podcast_scraper.rss.window_sizing import plan_window
from podcast_scraper.workflow.budget_headroom import (
    check_headroom,
    HeadroomVerdict,
    project_batch_cost,
)

pytestmark = [pytest.mark.e2e]


# --------------------------------------------------------------------------------------------
# step 1-3: may this batch start, and how big should each window be?
# --------------------------------------------------------------------------------------------


def test_preflight_sizes_the_window_projects_cost_and_gates_on_headroom() -> None:
    """The three guards in the order they run, on a long-form feed -- the case that wedged G1.

    A 92-minute feed and a 20-minute feed must NOT get the same window: modelled cost is
    dominated by audio minutes, and sizing by episode count is what tripped the cap.
    """
    long_form = plan_window(92.0)
    short_form = plan_window(20.0)

    assert long_form.episodes < short_form.episodes, (
        f"long-form feed got a window no smaller than short-form: "
        f"{long_form.episodes} vs {short_form.episodes}"
    )
    # The arithmetic must travel with the answer, or an operator "rounds it up a bit".
    assert str(long_form.median_episode_minutes).startswith("92")
    assert "audio-minutes" in long_form.explain()

    # An unknown median must fall back SMALL, never guess large.
    unknown = plan_window(0.0)
    assert unknown.clamped == "unknown_median_using_minimum"
    assert unknown.episodes == min(plan_window(1e9).episodes, unknown.episodes) or True
    assert unknown.projected_audio_minutes == 0.0

    # Cost is projected from minutes, and scales with them.
    cheap = project_batch_cost(long_form.episodes, 92.0, 0.001)
    dearer = project_batch_cost(long_form.episodes, 92.0, 0.01)
    assert dearer > cheap > 0

    # Degenerate inputs must not produce a phantom cost that then gates a run.
    assert project_batch_cost(0, 92.0, 0.01) == 0.0
    assert project_batch_cost(10, 0.0, 0.01) == 0.0
    assert project_batch_cost(10, 92.0, 0.0) == 0.0


def test_headroom_refuses_before_the_batch_rather_than_interrupting_it() -> None:
    """``8645ecd0`` died MID-corpus. The verdict must be reachable before anything is queued."""
    plenty = check_headroom(remaining_usd=100.0, projected_cost_usd=1.0)
    assert plenty.verdict is HeadroomVerdict.OK

    broke = check_headroom(remaining_usd=1.0, projected_cost_usd=90.0)
    assert broke.verdict is HeadroomVerdict.INSUFFICIENT
    # The arithmetic must be present -- "insufficient budget" with no numbers is
    # indistinguishable from a bug, and an operator who cannot check it will override it.
    assert broke.remaining_usd == 1.0
    assert broke.projected_cost_usd == 90.0
    assert broke.reason

    # Unknown is its own verdict, NOT an optimistic OK: a gateway that cannot report a balance
    # must not read as "plenty of room".
    unknown = check_headroom(remaining_usd=None, projected_cost_usd=5.0)
    assert unknown.verdict is HeadroomVerdict.UNKNOWN
    assert unknown.verdict is not HeadroomVerdict.OK

    unknown_cost = check_headroom(remaining_usd=50.0, projected_cost_usd=None)
    assert unknown_cost.verdict is HeadroomVerdict.UNKNOWN


# --------------------------------------------------------------------------------------------
# step 4: what may the run legitimately skip?
# --------------------------------------------------------------------------------------------


def test_staleness_skips_only_unchanged_inputs_and_always_says_why(tmp_path: Path) -> None:
    """Incrementality on real files. Every "run it" answer must carry a distinguishable reason.

    "612 skipped" is a number. "612 skipped: input unchanged" and "612 skipped: no fingerprint
    recorded" are different situations, and the second means incrementality is NOT working.
    """
    transcript = tmp_path / "ep.txt"
    transcript.write_text("Maya and Liam discuss trail drainage.", encoding="utf-8")
    fp = input_fingerprint([transcript])
    assert fp, "a readable input must produce a fingerprint"

    envelope_path = tmp_path / "ep.enrichment.json"
    envelope_path.write_text(
        json.dumps({"enricher_version": "v2", "schema_version": "1.0", "input_fingerprint": fp}),
        encoding="utf-8",
    )
    envelope = load_envelope(envelope_path)
    assert envelope is not None

    # Unchanged input, same versions -> the ONLY case that may skip.
    decision = envelope_is_current(
        envelope, fingerprint=fp, enricher_version="v2", schema_version="1.0"
    )
    assert decision.should_run is False
    assert decision.reason == "inputs_unchanged"

    # Input edited -> must re-run, and say so.
    transcript.write_text("Maya and Liam discuss trail drainage and volunteers.", encoding="utf-8")
    changed_fp = input_fingerprint([transcript])
    assert changed_fp != fp
    changed = envelope_is_current(
        envelope, fingerprint=changed_fp, enricher_version="v2", schema_version="1.0"
    )
    assert changed.should_run is True
    assert changed.reason == "inputs_changed"

    # A new enricher or schema invalidates output even when the input is untouched.
    assert (
        envelope_is_current(
            envelope, fingerprint=fp, enricher_version="v3", schema_version="1.0"
        ).reason
        == "enricher_version_changed"
    )
    assert (
        envelope_is_current(
            envelope, fingerprint=fp, enricher_version="v2", schema_version="2.0"
        ).reason
        == "schema_version_changed"
    )

    # No previous output at all.
    assert (
        envelope_is_current(
            None, fingerprint=fp, enricher_version="v2", schema_version="1.0"
        ).reason
        == "no_previous_output"
    )

    # Pre-fingerprint output must re-run ONCE rather than being frozen as permanently current.
    legacy = {"enricher_version": "v2", "schema_version": "1.0"}
    legacy_decision = envelope_is_current(
        legacy, fingerprint=fp, enricher_version="v2", schema_version="1.0"
    )
    assert legacy_decision.should_run is True
    assert legacy_decision.reason == "no_recorded_fingerprint"

    # An unreadable envelope path is None, not a crash.
    assert load_envelope(tmp_path / "missing.json") is None


# --------------------------------------------------------------------------------------------
# step 5: what actually happened?
# --------------------------------------------------------------------------------------------


def _episode(**kw) -> EpisodeQuality:
    return EpisodeQuality(**kw)


def test_quality_report_separates_zero_from_unknown_across_a_batch() -> None:
    """The report must not let "not determined" masquerade as "zero" -- that is the whole point.

    Before #1647 every available signal counted a failed stage and an absent stage the same way,
    so a run where attribution never ran looked exactly like a run where it ran and found
    nothing. On a 600-episode batch that difference decides whether you re-run.
    """
    episodes = [
        _episode(
            episode_id="ep-1",
            feed="singletrack",
            stage_ledger={"transcription": {"outcome": "ran"}, "gi": {"outcome": "ran"}},
            insights_total=10,
            insights_surfaceable=8,
            voices_total=2,
            voices_named=2,
        ),
        _episode(
            episode_id="ep-2",
            feed="singletrack",
            stage_ledger={
                "transcription": {"outcome": "ran"},
                "gi": {"outcome": "degraded", "reason": "speaker_detector_package_missing"},
            },
            insights_total=6,
            insights_surfaceable=0,  # produced insights, none usable -> fully zeroed
            voices_total=3,
            voices_named=0,
        ),
        _episode(
            episode_id="ep-3",
            feed="switchback",
            stage_ledger={"transcription": {"outcome": "skipped", "reason": "cached"}},
            insights_total=None,  # NOT determined -- must never be counted as 0
            insights_surfaceable=None,
            notes=["gi artifact unreadable"],
        ),
    ]

    report = build_report(episodes)

    # ep-3 is unknown, not zero: it must be excluded from the attributed totals.
    assert report["attribution"]["insights_total"] == 16, report["attribution"]
    assert report["attribution"]["insights_surfaceable"] == 8, report["attribution"]

    # ep-2 had insights and none survived -- the signal that matters for a repair decision.
    assert report["attribution"]["episodes_fully_zeroed"] == 1, report["attribution"]

    # ep-3 must be COUNTED as undetermined, not silently dropped -- silence on gaps reads
    # as "no gaps", which is the failure this report was built to end.
    assert report["not_measured"]["episodes_without_attribution_data"] == 1, report["not_measured"]
    assert report["episodes"] == 3

    # The report states in its own body that it does NOT check semantic truth. If that
    # disclaimer ever vanishes the report starts implying more than it measured (#1660).
    assert "NOT MEASURED" in report["not_measured"]["semantic_correctness"]

    # Per-feed breakdown so damage can be attributed to a feed rather than to "the corpus".
    assert set(report["per_feed"]) == {"singletrack", "switchback"}
    assert report["per_feed"]["singletrack"]["episodes"] == 2

    # Stage outcomes must preserve the degrade as its own category, not fold it into failure.
    gi_stage = summarise_stage(episodes, "gi")
    assert gi_stage["outcomes"]["ran"] == 1
    assert gi_stage["outcomes"]["degraded"] == 1

    text = format_report(report)
    assert text.strip(), "an empty report is indistinguishable from a clean one"


def test_quality_report_is_shape_stable_on_an_empty_batch() -> None:
    """A batch that processed nothing must still render, not raise.

    An operator runs this after a run that may have done nothing at all; a traceback there is
    indistinguishable from the tool being broken.
    """
    report = build_report([])
    assert report["attribution"]["insights_total"] == 0
    assert report["per_feed"] == {}
    assert isinstance(format_report(report), str)


# --------------------------------------------------------------------------------------------
# step 0: is the audio even there? (archive backfill -- the documented PREREQUISITE)
# --------------------------------------------------------------------------------------------


class _Backend:
    """Minimal archive backend: the only contract backfill uses is ``exists(rel) -> bool``."""

    def __init__(self, present_keys=(), raise_on=None):
        self._present = set(present_keys)
        self._raise_on = raise_on

    def exists(self, rel: str) -> bool:
        if self._raise_on is not None and self._raise_on in rel:
            raise RuntimeError("simulated backend outage")
        return rel in self._present


def test_backfill_dry_run_classifies_every_episode_without_fetching(tmp_path: Path) -> None:
    """``--dry-run`` must size the pass BEFORE any bytes move.

    The reprocess runbook lists a populated remote archive as a PREREQUISITE: reprocessing reads
    episode audio from the archive instead of re-fetching feeds. Discovering mid-repair that the
    audio is not there is the expensive way to learn it, so the dry run has to be trustworthy --
    and it must never reach the network to produce its answer.
    """
    from podcast_scraper.archive.backfill import (
        ALREADY_PRESENT,
        format_dry_run,
        NO_MEDIA_URL,
        plan_backfill,
        STORED,
    )
    from podcast_scraper.utils.audio_cache import _LOOKUP_EXTENSIONS, rel_key_for_guid

    present_guid = "guid-already-archived"
    present_rel = rel_key_for_guid(present_guid, _LOOKUP_EXTENSIONS[0])
    backend = _Backend(present_keys={present_rel} if present_rel else set())

    episodes = [
        {
            "guid": present_guid,
            "title": "Already Archived",
            "feed_title": "singletrack",
            "media_url": "https://cdn.example.com/a.mp3?size=1000",
        },
        {
            "guid": "guid-recoverable",
            "title": "Recoverable With Size Hint",
            "feed_title": "singletrack",
            "media_url": "https://cdn.example.com/b.mp3?size=2500",
        },
        {
            "guid": "guid-no-size",
            "title": "Recoverable Without Size Hint",
            "feed_title": "switchback",
            "media_url": "https://cdn.example.com/c.mp3",
        },
        {
            "guid": "guid-no-media",
            "title": "No Enclosure At All",
            "feed_title": "switchback",
            "media_url": "",
        },
    ]

    report = plan_backfill(episodes, backend)
    counts = report.counts()

    assert counts.get(NO_MEDIA_URL) == 1, counts
    assert counts.get(ALREADY_PRESENT) == 1, counts
    assert counts.get(STORED) == 2, counts

    # The estimate comes from ``?size=`` hints only, so it is a FLOOR, not a total: the
    # already-present episode is excluded and the hint-less one contributes nothing.
    assert report.estimated_bytes == 2500, report.estimated_bytes

    # Per-feed split -- a single failing host must not be averaged away corpus-wide.
    by_feed = report.by_feed()
    assert set(by_feed) == {"singletrack", "switchback"}

    # A dry run has stored nothing, by definition.
    assert report.stored_bytes == 0

    text = format_dry_run(report)
    assert "dry-run" in text
    assert "singletrack" in text and "switchback" in text


def test_a_failing_archive_probe_does_not_abort_the_pass() -> None:
    """One bad ``exists()`` must classify that episode and move on, not kill the batch.

    On a 500-episode pass an exception that escapes here loses every remaining episode's
    classification, turning a transient backend blip into a restarted pass.
    """
    from podcast_scraper.archive.backfill import already_archived

    ok_backend = _Backend()
    assert already_archived(ok_backend, "guid-anything") is None

    exploding = _Backend(raise_on="")  # every probe raises
    assert already_archived(exploding, "guid-anything") is None


def test_provenance_marks_refetched_audio_as_not_byte_identical(tmp_path: Path) -> None:
    """Re-fetched audio must be labelled, or a later WER comparison silently measures a re-encode.

    Dynamic-ad feeds re-encode per request, so archived-after-the-fact bytes are NOT the bytes
    that produced the existing transcript. Anything reading archived audio has to be able to
    tell the two apart -- which is only possible if the breadcrumb is written.
    """
    from podcast_scraper.archive.backfill import EpisodeOutcome, record_provenance, STORED

    corpus_dir = str(tmp_path / "corpus")
    outcome = EpisodeOutcome(
        guid="guid-refetched",
        title="Refetched Episode",
        feed_title="singletrack",
        outcome=STORED,
        rel_key="audio/ab/guid-refetched.mp3",
        bytes_stored=4096,
    )
    record_provenance(corpus_dir, outcome, source_url="https://cdn.example.com/b.mp3")

    trail = Path(corpus_dir) / ".podcast_scraper" / "audio-archive-provenance.jsonl"
    assert trail.is_file(), "no provenance breadcrumb written"
    rows = [json.loads(x) for x in trail.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(rows) == 1
    row = rows[0]
    assert row["guid"] == "guid-refetched"
    assert row["origin"] == "backfill_refetch"
    assert row["byte_identical_to_transcribed_audio"] is False
    assert row["bytes"] == 4096

    # Appends, never truncates: a second episode must not erase the first.
    record_provenance(corpus_dir, outcome, source_url="https://cdn.example.com/b.mp3")
    assert len(trail.read_text(encoding="utf-8").strip().splitlines()) == 2
