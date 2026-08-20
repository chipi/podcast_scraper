"""An episode persisted WITHOUT a summary must say so on the artifact, not only in a log (#1686).

Marko, 2026-08-20: "it's never acceptable to have an episode without the summary of a single one
... not mark it as a warning, but keep it as an error on metadata mark episode this erroneous."

WHAT WAS ACTUALLY WRONG. #1647 built a stage ledger precisely so "did this stage run?" stops being
a guess, and the five paths that end with no summary were never wired into it. The only
`summarization` entry the ledger ever carried was `deadline_exceeded_but_completed` — a summary
that was LATE but landed. So the ledger recorded the harmless case and stayed silent on the
harmful one, while the comment at the degradation site claimed it "still records the degradation".

The consequence is not cosmetic. Nothing downstream could FIND those episodes: the production
audit had to infer them from `summary: null`, which cannot distinguish "summarisation failed"
from "this artifact predates summaries", and `--skip-existing` treats them as complete forever.
A mark on the artifact is what makes the degraded set addressable at all.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast, TYPE_CHECKING
from unittest.mock import patch

import pytest

from podcast_scraper.exceptions import RecoverableSummarizationError
from podcast_scraper.workflow import metadata_generation as mg
from podcast_scraper.workflow.metrics import Metrics

if TYPE_CHECKING:
    from podcast_scraper.config import Config
    from podcast_scraper.models import Episode

pytestmark = [pytest.mark.unit]

EPISODE_IDX = 7


def _episode(**fields: object) -> "Episode":
    """A lightweight Episode double — the summary stage reads only a few attrs (idx/title/item)."""
    return cast("Episode", SimpleNamespace(**fields))


def _cfg(**fields: object) -> "Config":
    """A lightweight Config double — the stage only checks generate_summaries / dry_run."""
    return cast("Config", SimpleNamespace(**fields))


def _degrade(code: str, *, metrics) -> None:
    """Run the summary stage with a failure that degrades, and record into *metrics*."""
    episode = _episode(idx=EPISODE_IDX)
    cfg = _cfg(generate_summaries=True, dry_run=False)

    def _raise(**_):
        raise RecoverableSummarizationError(EPISODE_IDX, "boom", code=code)

    with (
        patch.object(mg, "_generate_episode_summary", _raise),
        patch.object(mg, "_capture_stage_exception", lambda exc, *, stage, level=None: None),
    ):
        meta, _elapsed, _cm = mg._generate_and_validate_summary(
            episode,
            "https://example.com/feed.xml",
            "transcript.txt",
            "/out",
            cfg,
            summary_provider=None,
            whisper_model=None,
            pipeline_metrics=metrics,
        )
    assert meta is None, "the episode must still be kept — this is the #1496 contract"


class TestTheLedgerRecordsAnEpisodeThatLostItsSummary:
    def test_the_stage_is_recorded_as_failed(self) -> None:
        metrics = Metrics()
        _degrade(RecoverableSummarizationError.TOKENIZER_THREADING, metrics=metrics)
        record = metrics.stage_outcomes_by_episode[EPISODE_IDX]["summarization"]
        assert record["outcome"] == "failed"

    def test_failed_not_degraded(self) -> None:
        """`degraded` means "produced output via a fallback path". There is no output.

        Calling this `degraded` would put it in the same bucket as
        `deadline_exceeded_but_completed` — a summary that was slow but real — and the whole
        point is that those two are not the same event.
        """
        metrics = Metrics()
        _degrade(RecoverableSummarizationError.PROVIDER_CONTENT_REJECTED, metrics=metrics)
        assert metrics.stage_outcomes_by_episode[EPISODE_IDX]["summarization"]["outcome"] != (
            "degraded"
        )

    @pytest.mark.parametrize(
        "code",
        [
            RecoverableSummarizationError.SCHEMA_INVALID_AFTER_REROLL,
            RecoverableSummarizationError.TOKENIZER_THREADING,
            RecoverableSummarizationError.PROVIDER_CONTENT_REJECTED,
            RecoverableSummarizationError.PROMPT_EXAMPLES_LEAKED,
        ],
    )
    def test_every_cause_keeps_its_own_slug(self, code: str) -> None:
        """A single "summary failed" count cannot tell a transient from a terminal failure.

        These four want different responses — retry the transient one, do not retry a content
        rejection — so the ledger has to carry WHICH, not just THAT.
        """
        metrics = Metrics()
        _degrade(code, metrics=metrics)
        assert metrics.stage_outcomes_by_episode[EPISODE_IDX]["summarization"]["reason"] == code

    def test_an_untagged_path_is_visible_rather_than_absent(self) -> None:
        """A degradation path added later that forgets its slug must still appear in the ledger.

        Defaulting to `unspecified` keeps it countable. Defaulting to "no record" would make a
        new bug look exactly like a healthy corpus, which is the failure this whole issue is.
        """
        metrics = Metrics()
        episode = _episode(idx=EPISODE_IDX)
        cfg = _cfg(generate_summaries=True, dry_run=False)

        def _raise(**_):
            raise RecoverableSummarizationError(EPISODE_IDX, "a path nobody tagged")

        with (
            patch.object(mg, "_generate_episode_summary", _raise),
            patch.object(mg, "_capture_stage_exception", lambda exc, *, stage, level=None: None),
        ):
            mg._generate_and_validate_summary(
                episode,
                "https://example.com/feed.xml",
                "transcript.txt",
                "/out",
                cfg,
                summary_provider=None,
                whisper_model=None,
                pipeline_metrics=metrics,
            )
        record = metrics.stage_outcomes_by_episode[EPISODE_IDX]["summarization"]
        assert record["outcome"] == "failed"
        assert record["reason"] == RecoverableSummarizationError.UNSPECIFIED

    def test_the_cause_is_readable_without_the_logs(self) -> None:
        metrics = Metrics()
        _degrade(RecoverableSummarizationError.TOKENIZER_THREADING, metrics=metrics)
        detail = metrics.stage_outcomes_by_episode[EPISODE_IDX]["summarization"]["detail"]
        assert "boom" in detail["message"]


class TestTheMarkSurvivesIntoTheWrittenArtifact:
    """A ledger entry that never reaches metadata.json would be the same silence, one layer in."""

    def test_it_reaches_the_processing_metadata(self) -> None:
        metrics = Metrics()
        _degrade(RecoverableSummarizationError.TOKENIZER_THREADING, metrics=metrics)
        ledger = mg._extract_episode_stage_ledger(metrics, EPISODE_IDX)
        assert ledger is not None
        assert ledger["summarization"].outcome == "failed"
        assert ledger["summarization"].reason == (RecoverableSummarizationError.TOKENIZER_THREADING)

    def test_a_healthy_episode_carries_no_summarization_failure(self) -> None:
        """The mark must be absent when nothing degraded — otherwise it is not a signal."""
        metrics = Metrics()
        # The success path reaches `get_episode_id_from_episode`, which the degrade paths never
        # do — so this episode needs the fields a real one carries.
        episode = _episode(idx=EPISODE_IDX, title="An Episode", item=None)
        cfg = _cfg(generate_summaries=True, dry_run=False)
        good = SimpleNamespace(title="A real summary", bullets=["something substantive"])

        with patch.object(mg, "_generate_episode_summary", lambda **_: (good, None)):
            meta, _elapsed, _cm = mg._generate_and_validate_summary(
                episode,
                "https://example.com/feed.xml",
                "transcript.txt",
                "/out",
                cfg,
                summary_provider=None,
                whisper_model=None,
                pipeline_metrics=metrics,
            )

        assert meta is good
        recorded = metrics.stage_outcomes_by_episode.get(EPISODE_IDX, {})
        assert recorded.get("summarization", {}).get("outcome") != "failed"


class TestTheLedgerNeverCostsUsTheEpisode:
    def test_a_broken_metrics_object_does_not_break_the_pipeline(self) -> None:
        """Telemetry is not worth an episode. #1496 keeps the episode; so must this."""

        class Hostile:
            def record_stage_outcome(self, *a, **k):
                raise RuntimeError("metrics backend down")

            def record_summarize_time(self, *a, **k):
                pass

        episode = _episode(idx=EPISODE_IDX)
        cfg = _cfg(generate_summaries=True, dry_run=False)

        def _raise(**_):
            raise RecoverableSummarizationError(EPISODE_IDX, "boom")

        with (
            patch.object(mg, "_generate_episode_summary", _raise),
            patch.object(mg, "_capture_stage_exception", lambda exc, *, stage, level=None: None),
        ):
            meta, _elapsed, _cm = mg._generate_and_validate_summary(
                episode,
                "https://example.com/feed.xml",
                "transcript.txt",
                "/out",
                cfg,
                summary_provider=None,
                whisper_model=None,
                pipeline_metrics=Hostile(),
            )
        assert meta is None


class TestTheTransientFailureGetsOneMoreAttempt:
    """The one cause the code itself calls transient was the only one never retried (#1686).

    A schema parse failure got the ADR-148 in-place re-roll. A tokenizer "Already borrowed" —
    whose own handler comment says "known threading issue ... can occur in parallel execution",
    i.e. a race between workers that a second pass can genuinely win — got nothing, and the
    episode was written summary-less. That is the least defensible way to lose a summary.
    """

    @staticmethod
    def _run(codes, *, metrics=None):
        """Fail with `codes[0]`, then behave as `codes[1]` (an exception class or a result)."""
        episode = _episode(idx=EPISODE_IDX, title="An Episode", item=None)
        cfg = _cfg(generate_summaries=True, dry_run=False)
        calls = {"n": 0}
        good = SimpleNamespace(title="Recovered", bullets=["a real bullet"])

        def _attempt(**_):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RecoverableSummarizationError(EPISODE_IDX, "first", code=codes[0])
            if codes[1] == "ok":
                return (good, None)
            raise RecoverableSummarizationError(EPISODE_IDX, "second", code=codes[1])

        with (
            patch.object(mg, "_generate_episode_summary", _attempt),
            patch.object(mg, "_capture_stage_exception", lambda exc, *, stage, level=None: None),
        ):
            meta, _e, _cm = mg._generate_and_validate_summary(
                episode,
                "https://example.com/feed.xml",
                "transcript.txt",
                "/out",
                cfg,
                summary_provider=None,
                whisper_model=None,
                pipeline_metrics=metrics,
            )
        return meta, calls["n"], good

    def test_a_tokenizer_race_is_retried_and_can_recover(self) -> None:
        meta, attempts, good = self._run((RecoverableSummarizationError.TOKENIZER_THREADING, "ok"))
        assert attempts == 2, "the transient cause must get a second attempt"
        assert meta is good, "and a summary recovered on retry must be KEPT"

    def test_a_content_rejection_is_not_retried(self) -> None:
        """The provider refused THIS input. A retry spends money to fail identically."""
        meta, attempts, _ = self._run(
            (RecoverableSummarizationError.PROVIDER_CONTENT_REJECTED, "ok")
        )
        assert attempts == 1
        assert meta is None

    def test_a_schema_failure_is_not_retried_again(self) -> None:
        """ADR-148 already re-rolled it in place; retrying here would make that two."""
        meta, attempts, _ = self._run(
            (RecoverableSummarizationError.SCHEMA_INVALID_AFTER_REROLL, "ok")
        )
        assert attempts == 1
        assert meta is None

    def test_an_unclassified_cause_is_not_retried(self) -> None:
        """ "Unknown" is not "transient" — it degrades visibly and someone classifies it."""
        meta, attempts, _ = self._run((RecoverableSummarizationError.UNSPECIFIED, "ok"))
        assert attempts == 1
        assert meta is None

    def test_the_retry_is_bounded_to_one(self) -> None:
        """A transient cause that recurs must not loop — the episode degrades after two."""
        meta, attempts, _ = self._run(
            (
                RecoverableSummarizationError.TOKENIZER_THREADING,
                RecoverableSummarizationError.TOKENIZER_THREADING,
            )
        )
        assert attempts == 2, "exactly one retry, not a loop"
        assert meta is None

    def test_a_failed_retry_still_marks_the_artifact(self) -> None:
        """The mark is what the whole issue is about; a spent retry must not swallow it."""
        metrics = Metrics()
        meta, _attempts, _ = self._run(
            (
                RecoverableSummarizationError.TOKENIZER_THREADING,
                RecoverableSummarizationError.TOKENIZER_THREADING,
            ),
            metrics=metrics,
        )
        assert meta is None
        record = metrics.stage_outcomes_by_episode[EPISODE_IDX]["summarization"]
        assert record["outcome"] == "failed"
        assert record["reason"] == RecoverableSummarizationError.TOKENIZER_THREADING

    def test_a_recovered_episode_is_not_marked_failed(self) -> None:
        """A summary that came back on retry is a SUCCESS. Marking it failed would put a
        healthy episode on the repair work-list forever."""
        metrics = Metrics()
        meta, attempts, good = self._run(
            (RecoverableSummarizationError.TOKENIZER_THREADING, "ok"), metrics=metrics
        )
        assert (attempts, meta) == (2, good)
        record = metrics.stage_outcomes_by_episode.get(EPISODE_IDX, {}).get("summarization", {})
        assert record.get("outcome") != "failed"
