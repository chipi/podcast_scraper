"""advisor #5: a swallowed RecoverableSummarizationError in the summary stage must reach Sentry
(capture_stage_exception(stage="summary")) while metadata generation continues without a summary.

#1632 added the second half of that contract — report it as a WARNING — reasoning that the branch
is the designed recovery (#1496) and that Sentry's default `error` made a healthy degradation
indistinguishable from a broken run.

#1686 REVERSED the severity here, and the reasoning is worth keeping straight because the two
issues are not actually in conflict. #1632 was right that a recovery IN PROGRESS is not a crash.
This line is not that: it is reached only once any retry is spent, at the moment the episode is
about to be persisted with no summary at all and nothing will ever revisit it. Marko, 2026-08-20:
"it's never acceptable to have an episode without the summary of a single one" — and production
had 8 such episodes, invisible precisely because the only signal was a warning nobody triages.

So severity now tracks RECOVERABILITY, not intent:
  - retry in flight   -> logged + recorded in the stage ledger, NO Sentry event
  - summary recovered -> nothing to report
  - summary LOST      -> one error

That is fewer events than #1632 removed, not more, while restoring the loudness for the only case
that warrants it. The episode is still kept — #1496 is untouched.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from podcast_scraper.exceptions import RecoverableSummarizationError
from podcast_scraper.workflow import metadata_generation as mg


@pytest.mark.unit
def test_summary_recoverable_error_is_captured_and_swallowed():
    episode = SimpleNamespace(idx=1)
    cfg = SimpleNamespace(generate_summaries=True, dry_run=False)
    captured: list[tuple[str, str | None]] = []

    def _raise(**_):
        raise RecoverableSummarizationError(1, "model OOM")

    with (
        patch.object(mg, "_generate_episode_summary", _raise),
        patch.object(
            mg,
            "_capture_stage_exception",
            lambda exc, *, stage, level=None: captured.append((stage, level)),
        ),
    ):
        meta, _elapsed, _cm = mg._generate_and_validate_summary(
            episode,
            "https://example.com/feed.xml",
            "transcript.txt",
            "/out",
            cfg,
            summary_provider=None,
            whisper_model=None,
        )

    assert meta is None  # continues without a summary (#1496 — the episode is still kept)
    # Reached Sentry tagged stage=summary, at ERROR: the summary is not degraded, it is gone.
    assert captured == [("summary", "error")]


@pytest.mark.unit
def test_a_summary_recovered_on_retry_reports_nothing():
    """The counterpart that makes the error meaningful: no event when nothing was lost.

    If a recovered episode still raised, the error would be noise again within a week and would
    be muted again — which is how #1632 happened in the first place.
    """
    episode = SimpleNamespace(idx=1, title="An Episode", item=None)
    cfg = SimpleNamespace(generate_summaries=True, dry_run=False)
    captured: list[tuple[str, str | None]] = []
    calls = {"n": 0}
    good = SimpleNamespace(title="Recovered", bullets=["a real bullet"])

    def _attempt(**_):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RecoverableSummarizationError(
                1, "already borrowed", code=RecoverableSummarizationError.TOKENIZER_THREADING
            )
        return (good, None)

    with (
        patch.object(mg, "_generate_episode_summary", _attempt),
        patch.object(
            mg,
            "_capture_stage_exception",
            lambda exc, *, stage, level=None: captured.append((stage, level)),
        ),
    ):
        meta, _elapsed, _cm = mg._generate_and_validate_summary(
            episode,
            "https://example.com/feed.xml",
            "transcript.txt",
            "/out",
            cfg,
            summary_provider=None,
            whisper_model=None,
        )

    assert (calls["n"], meta) == (2, good)
    assert captured == [], "a recovered summary is not an incident"
