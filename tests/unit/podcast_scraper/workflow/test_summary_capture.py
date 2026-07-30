"""advisor #5: a swallowed RecoverableSummarizationError in the summary stage must reach Sentry
(capture_stage_exception(stage="summary")) while metadata generation continues without a summary."""

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
    captured = []

    def _raise(**_):
        raise RecoverableSummarizationError(1, "model OOM")

    with (
        patch.object(mg, "_generate_episode_summary", _raise),
        patch.object(mg, "_capture_stage_exception", lambda exc, *, stage: captured.append(stage)),
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

    assert meta is None  # continues without a summary (recoverable)
    assert captured == ["summary"]  # the swallowed error reached Sentry, tagged stage=summary
