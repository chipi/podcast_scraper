"""New submissions must respect the operator pause flag (#1785 stop endpoint).

2026-08-28 incident: the pause flag only held promotion of already-QUEUED jobs, so the
03:00 scheduler cron — submitting into an idle queue — started immediately and ran a full
sweep the operator had explicitly braked (on a not-yet-deployed skip fix; another window
re-ingested). A free slot is not consent to run: while paused, every new submission lands
queued and waits for resume.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# No `importorskip("fastapi")`: fastapi is a [dev] dependency (pyproject: the server extras are
# "mirrored" into dev, :211), so it is present in the unit tier by construction. The guard was
# therefore dead — it could never skip — and `check-test-policy` rejects it because a unit test
# that LOOKS conditional on an extra hides whether the tier really covers this code.
from podcast_scraper.server.jobs import enqueue_enrichment_job, enqueue_pipeline_job
from podcast_scraper.server.queue_sweeper import pause_drain, resume_drain

pytestmark = [pytest.mark.unit]


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    (tmp_path / "viewer_operator.yaml").write_text("max_episodes: 1\n", encoding="utf-8")
    return tmp_path


def test_pipeline_submission_lands_queued_while_paused(corpus: Path) -> None:
    pause_drain(corpus)
    rec = enqueue_pipeline_job(corpus, corpus / "viewer_operator.yaml")
    assert rec["status"] == "queued", (
        "a submission during pause was started immediately — the scheduler-cron bypass "
        "(2026-08-28 03:00 fire) is back"
    )


def test_pipeline_submission_runs_when_not_paused(corpus: Path) -> None:
    pause_drain(corpus)
    resume_drain(corpus)
    rec = enqueue_pipeline_job(corpus, corpus / "viewer_operator.yaml")
    assert rec["status"] == "running"


def test_enrichment_submission_lands_queued_while_paused(corpus: Path) -> None:
    pause_drain(corpus)
    rec = enqueue_enrichment_job(corpus, operator_yaml=corpus / "viewer_operator.yaml")
    assert rec["status"] == "queued"


class TestSingleFeedHonoursPerFeedProfile:
    """`Run just this feed` must use the profile that feed pins (2026-08-28).

    The batch path resolves a per-feed ``profile:`` via merge_feed_entry_into_config. The
    single-feed path built argv from the operator YAML alone, so a feed pinned to the DGX
    profile would run on the corpus-wide one — the UI showing DGX while the run billed
    Deepgram, silently. These tests forbid that.
    """

    def _corpus(self, tmp_path: Path, spec: str) -> Path:
        (tmp_path / "viewer_operator.yaml").write_text(
            "profile: cloud_balanced\nmax_episodes: 1\n", encoding="utf-8"
        )
        (tmp_path / "feeds.spec.yaml").write_text(spec, encoding="utf-8")
        return tmp_path

    def test_feed_pinned_profile_wins_over_operator_yaml(self, tmp_path: Path) -> None:
        from podcast_scraper.server.jobs import build_pipeline_argv

        corpus = self._corpus(
            tmp_path,
            "feeds:\n"
            "  - https://old.example/f.xml\n"
            "  - url: https://new.example/f.xml\n"
            "    profile: cloud_with_dgx_primary\n",
        )
        argv = build_pipeline_argv(
            corpus, corpus / "viewer_operator.yaml", feed_url="https://new.example/f.xml"
        )
        assert "--profile" in argv
        assert argv[argv.index("--profile") + 1] == "cloud_with_dgx_primary", argv

    def test_unpinned_feed_keeps_the_corpus_profile(self, tmp_path: Path) -> None:
        from podcast_scraper.server.jobs import build_pipeline_argv

        corpus = self._corpus(
            tmp_path,
            "feeds:\n"
            "  - https://old.example/f.xml\n"
            "  - url: https://new.example/f.xml\n"
            "    profile: cloud_with_dgx_primary\n",
        )
        argv = build_pipeline_argv(
            corpus, corpus / "viewer_operator.yaml", feed_url="https://old.example/f.xml"
        )
        assert argv[argv.index("--profile") + 1] == "cloud_balanced", argv

    def test_missing_spec_does_not_break_submission(self, tmp_path: Path) -> None:
        from podcast_scraper.server.jobs import build_pipeline_argv

        (tmp_path / "viewer_operator.yaml").write_text(
            "profile: cloud_balanced\n", encoding="utf-8"
        )
        argv = build_pipeline_argv(
            tmp_path, tmp_path / "viewer_operator.yaml", feed_url="https://x.example/f.xml"
        )
        assert argv[argv.index("--profile") + 1] == "cloud_balanced"
