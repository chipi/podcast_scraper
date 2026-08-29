# mypy: disable-error-code="call-arg"
# Deliberate: Config(rss_url=...) — alias="rss"; populate-by-name accepts either at runtime.
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


class TestRequestLevelProfileOverride:
    """#1872 — corpus YAML < feed entry < THIS request. Highest wins, nothing persisted."""

    def _corpus(self, tmp_path: Path) -> Path:
        (tmp_path / "viewer_operator.yaml").write_text(
            "profile: cloud_balanced\nmax_episodes: 1\n", encoding="utf-8"
        )
        (tmp_path / "feeds.spec.yaml").write_text(
            "feeds:\n"
            "  - https://plain.example/f.xml\n"
            "  - url: https://pinned.example/f.xml\n"
            "    profile: cloud_with_dgx_primary\n",
            encoding="utf-8",
        )
        return tmp_path

    def _profile_in(self, argv: list[str]) -> str:
        return argv[argv.index("--profile") + 1]

    def test_request_profile_beats_the_corpus_yaml(self, tmp_path: Path) -> None:
        from podcast_scraper.server.jobs import build_pipeline_argv

        corpus = self._corpus(tmp_path)
        argv = build_pipeline_argv(
            corpus, corpus / "viewer_operator.yaml", profile_override="cloud_thin"
        )
        assert self._profile_in(argv) == "cloud_thin", argv

    def test_request_profile_beats_a_feed_pin(self, tmp_path: Path) -> None:
        """The whole point of the top layer: reprocess a pinned feed elsewhere, once."""
        from podcast_scraper.server.jobs import build_pipeline_argv

        corpus = self._corpus(tmp_path)
        argv = build_pipeline_argv(
            corpus,
            corpus / "viewer_operator.yaml",
            feed_url="https://pinned.example/f.xml",
            profile_override="cloud_thin",
        )
        assert self._profile_in(argv) == "cloud_thin", argv

    def test_without_a_request_profile_the_feed_pin_still_wins(self, tmp_path: Path) -> None:
        from podcast_scraper.server.jobs import build_pipeline_argv

        corpus = self._corpus(tmp_path)
        argv = build_pipeline_argv(
            corpus, corpus / "viewer_operator.yaml", feed_url="https://pinned.example/f.xml"
        )
        assert self._profile_in(argv) == "cloud_with_dgx_primary", argv

    def test_without_either_the_corpus_yaml_wins(self, tmp_path: Path) -> None:
        from podcast_scraper.server.jobs import build_pipeline_argv

        corpus = self._corpus(tmp_path)
        argv = build_pipeline_argv(
            corpus, corpus / "viewer_operator.yaml", feed_url="https://plain.example/f.xml"
        )
        assert self._profile_in(argv) == "cloud_balanced", argv

    def test_blank_override_is_a_no_op(self, tmp_path: Path) -> None:
        from podcast_scraper.server.jobs import build_pipeline_argv

        corpus = self._corpus(tmp_path)
        argv = build_pipeline_argv(corpus, corpus / "viewer_operator.yaml", profile_override="   ")
        assert self._profile_in(argv) == "cloud_balanced", argv


class TestRequestProfileValidation:
    """An unknown name must be REJECTED, not passed to argv (#1872).

    Config._resolve_profile only warns when a name matches nothing and then runs on
    defaults — so an unvalidated override would produce a run that looks configured and
    is not. That is the exact silent-mismatch class this guard exists to prevent.
    """

    def test_unknown_profile_name_raises(self) -> None:
        from podcast_scraper.server.profile_presets import validate_profile_name_allowed

        with pytest.raises(ValueError, match="not in the available profiles"):
            validate_profile_name_allowed("no-such-profile-xyz")

    def test_known_profile_name_passes_through(self) -> None:
        from podcast_scraper.server.profile_presets import (
            list_packaged_profile_names,
            validate_profile_name_allowed,
        )

        known = list_packaged_profile_names()
        assert known, "environment has no packaged profiles; test cannot be meaningful"
        assert validate_profile_name_allowed(known[0]) == known[0]

    def test_empty_is_the_no_op_default(self) -> None:
        from podcast_scraper.server.profile_presets import validate_profile_name_allowed

        assert validate_profile_name_allowed(None) is None
        assert validate_profile_name_allowed("  ") is None


class TestBatchOverrideBeatsFeedPins:
    """#1872 F1: in a BATCH run the request override must outrank per-feed pins.

    The batch loop cannot tell an operator-chosen override from the corpus default — both
    arrive as ``--profile`` — so a pinned feed silently ignored the override the API
    documents as winning. The flag is that missing signal.
    """

    def test_batch_override_sets_the_flag(self, tmp_path: Path) -> None:
        from podcast_scraper.server.jobs import build_pipeline_argv

        (tmp_path / "viewer_operator.yaml").write_text(
            "profile: cloud_balanced\n", encoding="utf-8"
        )
        argv = build_pipeline_argv(
            tmp_path, tmp_path / "viewer_operator.yaml", profile_override="cloud_thin"
        )
        assert "--profile-overrides-feed-pins" in argv, argv

    def test_single_feed_run_does_not_set_it(self, tmp_path: Path) -> None:
        """A single-feed run resolves the pin directly; the flag would be meaningless noise."""
        from podcast_scraper.server.jobs import build_pipeline_argv

        (tmp_path / "viewer_operator.yaml").write_text(
            "profile: cloud_balanced\n", encoding="utf-8"
        )
        argv = build_pipeline_argv(
            tmp_path,
            tmp_path / "viewer_operator.yaml",
            feed_url="https://x.example/f.xml",
            profile_override="cloud_thin",
        )
        assert "--profile-overrides-feed-pins" not in argv

    def test_no_override_no_flag(self, tmp_path: Path) -> None:
        from podcast_scraper.server.jobs import build_pipeline_argv

        (tmp_path / "viewer_operator.yaml").write_text(
            "profile: cloud_balanced\n", encoding="utf-8"
        )
        assert "--profile-overrides-feed-pins" not in build_pipeline_argv(
            tmp_path, tmp_path / "viewer_operator.yaml"
        )


def test_flag_makes_the_merge_ignore_a_feed_pin(tmp_path: Path, monkeypatch) -> None:
    """End of the chain: with the flag set, a pinned feed runs on the run's profile."""
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dummy-for-validation")
    from podcast_scraper import config as config_mod
    from podcast_scraper.rss.feeds_spec import merge_feed_entry_into_config, RssFeedEntry

    entry = RssFeedEntry(url="https://pinned.example/f", profile="cloud_with_dgx_primary")

    without = merge_feed_entry_into_config(
        config_mod.Config(rss_url="https://p.example/f", profile="cloud_balanced"), entry
    )
    assert without.transcription_provider == "tailnet_dgx_whisper"

    with_flag = merge_feed_entry_into_config(
        config_mod.Config(
            rss_url="https://p.example/f",
            profile="cloud_balanced",
            profile_overrides_feed_pins=True,
        ),
        entry,
    )
    assert with_flag.transcription_provider == "deepgram", "the pin ignored the run override"
