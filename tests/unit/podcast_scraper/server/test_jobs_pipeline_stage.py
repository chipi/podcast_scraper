"""``pipeline_stage`` on the Jobs API: validation, argv, and the pairing it must not omit.

Before this, reprocessing was CLI-only — ``POST /api/jobs`` accepted profile, max_episodes,
episode_offset and episode_order but had no way to say "re-derive the LLM stages". The operator
asked for it after the ``rederive_only`` rename.

Two properties carry the weight here:

* **The value reaches a subprocess argv**, so it is a trust boundary. It is checked against an
  explicit allowlist, never passed through. A caller string that got through could at best
  crash the child on argparse choices and at worst select a mode the API never meant to expose.
* **A reprocess stage MUST be paired with ``--reprocess-existing-only``.** Every reprocess
  stage coerces ``skip_existing=true``, so without existing-only scoping the work list is built
  from the LIVE feed and each already-ingested episode is then skipped — the run does nothing
  and exits 0. That silent no-op was measured end-to-end on 2026-09-01 ("Episodes to process:
  0 of 0"). An API that omitted the pairing would faithfully reproduce the bug this whole
  change set exists to remove.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper.server.jobs import (
    build_pipeline_argv,
    normalize_pipeline_stage,
    PIPELINE_STAGES_ALLOWED,
    PIPELINE_STAGES_REPROCESS,
)


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    (tmp_path / "viewer_operator.yaml").write_text("profile: cloud_balanced\n", encoding="utf-8")
    (tmp_path / "feeds.spec.yaml").write_text(
        "feeds:\n  - rss: https://example.com/f.rss\n", encoding="utf-8"
    )
    return tmp_path


def _argv(corpus: Path, **kw):
    return build_pipeline_argv(corpus, corpus / "viewer_operator.yaml", **kw)


class TestNormalisation:
    @pytest.mark.parametrize("stage", sorted(PIPELINE_STAGES_ALLOWED))
    def test_every_allowed_stage_survives(self, stage):
        assert normalize_pipeline_stage(stage) == stage

    def test_deprecated_alias_maps_to_the_canonical_name(self):
        """An API caller pinned to the old spelling must behave like a CLI caller."""
        assert normalize_pipeline_stage("enrich_only") == "rederive_only"

    @pytest.mark.parametrize("stage", [None, "", "   ", "full"])
    def test_absent_or_full_means_omit_the_flag(self, stage):
        """'full' is the default; expressing it as a flag is unnecessary but must not error.

        A UI that always sends the field should not have to special-case the normal run.
        """
        assert normalize_pipeline_stage(stage) is None

    @pytest.mark.parametrize(
        "stage",
        ["bogus", "REDERIVE_ONLY", "rederive", "--reprocess-existing-only", "-rf", "a;b"],
    )
    def test_unknown_and_injection_shaped_values_are_dropped(self, stage):
        assert normalize_pipeline_stage(stage) is None

    def test_the_allowlist_matches_what_the_cli_accepts(self):
        """A stage the API offers but argparse rejects would 400 at runtime, not at submit."""
        import argparse

        from podcast_scraper import cli

        parser = argparse.ArgumentParser()
        cli._add_pipeline_stage_arguments(parser)
        action = next(a for a in parser._actions if a.dest == "pipeline_stage")
        assert PIPELINE_STAGES_ALLOWED <= set(action.choices)


class TestArgv:
    def test_no_stage_means_no_flag(self, corpus):
        assert "--pipeline-stage" not in _argv(corpus)

    @pytest.mark.parametrize("stage", sorted(PIPELINE_STAGES_ALLOWED))
    def test_stage_is_emitted_as_a_flag_pair(self, corpus, stage):
        argv = _argv(corpus, pipeline_stage=stage)
        i = argv.index("--pipeline-stage")
        assert argv[i + 1] == stage

    @pytest.mark.parametrize("stage", sorted(PIPELINE_STAGES_REPROCESS))
    def test_reprocess_stages_get_existing_only_scoping(self, corpus, stage):
        """THE LOAD-BEARING ONE. Without this the reprocess selects nothing and exits 0."""
        argv = _argv(corpus, pipeline_stage=stage)
        assert "--reprocess-existing-only" in argv, (
            f"{stage} without --reprocess-existing-only builds its work list from the live "
            "feed, then skips every on-disk episode: a silent no-op"
        )

    @pytest.mark.parametrize("stage", ["audio_only", "download_only"])
    def test_partial_stages_do_NOT_get_existing_only(self, corpus, stage):
        """These ingest new episodes; scoping them to on-disk ones would defeat the point."""
        argv = _argv(corpus, pipeline_stage=stage)
        assert "--reprocess-existing-only" not in argv

    def test_the_alias_produces_the_canonical_flag_and_the_pairing(self, corpus):
        argv = _argv(corpus, pipeline_stage="enrich_only")
        i = argv.index("--pipeline-stage")
        assert argv[i + 1] == "rederive_only"
        assert "--reprocess-existing-only" in argv

    def test_unknown_stage_never_reaches_argv(self, corpus):
        argv = _argv(corpus, pipeline_stage="bogus")
        assert "--pipeline-stage" not in argv
        assert "bogus" not in argv

    def test_stage_composes_with_a_scoped_feed_run(self, corpus):
        argv = _argv(corpus, feed_url="https://example.com/f.rss", pipeline_stage="rederive_only")
        assert "--single-feed-uses-corpus-layout" in argv
        assert argv[argv.index("--pipeline-stage") + 1] == "rederive_only"
        assert "--reprocess-existing-only" in argv

    def test_existing_only_is_not_duplicated(self, corpus):
        argv = _argv(corpus, pipeline_stage="rederive_only")
        assert argv.count("--reprocess-existing-only") == 1
