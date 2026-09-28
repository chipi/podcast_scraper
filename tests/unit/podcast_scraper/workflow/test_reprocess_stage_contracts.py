"""Contracts every reprocess stage must satisfy — the guards that were missing on 2026-09-28.

WHY THIS FILE EXISTS. A feed-scoped ``rederive_only`` over 50 prod episodes re-derived 2 and
skipped 48, exiting 0. Fifty-seven unit files touch these stages and every one of them passed. The
defect lived in the COMPOSITION — which download route an episode takes — and nothing asserted
across that seam.

Three layers of legacy produced it, each individually defensible:

1. TWO DOWNLOAD ROUTES, REUSE LOGIC ON ONE. ``process_episode_download`` gained the
   "reuse the on-disk transcript" branch when rederive_only's original no-op was fixed. But it
   calls ``process_transcript_download`` FIRST for any episode whose publisher serves a transcript
   and returns that result, so the branch was unreachable for a direct-download feed. The 2
   episodes that worked were the only ones with no transcript URL.

2. RUN-LOCAL vs CORPUS-WIDE (D7), FIXED ONCE. ``_check_existing_transcript`` resolves presence
   corpus-wide; its sibling branch kept a run-local glob. Under
   ``--single-feed-uses-corpus-layout`` every run gets a FRESH run dir while the transcript lives
   in a prior one, so the glob always missed.

3. ONE CONSTANT, TWO PREDICATES. ``STAGES_THAT_NEVER_TRANSCRIBE`` is NAMED for "never calls an ASR
   provider" but DEFINED as "reaches the transcription stage via transcribe_missing=true and gets
   intercepted". ``rederive_only`` satisfies the name and not the definition, so three call sites
   bolt on ``or stage == "rederive_only"``. When the same workaround appears three times the
   constant is wrong, not the callers.

These tests are deliberately about the CONTRACT rather than any one function, so a fourth route or
a fifth stage cannot be added without satisfying them.

No network, no LLM, no audio.
"""

from __future__ import annotations

import pytest

from podcast_scraper import config
from podcast_scraper.server.jobs import PIPELINE_STAGES_REPROCESS

pytestmark = [pytest.mark.unit]


#: Every stage that works FROM an artifact already on disk and must never call an ASR provider.
#: Derived from the reprocess set rather than retyped, so the two cannot drift apart.
REPROCESS_STAGES = sorted(PIPELINE_STAGES_REPROCESS)


class TestTheReprocessStageSetIsWhatWeThinkItIs:
    """Pin the membership itself. Every assertion below is only as good as this list."""

    def test_the_four_known_reprocess_stages_are_present(self):
        assert set(REPROCESS_STAGES) == {
            "rederive_only",
            "relabel_only",
            "rediarize_only",
            "retranscript_only",
        }, (
            "A stage was added or removed. Every contract in this file is parametrised over this "
            "set — extend the contracts, do not just update this assertion."
        )

    def test_a_reprocess_stage_is_never_a_partial_stage(self):
        """``audio_only`` / ``download_only`` ingest; they are not reprocess modes."""
        assert not PIPELINE_STAGES_REPROCESS & {"audio_only", "download_only", "full"}

    def test_the_reuse_predicate_and_the_api_reprocess_set_cannot_drift(self):
        """Two constants, two modules, one truth. A stage in one and not the other loses behaviour.

        ``config.STAGES_REUSING_ON_DISK_ARTIFACTS`` drives the pipeline's "do not fetch, the on-disk
        artifact is the input" branches; ``server.jobs.PIPELINE_STAGES_REPROCESS`` drives the API's
        "pair this with --reprocess-existing-only" rule. They must name the same four stages.
        """
        assert config.STAGES_REUSING_ON_DISK_ARTIFACTS == PIPELINE_STAGES_REPROCESS

    def test_the_narrow_asr_set_is_a_strict_subset_of_the_reuse_set(self):
        """The mechanism set is contained in the semantic set, and is genuinely smaller.

        If these ever become equal, ``rederive_only`` has been added to
        ``STAGES_THAT_NEVER_TRANSCRIBE`` — which contradicts that constant's documented definition
        (it is about stages using the transcribe_missing=true interception trick) and would make the
        distinction this file exists to protect invisible again.
        """
        assert config.STAGES_THAT_NEVER_TRANSCRIBE < config.STAGES_REUSING_ON_DISK_ARTIFACTS
        assert config.STAGES_REUSING_ON_DISK_ARTIFACTS - config.STAGES_THAT_NEVER_TRANSCRIBE == {
            "rederive_only"
        }

    def test_no_call_site_still_carries_the_repeated_workaround(self):
        """The three-times-repeated ``or stage == "rederive_only"`` must not come back.

        A fourth repetition is how this became invisible in the first place: each site looked like a
        small local exception rather than a missing concept.
        """
        import inspect

        from podcast_scraper.workflow import episode_processor as ep, metadata_generation as mg

        for mod in (ep, mg):
            src = inspect.getsource(mod)
            or_form = (
                'STAGES_THAT_NEVER_TRANSCRIBE\n        or cfg.pipeline_stage == "rederive_only"'
            )
            assert or_form not in src
            assert 'STAGES_THAT_NEVER_TRANSCRIBE and stage != "rederive_only"' not in src, (
                f"{mod.__name__} reintroduced the workaround — use "
                "config.STAGES_REUSING_ON_DISK_ARTIFACTS instead"
            )


class TestEveryReprocessStageNeverCallsASR:
    """The predicate the constant is NAMED for. It must hold for ALL reprocess stages.

    ``rederive_only`` is the one that breaks if this is conflated with the transcribe_missing
    mechanism: it reaches its reuse branch with transcribe_missing=FALSE, while its three siblings
    set it TRUE purely to reach the interception point. Both are "never calls ASR"; only one shape
    is captured by ``STAGES_THAT_NEVER_TRANSCRIBE``'s docstring.
    """

    @pytest.mark.parametrize("stage", REPROCESS_STAGES)
    def test_no_asr_credential_is_demanded(self, stage, monkeypatch):
        """A stage that calls no ASR must never be refused for lacking an ASR key."""
        monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
        cfg = config.Config.model_validate(
            {
                "rss_url": "https://example.com/feed.xml",
                "transcription_provider": "deepgram",
                "deepgram_api_key": None,
                "pipeline_stage": stage,
            }
        )
        assert cfg.pipeline_stage == stage

    @pytest.mark.parametrize("stage", REPROCESS_STAGES)
    def test_the_stage_is_recognised_as_never_transcribing_by_SOME_declared_route(self, stage):
        """THE NAME-vs-DEFINITION MISMATCH, pinned.

        A stage qualifies either by being in the constant, or by coercing
        ``transcribe_missing=False`` (rederive_only's route). What must NEVER happen is a reprocess
        stage that satisfies NEITHER — that stage would reach a real ASR provider.
        """
        cfg = config.Config.model_validate(
            {"rss_url": "https://example.com/feed.xml", "pipeline_stage": stage}
        )
        in_constant = stage in config.STAGES_THAT_NEVER_TRANSCRIBE
        coerces_off = cfg.transcribe_missing is False
        assert in_constant or coerces_off, (
            f"{stage} is a reprocess stage that neither appears in "
            "STAGES_THAT_NEVER_TRANSCRIBE nor coerces transcribe_missing=False, so nothing stops "
            "it reaching an ASR provider."
        )

    def test_the_two_routes_are_genuinely_different_and_both_are_used(self):
        """Documents WHY one constant cannot express this, so nobody 'simplifies' it back.

        If this ever collapses to one route, the constant can absorb the other and the
        ``or stage == 'rederive_only'`` patches can go.
        """
        by_constant, by_coercion = set(), set()
        for stage in REPROCESS_STAGES:
            cfg = config.Config.model_validate(
                {"rss_url": "https://example.com/feed.xml", "pipeline_stage": stage}
            )
            if stage in config.STAGES_THAT_NEVER_TRANSCRIBE:
                by_constant.add(stage)
            if cfg.transcribe_missing is False:
                by_coercion.add(stage)
        assert by_coercion == {"rederive_only"}, (
            "rederive_only is the only stage that reaches its reuse branch with "
            f"transcribe_missing=False; got {sorted(by_coercion)}"
        )
        assert by_constant == {"relabel_only", "rediarize_only", "retranscript_only"}
        assert by_constant | by_coercion == set(REPROCESS_STAGES), "every stage must be covered"


class TestEveryReprocessStageIsScopedToOnDiskEpisodes:
    """A reprocess must never build its work list from the live feed."""

    @pytest.mark.parametrize("stage", REPROCESS_STAGES)
    def test_the_api_pairs_the_stage_with_reprocess_existing_only(self, stage, tmp_path):
        """Without the pairing the run selects from the feed and silently does nothing."""
        from podcast_scraper.server.jobs import build_pipeline_argv

        corpus = tmp_path / "corpus"
        corpus.mkdir()
        op = corpus / "viewer_operator.yaml"
        op.write_text("profile: local\n", encoding="utf-8")
        argv = build_pipeline_argv(corpus, op, run_id="j", pipeline_stage=stage)
        assert "--reprocess-existing-only" in argv, f"{stage} must be scoped to on-disk episodes"

    @pytest.mark.parametrize("stage", REPROCESS_STAGES)
    def test_the_stage_survives_normalisation(self, stage):
        from podcast_scraper.server.jobs import normalize_pipeline_stage

        assert normalize_pipeline_stage(stage) == stage


class TestSkipExistingMustNotSilenceAReprocess:
    """THE CROSS-CUTTING FAILURE MODE — all three layers expressed themselves through it.

    ``skip_existing`` is an INGEST guard: "I already have this episode, do not fetch it again."
    A reprocess stage inverts the meaning — already having the episode is the PRECONDITION, not a
    reason to skip. ``rederive_only`` coerces ``skip_existing=true`` (it must, or transcript reuse
    is disabled) and that same flag is what dropped 48 of 50 episodes.

    So: a reprocess stage that coerces skip_existing on MUST also provide a documented way past
    the per-episode skip, or it is a no-op by construction.
    """

    @pytest.mark.parametrize("stage", REPROCESS_STAGES)
    def test_a_stage_that_forces_skip_existing_has_a_documented_way_past_it(self, stage):
        cfg = config.Config.model_validate(
            {"rss_url": "https://example.com/feed.xml", "pipeline_stage": stage}
        )
        if not cfg.skip_existing:
            pytest.skip(f"{stage} does not coerce skip_existing")
        # rederive_only is the case: its way past the skip is the reuse branch in BOTH download
        # routes. Assert both exist, since having only one is the exact prod defect.
        import inspect

        from podcast_scraper.workflow import episode_processor as ep

        transcript_route = inspect.getsource(ep.process_transcript_download)
        audio_route = inspect.getsource(ep.process_episode_download)
        assert "rederive_only" in transcript_route, (
            "process_transcript_download has no rederive_only reuse branch — every episode whose "
            "publisher serves a transcript will be skipped (prod 2026-09-28: 48 of 50)."
        )
        assert "rederive_only" in audio_route, (
            "process_episode_download has no rederive_only reuse branch — episodes with no "
            "transcript URL will be skipped."
        )

    def test_both_download_routes_resolve_the_transcript_corpus_wide(self):
        """D7: a run-local lookup cannot see a prior run's transcript under corpus layout.

        STRUCTURAL TRIPWIRE, not the real proof. The behavioural tests live in
        ``tests/unit/workflow/test_rederive_only_reuses_transcripts.py``, which drives both routes
        against a real corpus whose transcript sits in a PRIOR run dir. This one exists because
        that behavioural coverage arrived only AFTER a route was found missing it — so it is cheap
        insurance that a THIRD route cannot be added with a fresh run-local glob and no test.
        """
        import inspect

        from podcast_scraper.workflow import episode_processor as ep

        for fn in (ep.process_transcript_download, ep.process_episode_download):
            src = inspect.getsource(fn)
            assert "_resolve_existing_transcript_for_rederive" in src, (
                f"{fn.__name__} must resolve the transcript with the corpus-wide resolver, not a "
                "run-local glob — under --single-feed-uses-corpus-layout the transcript lives in "
                "a PRIOR run dir, so a run-local lookup always misses (prod 2026-09-28)"
            )
