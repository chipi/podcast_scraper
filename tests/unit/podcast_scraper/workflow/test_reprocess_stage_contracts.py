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


#: Every stage that works FROM an artifact already on disk. Derived from the reprocess set rather
#: than retyped, so the two cannot drift apart.
REPROCESS_STAGES = sorted(PIPELINE_STAGES_REPROCESS)

#: ...and the ones among them that never call an ASR provider. ``repunctuate_only`` is the odd one
#: out: it calls the provider's CLIP call for the windows it repairs, so it needs a real provider
#: (and is checked on its own below) where these need none.
NEVER_CALLS_ASR_STAGES = sorted(set(REPROCESS_STAGES) - config.STAGES_CALLING_ASR_CLIPS_ONLY)

_DGX = {
    "transcription_provider": "tailnet_dgx_whisper",
    "transcription_fallback_provider": "openai",
    "openai_api_key": "sk-test-not-a-key",
    "dgx_tailnet_host": "dgx-llm-1.tail-test.ts.net",
}


def _stage_cfg(stage: str, **extra) -> config.Config:
    """A valid Config for ``stage``: the clip-calling stage needs the DGX provider, others none."""
    base = {"rss_url": "https://example.com/feed.xml", "pipeline_stage": stage}
    if stage in config.STAGES_CALLING_ASR_CLIPS_ONLY:
        base.update(_DGX)
    return config.Config.model_validate({**base, **extra})


class TestTheReprocessStageSetIsWhatWeThinkItIs:
    """Pin the membership itself. Every assertion below is only as good as this list."""

    def test_the_six_known_reprocess_stages_are_present(self):
        """`translate_only` joined on 2026-09-30 with the multilingual arc.

        It reuses the on-disk transcript and never transcribes, so it belongs in both constants,
        and every contract below is parametrised over this set — adding it here extends them
        rather than exempting it, which is what the message demands.

        `repunctuate_only` (#2187) is the sixth, and the first that calls a provider: for clips.
        """
        assert set(REPROCESS_STAGES) == {
            "rederive_only",
            "relabel_only",
            "rediarize_only",
            "retranscript_only",
            "translate_only",
            "repunctuate_only",
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
        distinction this file exists to protect invisible again. The same goes for
        ``repunctuate_only``, which calls the provider.
        """
        assert config.STAGES_THAT_NEVER_TRANSCRIBE < config.STAGES_REUSING_ON_DISK_ARTIFACTS
        assert config.STAGES_REUSING_ON_DISK_ARTIFACTS - config.STAGES_THAT_NEVER_TRANSCRIBE == {
            "rederive_only",
            "repunctuate_only",
        }
        assert not config.STAGES_CALLING_ASR_CLIPS_ONLY & config.STAGES_THAT_NEVER_TRANSCRIBE

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

    @pytest.mark.parametrize("stage", NEVER_CALLS_ASR_STAGES)
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

    @pytest.mark.parametrize("stage", NEVER_CALLS_ASR_STAGES)
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
        for stage in NEVER_CALLS_ASR_STAGES:
            cfg = _stage_cfg(stage)
            if stage in config.STAGES_THAT_NEVER_TRANSCRIBE:
                by_constant.add(stage)
            if cfg.transcribe_missing is False:
                by_coercion.add(stage)
        assert by_coercion == {"rederive_only"}, (
            "rederive_only is the only stage that reaches its reuse branch with "
            f"transcribe_missing=False; got {sorted(by_coercion)}"
        )
        assert by_constant == {
            "relabel_only",
            "rediarize_only",
            "retranscript_only",
            "translate_only",
        }
        assert by_constant | by_coercion == set(NEVER_CALLS_ASR_STAGES), "every stage is covered"
        assert set(NEVER_CALLS_ASR_STAGES) | config.STAGES_CALLING_ASR_CLIPS_ONLY == set(
            REPROCESS_STAGES
        ), "the one stage that does call the provider is accounted for on its own"


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
        cfg = _stage_cfg(stage)
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


class TestTheClipCallingStage:
    """``repunctuate_only`` is the one reprocess stage that calls a provider -- for clips.

    That makes it the one stage the "never transcribes" machinery must NOT cover. Each test names
    the failure it prevents: a stage that is built to run and exits 0 having done nothing.
    """

    def test_it_reaches_the_transcription_stage_but_is_not_in_the_swallow_set(self):
        cfg = _stage_cfg("repunctuate_only")
        assert cfg.pipeline_stage in config.STAGES_CALLING_ASR_CLIPS_ONLY
        assert cfg.pipeline_stage not in config.STAGES_THAT_NEVER_TRANSCRIBE
        assert cfg.transcribe_missing is True, "the routing trick: it must reach the dispatch"
        assert cfg.pipeline_stage in config.STAGES_REUSING_ON_DISK_ARTIFACTS

    def test_the_repair_switch_cannot_be_left_off_by_a_profile(self):
        """With it off every window would be skipped and the run would report success."""
        cfg = _stage_cfg("repunctuate_only", transcription_repair_unpunctuated_windows=False)
        assert cfg.transcription_repair_unpunctuated_windows is True

    def test_diarization_is_not_forced_on(self):
        """Speakers are carried from the stored segments; forcing it would demand a diarizer."""
        assert _stage_cfg("repunctuate_only", diarize=False).diarize is False

    def test_a_provider_without_a_clip_call_is_refused_at_validation(self):
        with pytest.raises(ValueError, match="transcribe_clip"):
            config.Config.model_validate(
                {
                    "rss_url": "https://example.com/feed.xml",
                    "pipeline_stage": "repunctuate_only",
                    "transcription_provider": "whisper",
                }
            )

    def test_silence_removal_is_refused_because_it_moves_the_audio_timeline(self):
        with pytest.raises(ValueError, match="preprocessing_silence_removal"):
            _stage_cfg("repunctuate_only", preprocessing_silence_removal=True)
        _stage_cfg(
            "repunctuate_only", preprocessing_silence_removal=True, preprocessing_enabled=False
        )

    def test_a_provider_that_cannot_be_built_stops_the_run(self, monkeypatch):
        """The set that swallows this error is the set this stage is deliberately not in."""
        from podcast_scraper.workflow import orchestration

        def _boom(_cfg, **_kw):
            raise ValueError("the DGX host is not configured")

        monkeypatch.setattr(orchestration, "_get_factory_function", lambda *a, **k: _boom)
        with pytest.raises(ValueError, match="DGX host"):
            orchestration._create_transcription_provider(_stage_cfg("repunctuate_only"))

    def test_the_provider_is_built_unwrapped(self, monkeypatch):
        from podcast_scraper.workflow import orchestration

        seen = {}

        class _Dgx:
            def initialize(self):
                pass

            def transcribe_clip(self, *_a, **_k):
                return {}

        def _factory(_cfg, **kw):
            seen.update(kw)
            return _Dgx()

        monkeypatch.setattr(orchestration, "_get_factory_function", lambda *a, **k: _factory)
        provider = orchestration._create_transcription_provider(_stage_cfg("repunctuate_only"))
        assert seen == {"_wrap_fallback": False}
        assert isinstance(provider, _Dgx)

    def test_a_provider_with_no_clip_call_fails_at_startup_not_per_window(self, monkeypatch):
        """THE SILENT NO-OP. Neither provider wrapper forwards ``transcribe_clip`` and
        ``_clip_call_of`` answers a wrapper with one INFO line, so a wrapped provider would skip
        every window of every episode and the run would exit 0."""
        from podcast_scraper.workflow import orchestration

        class _Wrapper:
            def initialize(self):
                pass

        monkeypatch.setattr(
            orchestration, "_get_factory_function", lambda *a, **k: lambda _c, **_kw: _Wrapper()
        )
        with pytest.raises(RuntimeError, match="transcribe_clip"):
            orchestration._create_transcription_provider(_stage_cfg("repunctuate_only"))

    def test_with_the_real_factory_a_fallback_ladder_does_not_hide_the_clip_call(self):
        """Why the provider is built unwrapped: the very config this stage needs wraps it."""
        from podcast_scraper.providers.tailnet_dgx.whisper_provider import (
            TailnetDgxWhisperTranscriptionProvider,
        )
        from podcast_scraper.workflow import orchestration

        wrapped = orchestration.create_transcription_provider(_stage_cfg("full", **_DGX))
        assert not callable(getattr(wrapped, "transcribe_clip", None)), (
            "if the wrapper forwarded the clip call the unwrapped build would be unnecessary; "
            "re-read orchestration._create_transcription_provider before relaxing it"
        )
        provider = orchestration._create_transcription_provider(_stage_cfg("repunctuate_only"))
        assert isinstance(provider, TailnetDgxWhisperTranscriptionProvider)
        assert callable(provider.transcribe_clip)

    def test_the_dispatch_runs_before_the_transcript_cache(self, monkeypatch, tmp_path):
        """A cache hit re-serves the pre-repair transcript the stage exists to replace."""
        from podcast_scraper.models import TranscriptionJob
        from podcast_scraper.workflow import episode_processor as ep

        def _must_not_run(*_a, **_k):
            raise AssertionError("the transcript cache / reuse was consulted before the dispatch")

        monkeypatch.setattr(ep, "_check_transcript_cache", _must_not_run)
        monkeypatch.setattr(ep, "_check_and_reuse_existing_transcript", _must_not_run)
        monkeypatch.setattr(
            ep, "_repunctuate_existing_transcript", lambda *_a, **_k: (True, "t/x.txt", 0)
        )
        job = TranscriptionJob(idx=1, ep_title="Ep", ep_title_safe="Ep", temp_media="")
        cfg = _stage_cfg("repunctuate_only", feed_declared_language="en")

        got = ep.transcribe_media_to_text(
            job, cfg, None, None, str(tmp_path), transcription_provider=object()
        )

        assert got == (True, "t/x.txt", 0)

    @pytest.mark.parametrize("skip_existing", [True, False])
    def test_it_reaches_its_job_whether_or_not_skip_existing_is_on(
        self, skip_existing, tmp_path, monkeypatch
    ):
        """The listed risk: ``skip_existing`` is an ingest guard, and the episode being PRESENT is
        this stage's precondition. The no-download job is queued either way, with no audio fetch."""
        import queue
        import xml.etree.ElementTree as ET

        from podcast_scraper import models
        from podcast_scraper.workflow import episode_processor as ep

        def _no_fetch(*_a, **_k):
            raise AssertionError("repunctuate_only fetched audio before knowing it had work")

        monkeypatch.setattr(ep, "_download_or_reuse_media", _no_fetch)
        corpus = tmp_path / "corpus"
        run = corpus / "run_20260101-000000_prior"
        (run / "transcripts").mkdir(parents=True)
        (run / "transcripts" / "0001 - Ep.txt").write_text("hello", encoding="utf-8")
        item = ET.Element("item")
        ET.SubElement(item, "guid").text = "g1"
        episode = models.Episode(idx=1, title="Ep", title_safe="Ep", item=item, transcript_urls=[])
        episode.media_url = "https://example.com/ep.mp3"
        cfg = _stage_cfg("repunctuate_only", output_dir=str(corpus), skip_existing=skip_existing)
        jobs: "queue.Queue" = queue.Queue()

        ep.process_episode_download(
            episode, cfg, str(tmp_path / "tmp"), str(corpus / "run_new"), None, jobs, None
        )

        assert jobs.qsize() == 1
        assert jobs.get().temp_media == ""
