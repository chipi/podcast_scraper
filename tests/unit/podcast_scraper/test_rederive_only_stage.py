"""``rederive_only``: the rename, the alias, and the ASR-credential gates it must not hit.

WHY THE RENAME. ``enrich_only`` collided with the unrelated corpus-level ``enrich`` command
(topic clusters, co-appearance). One re-derives ONE episode's LLM stages from its on-disk
transcript; the other rebuilds corpus-wide enrichments. Same word, different operation — and
the operator read the flag name and could not tell which thing was being worked on.

WHY THE GATES MATTER. Two independent places demanded a Deepgram key for reprocess modes that
never call a transcription provider, and each on its own made the mode unusable on the default
cloud profile:

1. ``_validate_deepgram_provider_requirements`` — ``relabel_only`` / ``rediarize_only`` set
   ``transcribe_missing=true`` deliberately, as a ROUTING TRICK so the episode reaches the
   transcription stage and ``_maybe_dispatch_reprocess_stage`` can intercept it. The validator
   read that flag as "will transcribe".
2. ``orchestration._create_transcription_provider`` — same missing key, raised at runtime.
   These stages use the provider only as the OPTIONAL formatter argument of
   ``_format_transcript_if_needed`` (whose signature defaults it to ``None``).

Found by running ``--pipeline-stage relabel_only`` on a real corpus while trying to establish
where relabel writes its artifacts: it died at config validation before doing anything, then
after fixing that, died again at provider init. Neither was visible to unit tests that call
the stage functions directly.
"""

from __future__ import annotations

import logging

import pytest

from podcast_scraper import config

_BASE = {"rss_url": "https://example.com/feed.xml", "transcription_provider": "deepgram"}


class TestTheRename:
    def test_canonical_name_is_accepted(self):
        cfg = config.Config.model_validate({**_BASE, "pipeline_stage": "rederive_only"})
        assert cfg.pipeline_stage == "rederive_only"

    def test_deprecated_alias_normalises_to_the_canonical_name(self, caplog):
        """The alias must be REWRITTEN, not merely tolerated.

        Every downstream check — the stage coercions, both ASR-credential gates, and the reuse
        branch in ``episode_processor`` — compares against the canonical string. If the alias
        survived normalisation, each of those would silently take the 'not a reprocess' path
        and the mode would go back to being a no-op.
        """
        caplog.set_level(logging.WARNING)
        cfg = config.Config.model_validate({**_BASE, "pipeline_stage": "enrich_only"})
        assert cfg.pipeline_stage == "rederive_only"
        assert any("DEPRECATED" in r.getMessage() for r in caplog.records)

    def test_the_alias_gets_the_same_coercions(self):
        """An old profile must behave IDENTICALLY, not just parse."""
        old = config.Config.model_validate({**_BASE, "pipeline_stage": "enrich_only"})
        new = config.Config.model_validate({**_BASE, "pipeline_stage": "rederive_only"})
        assert (old.pipeline_stage, old.transcribe_missing, old.skip_existing) == (
            new.pipeline_stage,
            new.transcribe_missing,
            new.skip_existing,
        )

    def test_the_stage_still_coerces_what_makes_it_work(self):
        cfg = config.Config.model_validate({**_BASE, "pipeline_stage": "rederive_only"})
        assert cfg.transcribe_missing is False, "must never reach an ASR call"
        assert cfg.skip_existing is True, "transcript reuse is gated on skip_existing"

    def test_canonical_name_emits_no_deprecation_noise(self, caplog):
        caplog.set_level(logging.WARNING)
        config.Config.model_validate({**_BASE, "pipeline_stage": "rederive_only"})
        assert not any("DEPRECATED" in r.getMessage() for r in caplog.records)

    def test_alias_map_points_somewhere_real(self):
        """Guards against an alias that maps to a stage the Literal does not accept."""
        import typing

        allowed = set(typing.get_args(config.Config.model_fields["pipeline_stage"].annotation))
        for old, new in config.DEPRECATED_PIPELINE_STAGE_ALIASES.items():
            assert new in allowed, f"{old} -> {new} is not a valid stage"
            assert old in allowed, f"{old} must stay accepted or old profiles break"


class TestNoAsrCredentialIsDemanded:
    """Gate 1: config validation."""

    @pytest.mark.parametrize(
        "stage", ["rederive_only", "enrich_only", "relabel_only", "rediarize_only"]
    )
    def test_reprocess_stages_build_without_a_deepgram_key(self, stage, monkeypatch):
        monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
        cfg = config.Config.model_validate({**_BASE, "pipeline_stage": stage})
        assert cfg.transcription_provider == "deepgram"

    def test_a_real_transcribing_run_still_requires_the_key(self, monkeypatch):
        """The skip must be narrow. `full` genuinely transcribes and must fail fast."""
        monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
        with pytest.raises(ValueError, match="Deepgram API key required"):
            config.Config.model_validate({**_BASE, "pipeline_stage": "full"})

    def test_the_never_transcribe_set_is_exactly_the_routing_trick_stages(self):
        """``rederive_only`` is NOT in this set, and that is deliberate.

        It reaches the skip via ``transcribe_missing=False`` — the honest route. Only the two
        stages that set the flag TRUE while never calling a provider need the extra exemption.
        Adding rederive_only here would hide a regression: if its coercion ever stopped
        setting transcribe_missing=False, this set would mask it.
        """
        assert config.STAGES_THAT_NEVER_TRANSCRIBE == frozenset({"relabel_only", "rediarize_only"})


class TestProviderInitIsNonFatalForThoseStages:
    """Gate 2: runtime provider construction."""

    @staticmethod
    def _cfg(stage):
        return config.Config.model_validate({**_BASE, "pipeline_stage": stage})

    @pytest.mark.parametrize("stage", ["relabel_only", "rediarize_only"])
    def test_init_failure_returns_none_instead_of_raising(self, stage, monkeypatch, caplog):
        from podcast_scraper.workflow import orchestration

        def _boom(_cfg):
            raise ValueError("Deepgram API key required for transcription_provider='deepgram'.")

        monkeypatch.setattr(orchestration, "_get_factory_function", lambda *a, **k: _boom)
        caplog.set_level(logging.WARNING)

        got = orchestration._create_transcription_provider(self._cfg(stage))

        assert got is None, (
            "a stage that never calls a transcription provider must not abort the run because "
            "one could not be built"
        )
        assert any("never calls one" in r.getMessage() for r in caplog.records)

    def test_a_transcribing_run_still_fails_fast(self, monkeypatch):
        """Fail-fast is the correct behaviour when the provider WILL be used."""
        from podcast_scraper.workflow import orchestration

        def _boom(_cfg):
            raise ValueError("boom")

        monkeypatch.setattr(orchestration, "_get_factory_function", lambda *a, **k: _boom)
        cfg = config.Config.model_validate(
            {**_BASE, "deepgram_api_key": "k", "pipeline_stage": "full"}
        )
        with pytest.raises(ValueError):
            orchestration._create_transcription_provider(cfg)

    def test_rederive_only_never_reaches_provider_construction_at_all(self):
        """It short-circuits on transcribe_missing=False, before any factory call."""
        from podcast_scraper.workflow import orchestration

        assert orchestration._create_transcription_provider(self._cfg("rederive_only")) is None
