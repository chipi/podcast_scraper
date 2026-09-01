"""``--pipeline-stage`` must reach config pre-validation, so stage coercions beat validators.

The bug this pins: a validator gated on a field that the STAGE coerces will see the
pre-coercion value unless the stage participates in the same early merge ``--profile`` does.
Concretely, ``rederive_only`` coerces ``transcribe_missing=False`` and the Deepgram-key validator
keys off exactly that — so ``--pipeline-stage rederive_only`` was refused for lacking an ASR
credential the run would never use. That turned "iterate on the LLM stages locally" into
"export a fake Deepgram key", which is the kind of workaround that then gets committed.

Two halves, tested separately because they fail independently:
  * the validator honours the coerced flag (``TestDeepgramKeySkippedWhenNotTranscribing``)
  * the flag actually arrives coerced (``TestStageReachesPreValidation``)
"""

from __future__ import annotations

import argparse

import pytest

from podcast_scraper import cli, config

_BASE = {
    "rss_url": "https://example.com/feed.xml",
    "transcription_provider": "deepgram",
    "deepgram_api_key": None,
}


class TestDeepgramKeySkippedWhenNotTranscribing:
    def test_no_key_needed_when_transcribe_missing_is_false(self):
        cfg = config.Config.model_validate({**_BASE, "transcribe_missing": False})
        assert cfg.transcription_provider == "deepgram"

    def test_key_still_required_when_the_run_will_transcribe(self, monkeypatch):
        """The skip must be narrow: a real ASR run without a key still fails fast."""
        monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
        with pytest.raises(ValueError, match="Deepgram API key required"):
            config.Config.model_validate({**_BASE, "transcribe_missing": True})

    def test_default_is_still_to_require_the_key(self, monkeypatch):
        """Omitting ``transcribe_missing`` must not inherit the permissive branch."""
        monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
        with pytest.raises(ValueError, match="Deepgram API key required"):
            config.Config.model_validate(dict(_BASE))


class TestStageReachesPreValidation:
    @staticmethod
    def _stage(argv):
        return cli._argv_cli_pipeline_stage(argv)

    @pytest.mark.parametrize(
        "argv,expected",
        [
            (["--pipeline-stage", "rederive_only"], "rederive_only"),
            (["--pipeline-stage=rederive_only"], "rederive_only"),
            (["--other", "x", "--pipeline-stage", "relabel_only"], "relabel_only"),
            ([], None),
            (["--pipeline-stage-ish", "nope"], None),  # must not prefix-match a longer flag
        ],
        ids=["space", "equals", "positional", "absent", "no-prefix-match"],
    )
    def test_reads_the_flag_from_argv(self, argv, expected):
        assert self._stage(argv) == expected


class TestExplicitFlagBeatsTheFile:
    """Precedence, driven through the REAL merge (``_load_and_merge_config``).

    The first version of this test rebuilt the merge inline — set a dict, call the argv reader,
    assign, assert. It passed against the buggy ``if cli_stage is not None and "pipeline_stage"
    not in validate_data`` guard, because a mirror of the logic cannot disagree with the logic.
    Mutation-testing caught it. Anything asserting precedence has to go through the function
    that decides precedence.

    The bug being pinned: a YAML pinning ``pipeline_stage: full`` silently defeating an
    explicit ``--pipeline-stage rederive_only``, which then surfaces as a Deepgram-key error and
    reads as a credentials problem rather than a precedence one.
    """

    _YAML = {
        "rss_url": "https://example.com/feed.xml",
        "transcription_provider": "deepgram",
        "pipeline_stage": "full",
    }

    @staticmethod
    def _merge(yaml_data, argv, monkeypatch):
        parser = argparse.ArgumentParser()
        cli._add_common_arguments(parser)
        cli._add_pipeline_stage_arguments(parser)
        monkeypatch.setattr(cli.config, "load_config_file", lambda _p: dict(yaml_data))
        return cli._load_and_merge_config(parser, "operator.yaml", argv)

    def test_cli_stage_overrides_a_yaml_pin(self, monkeypatch):
        monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
        args = self._merge(self._YAML, ["--pipeline-stage", "rederive_only"], monkeypatch)
        assert args.pipeline_stage == "rederive_only"

    def test_yaml_pin_survives_when_no_cli_flag_is_given(self, monkeypatch):
        """The override must be an override, not a blanket reset to a default."""
        monkeypatch.setenv("DEEPGRAM_API_KEY", "k")
        args = self._merge(self._YAML, [], monkeypatch)
        assert args.pipeline_stage == "full"

    def test_no_asr_key_is_demanded_for_the_overridden_stage(self, monkeypatch):
        """The whole reason precedence matters: yaml says full, CLI says rederive_only.

        With the buggy guard this raised "Deepgram API key required" — pre-validation still
        saw ``full``, so the key validator ran for a stage that calls no ASR provider.
        """
        monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
        self._merge(self._YAML, ["--pipeline-stage", "rederive_only"], monkeypatch)


class TestEndToEndEnrichOnlyNeedsNoAsrCredential:
    def test_rederive_only_config_builds_without_a_deepgram_key(self, monkeypatch):
        """The whole point, in one assertion: the two halves compose."""
        monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
        cfg = config.Config.model_validate({**_BASE, "pipeline_stage": "rederive_only"})
        assert cfg.transcribe_missing is False, "rederive_only must coerce transcribe_missing"
