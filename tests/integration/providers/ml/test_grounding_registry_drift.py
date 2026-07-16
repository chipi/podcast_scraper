"""The grounding stage must agree with itself in three places.

Who finds the quote that backs an insight was, until now, decided by a config fallback that no
registry entry could contradict. The result: every LLM profile silently grounded with the ML
QA + NLI stack — the grounder built for the *local* profiles — and grounded 8% of its insights
instead of 82%.

Three things must not drift apart:

  1. the registry PRESET      (grounding= names a StageOption)
  2. the summariser           (an LLM summariser grounds with itself; a local one uses ML QA+NLI)
  3. the resolved Config      (what a run actually loads)

Any two agreeing while the third quietly disagrees is exactly how this got shipped.
"""

from __future__ import annotations

import pathlib

import pytest

from podcast_scraper.config import Config, GIL_EVIDENCE_ALIGN_SUMMARY_PROVIDERS
from podcast_scraper.providers.ml.model_registry import (
    _PROFILE_PRESETS,
    _SUMMARY_OPTIONS,
    get_grounding_option,
    get_grounding_options,
)

PROFILE_DIR = pathlib.Path("config/profiles")

LLM_GROUNDER = "llm_matched_to_summary"
ML_GROUNDER = "ml_qa_nli"


class TestGroundingStageIsRegistered:
    def test_both_options_exist_and_are_measured(self) -> None:
        options = get_grounding_options()
        assert set(options) == {LLM_GROUNDER, ML_GROUNDER}
        for opt in options.values():
            # A StageOption without a measurement is an opinion. This stage shipped for months on
            # an unmeasured default; every option must now carry its evidence.
            assert opt.research_ref, f"{opt.option_id} has no research_ref"
            assert opt.headline_metric, f"{opt.option_id} has no headline_metric"
            assert opt.measured_at, f"{opt.option_id} has no measured_at"

    def test_the_llm_grounder_is_primary(self) -> None:
        assert get_grounding_option(LLM_GROUNDER).tier == "primary"
        assert get_grounding_option(ML_GROUNDER).tier == "fallback"

    def test_research_ref_points_at_a_real_report(self) -> None:
        for opt in get_grounding_options().values():
            ref = opt.research_ref
            assert ref is not None, f"{opt.option_id} has no research_ref"
            assert pathlib.Path(ref).is_file(), f"{ref} does not exist"


class TestPresetGrounderMatchesItsSummariser:
    """An LLM summariser must ground with itself; a local one must use the ML stack."""

    @pytest.mark.parametrize("preset_name", sorted(_PROFILE_PRESETS))
    def test_preset(self, preset_name: str) -> None:
        preset = _PROFILE_PRESETS[preset_name]
        summary_opt = _SUMMARY_OPTIONS.get(preset.summary)
        assert summary_opt is not None, f"{preset_name} names an unknown summary option"

        provider = summary_opt.provider
        expected = LLM_GROUNDER if provider in GIL_EVIDENCE_ALIGN_SUMMARY_PROVIDERS else ML_GROUNDER

        assert preset.grounding == expected, (
            f"preset {preset_name} summarises with {provider!r} but grounds with "
            f"{preset.grounding!r} — expected {expected!r}"
        )


class TestResolvedConfigAgreesWithTheRegistry:
    """The registry can say one thing and the runtime do another. Pin the runtime too."""

    @pytest.mark.parametrize(
        "profile",
        sorted(p.stem for p in PROFILE_DIR.glob("*.yaml")) if PROFILE_DIR.is_dir() else [],
    )
    def test_profile_resolves_to_its_registry_grounder(self, profile: str) -> None:
        try:
            cfg = Config.model_validate({"profile": profile, "generate_gi": True})
        except Exception:  # noqa: BLE001 — profiles that cannot build are another test's problem
            pytest.skip(f"profile {profile} does not construct")

        summary = cfg.summary_provider
        if summary in GIL_EVIDENCE_ALIGN_SUMMARY_PROVIDERS:
            # llm_matched_to_summary: the grounder IS the summarising LLM
            assert cfg.quote_extraction_provider == summary
            assert cfg.entailment_provider == summary
        else:
            # ml_qa_nli: the local extractive-QA + NLI models
            assert cfg.quote_extraction_provider == "transformers"
            assert cfg.entailment_provider == "transformers"

    def test_ml_option_pins_the_models_the_runtime_loads(self) -> None:
        opt = get_grounding_option(ML_GROUNDER)
        settings = opt.extra_settings or {}
        cfg = Config.model_validate({"profile": "dev", "generate_gi": True})
        assert cfg.gi_qa_model == settings["qa_model"]
        assert cfg.gi_nli_model == settings["nli_model"]
        assert cfg.gi_qa_window_chars == settings["qa_window_chars"]
