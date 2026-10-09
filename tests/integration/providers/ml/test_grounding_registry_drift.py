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
from podcast_scraper.providers.ml import model_registry
from podcast_scraper.providers.ml.model_registry import (
    _PROFILE_PRESETS,
    _SUMMARY_OPTIONS,
    get_grounding_option,
    get_grounding_options,
    StageOption,
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
            # an unmeasured default; every option must now publish what was measured, and when.
            # The report behind it is cited in the private eval project (ADR-162).
            assert opt.headline_metric, f"{opt.option_id} has no headline_metric"
            assert opt.measured_at, f"{opt.option_id} has no measured_at"

    def test_the_llm_grounder_is_primary(self) -> None:
        assert get_grounding_option(LLM_GROUNDER).tier == "primary"
        assert get_grounding_option(ML_GROUNDER).tier == "fallback"


def _all_options() -> list[StageOption]:
    return [
        opt
        for name in sorted(dir(model_registry))
        if name.startswith("get_") and name.endswith("_options")
        for opt in getattr(model_registry, name)().values()
    ]


class TestCitationsArePublic:
    """The registry cites only what a reader of this repo can open.

    Arc 2 (#2134) moved the eval reports to the private repo, and every
    citation to them was left as a repo-relative path that no longer resolved;
    19 rotted before one assertion noticed. The repo-qualified spelling that
    replaced them still named private documents from public code, which ADR-162
    rules out. So an option publishes its claim (``headline_metric``,
    ``measured_at``) here, and the eval project holds the report behind it and
    checks, against its pinned build, that every claim has one.

    What stays here is a public decision doc, which must resolve, or an issue.
    """

    def test_every_research_ref_is_a_public_doc_or_an_issue(self) -> None:
        options = _all_options()
        assert len(options) > 40, "the stage accessors changed; this test reads none of them"
        for opt in options:
            ref = opt.research_ref
            if ref is None:
                continue
            if ref.startswith("#"):
                assert ref[1:].isdigit(), f"{opt.option_id}: {ref!r} is not an issue number"
            elif ref.startswith("docs/"):
                assert pathlib.Path(ref).is_file(), f"{opt.option_id}: {ref} does not exist"
                assert not ref.startswith(
                    "docs/guides/eval-reports/"
                ), f"{opt.option_id}: eval reports are private; cite them in the eval project"
            else:
                raise AssertionError(
                    f"{opt.option_id}: research_ref {ref!r} is not a public doc or an issue"
                )

    def test_every_option_publishes_its_claim(self) -> None:
        for opt in _all_options():
            assert opt.headline_metric, f"{opt.option_id} has no headline_metric"
            assert opt.measured_at, f"{opt.option_id} has no measured_at"

    def test_the_registry_names_no_private_document(self) -> None:
        """Comments included: the research arguments that were ``# rationale:`` pointers
        are listed in the eval project's evidence map, beside the reports they cite."""
        src = pathlib.Path("src/podcast_scraper/providers/ml/model_registry.py").read_text()
        for marker in (
            "podcast-scraper-eval-data",
            "eval-data/",
            "private eval repo",
            "rationale/",
        ):
            assert marker not in src, f"model_registry.py names the private eval repo ({marker!r})"


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


#: Provider keys the cloud profiles need in order to construct at all.
#:
#: This module used to get them by accident: ``test_gate_model_rides_the_route`` set every provider
#: key at MODULE level with ``os.environ.setdefault``, which leaked into the whole pytest process,
#: so importing that file silently supplied the keys here. Scoping that leak (it was breaking
#: ``test_deepseek_provider``'s "missing key raises" assertion) would have turned 13 profiles —
#: every cloud_* and bakeoff_* one — into ``skip: does not construct``, i.e. a silent 13-profile
#: hole in the drift check, which is a worse bug than the one being fixed. So the keys are
#: declared HERE, where they are needed, and scoped so they cannot leak in turn.
_DUMMY_KEY_VARS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "DEEPGRAM_API_KEY",
    "DEEPSEEK_API_KEY",
    "GROQ_API_KEY",
    "GROK_API_KEY",
    "MISTRAL_API_KEY",
    "LITELLM_API_KEY",
    "QWEN_API_KEY",
    "DASHSCOPE_API_KEY",
)


@pytest.fixture(autouse=True)
def _dummy_provider_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dummy provider keys for every test in this module, unwound after each test."""
    for var in _DUMMY_KEY_VARS:
        monkeypatch.setenv(var, "dummy-for-validation")


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
