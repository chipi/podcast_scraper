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
import re

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
        """A citation must be one of three honest forms, and public ones must resolve.

        Arc 2 (#2134) moved the eval reports to the private repo, and every
        citation to them was left as a repo-relative path that no longer resolved.
        19 rotted before one assertion noticed.

        The first fix pointed them at ``eval-data/``, which was worse than it
        looked: that directory exists in maybe one working tree out of five, so
        the existence check almost never ran, and the value read like a local
        path that normally is not there.

        So a ref must now declare which KIND of citation it is:

          * ``podcast-scraper-eval-data:<path>`` — a report in the private research
            repo. Unverifiable from here by construction; the check that resolves
            it lives in that repo, which always has both the registry (installed
            from the pin) and the reports.
          * ``docs/...`` — a public decision doc. MUST resolve here, which is what
            catches a bare ``docs/guides/eval-reports/...`` — the arc-2 bug.
          * ``#1234`` — a GitHub issue.

        Anything else is a typo or a path smuggled in without saying so.
        """
        for opt in get_grounding_options().values():
            ref = opt.research_ref
            assert ref is not None, f"{opt.option_id} has no research_ref"
            if ref.startswith("podcast-scraper-eval-data:"):
                tail = ref.split(":", 1)[1]
                assert tail.endswith(".md"), f"{opt.option_id}: {ref!r} names no document"
            elif ref.startswith("#"):
                assert ref[1:].isdigit(), f"{opt.option_id}: {ref!r} is not an issue number"
            elif ref.startswith("docs/"):
                assert pathlib.Path(ref).is_file(), (
                    f"{opt.option_id}: {ref} does not exist. If it moved to the research "
                    "repo, cite it as podcast-scraper-eval-data:<path>."
                )
            else:
                raise AssertionError(
                    f"{opt.option_id}: research_ref {ref!r} is not a recognised citation. "
                    "Use podcast-scraper-eval-data:<path>, docs/<path>, or #<issue>."
                )


class TestRationalePointersAreWellFormed:
    """A `rationale:` pointer is a citation too, and rots the same way.

    Five research arguments moved to the private repo so they sit beside the
    reports they cite — they were prose in `#` comments, one of them copied
    verbatim into three presets. What stayed here is the operative fact plus a
    pointer.

    A comment is not importable, so the private repo's `registry-refs-check`
    cannot resolve these until its pin advances. This asserts what CAN be
    asserted here: that every pointer is repo-qualified and names a markdown
    file. It is the same three-form rule `research_ref` follows, and it catches
    the failure that actually happened — a citation left as a bare path.
    """

    def test_every_rationale_pointer_is_repo_qualified(self) -> None:
        src = pathlib.Path("src/podcast_scraper/providers/ml/model_registry.py").read_text()
        pointers = re.findall(r"#\s*rationale:\s*(\S+)", src)
        assert pointers, "no rationale pointers found — did the comment form change?"
        for ref in pointers:
            assert ref.startswith("podcast-scraper-eval-data:"), (
                f"rationale pointer {ref!r} is not repo-qualified. The rationale files "
                "live in chipi/podcast-scraper-eval-data; cite them as "
                "podcast-scraper-eval-data:<path>."
            )
            assert ref.endswith(".md"), f"rationale pointer {ref!r} names no document"

    def test_the_duplicated_argument_is_cited_once_per_preset(self) -> None:
        """The v3-not-v25 argument was copied into three presets verbatim.

        Three copies meant correcting one left two stale. They now point at one
        file; this keeps that true.
        """
        src = pathlib.Path("src/podcast_scraper/providers/ml/model_registry.py").read_text()
        n = src.count("rationale/gi_v3_not_v25.md")
        assert n == 3, f"expected the 3 cloud presets to cite the shared rationale, found {n}"


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
