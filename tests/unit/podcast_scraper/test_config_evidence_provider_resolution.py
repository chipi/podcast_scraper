"""Pin which models ground the insights, for every profile.

The design: an LLM summariser grounds with that same LLM; a local (transformers / summllama)
summariser grounds with the ML extractive-QA + NLI models. The two grounders are not
interchangeable — the ML one is built for the ML profiles.

That design was expressed in code (``GIL_EVIDENCE_ALIGN_SUMMARY_PROVIDERS`` +
``gil_evidence_match_summary_provider``, default True) but never reached production.
``_auto_promote_evidence_providers`` is a ``mode="before"`` validator, so it runs BEFORE the profile
is merged: a ``summary_provider`` arriving *from a profile* was invisible to it, the align never
fired, and all nine LLM profiles silently fell back to the ML grounder — which returns a single
answer-fragment per insight and grounds 0% of them.

It went unnoticed because the eval harness passed ``summary_provider`` explicitly, so evals
accidentally ran the INTENDED design while production ran the broken fallback.

These tests pin the resolved grounder per profile so the two can never drift apart again.
"""

from __future__ import annotations

import pathlib

import pytest

from podcast_scraper.config import Config, GIL_EVIDENCE_ALIGN_SUMMARY_PROVIDERS

PROFILE_DIR = pathlib.Path("config/profiles")


def _resolved(**kwargs: object) -> tuple[str, str]:
    # model_validate, not Config(...): ``profile`` is an extra input key consumed by a
    # mode="before" validator, not a declared field.
    cfg = Config.model_validate({"generate_gi": True, **kwargs})
    return cfg.quote_extraction_provider, cfg.entailment_provider


class TestEvidenceAlignsToTheSummariser:
    def test_llm_summariser_grounds_with_itself_when_passed_explicitly(self) -> None:
        assert _resolved(summary_provider="ollama") == ("ollama", "ollama")

    def test_llm_summariser_grounds_with_itself_when_it_comes_FROM_A_PROFILE(self) -> None:
        """The regression that mattered: a profile-sourced summary_provider must align too.

        experiment_dgx_only summarises with qwen, so it must ground with qwen. It used to fall
        through to the ML QA/NLI grounder — the stack meant for the ML profiles — and ground
        nothing.
        """
        assert _resolved(profile="experiment_dgx_only") == ("ollama", "ollama")

    def test_local_summariser_keeps_the_ML_grounder(self) -> None:
        """The ML grounder is correct for the ML profiles — that is what it is for."""
        assert _resolved(profile="test_default") == ("transformers", "transformers")

    def test_explicit_pin_survives_the_align(self) -> None:
        """A deliberate pin must not be clobbered.

        The align used to test ``quote == "transformers"``, which cannot distinguish "I did not
        choose" from "I chose the ML stack" — so an explicit pin was silently overwritten.
        """
        assert _resolved(
            summary_provider="ollama",
            quote_extraction_provider="transformers",
            entailment_provider="transformers",
        ) == ("transformers", "transformers")

    def test_opt_out_disables_the_align(self) -> None:
        assert _resolved(summary_provider="ollama", gil_evidence_match_summary_provider=False) == (
            "transformers",
            "transformers",
        )


class TestEveryShippedProfileGroundsWithItsIntendedStack:
    """No profile may silently ground with the other stack."""

    @pytest.mark.parametrize(
        "profile",
        sorted(p.stem for p in PROFILE_DIR.glob("*.yaml")) if PROFILE_DIR.is_dir() else [],
    )
    def test_profile_grounder_matches_its_summariser(self, profile: str) -> None:
        try:
            cfg = Config.model_validate({"profile": profile, "generate_gi": True})
        except Exception:  # noqa: BLE001 — a profile that cannot build is another test's problem
            pytest.skip(f"profile {profile} does not construct")

        summary = cfg.summary_provider
        expected = summary if summary in GIL_EVIDENCE_ALIGN_SUMMARY_PROVIDERS else "transformers"

        assert cfg.quote_extraction_provider == expected, (
            f"{profile} summarises with {summary} but extracts quotes with "
            f"{cfg.quote_extraction_provider}"
        )
        assert cfg.entailment_provider == expected, (
            f"{profile} summarises with {summary} but scores entailment with "
            f"{cfg.entailment_provider}"
        )


class TestEvalMatchesProduction:
    """The eval must run the pipeline production runs, not a reimplementation of it.

    The harness used to re-derive the evidence stack itself. Its copy happened to be RIGHT (it
    aligned to the LLM) while production was wrong — so the eval measured the intended design and
    production shipped the broken one, and the two numbers could never be reconciled.
    """

    def test_eval_inherits_the_production_evidence_stack(self) -> None:
        from podcast_scraper.evaluation.eval_gi_kg_runtime import (
            merge_eval_task_into_summarizer_config as merge,
        )

        prod = Config.model_validate({"profile": "experiment_dgx_only", "generate_gi": True})
        cell = merge(prod, "grounded_insights", {"gi_insight_source": "provider"})

        assert (cell.quote_extraction_provider, cell.entailment_provider) == (
            prod.quote_extraction_provider,
            prod.entailment_provider,
        )

    def test_an_experiment_can_still_name_the_grounder_on_purpose(self) -> None:
        """Comparing grounders must remain possible — but only by asking for one."""
        from podcast_scraper.evaluation.eval_gi_kg_runtime import (
            merge_eval_task_into_summarizer_config as merge,
        )

        base = Config.model_validate({"profile": "experiment_dgx_only", "generate_gi": True})
        cell = merge(
            base,
            "grounded_insights",
            {
                "quote_extraction_provider": "transformers",
                "entailment_provider": "transformers",
            },
        )
        assert (cell.quote_extraction_provider, cell.entailment_provider) == (
            "transformers",
            "transformers",
        )


class TestNliThresholdMatchesItsEntailer:
    """gi_nli_entailment_min: 0.75 is calibrated for an LLM answering the evidence-framed prompt on
    a graded scale. It is NOT a threshold for the ML NLI head, whose softmax is a different scale.
    The threshold and the model that consumes it must move together.
    """

    def test_prod_dgx_entailer_is_the_llm_the_threshold_was_calibrated_for(self) -> None:
        cfg = Config.model_validate({"profile": "experiment_dgx_only", "generate_gi": True})
        assert cfg.gi_nli_entailment_min == pytest.approx(0.75)
        assert cfg.entailment_provider == "ollama"
