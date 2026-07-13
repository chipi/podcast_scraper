"""Pin which models actually ground the insights, per profile.

The evidence stack (who retrieves the quote, who judges entailment) was resolving differently in
production than in every eval, and nothing caught it. ``_auto_promote_evidence_providers`` is a
``mode="before"`` validator, so it runs BEFORE the profile is merged: a ``summary_provider`` that
arrives *from a profile* is invisible to it, the align never fires, and grounding silently stays on
``transformers`` -- while an eval, which passes ``summary_provider`` explicitly, gets ``ollama``.

The result: every grounding number we measured described a retriever production does not run, and
``gi_nli_entailment_min`` shipped calibrated against the wrong model's score distribution.

These tests assert the resolved providers rather than the intent, so the divergence is stated out
loud and any future change to it has to be deliberate.
"""

from __future__ import annotations

import pytest

from podcast_scraper.config import Config


def _resolved(**kwargs: object) -> tuple[str, str]:
    # model_validate, not Config(...): ``profile`` is an extra input key consumed by a
    # mode="before" validator, not a declared field.
    cfg = Config.model_validate({"generate_gi": True, **kwargs})
    return cfg.quote_extraction_provider, cfg.entailment_provider


class TestEvidenceProviderResolution:
    def test_explicit_llm_summary_provider_aligns_evidence_to_it(self) -> None:
        """Passed explicitly (what the eval harness does), the align fires."""
        assert _resolved(summary_provider="ollama") == ("ollama", "ollama")

    def test_profile_sourced_summary_provider_does_NOT_align(self) -> None:
        """Coming from a profile (how production runs), the align does not fire.

        The validator runs before the profile merge, so it never sees the profile's
        summary_provider. Every shipped profile therefore summarises with an LLM and grounds with
        the local QA/NLI stack. That is the behaviour the corpus was built on; this test states it
        so a future "fix" to the asymmetry cannot flip it silently.
        """
        assert _resolved(profile="local") == ("transformers", "transformers")

    def test_explicit_pin_survives_the_align(self) -> None:
        """A pin must not be clobbered.

        The align used to test ``quote == "transformers"``, which cannot distinguish "I did not
        choose" from "I chose the local stack" — so an explicit pin was silently overwritten with
        the LLM, the one combination #1179 says destroys grounding.
        """
        assert _resolved(
            summary_provider="ollama",
            quote_extraction_provider="transformers",
            entailment_provider="transformers",
        ) == ("transformers", "transformers")

    def test_prod_dgx_pins_its_stack_explicitly(self) -> None:
        """prod_dgx_only must not depend on the accident: an explicit summary_provider cannot
        flip its grounding stack."""
        assert _resolved(profile="prod_dgx_only") == ("transformers", "transformers")
        assert _resolved(profile="prod_dgx_only", summary_provider="ollama") == (
            "transformers",
            "transformers",
        )


class TestEvalMatchesProduction:
    """The eval must run the pipeline production runs, not a reimplementation of it.

    The GI pipeline has two independent stages: write the insight, then find the quote that backs
    it. The eval harness used to re-derive who does the second stage, and its copy disagreed with
    the real one — it pointed quote-extraction at the summarising LLM while production left it on
    the local QA/NLI models. Every GI eval in this repo's history measured a grounding stack
    production does not run.
    """

    def test_eval_inherits_the_production_evidence_stack(self) -> None:
        from podcast_scraper.evaluation.eval_gi_kg_runtime import (
            merge_eval_task_into_summarizer_config as merge,
        )

        prod = Config.model_validate({"profile": "prod_dgx_only", "generate_gi": True})
        base = prod.model_copy(update={"summary_provider": "ollama"})
        cell = merge(base, "grounded_insights", {"gi_insight_source": "provider"})

        assert (cell.quote_extraction_provider, cell.entailment_provider) == (
            prod.quote_extraction_provider,
            prod.entailment_provider,
        )

    def test_an_experiment_can_still_name_the_grounder_on_purpose(self) -> None:
        """Choosing an LLM to ground must be possible — but only by asking for it."""
        from podcast_scraper.evaluation.eval_gi_kg_runtime import (
            merge_eval_task_into_summarizer_config as merge,
        )

        base = Config.model_validate({"profile": "prod_dgx_only", "generate_gi": True}).model_copy(
            update={"summary_provider": "ollama"}
        )
        cell = merge(
            base,
            "grounded_insights",
            {"quote_extraction_provider": "ollama", "entailment_provider": "ollama"},
        )
        assert (cell.quote_extraction_provider, cell.entailment_provider) == ("ollama", "ollama")


class TestNliThresholdCalibrationMatchesItsModel:
    """gi_nli_entailment_min: 0.75 is calibrated for qwen's graded LLM entailment (see the comment
    in prod_dgx_only.yaml). A transformers NLI model emits a softmax probability on a different
    scale, so the threshold and the model that consumes it must be chosen together.
    """

    def test_prod_dgx_threshold_and_entailer_are_recorded_together(self) -> None:
        cfg = Config.model_validate({"profile": "prod_dgx_only", "generate_gi": True})
        assert cfg.gi_nli_entailment_min == pytest.approx(0.75)
        # If this flips to "ollama", the 0.75 calibration becomes valid and this test should be
        # updated deliberately -- not discovered later in a corpus.
        assert cfg.entailment_provider == "transformers"
