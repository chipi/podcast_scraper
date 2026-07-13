"""Minimal runtime ``Config`` builders for GIL and KG experiment tasks."""

from __future__ import annotations

from typing import Any, cast, Dict, Literal, Optional

from podcast_scraper import config_constants
from podcast_scraper.config import Config

_TASK_GI = "grounded_insights"
_TASK_KG = "knowledge_graph"


def runtime_config_for_grounded_insights_eval(params: Optional[Dict[str, Any]]) -> Config:
    """Build ``Config`` for GIL eval (transcript → ``build_artifact``).

    Args:
        params: Optional ``ExperimentConfig.params`` (e.g. ``gi_insight_source``).

    Returns:
        Validated ``Config`` with summaries disabled and GIL enabled.
    """
    p = params or {}
    raw = p.get("gi_insight_source", "stub")
    if raw not in ("stub", "provider"):
        raw = "stub"
    source = cast(Literal["stub", "provider"], raw)
    require_grounding = bool(p.get("gi_require_grounding", False))
    max_insights = int(
        p.get("gi_max_insights", config_constants.DEFAULT_SUMMARY_BULLETS_DOWNSTREAM_MAX)
        or config_constants.DEFAULT_SUMMARY_BULLETS_DOWNSTREAM_MAX
    )
    return Config.model_validate(
        {
            "rss": "",
            "generate_metadata": True,
            "generate_summaries": False,
            "generate_gi": True,
            "generate_kg": False,
            "gi_insight_source": source,
            "gi_require_grounding": require_grounding,
            "gi_max_insights": max_insights,
            "transcribe_missing": False,
        }
    )


def runtime_config_for_knowledge_graph_eval(params: Optional[Dict[str, Any]]) -> Config:
    """Build ``Config`` for KG eval (transcript → ``kg.build_artifact``).

    Args:
        params: Optional ``ExperimentConfig.params`` (e.g. ``kg_extraction_source``).

    Returns:
        Validated ``Config`` with summaries disabled and KG enabled.
    """
    p = params or {}
    raw = p.get("kg_extraction_source", "stub")
    if raw not in ("stub", "provider"):
        raw = "stub"
    source = cast(Literal["stub", "provider"], raw)
    payload: Dict[str, Any] = {
        "rss": "",
        "generate_metadata": True,
        "generate_summaries": False,
        "generate_gi": False,
        "generate_kg": True,
        "kg_extraction_source": source,
        "transcribe_missing": False,
    }
    # #1035 — NER pre-pass opt-in for KG eval cells
    if p.get("kg_extraction_use_ner_prepass") is not None:
        payload["kg_extraction_use_ner_prepass"] = bool(p["kg_extraction_use_ner_prepass"])
    return Config.model_validate(payload)


def merge_eval_task_into_summarizer_config(
    base: Config,
    task: str,
    params: Optional[Dict[str, Any]],
) -> Config:
    """Copy summarization runtime ``Config`` and enable GI or KG for eval.

    Used when the experiment regenerates a summary per episode, then runs
    ``gi.build_artifact`` / ``kg.build_artifact`` with the same provider.

    Args:
        base: Config used to construct the summarization provider (API keys, models).
        task: ``grounded_insights`` or ``knowledge_graph``.
        params: Optional ``ExperimentConfig.params`` (GI/KG knobs).

    Returns:
        New validated ``Config`` with ``generate_gi`` or ``generate_kg`` set.

    Raises:
        ValueError: If ``task`` is not a GI/KG eval task.
    """
    p = params or {}
    if task == _TASK_GI:
        raw_src = p.get("gi_insight_source", "provider")
        if raw_src not in ("stub", "provider"):
            raw_src = "provider"
        gi_src = cast(Literal["stub", "provider"], raw_src)
        require = p.get("gi_require_grounding", True)
        max_insights = p.get("gi_max_insights")
        updates: Dict[str, Any] = {
            "generate_gi": True,
            "generate_kg": False,
            "gi_insight_source": gi_src,
            "gi_require_grounding": bool(require),
        }
        if max_insights is not None:
            updates["gi_max_insights"] = int(max_insights)
        # Value gate. Unforwarded params are dropped silently by this allowlist, so a gate
        # configured in the experiment YAML would sit inert and the cell would look like a
        # no-op result rather than a misconfiguration.
        value_gate = p.get("gi_value_gate_enabled")
        if value_gate is not None:
            updates["gi_value_gate_enabled"] = bool(value_gate)
        min_tier = p.get("gi_value_gate_min_tier")
        if isinstance(min_tier, int) and 0 <= min_tier <= 3:
            updates["gi_value_gate_min_tier"] = min_tier
        # The judge pin MUST be forwarded. Without it each arm grades its own output, and
        # self-grading is ~6x more lenient (qwen drops 4% of its own insights; anthropic drops 26%
        # of the same ones) — so a head-to-head would compare two different strictnesses.
        gate_judge = p.get("gi_value_gate_provider")
        if isinstance(gate_judge, str) and gate_judge:
            updates["gi_value_gate_provider"] = gate_judge
        gate_model = p.get("gi_value_gate_model")
        if isinstance(gate_model, str) and gate_model:
            updates["gi_value_gate_model"] = gate_model
        chunk_chars = p.get("gi_insight_chunk_chars")
        if isinstance(chunk_chars, int) and chunk_chars >= 0:
            updates["gi_insight_chunk_chars"] = chunk_chars
        dedupe_t = p.get("gi_insight_dedupe_threshold")
        if isinstance(dedupe_t, (int, float)):
            updates["gi_insight_dedupe_threshold"] = float(dedupe_t)
        # #698 GIL evidence-stack bundling — forward mode flags from the
        # experiment YAML's ``params:`` dict to the runtime Config so the
        # bundled dispatch in ``gi/pipeline.py`` actually fires for matrix
        # cells. Defaults preserve pre-#698 behaviour.
        quote_mode = p.get("gil_evidence_quote_mode")
        if quote_mode in ("staged", "bundled"):
            updates["gil_evidence_quote_mode"] = quote_mode
        nli_mode = p.get("gil_evidence_nli_mode")
        if nli_mode in ("staged", "bundled"):
            updates["gil_evidence_nli_mode"] = nli_mode
        chunk = p.get("gil_evidence_nli_chunk_size")
        if isinstance(chunk, int) and 1 <= chunk <= 100:
            updates["gil_evidence_nli_chunk_size"] = chunk
        # Auto-align evidence providers to match summary_provider when the
        # summarizer is an LLM (same logic as Config._auto_promote_evidence_providers
        # which only runs at construction time, not on model_copy).
        summary_prov = getattr(base, "summary_provider", "transformers")
        if summary_prov != "transformers":
            quote_prov = p.get("quote_extraction_provider", summary_prov)
            entail_prov = p.get("entailment_provider", summary_prov)
            updates["quote_extraction_provider"] = quote_prov
            updates["entailment_provider"] = entail_prov
        return base.model_copy(update=updates)
    if task == _TASK_KG:
        raw_kg = p.get("kg_extraction_source", "provider")
        if raw_kg not in ("stub", "provider"):
            raw_kg = "provider"
        kg_src = cast(Literal["stub", "provider"], raw_kg)
        updates_kg: Dict[str, Any] = {
            "generate_kg": True,
            "generate_gi": False,
            "kg_extraction_source": kg_src,
        }
        if p.get("kg_max_topics") is not None:
            updates_kg["kg_max_topics"] = int(p["kg_max_topics"])
        if p.get("kg_max_entities") is not None:
            updates_kg["kg_max_entities"] = int(p["kg_max_entities"])
        # #1035 — NER pre-pass opt-in for KG eval cells
        if p.get("kg_extraction_use_ner_prepass") is not None:
            updates_kg["kg_extraction_use_ner_prepass"] = bool(p["kg_extraction_use_ner_prepass"])
        return base.model_copy(update=updates_kg)
    raise ValueError(f"merge_eval_task_into_summarizer_config: unsupported task {task!r}")
