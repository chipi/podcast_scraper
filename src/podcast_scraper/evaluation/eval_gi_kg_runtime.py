"""Minimal runtime ``Config`` builders for GIL and KG experiment tasks."""

from __future__ import annotations

from typing import Any, cast, Dict, Literal, Optional

from podcast_scraper import config_constants
from podcast_scraper.config import Config


def runtime_config_for_grounded_insights_eval(params: Optional[Dict[str, Any]]) -> Config:
    """Build ``Config`` for GIL eval (transcript → ``build_artifact``).

    Args:
        params: Optional ``ExperimentConfig.params`` (e.g. ``gi_insight_source``).

    Returns:
        Validated ``Config`` with summaries disabled and GIL enabled.
    """
    p = params or {}
    raw = p.get("gi_insight_source", "stub")
    if raw not in ("stub", "provider", "summary_bullets"):
        raw = "stub"
    source = cast(Literal["stub", "provider", "summary_bullets"], raw)
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
    if raw not in ("stub", "provider", "summary_bullets"):
        raw = "stub"
    source = cast(Literal["stub", "provider", "summary_bullets"], raw)
    return Config.model_validate(
        {
            "rss": "",
            "generate_metadata": True,
            "generate_summaries": False,
            "generate_gi": False,
            "generate_kg": True,
            "kg_extraction_source": source,
            "transcribe_missing": False,
        }
    )
