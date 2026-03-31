"""Derived labels for GIL artifact provenance (gi.json top-level model_version)."""

from __future__ import annotations

from typing import Any, Optional

# Map summary_provider string -> Config attribute holding default summary model id.
_SUMMARY_MODEL_ATTR_BY_PROVIDER: dict[str, str] = {
    "openai": "openai_summary_model",
    "gemini": "gemini_summary_model",
    "anthropic": "anthropic_summary_model",
    "ollama": "ollama_summary_model",
    "deepseek": "deepseek_summary_model",
    "grok": "grok_summary_model",
    "mistral": "mistral_summary_model",
}


def _summary_model_from_cfg(cfg: Any) -> str:
    """Best-effort summarization model id from Config when provider has no .summary_model."""
    sp = getattr(cfg, "summary_provider", None)
    if sp in ("transformers", "hybrid_ml"):
        for key in ("summary_model", "summary_reduce_model"):
            v = getattr(cfg, key, None)
            if v:
                return str(v)
        return str(sp)
    attr = _SUMMARY_MODEL_ATTR_BY_PROVIDER.get(str(sp or ""), "")
    if attr:
        v = getattr(cfg, attr, None)
        if v:
            return str(v)
    return "unknown"


def _insight_lineage_model_id(cfg: Any, summary_provider: Optional[Any]) -> str:
    """Model id associated with insight text (bullets or generate_insights)."""
    if summary_provider is not None:
        sm = getattr(summary_provider, "summary_model", None)
        if sm is not None and str(sm).strip():
            return str(sm).strip()
    return _summary_model_from_cfg(cfg)


def resolve_gil_artifact_model_version(
    cfg: Any,
    summary_provider: Optional[Any],
    *,
    gi_insight_source: str,
) -> str:
    """Return gi.json ``model_version`` from pipeline state (no duplicate config field).

    Args:
        cfg: Resolved ``Config``.
        summary_provider: Summarization provider instance if available.
        gi_insight_source: ``stub`` | ``summary_bullets`` | ``provider``.

    Returns:
        Non-empty model identifier string for artifact provenance.
    """
    source = (gi_insight_source or "stub").strip().lower()
    if source == "stub":
        return "stub"
    if source in ("summary_bullets", "provider"):
        mid = _insight_lineage_model_id(cfg, summary_provider)
        return mid if mid and mid != "unknown" else "unknown"
    return "stub"
