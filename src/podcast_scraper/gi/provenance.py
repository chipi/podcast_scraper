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
    "groq": "groq_summary_model",
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
    """Model id for insight text produced from summary bullets (summarization model)."""
    if summary_provider is not None:
        sm = getattr(summary_provider, "summary_model", None)
        if isinstance(sm, str) and sm.strip():
            return sm.strip()
    return _summary_model_from_cfg(cfg)


def _provider_insight_lineage_model_id(cfg: Any, provider: Optional[Any]) -> str:
    """Model id for the insight provider (generate_insights), when distinct from summary."""
    if provider is not None:
        im = getattr(provider, "insight_model", None)
        if isinstance(im, str) and im.strip():
            return im.strip()
    return _insight_lineage_model_id(cfg, provider)


def resolve_gil_artifact_model_version(
    cfg: Any,
    lineage_provider: Optional[Any],
) -> str:
    """Return gi.json ``model_version`` from pipeline state (no duplicate config field).

    There is one source of insights — the provider — so the model identifier is the provider's,
    or ``"unknown"`` when it cannot be determined.

    This used to take a ``gi_insight_source`` argument and return a fixed placeholder label when it
    was anything other than ``"provider"``. That stamped a fake lineage onto real artifacts: the
    field defaulted to that placeholder, so an episode could carry a fabricated lineage in its
    provenance while the corpus counted it as processed (#1657). Both the argument and that
    return value are gone.

    Args:
        cfg: Resolved ``Config``.
        lineage_provider: Summarization/insight provider instance.

    Returns:
        Non-empty model identifier string for artifact provenance.
    """
    mid = _provider_insight_lineage_model_id(cfg, lineage_provider)
    return mid if mid and mid != "unknown" else "unknown"
