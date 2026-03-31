"""Optional dependency checks for GIL grounding (fail fast at pipeline start)."""

from __future__ import annotations

from typing import Any, Optional, Tuple, TYPE_CHECKING
from unittest.mock import Mock

from podcast_scraper.exceptions import ProviderDependencyError

if TYPE_CHECKING:
    from podcast_scraper import config

_LOCAL_ENTAILMENT_BACKENDS = frozenset({"transformers", "hybrid_ml"})


def _provider_field_str(cfg: Any, name: str, default: str) -> str:
    """Read a provider string from cfg; MagicMock / non-str -> default (for unit tests)."""
    raw = getattr(cfg, name, None)
    if isinstance(raw, Mock):
        return default
    if not isinstance(raw, str) or not raw.strip():
        return default
    return raw.strip()


def validate_gil_grounding_dependencies(cfg: "config.Config") -> None:
    """Raise ProviderDependencyError if local NLI is configured but deps are missing.

    Local entailment (CrossEncoder) requires the ``sentence-transformers`` package
    (declared under the ``ml`` optional extra in pyproject.toml).

    Args:
        cfg: Resolved pipeline configuration.

    Raises:
        ProviderDependencyError: When entailment uses transformers/hybrid_ml but
            ``sentence_transformers`` is not importable.
    """
    if not getattr(cfg, "generate_gi", False):
        return
    if not getattr(cfg, "gi_require_grounding", True):
        return
    ent = getattr(cfg, "entailment_provider", "transformers")
    if ent not in _LOCAL_ENTAILMENT_BACKENDS:
        return
    try:
        import sentence_transformers  # noqa: F401
    except ImportError as exc:
        raise ProviderDependencyError(
            message=(
                "GIL entailment is set to a local backend (transformers/hybrid_ml) but "
                "sentence-transformers is not installed."
            ),
            provider="GIL/evidence",
            dependency="sentence-transformers",
            suggestion='Install: pip install -e ".[ml]"',
        ) from exc


def create_gil_evidence_providers(
    cfg: "config.Config",
    summary_provider: Optional[Any] = None,
) -> Tuple[Any, Any]:
    """Build quote extraction and entailment provider instances from ``cfg``.

    Mirrors ``metadata_generation`` wiring: reuse ``summary_provider`` when
    ``quote_extraction_provider`` / ``entailment_provider`` match
    ``summary_provider``; otherwise create via ``create_summarization_provider``.

    Args:
        cfg: Resolved configuration (must include provider string fields).
        summary_provider: Optional existing summarization instance to reuse.

    Returns:
        ``(quote_extraction_provider, entailment_provider)``.
    """
    from podcast_scraper.summarization.factory import create_summarization_provider

    quote_key = _provider_field_str(cfg, "quote_extraction_provider", "transformers")
    entail_key = _provider_field_str(cfg, "entailment_provider", "transformers")
    summary_key = _provider_field_str(cfg, "summary_provider", "transformers")

    needs_summary_instance = quote_key == summary_key or entail_key == summary_key
    effective_summary = summary_provider
    if needs_summary_instance and effective_summary is None:
        effective_summary = create_summarization_provider(cfg)
        if hasattr(effective_summary, "initialize"):
            effective_summary.initialize()

    if quote_key == summary_key:
        quote_extraction_provider = effective_summary
    else:
        quote_extraction_provider = create_summarization_provider(
            cfg, provider_type_override=quote_key
        )
        if hasattr(quote_extraction_provider, "initialize"):
            quote_extraction_provider.initialize()

    if entail_key == summary_key:
        entailment_provider = effective_summary
    elif entail_key == quote_key:
        entailment_provider = quote_extraction_provider
    else:
        entailment_provider = create_summarization_provider(cfg, provider_type_override=entail_key)
        if hasattr(entailment_provider, "initialize"):
            entailment_provider.initialize()

    return quote_extraction_provider, entailment_provider
