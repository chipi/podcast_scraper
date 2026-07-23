"""Model governance (ADR-124 / #1258): only registry-sanctioned models may run.

The registry's StageOptions are the source of truth for vetted models. When
``enforce_model_governance`` is on (a reprocess/prod profile sets it — opt-in so tests and
experiment mode with base.en / ad-hoc eval models are unaffected), every ACTIVE model — the model
the CONFIGURED provider for each stage will actually run — must appear in that stage's sanctioned
set. An unsanctioned model raises :class:`UnsanctionedModelError` (stable code
``MODEL_NOT_SANCTIONED``) instead of silently building a corpus with an unvetted model — the
reprocess-once economics do not survive discovering, after 1000 episodes, that a typo'd or
un-benchmarked model produced them.
"""

from __future__ import annotations

import logging
from typing import Any, FrozenSet, List, Optional, Tuple

logger = logging.getLogger(__name__)

MODEL_NOT_SANCTIONED = "MODEL_NOT_SANCTIONED"


class UnsanctionedModelError(RuntimeError):
    """A model that is not in the registry's sanctioned set was configured for a stage.

    Carries a stable ``code`` (``MODEL_NOT_SANCTIONED``) plus the offending ``stage`` / ``field`` /
    ``model`` and the ``sanctioned`` allow-list, so callers (API layers, the CLI) can surface a
    specific, machine-identifiable error. Deliberately NOT a ``ValueError``: raised from a pydantic
    ``model_validator`` it must propagate as itself (pydantic wraps ValueError/AssertionError into a
    generic ValidationError, which would erase the type + code).
    """

    code = MODEL_NOT_SANCTIONED

    def __init__(self, *, stage: str, field: str, model: str, sanctioned: FrozenSet[str]) -> None:
        self.stage = stage
        self.field = field
        self.model = model
        self.sanctioned = sorted(sanctioned)
        super().__init__(
            f"[{self.code}] {stage} model {model!r} (config field '{field}') is not "
            f"registry-sanctioned. Sanctioned {stage} models: {self.sanctioned}. Add a StageOption "
            f"to the registry (make profiles-materialize) or use a sanctioned model."
        )


# The config field that carries the ACTIVE model for each (stage, provider).
# ``{provider}_{stage}_model`` is the common shape; DGX/deepgram/moss use their own names.
_TRANSCRIPTION_MODEL_FIELD = {
    "tailnet_dgx_whisper": "dgx_whisper_model",
    "whisper": "whisper_model",
    "moss": "moss_model",
    "openai": "openai_transcription_model",
    "gemini": "gemini_transcription_model",
    "mistral": "mistral_transcription_model",
    "deepgram": "deepgram_model",
}
_DIARIZATION_MODEL_FIELD = {
    "tailnet_dgx": "dgx_diarize_model",
    "local": "diarization_model",
    "deepgram": "deepgram_diarization_model",
}
# Summary providers whose model lives in ``{provider}_summary_model``. Local ML summarisers
# (transformers / summllama / hybrid_ml) run a mode-pinned model, not a free-form field, so they are
# not gated here.
_SUMMARY_CLOUD_PROVIDERS = frozenset(
    {"gemini", "openai", "anthropic", "deepseek", "grok", "mistral", "ollama"}
)


def sanctioned_models(stage: str) -> FrozenSet[str]:
    """Registry-sanctioned model names for ``stage`` (from its StageOptions' ``.model``)."""
    from . import model_registry as mr

    accessor = {
        "transcription": mr.get_transcription_options,
        "summary": mr.get_summary_options,
        "diarization": mr.get_diarization_options,
        "gi": mr.get_gi_options,
        "kg": mr.get_kg_options,
        "ner": mr.get_ner_options,
        "clustering": mr.get_clustering_options,
        "grounding": mr.get_grounding_options,
    }.get(stage)
    if accessor is None:
        return frozenset()
    return frozenset(o.model for o in accessor().values() if o.model)


def active_models(cfg: Any) -> List[Tuple[str, str, str]]:
    """The ``(stage, field, model)`` triples the CONFIGURED providers will actually run.

    Only the active model per stage is returned — the many ``{provider}_*_model`` fields for
    providers that are NOT selected are irrelevant (they never run), so gating them would falsely
    reject a config that merely carries a default for an unused provider.
    """
    out: List[Tuple[str, str, str]] = []

    def _add(stage: str, field: Optional[str]) -> None:
        if not field:
            return
        model = getattr(cfg, field, None)
        if isinstance(model, str) and model.strip():
            out.append((stage, field, model.strip()))

    tp = getattr(cfg, "transcription_provider", "")
    _add("transcription", _TRANSCRIPTION_MODEL_FIELD.get(tp))
    # ADR-123 coverage failover model (a whisper model on the same DGX service) — gated too.
    _add("transcription", "transcription_coverage_failover_model")

    sp = getattr(cfg, "summary_provider", "")
    if sp in _SUMMARY_CLOUD_PROVIDERS:
        _add("summary", f"{sp}_summary_model")
    _add("summary", "summary_model")  # canonical override, when set

    _add("diarization", _DIARIZATION_MODEL_FIELD.get(getattr(cfg, "diarization_provider", "")))

    return out


def assert_models_sanctioned(cfg: Any) -> None:
    """Raise :class:`UnsanctionedModelError` for the first active model not in its sanctioned set.

    No-op unless ``cfg.enforce_model_governance`` is true. Stages whose sanctioned set is empty
    (the registry declares no options for them) are skipped — governance cannot judge what the
    registry does not describe, and an empty allow-list must not reject everything.
    """
    if not getattr(cfg, "enforce_model_governance", False):
        return
    for stage, field, model in active_models(cfg):
        allowed = sanctioned_models(stage)
        if not allowed:
            logger.debug("model governance: no sanctioned %s models in registry; skipping", stage)
            continue
        if model not in allowed:
            raise UnsanctionedModelError(stage=stage, field=field, model=model, sanctioned=allowed)


__all__ = [
    "MODEL_NOT_SANCTIONED",
    "UnsanctionedModelError",
    "active_models",
    "assert_models_sanctioned",
    "sanctioned_models",
]
