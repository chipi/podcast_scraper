"""NLI cross-encoder loader for GIL evidence stack (Issue #435).

Lazy-loads a cross-encoder for premise/hypothesis → entailment score.
Used to validate that a quote supports an insight.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from .model_registry import ModelRegistry

logger = logging.getLogger(__name__)

_nli_model: Optional[object] = None  # CrossEncoder or pipeline


def _scalar_to_float(value: object, fallback: float = 0.0) -> float:
    """Convert tensor or scalar to float; avoid .item() on meta tensors (e.g. GIL + API-only)."""
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except RuntimeError as e:
            if "meta" in str(e).lower() or "item" in str(e).lower():
                logger.debug("Score on meta device or non-materialized tensor: %s", e)
                return fallback
            raise
    if hasattr(value, "__len__") and len(value):
        return _scalar_to_float(value[0], fallback)  # type: ignore[index]
    return fallback


def _get_device(device: Optional[str]) -> str:
    if device is not None and device.strip():
        return device.strip().lower()
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def load_nli_model(
    model_id: str,
    device: Optional[str] = None,
) -> object:
    """Load NLI cross-encoder model.

    Args:
        model_id: Alias (e.g. nli-deberta-base) or full HF ID.
        device: Device (cpu, cuda, mps) or None for auto.

    Returns:
        CrossEncoder instance (sentence_transformers).
    """
    from sentence_transformers import CrossEncoder

    resolved = ModelRegistry.resolve_evidence_model_id(model_id)
    dev = _get_device(device)
    logger.info("Loading NLI model %s on %s", resolved, dev)
    model = CrossEncoder(resolved, device=dev)
    return model


def get_nli_model(
    model_id: str,
    device: Optional[str] = None,
) -> object:
    """Return cached NLI model or load and cache it (lazy, singleton per process)."""
    global _nli_model
    if _nli_model is None:
        _nli_model = load_nli_model(model_id, device=device)
    return _nli_model


def entailment_score(
    premise: str,
    hypothesis: str,
    model_id: str,
    device: Optional[str] = None,
) -> float:
    """Score entailment of hypothesis given premise (0–1, higher = more entailment).

    Args:
        premise: Evidence text (e.g. quote).
        hypothesis: Claim (e.g. insight).
        model_id: Model alias or full HF ID.
        device: Device or None for auto.

    Returns:
        Entailment score (typically raw logit or softmax for entailment label).
    """
    model = get_nli_model(model_id, device=device)
    score = model.predict([[premise, hypothesis]])
    return _scalar_to_float(score)


def entailment_scores_batch(
    pairs: List[Tuple[str, str]],
    model_id: str,
    device: Optional[str] = None,
) -> List[float]:
    """Batch premise/hypothesis pairs; returns list of entailment scores."""
    model = get_nli_model(model_id, device=device)
    scores = model.predict(pairs)
    if hasattr(scores, "tolist"):
        try:
            return [float(x) for x in scores.tolist()]
        except RuntimeError as e:
            if "meta" in str(e).lower() or "item" in str(e).lower():
                logger.debug("Scores on meta device: %s", e)
                return [0.0] * len(pairs)
            raise
    return [_scalar_to_float(s) for s in scores]
