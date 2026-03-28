"""Extractive QA pipeline for GIL evidence stack (Issue #435).

Lazy-loads a transformers question-answering pipeline; returns answer spans
with character offsets and scores. Used for "What evidence supports: {insight}?"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, cast, Dict, List, Optional

from .model_registry import ModelRegistry

logger = logging.getLogger(__name__)

_qa_pipeline: Optional[object] = None  # Pipeline instance


def _safe_score_float(value: Any, fallback: float = 0.0) -> float:
    """Convert pipeline score (tensor or scalar) to float; avoid .item() on meta tensors."""
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except RuntimeError as e:
            if "meta" in str(e).lower() or "item" in str(e).lower():
                logger.debug("QA score on meta device: %s", e)
                return fallback
            raise
    return fallback


@dataclass
class QASpan:
    """Single extractive QA result: answer text and character offsets."""

    answer: str
    start: int
    end: int
    score: float


def _get_device(device: Optional[str]) -> str:
    """Resolve device for the HF QA pipeline.

    MPS is avoided: auto device selection used to pick ``mps``, but the QA stack
    can end up on ``meta`` or hit unsupported ops; CUDA when available, else CPU.
    """
    if device is not None and device.strip():
        dev = device.strip().lower()
        if dev == "mps":
            return "cpu"
        return dev
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def load_qa_pipeline(
    model_id: str,
    device: Optional[str] = None,
) -> object:
    """Load HuggingFace QA pipeline.

    Args:
        model_id: Alias (e.g. roberta-squad2) or full HF ID.
        device: Device (cpu, cuda, mps) or None for auto.

    Returns:
        transformers Pipeline for question-answering.
    """
    from transformers import pipeline

    resolved = ModelRegistry.resolve_evidence_model_id(model_id)
    dev = _get_device(device)
    logger.info("Loading extractive QA model %s on %s", resolved, dev)
    # QA pipeline: device -1 = CPU, 0 = CUDA; MPS often unsupported for QA
    pipe_device = 0 if dev == "cuda" else -1
    pipe = pipeline(
        "question-answering",
        model=resolved,
        device=pipe_device,
    )
    return pipe


def get_qa_pipeline(
    model_id: str,
    device: Optional[str] = None,
) -> object:
    """Return cached QA pipeline or load and cache it (lazy, singleton per process)."""
    global _qa_pipeline
    if _qa_pipeline is None:
        _qa_pipeline = load_qa_pipeline(model_id, device=device)
    return _qa_pipeline


def answer(
    context: str,
    question: str,
    model_id: str,
    device: Optional[str] = None,
) -> QASpan:
    """Single question over context; returns best answer span."""
    pipe = get_qa_pipeline(model_id, device=device)
    pipe_fn: Callable[..., Dict[str, Any]] = cast(Callable[..., Dict[str, Any]], pipe)
    out = pipe_fn(question=question, context=context, max_answer_len=512)
    return QASpan(
        answer=out["answer"],
        start=out["start"],
        end=out["end"],
        score=_safe_score_float(out["score"]),
    )


def answer_multi(
    context: str,
    questions: List[str],
    model_id: str,
    device: Optional[str] = None,
) -> List[QASpan]:
    """Multiple questions over the same context; returns one span per question."""
    pipe = get_qa_pipeline(model_id, device=device)
    pipe_fn: Callable[..., Dict[str, Any]] = cast(Callable[..., Dict[str, Any]], pipe)
    results: List[QASpan] = []
    for q in questions:
        out = pipe_fn(question=q, context=context, max_answer_len=512)
        results.append(
            QASpan(
                answer=out["answer"],
                start=out["start"],
                end=out["end"],
                score=_safe_score_float(out["score"]),
            )
        )
    return results
