"""Sentence-embedding loader for GIL evidence stack (Issue #435).

Lazy-loads a sentence-transformers model; exposes encode() and optional
cosine_similarity. Load only when GIL or an embedding-dependent feature is enabled.
"""

from __future__ import annotations

import logging
from typing import Any, cast, Iterable, List, Optional, Union

from .model_registry import ModelRegistry

logger = logging.getLogger(__name__)

_embedding_model: Optional[object] = None  # SentenceTransformer instance


def _get_device(device: Optional[str]) -> str:
    """Resolve device string; None means auto (prefer MPS/CUDA if available)."""
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


def load_embedding_model(
    model_id: str,
    device: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> object:
    """Load sentence-transformers model; return the model instance.

    Args:
        model_id: Alias (e.g. minilm-l6) or full HF ID.
        device: Device (cpu, cuda, mps) or None for auto.
        cache_dir: Optional cache directory for downloads.

    Returns:
        SentenceTransformer instance.
    """
    from sentence_transformers import SentenceTransformer

    resolved = ModelRegistry.resolve_evidence_model_id(model_id)
    dev = _get_device(device)
    logger.info("Loading embedding model %s on %s", resolved, dev)
    model = SentenceTransformer(resolved, device=dev, cache_folder=cache_dir)
    return model


def get_embedding_model(
    model_id: str,
    device: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> object:
    """Return cached embedding model or load and cache it (lazy, singleton per process)."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = load_embedding_model(model_id, device=device, cache_dir=cache_dir)
    return _embedding_model


def encode(
    texts: Union[str, List[str]],
    model_id: str,
    device: Optional[str] = None,
    cache_dir: Optional[str] = None,
    normalize: bool = True,
) -> Union[List[float], "List[List[float]]"]:
    """Encode text(s) to embedding vectors.

    Args:
        texts: Single string or list of strings.
        model_id: Model alias or full HF ID.
        device: Device or None for auto.
        cache_dir: Optional cache directory.
        normalize: If True, L2-normalize embeddings (default for cosine similarity).

    Returns:
        Single list of floats if one text; list of lists if multiple texts.
    """
    model = get_embedding_model(model_id, device=device, cache_dir=cache_dir)
    if isinstance(texts, str):
        texts = [texts]
    vectors = model.encode(texts, normalize_embeddings=normalize)

    def to_list(v: object) -> List[float]:
        if hasattr(v, "tolist"):
            return cast(List[float], getattr(v, "tolist")())
        return cast(List[float], list(cast(Iterable[Any], v)))

    if len(texts) == 1:
        return cast(Union[List[float], List[List[float]]], to_list(vectors[0]))
    return [to_list(v) for v in vectors]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two L2-normalized vectors."""
    if len(a) != len(b):
        raise ValueError("Vectors must have same length")
    dot = sum(x * y for x, y in zip(a, b))
    return float(dot)
