"""Sentence-embedding loader for GIL evidence stack (Issue #435).

Lazy-loads sentence-transformers models with a keyed per-process cache (resolved
model id + device + cache dir). Exposes encode() and optional cosine_similarity.
Load only when GIL, semantic search, or another embedding-dependent feature is enabled.
"""

from __future__ import annotations

import logging
from typing import Any, cast, Dict, Iterable, List, Optional, Tuple, Union

from .model_registry import ModelRegistry

logger = logging.getLogger(__name__)

# SentenceTransformer instances keyed by (resolved_id, device, cache_dir_key, allow_download).
_embedding_models: Dict[Tuple[str, str, str, bool], object] = {}


def _effective_cache_folder(cache_dir: Optional[str]) -> str:
    """Resolve hub cache dir for SentenceTransformer (same rules as preload).

    If *cache_dir* is omitted, use ``get_transformers_cache_dir()`` so loads hit the project
    ``.cache/huggingface/hub`` tree when present. ``cache_folder=None`` on SentenceTransformer
    would otherwise use the global HF default and miss repo-local preloads.
    """
    if isinstance(cache_dir, str) and cache_dir.strip():
        return cache_dir.strip()
    from podcast_scraper.cache.directories import get_transformers_cache_dir

    return str(get_transformers_cache_dir())


def _sentence_transformer_load_name(resolved_model_id: str) -> str:
    """Map registry/HF id to the name SentenceTransformer expects (avoids duplicate prefix)."""
    if resolved_model_id.startswith("sentence-transformers/"):
        return resolved_model_id.split("/", 1)[1]
    return resolved_model_id


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


def _cache_key(
    model_id: str,
    device: Optional[str],
    cache_dir: Optional[str],
    allow_download: bool,
) -> Tuple[str, str, str, bool]:
    resolved = ModelRegistry.resolve_evidence_model_id(model_id)
    dev = _get_device(device)
    cd = _effective_cache_folder(cache_dir)
    return (resolved, dev, cd, allow_download)


def load_embedding_model(
    model_id: str,
    device: Optional[str] = None,
    cache_dir: Optional[str] = None,
    *,
    allow_download: bool = False,
) -> object:
    """Load sentence-transformers model; return the model instance.

    Args:
        model_id: Alias (e.g. minilm-l6) or full HF ID.
        device: Device (cpu, cuda, mps) or None for auto.
        cache_dir: Optional cache directory for downloads.
        allow_download: If False (default), pass ``local_files_only=True`` so the run does
            not fetch from Hugging Face at inference time (aligns with preload policy).

    Returns:
        SentenceTransformer instance.
    """
    from sentence_transformers import SentenceTransformer

    resolved = ModelRegistry.resolve_evidence_model_id(model_id)
    load_name = _sentence_transformer_load_name(resolved)
    dev = _get_device(device)
    cache_folder = _effective_cache_folder(cache_dir)
    logger.info(
        "Loading embedding model %s on %s (cache_folder=%s)",
        load_name,
        dev,
        cache_folder,
    )
    st_kw: dict[str, Any] = {}
    if not allow_download:
        # sentence-transformers >= 3.x accepts local_files_only; 2.x does not.
        import inspect

        _st_params = set(inspect.signature(SentenceTransformer.__init__).parameters)
        if "local_files_only" in _st_params:
            st_kw["local_files_only"] = True
    # Reduce spam from sentence_transformers (e.g. redundant model-name warnings per batch).
    _st_loggers = [
        logging.getLogger("sentence_transformers"),
        logging.getLogger("sentence_transformers.SentenceTransformer"),
    ]
    _prev_levels = [lg.level for lg in _st_loggers]
    for lg in _st_loggers:
        lg.setLevel(logging.ERROR)
    try:
        model = SentenceTransformer(load_name, device=dev, cache_folder=cache_folder, **st_kw)
    finally:
        for lg, prev in zip(_st_loggers, _prev_levels):
            lg.setLevel(prev)
    return model


def get_embedding_model(
    model_id: str,
    device: Optional[str] = None,
    cache_dir: Optional[str] = None,
    *,
    allow_download: bool = False,
) -> object:
    """Return cached embedding model or load and cache it (lazy, keyed per process)."""
    key = _cache_key(model_id, device, cache_dir, allow_download)
    if key not in _embedding_models:
        _embedding_models[key] = load_embedding_model(
            model_id,
            device=device,
            cache_dir=cache_dir,
            allow_download=allow_download,
        )
    return _embedding_models[key]


def encode(
    texts: Union[str, List[str]],
    model_id: str,
    device: Optional[str] = None,
    cache_dir: Optional[str] = None,
    normalize: bool = True,
    *,
    return_numpy: bool = False,
    batch_size: int = 64,
    allow_download: bool = False,
) -> Union[List[float], List[List[float]], Any]:
    """Encode text(s) to embedding vectors.

    Args:
        texts: Single string or list of strings.
        model_id: Model alias or full HF ID.
        device: Device or None for auto.
        cache_dir: Optional cache directory.
        normalize: If True, L2-normalize embeddings (default for cosine similarity).
        return_numpy: If True, return numpy ndarray(s) from the model (no list conversion).
        batch_size: Batch size forwarded to ``model.encode`` (corpus-scale indexing).
        allow_download: Passed through to model load (default False: local_files_only).

    Returns:
        Single list of floats, list of lists, or ndarray(s) depending on input count
        and ``return_numpy``.
    """
    model = get_embedding_model(
        model_id, device=device, cache_dir=cache_dir, allow_download=allow_download
    )
    if isinstance(texts, str):
        texts_list = [texts]
    else:
        texts_list = list(texts)

    vectors = model.encode(
        texts_list,
        normalize_embeddings=normalize,
        batch_size=batch_size,
    )

    if return_numpy:
        if len(texts_list) == 1:
            return vectors[0] if hasattr(vectors, "__getitem__") else vectors
        return vectors

    def to_list(v: object) -> List[float]:
        if hasattr(v, "tolist"):
            return cast(List[float], getattr(v, "tolist")())
        return cast(List[float], list(cast(Iterable[Any], v)))

    if len(texts_list) == 1:
        row = vectors[0] if hasattr(vectors, "__getitem__") else vectors
        return cast(Union[List[float], List[List[float]]], to_list(row))
    return [to_list(v) for v in vectors]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two L2-normalized vectors."""
    if len(a) != len(b):
        raise ValueError("Vectors must have same length")
    dot = sum(x * y for x, y in zip(a, b))
    return float(dot)
