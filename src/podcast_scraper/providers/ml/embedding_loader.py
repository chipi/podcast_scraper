"""Sentence-embedding loader for GIL evidence stack (Issue #435).

Post-#382 Phase E: local sentence-transformers loading + cache lives on
:class:`EmbeddingEvidenceBackend` (in :mod:`.hf_evidence_backend`). The
remote/provider dispatch (Ollama endpoint, RFC-089 legacy endpoint) stays
here because it is orthogonal to the local backend abstraction.

Public API preserved: :func:`load_embedding_model`, :func:`get_embedding_model`,
:func:`encode`, :func:`cosine_similarity`.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, cast, ClassVar, Iterable, List, Optional, Union

from .hf_evidence_backend import HFEvidenceBackend

logger = logging.getLogger(__name__)


def _effective_cache_folder(cache_dir: Optional[str]) -> str:
    if isinstance(cache_dir, str) and cache_dir.strip():
        return cache_dir.strip()
    from podcast_scraper.cache.directories import get_transformers_cache_dir

    return str(get_transformers_cache_dir())


def _sentence_transformer_load_name(resolved_model_id: str) -> str:
    """Map registry/HF id to the name SentenceTransformer expects (avoids duplicate prefix)."""
    if resolved_model_id.startswith("sentence-transformers/"):
        return resolved_model_id.split("/", 1)[1]
    return resolved_model_id


class EmbeddingEvidenceBackend(HFEvidenceBackend):
    """sentence-transformers SentenceTransformer wrapper.

    Extras cache-keyed on: ``cache_dir`` (per-run override) and
    ``allow_download`` (offline vs. permissive load). This preserves the
    pre-Phase-E four-way key ``(id, device, cache_dir, allow_download)``
    used by the vector-index rebuild + LanceDB pipeline.
    """

    kind = "embedding"
    mps_supported = True

    _instances: ClassVar[dict] = {}
    _instances_lock: ClassVar[threading.Lock] = threading.Lock()

    def _load(self) -> None:
        import inspect

        from sentence_transformers import SentenceTransformer

        cache_dir = self.extras.get("cache_dir")
        allow_download = bool(self.extras.get("allow_download", False))
        load_name = _sentence_transformer_load_name(self.resolved_id)
        cache_folder = _effective_cache_folder(cache_dir)
        logger.debug(
            "Loading embedding model %s on %s (cache_folder=%s)",
            load_name,
            self.device,
            cache_folder,
        )

        st_kw: dict[str, Any] = {}
        if not allow_download:
            _st_params = set(inspect.signature(SentenceTransformer.__init__).parameters)
            if "local_files_only" in _st_params:
                st_kw["local_files_only"] = True

        _st_loggers = [
            logging.getLogger("sentence_transformers"),
            logging.getLogger("sentence_transformers.SentenceTransformer"),
        ]
        _prev_levels = [lg.level for lg in _st_loggers]
        for lg in _st_loggers:
            lg.setLevel(logging.ERROR)
        try:
            self.model = SentenceTransformer(  # nosec B615
                load_name, device=self.device, cache_folder=cache_folder, **st_kw
            )
        finally:
            for lg, prev in zip(_st_loggers, _prev_levels):
                lg.setLevel(prev)


# ---- Module-level thin wrappers (public API preserved) -------------------


def load_embedding_model(
    model_id: str,
    device: Optional[str] = None,
    cache_dir: Optional[str] = None,
    *,
    allow_download: bool = False,
) -> object:
    """Load sentence-transformers model — returns the underlying SentenceTransformer."""
    backend = EmbeddingEvidenceBackend(
        model_id,
        device=device,
        cache_dir=cache_dir,
        allow_download=allow_download,
    )
    backend._ensure_loaded()
    return backend.model


def get_embedding_model(
    model_id: str,
    device: Optional[str] = None,
    cache_dir: Optional[str] = None,
    *,
    allow_download: bool = False,
) -> object:
    """Return cached embedding model or load and cache it (keyed per process)."""
    # cache_dir default is a resolved absolute path — use the same shape when
    # caching so cache-hit works with cache_dir=None.
    resolved_cache = _effective_cache_folder(cache_dir)
    backend = EmbeddingEvidenceBackend.get_or_load(
        model_id,
        device=device,
        cache_dir=resolved_cache,
        allow_download=allow_download,
    )
    return backend.model


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
    remote_endpoint: Optional[str] = None,
    provider: Optional[str] = None,
) -> Union[List[float], List[List[float]], Any]:
    """Encode text(s) to embedding vectors — local, Ollama, or legacy remote endpoint.

    See :func:`load_embedding_model` for the local-load kwargs. Ollama /
    RFC-089-legacy dispatch happens here and doesn't touch the local backend.
    """
    resolved_provider = (provider or "sentence_transformers").strip().lower()

    if resolved_provider == "ollama":
        if not (isinstance(remote_endpoint, str) and remote_endpoint.strip()):
            raise ValueError(
                "vector_embedding_provider='ollama' requires vector_embedding_endpoint "
                "(the Ollama base URL, e.g. http://dgx:11434)."
            )
        from .embedding_ollama import encode_via_ollama

        rows = encode_via_ollama(
            texts,
            remote_endpoint.strip(),
            model_id=model_id,
            normalize=normalize,
        )
        return _shape_rows(rows, texts, return_numpy)

    if isinstance(remote_endpoint, str) and remote_endpoint.strip():
        logger.warning(
            "vector_embedding_endpoint set without provider='ollama' — using legacy DGX "
            "shim path (RFC-089 §D4, superseded by ADR-098). Switch profile to "
            "vector_embedding_provider: ollama for the new path."
        )
        from .embedding_remote import encode_via_endpoint

        rows = encode_via_endpoint(
            texts,
            remote_endpoint.strip(),
            model_id=model_id,
            normalize=normalize,
        )
        return _shape_rows(rows, texts, return_numpy)

    model = get_embedding_model(
        model_id,
        device=device,
        cache_dir=cache_dir,
        allow_download=allow_download,
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


def _shape_rows(
    rows: List[List[float]],
    texts: Union[str, List[str]],
    return_numpy: bool,
) -> Union[List[float], List[List[float]], Any]:
    if isinstance(texts, str):
        single = rows[0]
        return single if not return_numpy else __import__("numpy").array(single)
    if return_numpy:
        return __import__("numpy").array(rows)
    return rows


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two L2-normalized vectors."""
    if len(a) != len(b):
        raise ValueError("Vectors must have same length")
    dot = sum(x * y for x, y in zip(a, b))
    return float(dot)
