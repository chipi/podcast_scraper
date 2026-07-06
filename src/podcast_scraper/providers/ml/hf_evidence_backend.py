"""Shared base for GIL evidence-stack model backends (QA, NLI, embedding).

Introduced under #382 (Phase E) to collapse the three parallel loader/cache
idioms in :mod:`extractive_qa`, :mod:`nli_loader`, and :mod:`embedding_loader`
into one shape. Aligns the local evidence stack with the "just different
profiles and providers" abstraction the AI-provider stack already follows.

The base class owns:

- Device resolution (``_resolve_device``) with an opt-in ``mps_supported``
  class flag (QA coerces MPS→CPU because attention kernels are unreliable;
  NLI and embedding accept MPS/CUDA/CPU freely).
- A per-subclass instance cache keyed by ``(resolved_model_id, device, extras)``.
  Threading lock prevents concurrent loads that exhaust memory on CI (the
  meta-tensor crash we hit at ``processing_parallelism > 1``).
- Standard ``from_pretrained`` kwargs: ``cache_dir`` (from
  :func:`podcast_scraper.cache.get_transformers_cache_dir`), ``local_files_only=True``,
  ``trust_remote_code=False``, ``low_cpu_mem_usage=False``.

Subclasses provide:

- ``kind`` — one of ``"qa" | "nli" | "embedding"`` (used in cache key + logging).
- ``mps_supported`` — class-level bool.
- ``_load()`` — populate the instance with model / tokenizer state.
- Task methods (``answer_top_k``, ``predict``, ``encode``, …).

Public API from the pre-Phase-E modules (``extractive_qa.answer_candidates``,
``nli_loader.entailment_score``, ``embedding_loader.encode``, …) is preserved
via thin module-level wrappers so callers and tests need no changes.
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, Optional, Tuple

from .model_registry import ModelRegistry

logger = logging.getLogger(__name__)

# Cache key: (kind, resolved_model_id, device, sorted-tuple-of-extra-kwargs).
# The ``kind`` is included so evidence stacks in the same process can co-exist
# in one flat cache namespace (each subclass gets its own dict below).
CacheKey = Tuple[str, str, str, Tuple[Tuple[str, Any], ...]]


def resolve_evidence_device(device: Optional[str], *, mps_supported: bool = True) -> str:
    """Resolve a device string for an evidence model.

    Callable independently of a backend instance (used in cache-key
    construction before an instance is built).
    """
    if device is not None and device.strip():
        dev = device.strip().lower()
        if dev == "mps" and not mps_supported:
            return "cpu"
        return dev
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if mps_supported and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def standard_hf_load_kwargs() -> Dict[str, Any]:
    """Return the ``from_pretrained`` kwargs every evidence load uses.

    Kept as a function (not a class attr) so ``get_transformers_cache_dir()``
    is called at load time — respects mid-run ``HF_HUB_CACHE`` env changes
    the preload script makes.
    """
    from ...cache import get_transformers_cache_dir

    return {
        "cache_dir": str(get_transformers_cache_dir().resolve()),
        "local_files_only": True,
        "trust_remote_code": False,  # Security: no remote code (#379)
        "low_cpu_mem_usage": False,  # #539: lazy meta init breaks re-loads
    }


class HFEvidenceBackend(ABC):
    """Base class for GIL evidence-stack model backends.

    Concrete subclasses (:class:`QAEvidenceBackend`, :class:`NLIEvidenceBackend`,
    :class:`EmbeddingEvidenceBackend`) declare their own ``_instances`` /
    ``_instances_lock`` class attributes so each subclass owns an
    independent process-wide cache. This is by design — the caches are
    strongly typed at the subclass level (a QA backend has a QA model +
    QA-specific tokenizer configuration; conflating with an NLI backend
    would be a type error at call sites).
    """

    kind: ClassVar[str] = "evidence"
    mps_supported: ClassVar[bool] = True

    # Subclasses MUST override these to their own dict + lock. Using the
    # base class dict would silently share instances across subclasses.
    _instances: ClassVar[Dict[CacheKey, "HFEvidenceBackend"]] = {}
    _instances_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self,
        model_id: str,
        device: Optional[str] = None,
        **extras: Any,
    ) -> None:
        self.resolved_id = ModelRegistry.resolve_evidence_model_id(model_id)
        self.device = resolve_evidence_device(device, mps_supported=self.mps_supported)
        self.extras = extras
        self._loaded = False

    # ---- Cache API ----------------------------------------------------

    @classmethod
    def _build_cache_key(
        cls,
        model_id: str,
        device: Optional[str],
        extras: Optional[Dict[str, Any]] = None,
    ) -> CacheKey:
        resolved = ModelRegistry.resolve_evidence_model_id(model_id)
        dev = resolve_evidence_device(device, mps_supported=cls.mps_supported)
        extras_key = tuple(sorted((extras or {}).items()))
        return (cls.kind, resolved, dev, extras_key)

    @classmethod
    def get_or_load(
        cls,
        model_id: str,
        device: Optional[str] = None,
        **extras: Any,
    ) -> "HFEvidenceBackend":
        """Return the cached backend for (model_id, device, extras) or load one.

        Thread-safe: holding ``cls._instances_lock`` for the entire load
        prevents concurrent loads exhausting memory on CI. The one-shot
        eager load happens INSIDE the lock so callers never see a
        half-constructed instance.
        """
        key = cls._build_cache_key(model_id, device, extras)
        with cls._instances_lock:
            existing = cls._instances.get(key)
            if existing is not None:
                return existing
            logger.info(
                "Loading %s evidence model %s on %s",
                cls.kind,
                key[1],
                key[2],
            )
            inst = cls(model_id, device=device, **extras)
            inst._ensure_loaded()
            cls._instances[key] = inst
            return inst

    @classmethod
    def clear_cache(cls) -> None:
        """Drop every cached instance of this subclass (multi-feed hygiene, #539)."""
        cls._instances.clear()

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self._load()
            self._loaded = True

    # ---- Subclass contract -------------------------------------------

    @abstractmethod
    def _load(self) -> None:
        """Populate the instance with model / tokenizer state.

        Implementations should read :attr:`resolved_id`, :attr:`device`,
        :attr:`extras`, and use :func:`standard_hf_load_kwargs` for the
        ``from_pretrained`` kwargs unless they have a reason to deviate.
        """
        raise NotImplementedError
