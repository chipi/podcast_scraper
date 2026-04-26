"""Centralized ML model loading and downloading.

This module provides the ONLY place where ML models can be downloaded.
All model downloads go through these functions - libraries are never allowed
to download on their own (always use local_files_only=True when loading).

Both the CLI workflow and the preload script use these internal functions.
"""

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional

from ... import config
from ...cache import (
    get_transformers_cache_dir,
    get_whisper_cache_dir,
)
from ...utils.log_redaction import format_exception_for_log

logger = logging.getLogger(__name__)


def preload_whisper_models(model_names: Optional[List[str]] = None) -> None:
    """Preload Whisper models (centralized download function).

    This is the ONLY place where Whisper models can be downloaded.
    All other code must use local_files_only=True when loading.

    Args:
        model_names: List of Whisper model names to preload.
                    If None, uses WHISPER_MODELS env var or defaults to test default.
    """
    try:
        import whisper
    except ImportError:
        logger.error("openai-whisper not installed. Install with: pip install openai-whisper")
        raise

    if model_names is None:
        # Get from environment variable (comma-separated) or use default
        env_models = os.environ.get("WHISPER_MODELS", "").strip()
        if env_models:
            model_names = [m.strip() for m in env_models.split(",") if m.strip()]
        else:
            # Default: tiny.en for local dev/tests (smallest, fastest)
            model_names = [config.TEST_DEFAULT_WHISPER_MODEL]

    if not model_names:
        logger.debug("Skipping Whisper model preloading (no models specified)")
        return

    logger.info("Preloading Whisper models...")

    # Use get_whisper_cache_dir() which checks for local cache first,
    # then falls back to ~/.cache/whisper/ (which CI caches between jobs)
    whisper_cache = get_whisper_cache_dir()

    # Ensure cache directory exists
    whisper_cache.mkdir(parents=True, exist_ok=True)

    whisper_cache_str = str(whisper_cache)
    for model_name in model_names:
        logger.info(f"  - {model_name}...")
        try:
            model_file = whisper_cache / f"{model_name}.pt"

            # Use download_root parameter to cache to local directory
            # This is the ONLY place where whisper downloads models
            model = whisper.load_model(model_name, download_root=whisper_cache_str)
            assert model is not None, f"Model {model_name} loaded but is None"
            assert (
                hasattr(model, "dims") and model.dims is not None
            ), f"Model {model_name} missing dims attribute"

            if model_file.exists():
                file_size_mb = model_file.stat().st_size / (1024 * 1024)
                logger.info(f"  ✓ Whisper {model_name} cached ({file_size_mb:.1f} MB)")
            else:
                logger.info(f"  ✓ Whisper {model_name} cached")
        except Exception as e:
            logger.error(
                "  ✗ Failed to preload Whisper model %s: %s",
                model_name,
                format_exception_for_log(e),
            )
            raise


def preload_transformers_models(model_names: Optional[List[str]] = None) -> None:
    """Preload Transformers models (centralized download function).

    This is the ONLY place where Transformers models can be downloaded.
    All other code must use local_files_only=True when loading.

    Args:
        model_names: List of Transformers model names to preload.
                    If None, uses TRANSFORMERS_MODELS env var or defaults to common models.
    """
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError:
        logger.error("transformers not installed. Install with: pip install transformers torch")
        raise

    if model_names is None:
        # Get from environment variable (comma-separated) or use default
        env_models = os.environ.get("TRANSFORMERS_MODELS", "").strip()
        if env_models:
            model_names = [m.strip() for m in env_models.split(",") if m.strip()]
        else:
            # Default: preload only test defaults (small, fast models for local dev/testing)
            model_names = [
                config.TEST_DEFAULT_SUMMARY_MODEL,
                config.TEST_DEFAULT_SUMMARY_REDUCE_MODEL,
            ]

    if not model_names:
        logger.debug("Skipping Transformers model preloading (no models specified)")
        return

    logger.info("Preloading Transformers models...")

    # Use get_transformers_cache_dir() which respects HF_HUB_CACHE env var
    cache_dir = get_transformers_cache_dir()

    # Ensure cache directory exists
    cache_dir.mkdir(parents=True, exist_ok=True)

    for model_name in model_names:
        logger.info(f"  - {model_name}...")

        # Check if already cached
        model_cache_name = model_name.replace("/", "--")
        model_cache_path = cache_dir / f"models--{model_cache_name}"
        if model_cache_path.exists():
            existing_size = sum(
                f.stat().st_size for f in model_cache_path.rglob("*") if f.is_file()
            )
            existing_size_mb = existing_size / (1024 * 1024)
            logger.info(f"    Status: Already cached ({existing_size_mb:.1f} MB)")
        else:
            logger.info("    Status: Not cached, downloading...")

        try:
            # Apply retry policy for transient errors (Issue #379)
            from requests.exceptions import (
                ConnectionError,
                HTTPError,
                RequestException,
                Timeout,
            )

            from ...utils.retry import retry_with_exponential_backoff

            # Define retryable exceptions for model loading
            retryable_exceptions = (
                ConnectionError,
                HTTPError,
                Timeout,
                RequestException,
                OSError,  # Network/IO errors
            )

            # Retry wrapper for tokenizer loading
            # Pegasus, LongT5, FLAN-T5 (pinned rev) may not have safetensors; use PyTorch
            model_lower = model_name.lower()
            use_safetensors = (
                "pegasus" not in model_lower
                and "long-t5" not in model_lower
                and "longt5" not in model_lower
                and "flan-t5" not in model_lower
            )
            from ...config_constants import get_pinned_revision_for_model

            revision = get_pinned_revision_for_model(model_name)
            tokenizer_kw: dict = {
                "cache_dir": str(cache_dir),
                "local_files_only": False,  # nosec B615
                "use_safetensors": use_safetensors,
            }
            if revision:
                tokenizer_kw["revision"] = revision
            model_kw: dict = {
                "cache_dir": str(cache_dir),
                "local_files_only": False,  # nosec B615
                "use_safetensors": use_safetensors,
            }
            if revision:
                model_kw["revision"] = revision

            def _load_tokenizer():
                return AutoTokenizer.from_pretrained(model_name, **tokenizer_kw)  # nosec B615

            # Retry wrapper for model loading
            def _load_model():
                return AutoModelForSeq2SeqLM.from_pretrained(model_name, **model_kw)  # nosec B615

            # This is the ONLY place where transformers downloads models
            # Use retry with exponential backoff for transient errors
            retry_with_exponential_backoff(
                _load_tokenizer,
                max_retries=3,
                initial_delay=1.0,
                max_delay=30.0,
                retryable_exceptions=retryable_exceptions,
            )

            retry_with_exponential_backoff(
                _load_model,
                max_retries=3,
                initial_delay=1.0,
                max_delay=30.0,
                retryable_exceptions=retryable_exceptions,
            )

            # Calculate final size after download
            if model_cache_path.exists():
                total_size = sum(
                    f.stat().st_size for f in model_cache_path.rglob("*") if f.is_file()
                )
                size_mb = total_size / (1024 * 1024)
                logger.info(f"  ✓ Downloaded and cached: {size_mb:.1f} MB")
            else:
                logger.info(f"  ✓ Downloaded and cached: {model_name}")
        except Exception as e:
            logger.error(
                "  ✗ Failed to preload Transformers model %s: %s",
                model_name,
                format_exception_for_log(e),
            )
            raise


def build_huggingface_qa_pipeline(
    model_id: str,
    *,
    device: int,
    local_files_only: bool,
) -> Any:
    """Instantiate a Hugging Face ``question-answering`` pipeline with our cache dir.

    Newer ``transformers`` merges a top-level ``local_files_only`` into hub kwargs; if the
    same flag is also inside ``model_kwargs``, ``AutoConfig.from_pretrained`` raises
    "multiple values for keyword argument 'local_files_only'". Older versions expect the
    flag only in ``model_kwargs``. Preload (downloads allowed) uses cache_dir only; offline
    load tries ``model_kwargs`` first, then the top-level argument.

    Args:
        model_id: Hub id or local path passed to ``pipeline(model=...)``.
        device: ``pipeline`` device index (e.g. ``-1`` for CPU).
        local_files_only: If True, do not hit the hub (cache must be populated).

    Returns:
        The transformers QA pipeline instance.
    """
    cache_dir = str(get_transformers_cache_dir().resolve())
    # low_cpu_mem_usage=False avoids lazy meta-device init paths on some torch/transformers
    # pairs that break a second Whisper load in the same process (GitHub #539).
    if not local_files_only:
        return _call_transformers_qa_pipeline(
            "question-answering",
            model=model_id,
            device=device,
            model_kwargs={"cache_dir": cache_dir, "low_cpu_mem_usage": False},
        )
    model_kw: dict[str, Any] = {
        "local_files_only": True,
        "cache_dir": cache_dir,
        "low_cpu_mem_usage": False,
    }
    try:
        return _call_transformers_qa_pipeline(
            "question-answering",
            model=model_id,
            device=device,
            model_kwargs=model_kw,
        )
    except TypeError as exc:
        if "multiple values" not in str(exc) or "local_files_only" not in str(exc):
            raise
        return _call_transformers_qa_pipeline(
            "question-answering",
            model=model_id,
            device=device,
            model_kwargs={"cache_dir": cache_dir, "low_cpu_mem_usage": False},
            local_files_only=True,
        )


def _call_transformers_qa_pipeline(*args: Any, **kwargs: Any) -> Any:
    """Module-level indirection for ``transformers.pipeline``.

    Tests must patch *this* symbol — not ``transformers.pipeline``. The
    top-level ``transformers`` module is a ``_LazyModule`` (transformers
    >= 4.40) whose ``__getattr__`` resolves ``pipeline`` from a submodule,
    so ``monkeypatch.setattr("transformers.pipeline", fake)`` is silently
    bypassed by the function-local ``from transformers import pipeline``
    inside :func:`build_huggingface_qa_pipeline`. Issue #677.
    """
    from transformers import pipeline

    return pipeline(*args, **kwargs)


def _download_qa_pipeline_for_cache(model_id: str) -> None:
    """Run transformers QA pipeline once to populate the HF cache.

    Extracted as a module-level hook so unit tests can patch it without
    fighting ``from transformers import pipeline`` name binding.
    """
    build_huggingface_qa_pipeline(model_id, device=-1, local_files_only=False)


def _download_nli_cross_encoder_for_cache(model_id: str) -> None:
    """Instantiate CrossEncoder once to populate the HF cache (test seam)."""
    import inspect

    from sentence_transformers import CrossEncoder

    cache_dir = str(get_transformers_cache_dir().resolve())
    # ST 2.x CrossEncoder doesn't accept local_files_only or cache_folder.
    ce_params = set(inspect.signature(CrossEncoder.__init__).parameters)
    ce_kw: dict = {}
    if "local_files_only" in ce_params:
        ce_kw["local_files_only"] = False
    if "cache_folder" in ce_params:
        ce_kw["cache_folder"] = cache_dir
    CrossEncoder(model_id, **ce_kw)


def _download_sentence_transformer_for_cache(model_id: str) -> None:
    """Instantiate SentenceTransformer once to populate the HF cache (test seam)."""
    from sentence_transformers import SentenceTransformer

    cache_dir = str(get_transformers_cache_dir().resolve())
    SentenceTransformer(model_id, cache_folder=cache_dir)


def is_evidence_model_cached(model_id: str) -> bool:
    """Return True if the given evidence-stack model is already in the HF hub cache dir.

    Covers embedding (sentence-transformers), extractive QA, and NLI checkpoints.

    Args:
        model_id: Alias (e.g. minilm-l6, roberta-squad2) or full HF ID.

    Returns:
        True if the resolved model exists under the transformers/HF cache.
    """
    from .model_registry import ModelRegistry

    try:
        resolved = ModelRegistry.resolve_evidence_model_id(model_id)
    except ValueError:
        return False
    cache_dir = get_transformers_cache_dir()
    model_cache_name = resolved.replace("/", "--")
    model_cache_path = cache_dir / f"models--{model_cache_name}"
    return model_cache_path.exists()


def preload_evidence_models(
    qa_models: Optional[List[str]] = None,
    nli_models: Optional[List[str]] = None,
    embedding_models: Optional[List[str]] = None,
) -> None:
    """Preload GIL evidence-stack models (embedding, QA, NLI) to the central cache.

    This is the ONLY place where evidence-stack models can be downloaded.
    Uses the same HF cache as Transformers (get_transformers_cache_dir).

    Args:
        qa_models: List of QA model aliases or HF IDs; if None along with the
            other two arguments, uses default from config_constants; if this
            call passes any explicit list (possibly empty), omitted arguments
            are treated as empty lists (partial preload for setup.py).
        nli_models: List of NLI model aliases or HF IDs; same semantics as qa_models.
        embedding_models: Sentence-transformers embedding IDs; same semantics.
    """
    from ... import config_constants
    from .model_registry import ModelRegistry

    all_none = qa_models is None and nli_models is None and embedding_models is None
    if all_none:
        qa_models = [config_constants.DEFAULT_EXTRACTIVE_QA_MODEL]
        nli_models = [config_constants.DEFAULT_NLI_MODEL]
        embedding_models = [config_constants.DEFAULT_EMBEDDING_MODEL]
    else:
        qa_models = list(qa_models or [])
        nli_models = list(nli_models or [])
        embedding_models = list(embedding_models or [])

    cache_dir = get_transformers_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_dir_str = str(cache_dir.resolve())
    if os.environ.get("HF_HUB_CACHE") != cache_dir_str:
        os.environ["HF_HUB_CACHE"] = cache_dir_str
        logger.info("Set HF_HUB_CACHE for evidence model preload: %s", cache_dir_str)

    for alias in embedding_models:
        resolved = ModelRegistry.resolve_evidence_model_id(alias)
        model_cache_path = cache_dir / f"models--{resolved.replace('/', '--')}"
        if model_cache_path.exists():
            logger.info("  Embedding model %s already cached", resolved)
            continue
        logger.info("  Preloading embedding model %s...", resolved)
        try:
            _download_sentence_transformer_for_cache(resolved)
            logger.info("  ✓ Embedding model %s cached", resolved)
        except Exception as e:
            logger.error(
                "  ✗ Failed to preload embedding model %s: %s",
                resolved,
                format_exception_for_log(e),
            )
            raise

    for alias in qa_models:
        resolved = ModelRegistry.resolve_evidence_model_id(alias)
        model_cache_path = cache_dir / f"models--{resolved.replace('/', '--')}"
        if model_cache_path.exists():
            logger.info("  QA model %s already cached", resolved)
            continue
        logger.info("  Preloading QA model %s...", resolved)
        try:
            _download_qa_pipeline_for_cache(resolved)
            logger.info("  ✓ QA model %s cached", resolved)
        except Exception as e:
            logger.error(
                "  ✗ Failed to preload QA model %s: %s",
                resolved,
                format_exception_for_log(e),
            )
            raise

    for alias in nli_models:
        resolved = ModelRegistry.resolve_evidence_model_id(alias)
        model_cache_path = cache_dir / f"models--{resolved.replace('/', '--')}"
        if model_cache_path.exists():
            logger.info("  NLI model %s already cached", resolved)
            continue
        logger.info("  Preloading NLI model %s...", resolved)
        try:
            _download_nli_cross_encoder_for_cache(resolved)
            logger.info("  ✓ NLI model %s cached", resolved)
        except Exception as e:
            logger.error(
                "  ✗ Failed to preload NLI model %s: %s",
                resolved,
                format_exception_for_log(e),
            )
            raise
