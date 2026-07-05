"""Shared HF seq2seq loader + generator (BART / LED / Pegasus / LongT5 / FLAN-T5).

Introduced under #382 (Phase F) to collapse the two parallel "load HF
seq2seq checkpoint + generate summary" idioms that lived in
:mod:`summarizer` (as ``SummaryModel``) and :mod:`hybrid_ml_provider`
(as ``TransformersReduceBackend``). Both consumers now instantiate an
:class:`HFSeq2SeqBackend` and lean on it for:

- Snapshot-first checkpoint loading (avoids transformers checkpoint-
  discovery bugs on PyTorch-only cache — see #539 lineage).
- Model-family dispatch (Pegasus / LED / BART / AutoModel).
- Device placement with meta-tensor + OOM fallback semantics.
- ``model.generate()`` wrapped by ``GenerationConfig``.

Unlike :class:`HFEvidenceBackend`, this class does NOT expose a shared
process-wide cache — every ``SummaryModel`` / ``TransformersReduceBackend``
instance is already the caller's cache handle (they own the lifecycle
of the loaded weights, cleaned up between feeds via
``clear_qa_pipeline_cache`` / ``cleanup()``).

Consumers keep their existing public shapes; only the load + generate
plumbing moves in.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


def _detect_default_device() -> str:
    """Auto-detect device — MPS > CUDA > CPU."""
    try:
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _default_use_safetensors(model_id: str) -> bool:
    """Model families whose HF cache historically lacks a safetensors file.

    Pegasus / LongT5 / LED / FLAN-T5 (pinned revisions) — offline load
    with ``use_safetensors=True`` triggers API calls that break
    ``local_files_only=True``. Mirror the pre-Phase-F exclusion list.
    """
    lower = model_id.lower()
    if "pegasus" in lower or "long-t5" in lower or "longt5" in lower:
        return False
    if "flan-t5" in lower:
        return False
    if "led" in lower or "longformer" in lower:
        return False
    return True


class HFSeq2SeqBackend:
    """Load + generate for an HF seq2seq checkpoint (BART / LED / Pegasus / T5 family).

    Instance lifecycle:

    1. ``__init__`` — store config; nothing loaded yet.
    2. ``load()`` — populate ``self.model`` and ``self.tokenizer``; move
       model to ``self.device``; call ``.eval()``.
    3. ``generate(...)`` — tokenize → ``model.generate()`` → decode.
    4. ``to(device)`` — move already-loaded model (OOM fallback).
    5. ``unload()`` — drop references for GC.

    Callers construct this per model they intend to use. The pre-Phase-F
    call sites (``SummaryModel``, ``TransformersReduceBackend``) hold one
    backend each and reuse it across many ``generate()`` calls.
    """

    def __init__(
        self,
        model_id: str,
        *,
        device: Optional[str] = None,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        use_safetensors: Optional[bool] = None,
        low_cpu_mem_usage: bool = False,
        family_class: Optional[Callable[..., Any]] = None,
        retry_wrapper: Optional[Callable[[Callable[[], Any], str, str], Any]] = None,
    ) -> None:
        self.model_id = model_id
        self.device = device or _detect_default_device()
        self.revision = revision
        self.cache_dir = cache_dir
        self.use_safetensors = (
            use_safetensors if use_safetensors is not None else _default_use_safetensors(model_id)
        )
        # ``low_cpu_mem_usage=False`` avoids the lazy meta-init path that leaves tied
        # weights on the meta device and breaks a second-load-in-process (#539).
        self.low_cpu_mem_usage = low_cpu_mem_usage
        # Optional model-family class override; defaults to AutoModelForSeq2SeqLM.
        # SummaryModel passes BartForConditionalGeneration / LEDForConditionalGeneration
        # explicitly so we get the specialised class-init path (and its warnings).
        self.family_class = family_class
        # Optional retry wrapper — SummaryModel passes ``_load_with_retry_summarizer``
        # to keep the cache-clear-and-retry behavior on OSError/safetensors mismatch.
        # If None, from_pretrained is called directly.
        self._retry_wrapper = retry_wrapper
        self.model: Any = None
        self.tokenizer: Any = None
        self._loaded = False

    # ---- Loading ------------------------------------------------------

    def load(self) -> None:
        """Load tokenizer + model onto ``self.device``.

        Snapshot-first: if the pinned revision snapshot exists in the
        transformers cache tree, load from the resolved directory to
        avoid checkpoint-file discovery bugs (transformers has raised
        ``checkpoint_files[0] is None`` when a PyTorch-only cache is
        probed with mixed safetensors expectations).
        """
        if self._loaded:
            return

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        from ...cache import get_transformers_cache_dir, get_transformers_snapshot_path

        effective_cache_dir = self.cache_dir
        if effective_cache_dir is None:
            effective_cache_dir = str(get_transformers_cache_dir())
        cache_path = Path(effective_cache_dir)

        # Tokenizer — always from the (repo id + optional revision) path.
        tokenizer_kw: Dict[str, Any] = {
            "cache_dir": effective_cache_dir,
            "local_files_only": True,
            "trust_remote_code": False,
            "use_safetensors": self.use_safetensors,
        }
        if self.revision:
            tokenizer_kw["revision"] = self.revision
        load_tokenizer: Callable[[], Any] = lambda: AutoTokenizer.from_pretrained(  # noqa: E731
            self.model_id, **tokenizer_kw  # nosec B615
        )
        self.tokenizer = self._maybe_retry(load_tokenizer, "tokenizer")

        # Model — try snapshot-first, fall back to (repo id + revision).
        from typing import cast as _cast

        model_class = _cast(Any, self.family_class or AutoModelForSeq2SeqLM)
        snapshot_path = get_transformers_snapshot_path(
            self.model_id, revision=self.revision, cache_dir=cache_path
        )
        if snapshot_path is None and self.revision:
            snapshot_path = get_transformers_snapshot_path(
                self.model_id, revision=None, cache_dir=cache_path
            )

        if snapshot_path is not None:
            snapshot_str = str(snapshot_path.resolve())
            load_model: Callable[[], Any] = lambda: model_class.from_pretrained(  # noqa: E731
                snapshot_str,
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=self.use_safetensors,
                low_cpu_mem_usage=self.low_cpu_mem_usage,
            )
            try:
                self.model = self._maybe_retry(load_model, "model (snapshot)")
            except OSError as e:
                if "no file named" not in str(e):
                    raise
                # Pinned snapshot may lack weights; try main.
                fallback = get_transformers_snapshot_path(
                    self.model_id, revision=None, cache_dir=cache_path
                )
                if fallback is None or fallback == snapshot_path:
                    raise
                logger.warning(
                    "Snapshot %s missing model weights (%s); loading from main.",
                    snapshot_path.name,
                    e,
                )
                fallback_str = str(fallback.resolve())
                load_fallback: Callable[[], Any] = (
                    lambda: model_class.from_pretrained(  # noqa: E731
                        fallback_str,
                        local_files_only=True,
                        trust_remote_code=False,
                        use_safetensors=True,  # main usually has safetensors
                        low_cpu_mem_usage=self.low_cpu_mem_usage,
                    )
                )
                self.model = self._maybe_retry(load_fallback, "model (fallback)")
        else:
            model_kw: Dict[str, Any] = {
                "cache_dir": effective_cache_dir,
                "local_files_only": True,
                "trust_remote_code": False,
                "use_safetensors": self.use_safetensors,
                "low_cpu_mem_usage": self.low_cpu_mem_usage,
            }
            if self.revision:
                model_kw["revision"] = self.revision
            load_model_repo: Callable[[], Any] = lambda: model_class.from_pretrained(  # noqa: E731
                self.model_id, **model_kw  # nosec B615
            )
            self.model = self._maybe_retry(load_model_repo, "model")

        # Move to device with the meta-tensor + OOM fallback semantics we've
        # relied on for MPS-first machines.
        self.to(self.device, initial=True)
        self.model.eval()
        self._loaded = True

    def _maybe_retry(self, fn: Callable[[], Any], what: str) -> Any:
        """Run ``fn`` under the caller's retry wrapper if provided; else call directly."""
        if self._retry_wrapper is not None:
            return self._retry_wrapper(fn, self.model_id, what)
        return fn()

    # ---- Device handling ----------------------------------------------

    def to(self, device: str, *, initial: bool = False) -> None:
        """Move the loaded model to ``device`` with meta-tensor + OOM fallback.

        ``initial=True`` (from :meth:`load`) allows a device-mismatch
        exception (MPS/CUDA unavailable at runtime) to fall back to CPU.
        ``initial=False`` (from callers doing OOM recovery) narrows the
        fallback to "OOM-like" errors only — anything else re-raises so
        the caller can decide.
        """
        if self.model is None:
            raise RuntimeError("Backend not loaded; call load() first")

        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()):
            try:
                self.model = self.model.to(device)
                self.device = device
                return
            except NotImplementedError as e:
                if "meta tensor" in str(e).lower():
                    logger.warning(
                        "Model has meta-backed tied weights; re-tying and retrying %s on %s",
                        self.model_id,
                        device,
                    )
                    if hasattr(self.model, "tie_weights"):
                        self.model.tie_weights()
                    self.model = self.model.to(device)
                    self.device = device
                    return
                raise
            except (RuntimeError, Exception) as e:
                error_msg = str(e).lower()
                oom_like = (
                    "out of memory" in error_msg
                    or "invalid buffer size" in error_msg
                    or "not implemented" in error_msg
                    or "unsupported" in error_msg
                )
                fallback_allowed = initial or (device in ("mps", "cuda") and oom_like)
                if fallback_allowed and device != "cpu":
                    logger.warning(
                        "Device fallback: %s failed (%s). Falling back to CPU.",
                        device,
                        e,
                    )
                    self.model = self.model.to("cpu")
                    self.device = "cpu"
                    return
                raise

    # ---- Generation ---------------------------------------------------

    def generate(
        self,
        input_text: str,
        gen_config: Any,
        *,
        max_input_tokens: Optional[int] = None,
        truncation: bool = True,
    ) -> str:
        """Tokenize ``input_text`` → ``model.generate(...)`` → decode → strip.

        ``gen_config`` is a fully-populated
        :class:`transformers.GenerationConfig` — callers decide beam count,
        length caps, ngram penalties, etc.

        ``max_input_tokens=None`` uses the model's configured encoder max
        (``max_position_embeddings`` or ``max_encoder_position_embeddings``
        with a 1024 fallback), matching the pre-Phase-F behavior.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Backend not loaded; call load() first")

        import torch

        if max_input_tokens is None:
            max_input_tokens = (
                getattr(self.model.config, "max_position_embeddings", None)
                or getattr(self.model.config, "max_encoder_position_embeddings", None)
                or 1024
            )

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=truncation,
            max_length=max_input_tokens,
        )
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, generation_config=gen_config)
        decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return str(decoded).strip()

    # ---- External-load adoption --------------------------------------

    def adopt(self, model: Any, tokenizer: Any, device: Optional[str] = None) -> None:
        """Mark the backend as loaded using externally-instantiated artifacts.

        Used by :class:`SummaryModel` for the Pegasus code path — that
        loader has custom missing-keys validation semantics that don't
        map cleanly onto :meth:`load`, so the caller instantiates model
        + tokenizer their own way and hands them here.

        After adoption, :meth:`generate` / :meth:`to` / :meth:`unload`
        work exactly as if :meth:`load` had populated the state.
        """
        self.model = model
        self.tokenizer = tokenizer
        if device is not None:
            self.device = device
        if self.model is not None and hasattr(self.model, "eval"):
            self.model.eval()
        self._loaded = True

    # ---- Teardown -----------------------------------------------------

    def unload(self) -> None:
        """Drop model + tokenizer references (multi-feed hygiene, #539)."""
        self.model = None
        self.tokenizer = None
        self._loaded = False
