"""Hybrid MAP-REDUCE provider.

Implements issue #352: classic summarizer MAP phase (compression) and
instruction-tuned REDUCE phase (abstraction + structure).

Initial implementation focuses on:
- MAP: local classic seq2seq model via existing summarizer helpers (LED/LongT5/etc.)
- REDUCE Tier 1: FLAN-T5 via transformers (no Ollama/llama.cpp required)
- REDUCE Tier 2: optional Ollama backend reusing existing Ollama provider

This is intentionally minimal and testable: unit tests mock model loading and
the summarizer helpers to avoid downloading large models in CI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, cast, Dict, List, Optional, Protocol

from ...cleaning import HybridCleaner, LLMBasedCleaner, PatternBasedCleaner
from ...cleaning.base import TranscriptCleaningProcessor
from ...preprocessing.profiles import apply_profile_with_stats
from ...summarization.base import SummarizationProvider
from ...utils.log_redaction import format_exception_for_log
from ...utils.protocol_verification import verify_protocol_compliance
from ..ollama.ollama_provider import OllamaProvider
from . import summarizer
from .model_registry import ModelRegistry
from .summarizer import SummaryModel

logger = logging.getLogger(__name__)


class InferenceBackend(Protocol):
    """Backend interface for REDUCE inference (instruction-tuned)."""

    def initialize(self) -> None:
        """Load models and prepare the backend for REDUCE inference."""
        ...

    def reduce(
        self,
        notes: str,
        instruction: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> HybridReduceResult:
        """Run instruction-tuned reduction on MAP-phase notes."""
        ...

    def cleanup(self) -> None:
        """Release resources held by the backend."""
        ...


@dataclass(frozen=True)
class HybridReduceResult:
    """Result from the REDUCE backend (for metadata and debugging)."""

    text: str
    backend: str
    model: str


class TransformersReduceBackend:
    """Tier 1 REDUCE backend using transformers (e.g. FLAN-T5)."""

    def __init__(
        self,
        model_name: str,
        device: Optional[str],
        cache_dir: Optional[str],
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self._pipeline: Any = None

    def initialize(self) -> None:
        """Load tokenizer and seq2seq model; build the text2text-generation pipeline."""
        if self._pipeline is not None:
            return

        # Lazy imports to keep base imports light
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

        device = self.device
        if device is None:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        # Use the same offline-by-default behavior as the rest of the local ML stack.
        from pathlib import Path

        from ...cache import (
            get_transformers_cache_dir,
            get_transformers_snapshot_path,
        )
        from ...config_constants import get_pinned_revision_for_model

        effective_cache_dir = self.cache_dir
        if effective_cache_dir is None:
            effective_cache_dir = str(get_transformers_cache_dir())
        cache_path = Path(effective_cache_dir)

        revision = get_pinned_revision_for_model(self.model_name)
        # FLAN-T5 pinned revision may be PyTorch-only (no safetensors)
        model_lower = self.model_name.lower()
        use_safetensors = "flan-t5" not in model_lower
        tokenizer_kw: Dict[str, Any] = dict(
            cache_dir=effective_cache_dir,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=use_safetensors,
        )
        if revision:
            tokenizer_kw["revision"] = revision
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, **tokenizer_kw)  # nosec B615

        # Prefer loading from snapshot directory to avoid transformers checkpoint
        # discovery bugs (e.g. checkpoint_files[0] None with PyTorch-only cache).
        snapshot_path = get_transformers_snapshot_path(
            self.model_name, revision=revision, cache_dir=cache_path
        )
        if snapshot_path is None and revision:
            snapshot_path = get_transformers_snapshot_path(
                self.model_name, revision=None, cache_dir=cache_path
            )
        if snapshot_path is not None:
            # Resolve to absolute path so symlinks in the snapshot resolve correctly
            snapshot_str = str(snapshot_path.resolve())
            try:
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    snapshot_str,
                    local_files_only=True,
                    trust_remote_code=False,
                    use_safetensors=use_safetensors,
                )  # nosec B615
            except OSError as e:
                if "no file named" in str(e):
                    # Pinned revision snapshot may exist but lack weights; try main
                    fallback = get_transformers_snapshot_path(
                        self.model_name, revision=None, cache_dir=cache_path
                    )
                    if fallback is not None and fallback != snapshot_path:
                        logger.warning(
                            "Snapshot %s missing model weights (%s); loading from main.",
                            snapshot_path.name,
                            e,
                        )
                        # Main snapshot often has safetensors; try True for fallback
                        model = AutoModelForSeq2SeqLM.from_pretrained(
                            str(fallback.resolve()),
                            local_files_only=True,
                            trust_remote_code=False,
                            use_safetensors=True,
                        )  # nosec B615
                    else:
                        raise
                else:
                    raise
        else:
            model_kw: Dict[str, Any] = dict(
                cache_dir=effective_cache_dir,
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=use_safetensors,
            )
            if revision:
                model_kw["revision"] = revision
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, **model_kw)  # nosec B615

        if device in ("mps", "cuda"):
            model = model.to(device)
        pipeline_device = 0 if device == "cuda" else "mps" if device == "mps" else -1
        # transformers 5.x overloads are stricter; runtime call is valid
        _pipe = cast(Any, pipeline)
        self._pipeline = _pipe(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            device=pipeline_device,
        )

    def reduce(
        self,
        notes: str,
        instruction: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> HybridReduceResult:
        """Run seq2seq generation on instruction plus NOTES (MAP output)."""
        if self._pipeline is None:
            raise RuntimeError(
                "TransformersReduceBackend not initialized. Call initialize() first."
            )

        params = params or {}
        max_new_tokens = int(params.get("max_new_tokens") or 600)
        num_beams = int(params.get("num_beams") or 4)
        do_sample = bool(params.get("do_sample") or False)

        prompt = f"{instruction.strip()}\n\nNOTES:\n{notes.strip()}"
        outputs = self._pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
        )
        if not outputs:
            return HybridReduceResult(text="", backend="transformers", model=self.model_name)
        # transformers returns list[{"generated_text": "..."}]
        return HybridReduceResult(
            text=str(outputs[0].get("generated_text", "")).strip(),
            backend="transformers",
            model=self.model_name,
        )

    def cleanup(self) -> None:
        """Drop the pipeline reference for garbage collection."""
        # Best-effort cleanup (avoid heavy torch imports if not needed)
        self._pipeline = None


class OllamaReduceBackend:
    """Tier 2 REDUCE backend using the existing Ollama provider."""

    def __init__(self, cfg: Any, model_name: str) -> None:
        self._cfg = cfg
        self._model_name = model_name
        self._provider: Optional[OllamaProvider] = None

    def initialize(self) -> None:
        """Construct and initialize ``OllamaProvider`` for the configured reduce model."""
        if self._provider is not None:
            return

        # Reuse OllamaProvider implementation by constructing a cfg copy that uses ollama.
        cfg = self._cfg.model_copy(
            update={
                "summary_provider": "ollama",
                "ollama_summary_model": self._model_name,
            }
        )
        self._provider = OllamaProvider(cfg)
        self._provider.initialize()

    def reduce(
        self,
        notes: str,
        instruction: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> HybridReduceResult:
        """Call Ollama summarization with instruction as prompt and notes as text."""
        if self._provider is None:
            raise RuntimeError("OllamaReduceBackend not initialized. Call initialize() first.")
        user_params = dict(params or {})
        # OllamaProvider uses {max_length,min_length}.
        # Map from local-style reduce params if present.
        if "max_length" not in user_params and "max_new_tokens" in user_params:
            user_params["max_length"] = user_params["max_new_tokens"]
        if "min_length" not in user_params and "min_new_tokens" in user_params:
            user_params["min_length"] = user_params["min_new_tokens"]

        result = self._provider.summarize(
            text=notes,
            params={"prompt": instruction, **user_params},
        )
        return HybridReduceResult(
            text=str(result.get("summary", "")).strip(),
            backend="ollama",
            model=self._model_name,
        )

    def cleanup(self) -> None:
        """Release the Ollama provider instance."""
        if self._provider is not None:
            self._provider.cleanup()
        self._provider = None


class LlamaCppReduceBackend:
    """Tier 2 REDUCE via llama.cpp (GGUF). Requires ``pip install podcast-scraper[ml]``."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
    ) -> None:
        self.model_path = model_path
        self.n_ctx = n_ctx
        self._llm: Any = None

    def initialize(self) -> None:
        """Load the GGUF model via llama-cpp-python."""
        if self._llm is not None:
            return
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise ImportError(
                "llama-cpp-python is required for hybrid_reduce_backend='llama_cpp'. "
                "Install with: pip install podcast-scraper[ml]"
            ) from e

        self._llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            verbose=False,
        )

    def reduce(
        self,
        notes: str,
        instruction: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> HybridReduceResult:
        """Generate text from instruction and notes using the loaded Llama model."""
        if self._llm is None:
            raise RuntimeError("LlamaCppReduceBackend not initialized. Call initialize() first.")
        params = params or {}
        max_tokens = int(params.get("max_new_tokens", params.get("max_tokens", 512)))
        prompt = f"{instruction}\n\n{notes}"
        out = self._llm(
            prompt,
            max_tokens=max_tokens,
            stop=["</s>", "\n\n\n"],
            echo=False,
        )
        text = ""
        if out and "choices" in out and len(out["choices"]) > 0:
            text = (out["choices"][0].get("text") or "").strip()
        return HybridReduceResult(
            text=text,
            backend="llama_cpp",
            model=self.model_path,
        )

    def cleanup(self) -> None:
        """Drop the llama.cpp handle."""
        self._llm = None


class HybridMLProvider:
    """Hybrid MAP-REDUCE summarization provider."""

    cleaning_processor: TranscriptCleaningProcessor  # Type annotation for mypy

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self._initialized = False
        self._map_model: Optional[SummaryModel] = None
        self._reduce_backend: Optional[InferenceBackend] = None
        # Issue #387: same attribute as MLProvider so metadata reconciliation reuses NER
        self._spacy_nlp: Optional[Any] = None

        cleaning_strategy = getattr(cfg, "transcript_cleaning_strategy", "hybrid")
        if cleaning_strategy == "pattern":
            self.cleaning_processor = PatternBasedCleaner()  # type: ignore[assignment]
        elif cleaning_strategy == "llm":
            self.cleaning_processor = LLMBasedCleaner()  # type: ignore[assignment]
        else:
            self.cleaning_processor = HybridCleaner()  # type: ignore[assignment]

        verify_protocol_compliance(self, SummarizationProvider, "SummarizationProvider")

    def initialize(self) -> None:
        """Load the MAP model and construct and initialize the REDUCE backend."""
        if self._initialized:
            return

        map_model_name = summarizer.resolve_model_name(str(self.cfg.hybrid_map_model))
        map_device = (
            self.cfg.hybrid_map_device or self.cfg.summarization_device or self.cfg.summary_device
        )
        self._map_model = SummaryModel(
            model_name=map_model_name,
            device=map_device,
            cache_dir=getattr(self.cfg, "summary_cache_dir", None),
        )

        reduce_backend = getattr(self.cfg, "hybrid_reduce_backend", "transformers")
        reduce_model = str(getattr(self.cfg, "hybrid_reduce_model", "google/flan-t5-base"))
        reduce_device = (
            self.cfg.hybrid_reduce_device
            or self.cfg.summarization_device
            or self.cfg.summary_device
        )

        if reduce_backend == "transformers":
            self._reduce_backend = TransformersReduceBackend(
                model_name=reduce_model,
                device=reduce_device,
                cache_dir=getattr(self.cfg, "summary_cache_dir", None),
            )
        elif reduce_backend == "ollama":
            self._reduce_backend = OllamaReduceBackend(cfg=self.cfg, model_name=reduce_model)
        elif reduce_backend == "llama_cpp":
            # hybrid_reduce_model = path to GGUF file (e.g. /path/to/model.gguf)
            n_ctx = int(getattr(self.cfg, "hybrid_llama_n_ctx", 4096) or 4096)
            self._reduce_backend = LlamaCppReduceBackend(
                model_path=reduce_model,
                n_ctx=n_ctx,
            )
        else:
            raise ValueError(
                f"Unsupported hybrid_reduce_backend={reduce_backend!r}. "
                "Supported: transformers, ollama, llama_cpp."
            )

        self._reduce_backend.initialize()

        self._spacy_nlp = None
        if getattr(self.cfg, "auto_speakers", False):
            from . import speaker_detection

            logger.debug("Initializing spaCy NER for hybrid ML (reconciliation reuse)")
            self._spacy_nlp = speaker_detection.get_ner_model(self.cfg)

        self._initialized = True

    def summarize(
        self,
        text: str,
        episode_title: Optional[str] = None,
        episode_description: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        pipeline_metrics: Any = None,
        call_metrics: Any = None,
    ) -> Dict[str, Any]:
        """Chunk and summarize transcript (MAP), then instruction-tuned REDUCE to final summary.

        ``params["preprocessing_profile"]`` selects the registered preprocessing profile
        applied inside this method (default ``cleaning_v4``). The workflow passes
        ``cleaning_hybrid_after_pattern`` when ``transcript_cleaning_strategy`` is
        ``pattern`` so sponsor/outro work is not duplicated (Issue #419).
        """
        if not self._initialized or self._map_model is None or self._reduce_backend is None:
            raise RuntimeError("HybridMLProvider not initialized. Call initialize() first.")

        params = params or {}

        # Resolution order (high -> low priority):
        #   1. Explicit params dict
        #   2. Config.ml_preprocessing_profile (#634 Scope 2)
        #   3. "cleaning_v4" hard fallback
        preprocessing_profile = (
            params.get("preprocessing_profile")
            or getattr(self.cfg, "ml_preprocessing_profile", None)
            or "cleaning_v4"
        )
        cleaned_text, _stats = apply_profile_with_stats(text, preprocessing_profile)

        # MAP: classic compression (chunk summaries)
        map_model_max = ModelRegistry.get_capabilities(
            self._map_model.model_name, self._map_model.model
        ).max_input_tokens
        chunk_size = int(params.get("chunk_size") or self.cfg.summary_chunk_size or map_model_max)
        chunk_size = max(1, min(chunk_size, map_model_max - summarizer.MODEL_MAX_BUFFER))

        # Build chunks + map summaries using existing internal helpers (tested elsewhere)
        chunks, _effective_chunk_size = summarizer._prepare_chunks(
            self._map_model,
            cleaned_text,
            chunk_size=chunk_size,
            use_word_chunking=False,
            word_chunk_size=int(getattr(self.cfg, "summary_word_chunk_size", 900) or 900),
            word_overlap=int(getattr(self.cfg, "summary_word_overlap", 150) or 150),
        )
        chunks = summarizer._merge_tiny_chunks(self._map_model, chunks)

        map_params = dict(getattr(self.cfg, "summary_map_params", {}) or {})
        map_max_new_tokens = params.get("map_max_new_tokens") or map_params.get("max_new_tokens")
        map_min_new_tokens = params.get("map_min_new_tokens") or map_params.get("min_new_tokens")
        map_num_beams = params.get("map_num_beams") or map_params.get("num_beams")
        map_no_repeat_ngram_size = params.get("map_no_repeat_ngram_size") or map_params.get(
            "no_repeat_ngram_size"
        )
        map_length_penalty = params.get("map_length_penalty") or map_params.get("length_penalty")
        map_early_stopping = params.get("map_early_stopping") or map_params.get("early_stopping")
        map_repetition_penalty = params.get("map_repetition_penalty") or map_params.get(
            "repetition_penalty"
        )

        chunk_summaries = summarizer._summarize_chunks_map(
            model=self._map_model,
            chunks=chunks,
            max_length=int(map_max_new_tokens or 200),
            min_length=int(map_min_new_tokens or 80),
            prompt=None,
            batch_size=None,
            use_word_chunking=False,
            word_chunk_size=int(getattr(self.cfg, "summary_word_chunk_size", 900) or 900),
            word_overlap=int(getattr(self.cfg, "summary_word_overlap", 150) or 150),
            chunk_size=chunk_size,
            map_max_new_tokens=int(map_max_new_tokens) if map_max_new_tokens is not None else None,
            map_min_new_tokens=int(map_min_new_tokens) if map_min_new_tokens is not None else None,
            map_num_beams=int(map_num_beams) if map_num_beams is not None else None,
            map_no_repeat_ngram_size=(
                int(map_no_repeat_ngram_size) if map_no_repeat_ngram_size is not None else None
            ),
            map_length_penalty=(
                float(map_length_penalty) if map_length_penalty is not None else None
            ),
            map_early_stopping=(
                bool(map_early_stopping) if map_early_stopping is not None else None
            ),
            map_repetition_penalty=(
                float(map_repetition_penalty) if map_repetition_penalty is not None else None
            ),
        )

        notes = summarizer._join_summaries_with_structure(chunk_summaries)

        instruction_style = getattr(self.cfg, "hybrid_reduce_instruction_style", None)
        if instruction_style == "paragraph":
            reduce_instruction = self._build_reduce_instruction_paragraph(
                episode_title=episode_title,
                episode_description=episode_description,
            )
        else:
            reduce_instruction = self._build_reduce_instruction(
                episode_title=episode_title,
                episode_description=episode_description,
            )
        reduce_params = dict(getattr(self.cfg, "summary_reduce_params", {}) or {})
        reduce_result = self._reduce_backend.reduce(
            notes=notes,
            instruction=reduce_instruction,
            params=reduce_params,
        )
        reduce_text = self._postprocess_reduce_output(reduce_result.text)

        if call_metrics is not None:
            call_metrics.finalize()

        metadata: Dict[str, Any] = {
            "provider": "hybrid_ml",
            "hybrid_map_model": self._map_model.model_name,
            "hybrid_reduce_model": str(getattr(self.cfg, "hybrid_reduce_model", "")),
            "hybrid_reduce_backend": str(getattr(self.cfg, "hybrid_reduce_backend", "")),
            "preprocessing_profile": preprocessing_profile,
            "map_chunks": len(chunks),
            "reduce_backend": reduce_result.backend,
            "reduce_model": reduce_result.model,
        }
        return {
            "summary": reduce_text,
            "summary_short": None,
            "metadata": metadata,
        }

    def generate_insights(
        self,
        text: str,
        episode_title: Optional[str] = None,
        max_insights: int = 5,
        params: Optional[Dict[str, Any]] = None,
        pipeline_metrics: Optional[Any] = None,
    ) -> List[str]:
        """Generate insight statements (GIL). Hybrid ML provider does not implement this.

        Returns empty list so GIL falls back to stub or summary_bullets when
        gi_insight_source=provider.
        """
        return []

    def extract_kg_graph(
        self,
        text: str,
        episode_title: Optional[str] = None,
        max_topics: int = 5,
        max_entities: int = 15,
        params: Optional[Dict[str, Any]] = None,
        pipeline_metrics: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """KG LLM extraction is not implemented for hybrid_ml summarization."""
        return None

    def extract_quotes(
        self,
        transcript: str,
        insight_text: str,
        **kwargs: Any,
    ) -> List[Any]:
        """Extract candidate quote spans (GIL QA). Uses same evidence stack as MLProvider."""
        from ...gi.grounding import QuoteCandidate
        from . import extractive_qa

        if not (transcript and insight_text):
            return []
        question = f"What evidence supports: {insight_text.strip()}"
        try:
            span = extractive_qa.answer(
                context=transcript,
                question=question,
                model_id=getattr(self.cfg, "gi_qa_model", "roberta-squad2"),
                device=getattr(self.cfg, "extractive_qa_device", None),
                window_chars=int(getattr(self.cfg, "gi_qa_window_chars", 0)),
                window_overlap_chars=int(getattr(self.cfg, "gi_qa_window_overlap_chars", 250)),
            )
        except Exception as e:
            logger.warning(
                "Extractive QA failed for extract_quotes: %s", format_exception_for_log(e)
            )
            return []
        verbatim = transcript[span.start : span.end] if span.end <= len(transcript) else span.answer
        return [
            QuoteCandidate(
                char_start=span.start,
                char_end=span.end,
                text=verbatim,
                qa_score=span.score,
            )
        ]

    def score_entailment(
        self,
        premise: str,
        hypothesis: str,
        **kwargs: Any,
    ) -> float:
        """Score entailment (GIL NLI). Uses same evidence stack as MLProvider."""
        from . import nli_loader

        if not (premise and hypothesis):
            return 0.0
        try:
            return nli_loader.entailment_score(
                premise=premise.strip(),
                hypothesis=hypothesis.strip(),
                model_id=getattr(self.cfg, "gi_nli_model", "nli-deberta-base"),
                device=getattr(self.cfg, "nli_device", None),
            )
        except Exception as e:
            logger.warning("NLI failed for score_entailment: %s", format_exception_for_log(e))
            return 0.0

    def cleanup(self) -> None:
        """Unload MAP weights and release the REDUCE backend."""
        self._spacy_nlp = None
        if self._reduce_backend is not None:
            self._reduce_backend.cleanup()
        self._reduce_backend = None
        if self._map_model is not None:
            summarizer.unload_model(self._map_model)
        self._map_model = None
        self._initialized = False

    @staticmethod
    def _build_reduce_instruction(
        episode_title: Optional[str],
        episode_description: Optional[str],
    ) -> str:
        title_line = f"Episode title: {episode_title.strip()}" if episode_title else ""
        desc_line = (
            f"Episode description: {episode_description.strip()}" if episode_description else ""
        )
        context = "\n".join([line for line in (title_line, desc_line) if line]).strip()
        header = f"{context}\n\n" if context else ""

        return (
            header
            + "You are a helpful podcast summarizer.\n"
            + "Using ONLY the NOTES below, write a structured summary in Markdown with "
            + "EXACT headings:\n"
            + "## Takeaways\n"
            + "- ...\n\n"
            + "## Outline\n"
            + "- ...\n\n"
            + "## Actions\n"
            + "- ...\n\n"
            + "Rules:\n"
            + "- No preamble.\n"
            + "- Do not mention that you were given notes.\n"
            + "- Do not include any other headings.\n"
            + "- Keep bullets concise.\n"
            + "- Omit sponsorships, ads, housekeeping, and generic intros/outros.\n\n"
            + "NOTES:\n{{ transcript }}"
        )

    @staticmethod
    def _build_reduce_instruction_paragraph(
        episode_title: Optional[str],
        episode_description: Optional[str],
    ) -> str:
        """Silver REDUCE instruction: paragraphs, no fixed headings (align with OpenAI long_v1)."""
        title_line = f"Episode title: {episode_title.strip()}" if episode_title else ""
        desc_line = (
            f"Episode description: {episode_description.strip()}" if episode_description else ""
        )
        context = "\n".join([line for line in (title_line, desc_line) if line]).strip()
        header = f"{context}\n\n" if context else ""

        return (
            header
            + "You are an expert at creating concise, informative summaries of podcast episodes.\n"
            + "Using ONLY the NOTES below, write a detailed summary in 4 to 6 paragraphs.\n\n"
            + "Content: Focus on key insights, decisions, arguments, and lessons learned. "
            + "Emphasize cause-and-effect, reasoning, and takeaways.\n"
            + "Filter out: sponsorships, ads, housekeeping, generic intros/outros. "
            + "Do not use quotes or speaker names. Do not invent information not in the notes.\n"
            + "Style: Write in a neutral, professional voice with clear paragraph breaks. "
            + "No preamble, no markdown headings, no bullet lists. "
            + "Start directly with the summary.\n\n"
            + "NOTES:\n{{ transcript }}"
        )

    @staticmethod
    def _postprocess_reduce_output(text: str) -> str:
        cleaned = (text or "").strip()
        if not cleaned:
            return ""
        # Best-effort scaffold leakage stripping
        cleaned = cleaned.replace("NOTES:", "").strip()
        return cleaned
