"""SummLlama3.2-3B summarization provider (#571).

Single-pass causal LM summarization using DISLab/SummLlama3.2-3B.
No-daemon alternative to BART+LED — no MPS contention, no MAP/REDUCE.

Usage:
    Set ``summary_provider: summllama`` in config.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Default model ID on HuggingFace
DEFAULT_SUMMLLAMA_MODEL = "DISLab/SummLlama3.2-3B"

# Prompt templates (same as standalone runner, tuned in autoresearch v2)
_SYSTEM = "You write focused summaries of podcast transcripts."

_USER_PARAGRAPH = (
    "Please write a focused prose summary of the following podcast transcript "
    "in 4-6 paragraphs. Begin the first paragraph with a single sentence naming "
    "the episode's domain and its central argument or premise. Cover all major "
    "discussion segments in the order they appear. Preserve specific terminology "
    "verbatim — do not paraphrase named concepts. Anchor each paragraph in "
    "specific claims, data points, or named entities from the transcript. Ignore "
    "sponsorships, ads, and housekeeping. Do not use quotes or speaker names. Do "
    "not invent information not implied by the transcript.\n\nTranscript:\n\n"
)

_USER_BULLETS = (
    "Write a bullet-point summary of the following podcast transcript as 6-8 "
    "single-sentence bullets. Each bullet should cover a distinct major topic "
    "or claim in the order it appears. Preserve specific terminology verbatim — "
    "do not paraphrase named concepts. Anchor each bullet in specific claims, "
    "data points, or named entities from the transcript. Ignore sponsorships, "
    "ads, and housekeeping. Do not use quotes or speaker names. Do not invent "
    "information not implied by the transcript. Output only the bullets, one "
    "per line, each starting with '- '.\n\nTranscript:\n\n"
)


class SummLlamaProvider:
    """SummLlama3.2-3B summarization provider.

    Loads the model once on initialize(), reuses for all episodes.
    Supports paragraph and bullet styles via config.
    """

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self._initialized = False
        self._tokenizer: Any = None
        self._model: Any = None
        self._model_id = getattr(cfg, "summllama_model", DEFAULT_SUMMLLAMA_MODEL)
        self._device = getattr(cfg, "summllama_device", None) or "cpu"
        self._max_tokens = int(getattr(cfg, "summllama_max_tokens", 600))
        self._style = getattr(cfg, "summllama_summary_style", "bullets")
        self._max_input_chars = 40000

        # Required by SummarizationProvider protocol
        cleaning_strategy = getattr(cfg, "transcript_cleaning_strategy", "hybrid")
        if cleaning_strategy == "pattern":
            from ...cleaning.pattern_based import PatternBasedCleaner

            self.cleaning_processor = PatternBasedCleaner()  # type: ignore[assignment]
        elif cleaning_strategy == "llm":
            from ...cleaning.llm_based import LLMBasedCleaner

            self.cleaning_processor = LLMBasedCleaner()  # type: ignore[assignment]
        else:
            from ...cleaning.hybrid import HybridCleaner

            self.cleaning_processor = HybridCleaner()  # type: ignore[assignment]

    def initialize(self) -> None:
        """Load model and tokenizer."""
        if self._initialized:
            return

        import os

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

        # Auto-detect device
        if self._device == "auto" or self._device is None:
            if torch.backends.mps.is_available():
                self._device = "mps"
            elif torch.cuda.is_available():
                self._device = "cuda"
            else:
                self._device = "cpu"

        cache_dir = getattr(self.cfg, "summary_cache_dir", None)
        logger.info("Loading SummLlama model %s on %s", self._model_id, self._device)

        tk_kw: dict = {"trust_remote_code": False}
        model_kw: dict = {
            "torch_dtype": torch.float16,
            "trust_remote_code": False,
        }
        if cache_dir:
            tk_kw["cache_dir"] = cache_dir
            model_kw["cache_dir"] = cache_dir

        if self._device == "cpu":
            model_kw["device_map"] = "cpu"
            model_kw["torch_dtype"] = torch.float32
        else:
            model_kw["device_map"] = self._device

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id, **tk_kw)  # nosec B615
        self._model = AutoModelForCausalLM.from_pretrained(self._model_id, **model_kw)  # nosec B615
        self._model.eval()

        self._initialized = True
        logger.info("SummLlama loaded: %s on %s", self._model_id, self._device)

    def summarize(
        self,
        text: str,
        episode_title: Optional[str] = None,
        episode_description: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        pipeline_metrics: Any = None,
        call_metrics: Any = None,
    ) -> Dict[str, Any]:
        """Generate summary using SummLlama chat template."""
        import torch

        if not self._initialized:
            self.initialize()

        # Truncate input
        transcript = text[: self._max_input_chars] if text else ""

        # Build prompt
        style = (params or {}).get("style", self._style)
        if style == "paragraph":
            user_content = _USER_PARAGRAPH + transcript
        else:
            user_content = _USER_BULLETS + transcript

        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user_content},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        t0 = time.time()
        inputs = self._tokenizer(prompt, return_tensors="pt")
        if self._device != "cpu":
            inputs = inputs.to(self._device)

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self._max_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        elapsed = time.time() - t0
        response = self._tokenizer.decode(
            output[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        ).strip()

        logger.info(
            "SummLlama: %d chars → %d chars in %.1fs (style=%s)",
            len(transcript),
            len(response),
            elapsed,
            style,
        )

        return {
            "summary": response,
            "model": self._model_id,
            "style": style,
            "elapsed_seconds": round(elapsed, 2),
        }

    def cleanup(self) -> None:
        """Unload model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._initialized = False
        logger.info("SummLlama cleaned up")

    @property
    def is_initialized(self) -> bool:
        return self._initialized
