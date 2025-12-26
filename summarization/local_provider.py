"""Local transformers-based summarization provider implementation.

This module provides a SummarizationProvider implementation using local
PyTorch transformer models for episode summarization.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

# Import summarizer functions (keeping existing implementation)
from .. import config, summarizer

logger = logging.getLogger(__name__)


class TransformersSummarizationProvider:
    """Local transformers-based summarization provider.

    This provider uses local PyTorch transformer models (BART, PEGASUS, LED, etc.)
    for automatic episode summarization. It implements the SummarizationProvider protocol.
    """

    def __init__(self, cfg: config.Config):
        """Initialize transformers summarization provider.

        Args:
            cfg: Configuration object with summary_model, summary_reduce_model, etc.
        """
        self.cfg = cfg
        self._map_model: Optional[summarizer.SummaryModel] = None
        self._reduce_model: Optional[summarizer.SummaryModel] = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize summarization models.

        This method loads the MAP model (and optionally REDUCE model) using
        the configuration. It should be called before summarize() is used.
        """
        if self._initialized:
            return

        if not self.cfg.generate_summaries:
            logger.debug("Summary generation disabled, models not loaded")
            return

        logger.debug(
            "Initializing transformers summarization provider (model: %s)",
            self.cfg.summary_model or "default",
        )

        try:
            # Load MAP model (for chunk summarization)
            model_name = summarizer.select_summary_model(self.cfg)
            self._map_model = summarizer.SummaryModel(
                model_name=model_name,
                device=self.cfg.summary_device,
                cache_dir=self.cfg.summary_cache_dir,
            )
            logger.debug("Loaded MAP summary model: %s", model_name)

            # Load REDUCE model if different from MAP model (for final combine)
            reduce_model_name = summarizer.select_reduce_model(self.cfg, model_name)
            if reduce_model_name != model_name:
                self._reduce_model = summarizer.SummaryModel(
                    model_name=reduce_model_name,
                    device=self.cfg.summary_device,
                    cache_dir=self.cfg.summary_cache_dir,
                )
                logger.debug("Loaded REDUCE summary model: %s", reduce_model_name)
            else:
                # Use MAP model for REDUCE phase if they're the same
                self._reduce_model = self._map_model
                logger.debug("Using MAP model for REDUCE phase (same model)")

            self._initialized = True
            logger.debug("Transformers summarization provider initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize summarization models: %s", e)
            raise

    def summarize(
        self,
        text: str,
        episode_title: Optional[str] = None,
        episode_description: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Summarize text using MAP/REDUCE pattern.

        Args:
            text: Transcript text to summarize
            episode_title: Optional episode title (not used by transformers provider)
            episode_description: Optional episode description (not used by transformers provider)
            params: Optional parameters dict with:
                - max_length: Maximum summary length (default from config)
                - min_length: Minimum summary length (default from config)
                - chunk_size: Chunk size in tokens (default from config)
                - chunk_parallelism: Number of chunks to process in parallel
                  (CPU only, default from config)
                - use_word_chunking: Use word-based chunking (default: auto-detected)
                - word_chunk_size: Chunk size in words (default from config)
                - word_overlap: Overlap in words (default from config)
                - prompt: Optional instruction/prompt

        Returns:
            Dictionary with summary results:
            {
                "summary": str,
                "summary_short": Optional[str],
                "metadata": {
                    "model_used": str,
                    "reduce_model_used": Optional[str],
                    ...
                }
            }

        Raises:
            ValueError: If summarization fails
            RuntimeError: If provider not initialized
        """
        if not self._initialized or not self._map_model:
            raise RuntimeError("Provider not initialized. Call initialize() first.")

        # Extract parameters with defaults from config
        max_length = (params.get("max_length") if params else None) or self.cfg.summary_max_length
        min_length = (params.get("min_length") if params else None) or self.cfg.summary_min_length
        chunk_size = (params.get("chunk_size") if params else None) or self.cfg.summary_chunk_size
        # Chunk-level parallelism: Use chunk_parallelism from params, fallback to config
        chunk_parallelism = (
            params.get("chunk_parallelism") if params else None
        ) or self.cfg.summary_chunk_parallelism
        # batch_size is deprecated - use chunk_parallelism instead
        # Keep for backward compatibility if explicitly provided
        batch_size = params.get("batch_size") if params else None
        if batch_size is None:
            batch_size = chunk_parallelism if self._map_model.device == "cpu" else None
        use_word_chunking = params.get("use_word_chunking") if params else None
        word_chunk_size = (
            params.get("word_chunk_size") if params else None
        ) or self.cfg.summary_word_chunk_size
        word_overlap = (
            params.get("word_overlap") if params else None
        ) or self.cfg.summary_word_overlap
        prompt = params.get("prompt") if params else None

        # Auto-detect word chunking if not specified
        if use_word_chunking is None:
            model_name = (
                self._map_model.model_name if hasattr(self._map_model, "model_name") else ""
            )
            use_word_chunking = any(
                model_keyword in model_name.lower()
                for model_keyword in ["bart", "pegasus", "distilbart"]
            )

        # Use summarize_long_text for MAP/REDUCE pattern
        try:
            summary_text = summarizer.summarize_long_text(
                model=self._map_model,
                text=text,
                chunk_size=chunk_size or summarizer.BART_MAX_POSITION_EMBEDDINGS,
                max_length=max_length,
                min_length=min_length,
                batch_size=batch_size if self._map_model.device == "cpu" else None,
                prompt=prompt,
                use_word_chunking=use_word_chunking,
                word_chunk_size=word_chunk_size or summarizer.DEFAULT_WORD_CHUNK_SIZE,
                word_overlap=word_overlap or summarizer.DEFAULT_WORD_OVERLAP,
                reduce_model=self._reduce_model,
            )

            # Build metadata
            metadata: Dict[str, Any] = {
                "model_used": self._map_model.model_name,
                "reduce_model_used": (
                    self._reduce_model.model_name
                    if self._reduce_model and self._reduce_model != self._map_model
                    else None
                ),
                "device": self._map_model.device,
            }

            return {
                "summary": summary_text,
                "summary_short": None,  # Transformers provider doesn't generate short summaries
                "metadata": metadata,
            }
        except Exception as e:
            logger.error("Summarization failed: %s", e)
            raise ValueError(f"Summarization failed: {e}") from e

    def cleanup(self) -> None:
        """Cleanup and unload models.

        This method unloads the models to free memory. It should be called
        when the provider is no longer needed.
        """
        if not self._initialized:
            return

        logger.debug("Cleaning up summarization models")

        # Note: SummaryModel doesn't have explicit cleanup, but we can
        # set references to None to help GC
        if self._reduce_model and self._reduce_model != self._map_model:
            # Only cleanup reduce_model if it's different from map_model
            self._reduce_model = None

        self._map_model = None
        self._reduce_model = None
        self._initialized = False

        logger.debug("Summarization models cleaned up")

    @property
    def map_model(self) -> Optional[summarizer.SummaryModel]:
        """Get the MAP model instance (for backward compatibility).

        Returns:
            SummaryModel instance or None if not initialized
        """
        return self._map_model

    @property
    def reduce_model(self) -> Optional[summarizer.SummaryModel]:
        """Get the REDUCE model instance (for backward compatibility).

        Returns:
            SummaryModel instance or None if not initialized
        """
        return self._reduce_model

    @property
    def is_initialized(self) -> bool:
        """Check if provider is initialized.

        Returns:
            True if provider is initialized, False otherwise
        """
        return self._initialized
