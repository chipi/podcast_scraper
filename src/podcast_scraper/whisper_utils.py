"""Shared utilities for Whisper model handling.

This module provides centralized logic for Whisper model selection and fallback chains,
making the code config-driven and eliminating duplication across multiple modules.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

from .config_constants import (
    FALLBACK_WHISPER_MODELS_EN,
    FALLBACK_WHISPER_MODELS_MULTILINGUAL,
    WHISPER_MODELS_WITH_EN_VARIANT,
)

logger = logging.getLogger(__name__)


def normalize_whisper_model_name(model_name: str, language: str | None) -> Tuple[str, List[str]]:
    """Normalize Whisper model name and build fallback chain.

    This function handles:
    1. Language-aware model selection (adds/removes .en suffix)
    2. Building fallback chain from largest to smallest models

    Args:
        model_name: Requested Whisper model name (e.g., "base", "large", "tiny.en")
        language: Language code (e.g., "en", "fr", "es") or None

    Returns:
        Tuple of (normalized_model_name, fallback_chain):
        - normalized_model_name: Model name after language-aware normalization
        - fallback_chain: List of models to try in order (largest to smallest)

    Examples:
        >>> normalize_whisper_model_name("base", "en")
        ("base.en", ["base.en", "tiny.en"])

        >>> normalize_whisper_model_name("large", "en")
        ("large", ["large", "medium.en", "small.en", "base.en", "tiny.en"])

        >>> normalize_whisper_model_name("base.en", "fr")
        ("base", ["base", "tiny"])

        >>> normalize_whisper_model_name("tiny.en", "en")
        ("tiny.en", ["tiny.en"])
    """
    is_english = language and language.lower() in ("en", "english")
    requested_model = model_name
    normalized_name = model_name

    # Step 1: Language-aware model selection
    if is_english:
        # For English, prefer .en variants (better performance)
        if normalized_name in WHISPER_MODELS_WITH_EN_VARIANT:
            normalized_name = f"{normalized_name}.en"
            logger.debug(
                "Language is English, preferring %s over %s", normalized_name, requested_model
            )
    else:
        # For non-English, ensure we use multilingual models (no .en suffix)
        if normalized_name.endswith(".en"):
            logger.debug(
                "Language is %s, using multilingual model instead of %s",
                language,
                normalized_name,
            )
            normalized_name = normalized_name[:-3]  # Remove .en suffix

    # Step 2: Build fallback chain (from requested model down to smallest)
    fallback_chain = [normalized_name]

    if is_english:
        # Use English-only fallback chain
        fallback_models = FALLBACK_WHISPER_MODELS_EN
    else:
        # Use multilingual fallback chain
        fallback_models = FALLBACK_WHISPER_MODELS_MULTILINGUAL

    # Find the index of the normalized model in the fallback chain
    # Note: fallback_models is ordered from smallest to largest: [tiny, base, small, medium, large]
    # We want fallback chain to go from requested model down to smallest
    try:
        current_index = fallback_models.index(normalized_name)
        # Add all smaller models as fallbacks (models before current_index, in reverse order)
        if current_index > 0:
            # Get smaller models and reverse to go from larger to smaller
            smaller_models = fallback_models[:current_index]
            fallback_chain.extend(reversed(smaller_models))
    except ValueError:
        # Model not in standard fallback chain (e.g., "large" without .en, "large-v2", "large-v3")
        # Try to find a matching prefix to determine size
        model_base = normalized_name.split(".")[0].split("-")[0]  # "large-v2" -> "large"
        try:
            # Find the index of the matching base model (with or without .en)
            # Try exact match first, then prefix match
            base_index = None
            for i, model in enumerate(fallback_models):
                model_base_name = model.split(".")[0].split("-")[0]
                if model_base_name == model_base:
                    base_index = i
                    break

            if base_index is not None:
                # Add all smaller models as fallbacks (models before matching base, reversed)
                if base_index > 0:
                    smaller_models = fallback_models[:base_index]
                    fallback_chain.extend(reversed(smaller_models))
            else:
                # No matching base found, no fallback (user specified exotic model)
                logger.debug(
                    "Model %s not in standard fallback chain, no fallback available",
                    normalized_name,
                )
        except (StopIteration, ValueError):
            # No matching base found, no fallback (user specified exotic model)
            logger.debug(
                "Model %s not in standard fallback chain, no fallback available",
                normalized_name,
            )

    # Remove duplicates while preserving order
    seen = set()
    unique_fallback = []
    for model in fallback_chain:
        if model not in seen:
            seen.add(model)
            unique_fallback.append(model)
    fallback_chain = unique_fallback

    if len(fallback_chain) > 1:
        logger.debug("Fallback chain for %s: %s", normalized_name, fallback_chain)

    return normalized_name, fallback_chain
