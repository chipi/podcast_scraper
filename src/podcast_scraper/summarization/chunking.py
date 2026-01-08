"""Text chunking utilities for summarization.

This module provides functions for splitting long texts into overlapping chunks
for map-reduce summarization workflows.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# Chunking configuration
CHUNK_OVERLAP_RATIO = 0.1  # 10% overlap between chunks for context continuity
DEFAULT_TOKEN_OVERLAP = 200  # Default token overlap (for token-based chunking)
DEFAULT_WORD_CHUNK_SIZE = (
    900  # Default chunk size in words (per SUMMARY_REVIEW.md: 800-1200 recommended)
)
DEFAULT_WORD_OVERLAP = 150  # Default overlap in words (per SUMMARY_REVIEW.md: 100-200 recommended)
MIN_WORD_CHUNK_SIZE = 800  # Minimum recommended word chunk size
MAX_WORD_CHUNK_SIZE = 1200  # Maximum recommended word chunk size
MIN_WORD_OVERLAP = 100  # Minimum recommended word overlap
MAX_WORD_OVERLAP = 200  # Maximum recommended word overlap
ENCODER_DECODER_TOKEN_CHUNK_SIZE = (
    600  # Forced token chunk size for encoder-decoder models (BART/PEGASUS)
)


def chunk_text_for_summarization(
    text: str,
    tokenizer: "AutoTokenizer",
    chunk_size: int,
    # Default token overlap (will be adjusted based on chunk_size)
    overlap: int = DEFAULT_TOKEN_OVERLAP,
) -> List[str]:
    """Split long text into overlapping chunks.

    Args:
        text: Input text
        tokenizer: Tokenizer instance for accurate token counting
        chunk_size: Target chunk size in tokens
        overlap: Overlap between chunks in tokens

    Returns:
        List of text chunks
    """
    # Tokenize to get accurate token counts
    tokens = tokenizer.encode(text, add_special_tokens=False)  # type: ignore[attr-defined]

    chunks = []
    start = 0
    total_tokens = len(tokens)

    while start < total_tokens:
        # Calculate chunk end (don't exceed total tokens)
        end = min(start + chunk_size, total_tokens)
        chunk_tokens = tokens[start:end]

        # Decode chunk tokens back to text
        chunk_text = tokenizer.decode(  # type: ignore[attr-defined]
            chunk_tokens, skip_special_tokens=True
        )
        chunks.append(chunk_text)

        # Move start forward: advance by (chunk_size - overlap) tokens
        # This ensures we process chunks efficiently with proper overlap
        advance = chunk_size - overlap
        if advance < 1:
            advance = 1  # Ensure we always advance

        new_start = start + advance

        # If we've reached or exceeded the end, we're done
        if new_start >= total_tokens:
            break

        start = new_start

    return chunks


def chunk_by_tokens(text: str, tokenizer: "AutoTokenizer", max_tokens: int = 600) -> List[str]:
    """Simple token-based chunking without overlap (for mini map-reduce).

    This function ensures chunks never exceed max_tokens, preventing truncation.
    Used specifically for mini map-reduce where we need guaranteed token limits.

    Args:
        text: Input text to chunk
        tokenizer: Tokenizer instance for encoding/decoding
        max_tokens: Maximum tokens per chunk (default: 600, safe for BART's 1024 limit)

    Returns:
        List of text chunks, each guaranteed to be <= max_tokens
    """
    ids = tokenizer.encode(text, add_special_tokens=False)  # type: ignore[attr-defined]
    chunks = []
    for i in range(0, len(ids), max_tokens):
        cs = ids[i : i + max_tokens]
        chunks.append(tokenizer.decode(cs, skip_special_tokens=True))  # type: ignore[attr-defined]
    return chunks


def chunk_text_words(
    text: str,
    chunk_size: int = DEFAULT_WORD_CHUNK_SIZE,
    overlap: int = DEFAULT_WORD_OVERLAP,
) -> List[str]:
    """Split long text into overlapping chunks using word-based approximation.

    Word-based chunking is recommended for encoder-decoder models (BART, PEGASUS)
    as it provides better semantic boundaries than token-based chunking.

    Args:
        text: Input text
        chunk_size: Target chunk size in words
            (MIN_WORD_CHUNK_SIZE-MAX_WORD_CHUNK_SIZE recommended for encoder-decoder)
        overlap: Overlap between chunks in words (MIN_WORD_OVERLAP-MAX_WORD_OVERLAP recommended)

    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []
    start = 0
    n = len(words)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == n:
            break
        start = end - overlap
        if start >= n:
            break

    return chunks


def check_if_needs_chunking(
    model,  # SummaryModel - avoid circular import
    text: str,
    chunk_size: int,
    max_length: int,
    min_length: int,
    prompt: Optional[str],
) -> Optional[str]:
    """Check if text can be summarized without chunking.

    Args:
        model: Summary model instance
        text: Input text
        chunk_size: Chunk size in tokens (from config)
        max_length: Max summary length
        min_length: Min summary length
        prompt: Optional prompt

    Returns:
        Summary if text fits without chunking, None otherwise
    """
    if not model.tokenizer:
        raise RuntimeError("Model tokenizer not available")

    # Check if text fits in configured chunk_size
    tokens = model.tokenizer.encode(text, add_special_tokens=False)  # type: ignore[attr-defined]
    total_tokens = len(tokens)

    if total_tokens <= chunk_size:
        # Text fits in one chunk
        result = model.summarize(text, max_length=max_length, min_length=min_length, prompt=prompt)
        return str(result) if result is not None else None

    return None


def prepare_chunks(
    model,  # SummaryModel - avoid circular import
    text: str,
    chunk_size: int,
    use_word_chunking: bool,
    word_chunk_size: int,
    word_overlap: int,
) -> Tuple[List[str], int]:
    """Prepare text chunks for summarization using token-based chunking.

    Args:
        model: Summary model instance
        text: Input text
        chunk_size: Effective chunk size in tokens (already capped for model)
        use_word_chunking: True if encoder-decoder heuristics requested word chunking (for logging)
        word_chunk_size: Original word chunk size (for logging)
        word_overlap: Original word overlap (for logging)

    Returns:
        Tuple of (chunks, effective_chunk_size_in_tokens)
    """
    if not model.tokenizer:
        raise RuntimeError("Model tokenizer not available")

    overlap = max(1, int(chunk_size * CHUNK_OVERLAP_RATIO))
    chunks = chunk_text_for_summarization(
        text,
        model.tokenizer,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    total_words = len(text.split())
    total_tokens = len(
        model.tokenizer.encode(text, add_special_tokens=False)  # type: ignore[attr-defined]
    )

    if use_word_chunking:
        logger.debug(
            "Encoder-decoder model detected (word chunking requested). "
            f"Forcing token chunking with chunk_size={chunk_size} tokens "
            f"(requested word_chunk_size={word_chunk_size} words, overlap={overlap} tokens)."
        )
    else:
        logger.debug(
            f"Using token-based chunking "
            f"(chunk_size={chunk_size} tokens, overlap={overlap} tokens)."
        )

    logger.debug(
        f"Split text into {len(chunks)} chunks for summarization "
        f"({total_words} words total, ~{total_tokens} tokens, chunk_size={chunk_size} tokens, "
        f"overlap={overlap} tokens)"
    )

    return chunks, chunk_size
