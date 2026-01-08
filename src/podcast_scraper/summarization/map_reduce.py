"""Map-reduce summarization workflow.

This module implements the map-reduce workflow for summarizing long texts:
1. Map: Summarize each chunk in parallel or sequentially
2. Reduce: Combine chunk summaries into final summary using various strategies
   (abstractive, hierarchical, or extractive)
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import as_completed, ThreadPoolExecutor
from typing import Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    # Avoid circular import - SummaryModel is defined in summarizer.py
    from ..summarizer import SummaryModel

from . import chunking, prompts

logger = logging.getLogger(__name__)

# Import constants from chunking module
CHUNK_OVERLAP_RATIO = chunking.CHUNK_OVERLAP_RATIO
ENCODER_DECODER_TOKEN_CHUNK_SIZE = chunking.ENCODER_DECODER_TOKEN_CHUNK_SIZE
DEFAULT_WORD_CHUNK_SIZE = chunking.DEFAULT_WORD_CHUNK_SIZE
DEFAULT_WORD_OVERLAP = chunking.DEFAULT_WORD_OVERLAP

# Import constants from prompts module
REDUCE_PROMPT_SHORT = prompts.REDUCE_PROMPT_SHORT
INSTRUCTION_LEAK_PATTERNS = prompts.INSTRUCTION_LEAK_PATTERNS

# Model context window limits
BART_MAX_POSITION_EMBEDDINGS = 1024  # Standard BART/PEGASUS model limit

# Token estimation
CHARS_PER_TOKEN_ESTIMATE = 4  # Rough estimate: 1 token ≈ 4 characters
CHARS_PER_TOKEN_FOR_LENGTH_CHECK = 8  # More conservative estimate for length validation

# Summarization thresholds
CHUNK_SUMMARY_MIN_TOKENS = 80  # Target lower bound for map summaries
CHUNK_SUMMARY_MAX_TOKENS = 160  # Target upper bound for map summaries
SECTION_SUMMARY_MIN_TOKENS = 80  # Target lower bound for hierarchical section summaries
SECTION_SUMMARY_MAX_TOKENS = 160  # Target upper bound for hierarchical section summaries
FINAL_SUMMARY_MIN_TOKENS = 200  # Target lower bound for final reduce
FINAL_SUMMARY_MAX_TOKENS = (
    480  # Target upper bound for final reduce (slightly higher for more detail)
)
EXTRACTIVE_APPROACH_THRESHOLD = (
    0.8  # Use extractive approach if combined summaries > 80% of model max
)
# Baseline threshold for short-context models (BART/PEGASUS)
MINI_MAP_REDUCE_THRESHOLD = 800
# Maximum tokens for mini map-reduce (used for short-context models)
MINI_MAP_REDUCE_MAX_TOKENS = 4000
# Mini map-reduce chunk size: Use 80% of model's max_position_embeddings
# to leave room for special tokens
MINI_MAP_CHUNK_SIZE_RATIO = 0.8  # Use 80% of model max for safety
MINI_MAP_MIN_CHUNK_SIZE = 512  # Never go below 512 tokens when chunking
# Treat models above this limit (LED) differently
LONG_CONTEXT_THRESHOLD = 4096
MINI_MAP_REDUCE_TRIGGER_RATIO = (
    0.6  # Trigger mini map-reduce once input exceeds 60% of usable context
)
MAX_HIERARCHICAL_PASSES = 4  # Maximum number of hierarchical chunk→summarize passes
SUMMARY_VALIDATION_THRESHOLD = 0.6  # Flag summary if length > 60% of input (likely failed)
REPETITIVE_SUMMARY_THRESHOLD = 0.8  # Flag summary if length > 80% of selected summaries
MAX_LENGTH_MULTIPLIER = 2  # Multiplier for safe max length calculation
FINAL_MAX_LENGTH_MULTIPLIER = 1.8  # Multiplier for final max length calculation
MODEL_MAX_BUFFER = 200  # Buffer to subtract from model max for safety
SAFE_MAX_LENGTH = 512  # Safe maximum length for final summarization

# Chunk selection thresholds
FEW_CHUNKS_THRESHOLD = 3  # Use all chunks if <= this many
MEDIUM_CHUNKS_THRESHOLD = 10  # Use first/middle/last if <= this many

# Progress reporting
PROGRESS_LOG_INTERVAL = 5  # Log progress every N chunks (reduced from 20 for better visibility)
SECONDS_PER_CHUNK_ESTIMATE = 3  # Rough estimate for time calculation

# Parallel processing
MAX_PARALLEL_WORKERS = 4  # Maximum number of parallel workers for CPU processing

# Text processing thresholds
MIN_TEXT_LENGTH = 50  # Minimum text length for processing
MIN_SENTENCE_LENGTH = 20  # Minimum sentence length for validation
MAX_REPETITIONS_THRESHOLD = 3  # Maximum repetitions before flagging as repetitive
MIN_SUMMARY_LENGTH_MULTIPLIER = 2  # Minimum summary should be at least 2x min_length


def validate_and_fix_repetitive_summary(summary: str) -> str:
    """Detect and fix repetitive/hallucinated summaries.

    Args:
        summary: Generated summary text

    Returns:
        Fixed summary (or original if no issues detected)
    """
    if not summary or len(summary) < MIN_TEXT_LENGTH:
        return summary

    # Split into sentences
    sentences = summary.split(". ")
    if len(sentences) < FEW_CHUNKS_THRESHOLD:
        return summary

    # Check for excessive repetition (same sentence repeated many times)
    sentence_counts: Dict[str, int] = {}
    for sent in sentences:
        sent_clean = sent.strip().lower()
        if len(sent_clean) > MIN_SENTENCE_LENGTH:  # Only check substantial sentences
            sentence_counts[sent_clean] = sentence_counts.get(sent_clean, 0) + 1

    # If any sentence appears more than threshold times, it's likely hallucination
    max_repetitions = max(sentence_counts.values()) if sentence_counts else 0
    if max_repetitions > MAX_REPETITIONS_THRESHOLD:
        logger.warning(
            f"Detected repetitive summary (max sentence repetition: {max_repetitions}). "
            "This indicates potential hallucination. Attempting to fix..."
        )

        # Remove duplicate sentences, keeping only unique ones in order
        seen: Set[str] = set()
        unique_sentences: List[str] = []
        for sent in sentences:
            sent_clean = sent.strip().lower()
            if sent_clean not in seen and len(sent_clean) > MIN_SENTENCE_LENGTH:
                seen.add(sent_clean)
                unique_sentences.append(sent.strip())

        if unique_sentences:
            fixed_summary = ". ".join(unique_sentences)
            if fixed_summary and not fixed_summary.endswith("."):
                fixed_summary += "."
            logger.debug(
                f"Fixed repetitive summary: reduced from {len(sentences)} "
                f"to {len(unique_sentences)} sentences"
            )
            return fixed_summary
        else:
            logger.error("Failed to fix repetitive summary - all sentences were duplicates")
            return summary

    # Check for very short repetitive patterns (like "What's the best way to do that?" repeated)
    words = summary.lower().split()
    if len(words) > 10:
        # Check for 5-gram repetition
        ngram_size = 5
        ngrams: List[str] = []
        for i in range(len(words) - ngram_size + 1):
            ngram = " ".join(words[i : i + ngram_size])
            ngrams.append(ngram)

        ngram_counts: Dict[str, int] = {}
        for ngram in ngrams:
            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1

        max_ngram_repetitions = max(ngram_counts.values()) if ngram_counts else 0
        if max_ngram_repetitions > 5:
            logger.warning(
                f"Detected repetitive n-grams (max repetition: {max_ngram_repetitions}). "
                "Summary likely contains hallucinations."
            )
            # Return empty summary rather than hallucinated content
            return ""

    return summary


def strip_instruction_leak(summary: str) -> str:
    """Remove sentences that look like leaked instructions from prompts."""
    if not summary:
        return summary

    # Split on sentence boundaries (., ?, !) followed by whitespace
    sentences = re.split(r"(?<=[\.\?\!])\s+", summary)
    filtered: List[str] = []

    for sent in sentences:
        s_lower = sent.lower()
        if any(pat in s_lower for pat in INSTRUCTION_LEAK_PATTERNS):
            continue
        filtered.append(sent.strip())

    cleaned = " ".join(s for s in filtered if s)
    return cleaned.strip()


def select_key_summaries(chunk_summaries: List[str]) -> List[str]:
    """Select representative chunk summaries for extractive approach.

    This function should ONLY be called in extractive paths.
    Abstractive paths must use ALL summaries, not a subset.

    Args:
        chunk_summaries: List of all chunk summaries

    Returns:
        Selected subset of summaries (representative chunks)
    """
    num_chunks = len(chunk_summaries)
    if num_chunks <= FEW_CHUNKS_THRESHOLD:
        return chunk_summaries
    elif num_chunks <= MEDIUM_CHUNKS_THRESHOLD:
        return [
            chunk_summaries[0],
            chunk_summaries[num_chunks // 2],
            chunk_summaries[-1],
        ]
    else:
        return [
            chunk_summaries[0],
            chunk_summaries[num_chunks // 4],
            chunk_summaries[num_chunks // 2],
            chunk_summaries[3 * num_chunks // 4],
            chunk_summaries[-1],
        ]


def join_summaries_with_structure(summaries: List[str]) -> str:
    """Join summaries with structural separation to preserve semantics."""
    return "\n\n".join(summaries)


def summarize_chunks_map(
    model: "SummaryModel",
    chunks: List[str],
    max_length: int,
    min_length: int,
    prompt: Optional[str],
    batch_size: Optional[int],
    use_word_chunking: bool,
    word_chunk_size: int,
    word_overlap: int,
    chunk_size: int,
) -> List[str]:
    """Map step: Summarize each chunk (parallel or sequential).

    Args:
        model: Summary model instance
        chunks: List of text chunks to summarize
        max_length: Max summary length per chunk
        min_length: Min summary length per chunk
        prompt: Optional prompt
        batch_size: Batch size for parallel processing (CPU only)
        use_word_chunking: Whether word-based chunking was used (for logging)
        word_chunk_size: Word chunk size (for logging)
        word_overlap: Word overlap (for logging)
        chunk_size: Token chunk size (for logging)

    Returns:
        List of chunk summaries
    """
    import time

    total_chunks = len(chunks)
    start_time = time.time()

    # Determine if we can parallelize based on device
    can_parallelize = model.device == "cpu"
    max_workers = 1
    if can_parallelize and batch_size and batch_size > 1:
        max_workers = min(batch_size, MAX_PARALLEL_WORKERS, total_chunks)
        if max_workers > 1:
            logger.debug(f"Using parallel processing with {max_workers} workers (CPU device)")

    # Estimate and log processing time
    estimated_minutes = (total_chunks * SECONDS_PER_CHUNK_ESTIMATE) // 60
    if max_workers > 1:
        estimated_minutes = estimated_minutes // max_workers
    overlap = int(chunk_size * CHUNK_OVERLAP_RATIO)
    chunk_max_length = min(chunk_size, max_length, CHUNK_SUMMARY_MAX_TOKENS)
    chunk_min_length = min(chunk_max_length, max(min_length, CHUNK_SUMMARY_MIN_TOKENS))
    logger.debug(
        f"[MAP-REDUCE CONFIG] Map stage: {total_chunks} chunks, chunk_size={chunk_size} tokens, "
        f"overlap={overlap} tokens, workers={max_workers}, "
        f"chunk_summary_range={chunk_min_length}-{chunk_max_length} tokens, "
        f"estimated time ~{estimated_minutes} minutes"
    )

    if max_workers > 1:
        # Parallel processing for CPU
        return summarize_chunks_parallel(
            model,
            chunks,
            chunk_max_length,
            chunk_min_length,
            prompt,
            max_workers,
            start_time,
        )
    else:
        # Sequential processing (GPU or single worker)
        return summarize_chunks_sequential(
            model,
            chunks,
            chunk_max_length,
            chunk_min_length,
            prompt,
            chunk_size,
            start_time,
        )


def summarize_chunks_parallel(
    model: "SummaryModel",
    chunks: List[str],
    chunk_max_length: int,
    chunk_min_length: int,
    prompt: Optional[str],
    max_workers: int,
    start_time: float,
) -> List[str]:
    """Summarize chunks in parallel (CPU only).

    Args:
        model: Summary model instance
        chunks: List of text chunks
        chunk_max_length: Max summary length per chunk
        chunk_min_length: Min summary length per chunk
        prompt: Optional prompt
        max_workers: Number of parallel workers
        start_time: Start time for progress tracking

    Returns:
        List of chunk summaries
    """
    import time

    total_chunks = len(chunks)

    def _summarize_chunk(chunk_idx_and_text):
        chunk_idx, chunk_text = chunk_idx_and_text
        try:
            return (
                chunk_idx,
                model.summarize(
                    chunk_text,
                    max_length=chunk_max_length,
                    min_length=chunk_min_length,
                    prompt=prompt,
                ),
            )
        except Exception as e:
            logger.error(f"Error summarizing chunk {chunk_idx}: {e}")
            return (chunk_idx, None)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all chunks
        future_to_chunk = {
            executor.submit(_summarize_chunk, (i, chunk)): i for i, chunk in enumerate(chunks, 1)
        }

        # Collect results as they complete
        completed = 0
        results = {}
        for future in as_completed(future_to_chunk):
            chunk_idx, summary = future.result()
            completed += 1
            results[chunk_idx] = summary

            if completed % PROGRESS_LOG_INTERVAL == 0 or completed == total_chunks:
                elapsed_total = time.time() - start_time
                avg_time = elapsed_total / completed
                remaining_chunks = total_chunks - completed
                eta_seconds = avg_time * remaining_chunks
                eta_min = int(eta_seconds // 60)
                eta_sec = int(eta_seconds % 60)
                logger.debug(
                    f"Completed {completed}/{total_chunks} chunks with MAP model: "
                    f"{model.model_name} ({avg_time:.1f}s avg, ETA: ~{eta_min}m {eta_sec}s)"
                )

    # Sort results by chunk index and collect summaries
    return [results[i] for i in sorted(results.keys()) if results[i]]


def summarize_chunks_sequential(
    model: "SummaryModel",
    chunks: List[str],
    chunk_max_length: int,
    chunk_min_length: int,
    prompt: Optional[str],
    chunk_size: int,
    start_time: float,
) -> List[str]:
    """Summarize chunks sequentially (GPU or single worker).

    Args:
        model: Summary model instance
        chunks: List of text chunks
        chunk_max_length: Max summary length per chunk
        chunk_min_length: Min summary length per chunk
        prompt: Optional prompt
        chunk_size: Chunk size (for error messages)
        start_time: Start time for progress tracking

    Returns:
        List of chunk summaries
    """
    import time

    chunk_summaries = []
    total_chunks = len(chunks)

    for i, chunk in enumerate(chunks, 1):
        try:
            chunk_start = time.time()
            if i == 1 or i % PROGRESS_LOG_INTERVAL == 0:
                logger.debug(
                    f"Processing chunk {i}/{total_chunks} with MAP model: {model.model_name}..."
                )
            summary = model.summarize(
                chunk,
                max_length=chunk_max_length,
                min_length=chunk_min_length,
                prompt=prompt,
            )
            chunk_elapsed = time.time() - chunk_start
            if summary:
                chunk_summaries.append(summary)
                if i % PROGRESS_LOG_INTERVAL == 0 or i == total_chunks:
                    elapsed_total = time.time() - start_time
                    avg_time = elapsed_total / i
                    remaining_chunks = total_chunks - i
                    eta_seconds = avg_time * remaining_chunks
                    logger.debug(
                        f"Completed {i}/{total_chunks} chunks "
                        f"({chunk_elapsed:.1f}s this chunk, {avg_time:.1f}s avg, "
                        f"ETA: ~{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s)"
                    )
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "invalid buffer size" in error_msg or "out of memory" in error_msg:
                logger.error(
                    f"Buffer size error on chunk {i}/{len(chunks)}: {e}. "
                    f"Chunk size ({chunk_size} tokens) may be too large for {model.device}. "
                    "Try reducing summary_chunk_size."
                )
                continue
            raise

    return chunk_summaries


def combine_summaries_reduce(
    model: "SummaryModel",
    chunk_summaries: List[str],
    max_length: int,
    min_length: int,
    prompt: Optional[str],
) -> str:
    """Reduce step: Combine chunk summaries into final summary.

    Args:
        model: Summary model instance
        chunk_summaries: List of chunk summaries to combine
        max_length: Max summary length
        min_length: Min summary length
        prompt: Optional prompt

    Returns:
        Final combined summary
    """
    if not chunk_summaries:
        logger.warning("No chunk summaries generated, returning empty summary")
        return ""

    # Combine chunk summaries with clear structure
    combined_text = join_summaries_with_structure(chunk_summaries)
    combined_chars = len(combined_text)
    combined_words = len(combined_text.split())
    if model.tokenizer:
        combined_tokens = len(
            model.tokenizer.encode(  # type: ignore[attr-defined]
                combined_text, add_special_tokens=False
            )
        )
    else:
        combined_tokens = combined_chars // CHARS_PER_TOKEN_ESTIMATE

    # Get model max length for decision making
    model_max = (
        getattr(model.model.config, "max_position_embeddings", BART_MAX_POSITION_EMBEDDINGS)
        if model.model and hasattr(model.model, "config")
        else BART_MAX_POSITION_EMBEDDINGS
    )

    usable_context = max(model_max - MODEL_MAX_BUFFER, MINI_MAP_REDUCE_THRESHOLD)

    # Short-context models (BART/PEGASUS) can still benefit from hierarchical reduce
    # on combined summaries that are much longer than their context window, because
    # we re-chunk in the mini map-reduce layer. For these models, use a fixed
    # ceiling (e.g. 4k tokens). Long-context models (LED) can use their full window.
    if model_max >= LONG_CONTEXT_THRESHOLD:
        # Long-context model (e.g. LED): allow up to usable_context tokens
        mini_map_reduce_ceiling = usable_context
    else:
        # Short-context model (e.g. BART/PEGASUS): allow hierarchical reduce
        # up to MINI_MAP_REDUCE_MAX_TOKENS (e.g. ~4k tokens) before extractive fallback
        mini_map_reduce_ceiling = MINI_MAP_REDUCE_MAX_TOKENS
    single_pass_limit = min(
        mini_map_reduce_ceiling,
        max(MINI_MAP_REDUCE_THRESHOLD, int(usable_context * MINI_MAP_REDUCE_TRIGGER_RATIO)),
    )

    # Decision logic with detailed logging
    if combined_tokens <= single_pass_limit:
        approach = "abstractive (single-pass)"
        reason = (
            f"combined_tokens ({combined_tokens}) <= single_pass_limit ({single_pass_limit}) "
            f"within usable_context ({usable_context})"
        )
    elif combined_tokens <= mini_map_reduce_ceiling:
        approach = "hierarchical reduce"
        reason = (
            f"combined_tokens ({combined_tokens}) > single_pass_limit ({single_pass_limit}); "
            f"attempting hierarchical reduce up to {MAX_HIERARCHICAL_PASSES} passes "
            f"(mini_map_reduce_ceiling={mini_map_reduce_ceiling})"
        )
    else:
        approach = "extractive"
        reason = (
            f"combined_tokens ({combined_tokens}) > mini_map_reduce_ceiling "
            f"({mini_map_reduce_ceiling}); "
            "using extractive fallback (representative chunks only)"
        )

    logger.debug(
        "[MAP-REDUCE VALIDATION] Reduce phase decision: "
        f"combined_input={combined_chars:,} chars, {combined_words:,} words, "
        f"~{combined_tokens:,} tokens, "
        f"model_max={model_max}, usable_context={usable_context}, "
        f"single_pass_limit={single_pass_limit}, "
        f"mini_map_reduce_ceiling={mini_map_reduce_ceiling}, "
        f"approach={approach}"
    )
    logger.debug(f"[MAP-REDUCE VALIDATION] Reduce phase decision reason: {reason}")

    final_reduce_max_length = int(
        min(FINAL_SUMMARY_MAX_TOKENS, model_max - MODEL_MAX_BUFFER, SAFE_MAX_LENGTH)
    )
    final_reduce_min_length = min(
        final_reduce_max_length,
        max(min_length, FINAL_SUMMARY_MIN_TOKENS),
    )
    reduce_prompt = REDUCE_PROMPT_SHORT if prompt is None else prompt

    # Decision tree:
    # 1. If <= threshold → single abstractive reduce (most efficient)
    # 2. If within ceiling → hierarchical reduce
    # 3. If > ceiling → extractive approach
    if combined_tokens > mini_map_reduce_ceiling:
        selected = select_key_summaries(chunk_summaries)
        logger.debug(
            "[MAP-REDUCE CONFIG] Extractive fallback: "
            f"summary_range={final_reduce_min_length}-{final_reduce_max_length} tokens"
        )

        return combine_summaries_extractive(
            model,
            selected,
            final_reduce_max_length,
            final_reduce_min_length,
            reduce_prompt,
            model_max,
        )

    if combined_tokens > single_pass_limit:
        return combine_summaries_mini_map_reduce(
            model,
            combined_text,
            chunk_summaries,
            final_reduce_max_length,
            final_reduce_min_length,
            reduce_prompt,
            combined_tokens,
            single_pass_limit,
        )

    # Single-pass abstractive reduce - use ALL summaries, no selection
    logger.debug(
        "[MAP-REDUCE CONFIG] Final reduce: "
        f"summary_range={final_reduce_min_length}-{final_reduce_max_length} tokens, "
        f"prompt={'REDUCE_PROMPT_SHORT' if prompt is None else 'custom'}"
    )
    try:
        return combine_summaries_abstractive(
            model,
            combined_text,
            chunk_summaries,
            final_reduce_max_length,
            final_reduce_min_length,
            reduce_prompt,
            model_max,
            combined_tokens,
        )
    except RuntimeError as e:
        error_msg = str(e).lower()
        if "invalid buffer size" in error_msg or "out of memory" in error_msg:
            # Abstractive failed - fall back to extractive (which does selection)
            logger.warning(
                "[MAP-REDUCE VALIDATION] Abstractive reduce failed, "
                "falling back to extractive approach"
            )
            return combine_summaries_extractive(
                model,
                select_key_summaries(chunk_summaries),
                final_reduce_max_length,
                final_reduce_min_length,
                reduce_prompt,
                model_max,
            )
        raise


def combine_summaries_extractive(
    model: "SummaryModel",
    selected_summaries: List[str],
    max_length: int,
    min_length: int,
    prompt: Optional[str],
    model_max: int,
) -> str:
    """Combine summaries using extractive approach (select representative chunks).

    Args:
        model: Summary model instance
        selected_summaries: List of pre-selected chunk summaries (representative set)
        max_length: Max summary length
        min_length: Min summary length
        prompt: Optional prompt
        model_max: Model's max position embeddings

    Returns:
        Final summary
    """
    combined_tokens = len("".join(selected_summaries)) // CHARS_PER_TOKEN_ESTIMATE
    logger.debug(
        "[MAP-REDUCE VALIDATION] 🔄 EXTRACTIVE APPROACH: Combined summaries too long "
        f"(~{combined_tokens} tokens). Selecting representative chunks (safety fallback)."
    )

    final_summary = join_summaries_with_structure(selected_summaries)

    # Only do one final pass if still too long
    if len(final_summary) > max_length * CHARS_PER_TOKEN_FOR_LENGTH_CHECK:
        logger.debug("Selected summaries still too long, doing one final summarization pass")
        safe_max_length = min(
            max_length * MAX_LENGTH_MULTIPLIER, model_max - MODEL_MAX_BUFFER, SAFE_MAX_LENGTH
        )
        try:
            final_summary = model.summarize(
                final_summary,
                max_length=safe_max_length,
                min_length=min_length,
                do_sample=False,
                prompt=prompt,
            )

            # Validate: if summary is suspiciously long, it might be hallucinating
            if len(final_summary) > len(selected_summaries) * REPETITIVE_SUMMARY_THRESHOLD:
                logger.warning(
                    "Final summary suspiciously long, using extractive summaries directly "
                    "(no further summarization to prevent hallucinations)"
                )
                return join_summaries_with_structure(selected_summaries)

            return final_summary
        except Exception as e:
            logger.warning(f"Final summarization failed ({e}), using extractive summaries directly")
            return join_summaries_with_structure(selected_summaries)
    else:
        logger.debug("Using extractive summaries directly (no further summarization)")
        return final_summary


def combine_summaries_mini_map_reduce(
    model: "SummaryModel",
    combined_text: str,
    chunk_summaries: List[str],
    max_length: int,
    min_length: int,
    prompt: Optional[str],
    combined_tokens: int,
    target_tokens: int,
    max_passes: int = MAX_HIERARCHICAL_PASSES,
) -> str:
    """Combine summaries using iterative mini map-reduce approach (fully abstractive).

    This implements a recursive/iterative abstractive approach:
    - Loop until combined summaries are small enough for single-pass abstractive reduce
    - Each iteration: chunk → summarize → join
    - Final iteration: single-pass abstractive reduce

    Args:
        model: Summary model instance
        combined_text: Combined chunk summaries text
        chunk_summaries: List of chunk summaries (for logging)
        max_length: Max summary length
        min_length: Min summary length
        prompt: Optional prompt
        combined_tokens: Token count in combined text
        target_tokens: Target token count for final single-pass reduce (model-aware threshold)
        max_passes: Maximum hierarchical passes before falling back to extractive approach

    Returns:
        Final summary
    """
    import time

    mini_map_start = time.time()

    # Get model's max position embeddings to calculate safe chunk size
    model_max = (
        getattr(model.model.config, "max_position_embeddings", BART_MAX_POSITION_EMBEDDINGS)
        if model.model and hasattr(model.model, "config")
        else BART_MAX_POSITION_EMBEDDINGS
    )

    # Calculate safe chunk size: use 80% of model max to leave room for special tokens
    mini_chunk_size_tokens = max(
        MINI_MAP_MIN_CHUNK_SIZE,
        min(int(model_max * MINI_MAP_CHUNK_SIZE_RATIO), model_max - MODEL_MAX_BUFFER),
    )

    # Ensure we have a tokenizer for token-based chunking
    if not model.tokenizer:
        raise RuntimeError("Model tokenizer not available for mini map-reduce token-based chunking")

    # Current working text and token count
    current_text = combined_text
    current_tokens = combined_tokens

    logger.debug(
        f"[MAP-REDUCE VALIDATION] ⚡ HIERARCHICAL REDUCE: "
        f"combined summaries ({current_tokens} tokens) exceed single-pass threshold "
        f"({target_tokens}), "
        f"executing up to {max_passes} chunk→summarize→join passes"
    )

    passes_run = 0
    last_section_summaries: List[str] = chunk_summaries

    for iteration in range(1, max_passes + 1):
        if current_tokens <= target_tokens:
            break

        iteration_start = time.time()
        passes_run += 1

        logger.debug(
            f"[MAP-REDUCE VALIDATION] ⚡ Hierarchical Iteration {iteration} "
            f"(REDUCE model: {model.model_name}): "
            f"Processing {current_tokens} tokens (threshold={target_tokens})"
        )

        # Step 1: Re-chunk current text into smaller chunks using token-based chunking
        mini_chunks = chunking.chunk_by_tokens(
            current_text, model.tokenizer, max_tokens=mini_chunk_size_tokens
        )

        # Validate chunk sizes to ensure they don't exceed model limit
        max_chunk_tokens = 0
        for i, chunk in enumerate(mini_chunks, 1):
            chunk_tokens = len(
                model.tokenizer.encode(  # type: ignore[attr-defined]
                    chunk, add_special_tokens=False
                )
            )
            max_chunk_tokens = max(max_chunk_tokens, chunk_tokens)
            if chunk_tokens > model_max:
                logger.error(
                    f"[MAP-REDUCE VALIDATION] ⚡ MINI MAP-REDUCE ERROR: "
                    f"Chunk {i} exceeds model limit: {chunk_tokens} tokens > {model_max} max. "
                    f"This should not happen with token-based chunking!"
                )

        logger.debug(
            f"[MAP-REDUCE VALIDATION] ⚡ Hierarchical Iteration {iteration} "
            f"Step 1 (REDUCE model: {model.model_name}): "
            f"Re-chunked into {len(mini_chunks)} section chunks "
            f"(target={mini_chunk_size_tokens} tokens, max_actual={max_chunk_tokens} tokens, "
            f"model_max={model_max})"
        )

        # Step 2: Map phase - summarize each chunk
        section_summaries = []
        section_max_length = int(
            min(
                SECTION_SUMMARY_MAX_TOKENS,
                model_max - MODEL_MAX_BUFFER,
                SAFE_MAX_LENGTH,
            )
        )
        section_max_length = max(section_max_length, SECTION_SUMMARY_MIN_TOKENS)
        section_min_length = min(
            section_max_length,
            max(min_length, SECTION_SUMMARY_MIN_TOKENS),
        )
        logger.debug(
            f"[MAP-REDUCE CONFIG] Hierarchical iteration {iteration}: "
            f"section_summary_range={section_min_length}-{section_max_length} tokens, "
            f"sections={len(mini_chunks)}"
        )
        for i, mini_chunk in enumerate(mini_chunks, 1):
            try:
                section_prompt = REDUCE_PROMPT_SHORT if prompt is None else prompt
                section_summary = model.summarize(
                    mini_chunk,
                    max_length=section_max_length,
                    min_length=section_min_length,
                    prompt=section_prompt,
                )
                if section_summary:
                    section_summaries.append(section_summary)
                    logger.debug(
                        f"[MAP-REDUCE VALIDATION] ⚡ Hierarchical Iteration {iteration} "
                        f"Step 2 (REDUCE model: {model.model_name}): "
                        f"Section {i}/{len(mini_chunks)} summarized "
                        f"({len(section_summary.split())} words)"
                    )
            except Exception as e:
                logger.debug(
                    f"[MAP-REDUCE VALIDATION] Mini map-reduce iteration {iteration}: "
                    f"failed to summarize section {i}: {e}"
                )
                continue

        if not section_summaries:
            logger.debug(
                f"[MAP-REDUCE VALIDATION] Hierarchical iteration {iteration}: "
                "No section summaries generated, falling back to extractive approach"
            )
            return combine_summaries_extractive(
                model,
                select_key_summaries(chunk_summaries),
                max_length,
                min_length,
                prompt,
                BART_MAX_POSITION_EMBEDDINGS,
            )

        # Step 3: Join summaries with newlines (preserves structure)
        current_text = join_summaries_with_structure(section_summaries)
        current_chars = len(current_text)
        current_words = len(current_text.split())
        if model.tokenizer:
            current_tokens = len(
                model.tokenizer.encode(  # type: ignore[attr-defined]
                    current_text, add_special_tokens=False
                )
            )
        else:
            current_tokens = current_chars // CHARS_PER_TOKEN_ESTIMATE

        last_section_summaries = section_summaries
        iteration_time = time.time() - iteration_start
        logger.debug(
            f"[MAP-REDUCE VALIDATION] ⚡ Hierarchical Iteration {iteration} "
            f"Step 3 (REDUCE model: {model.model_name}): "
            f"Summaries combined ({current_chars:,} chars, {current_words:,} words, "
            f"~{current_tokens:,} tokens) in {iteration_time:.1f}s"
        )

    if current_tokens > target_tokens:
        logger.debug(
            f"[MAP-REDUCE VALIDATION] Hierarchical reduce reached {iteration} passes "
            f"but still {current_tokens} tokens > threshold ({target_tokens}). "
            "Falling back to extractive approach."
        )
        return combine_summaries_extractive(
            model,
            select_key_summaries(chunk_summaries),
            max_length,
            min_length,
            prompt,
            model_max,
        )

    # Final abstractive reduce (now small enough for single pass)
    logger.debug(
        f"[MAP-REDUCE VALIDATION] ⚡ Hierarchical Final Step: "
        f"Combined summaries ({current_tokens} tokens) now <= threshold ({target_tokens}), "
        f"proceeding to single-pass abstractive reduce after {passes_run} iteration(s)"
    )

    final_summary = combine_summaries_abstractive(
        model,
        current_text,
        last_section_summaries,
        max_length,
        min_length,
        prompt,
        model_max,
        current_tokens,
    )

    total_mini_time = time.time() - mini_map_start

    logger.debug(
        f"[MAP-REDUCE VALIDATION] ⚡ MINI MAP-REDUCE COMPLETE: "
        f"total_time={total_mini_time:.1f}s ({iteration} iteration(s)), "
        f"input={combined_tokens} tokens -> output={len(final_summary.split())} words"
    )

    return final_summary


def combine_summaries_abstractive(
    model: "SummaryModel",
    combined_text: str,
    chunk_summaries: List[str],
    max_length: int,
    min_length: int,
    prompt: Optional[str],
    model_max: int,
    combined_tokens: int,
) -> str:
    """Combine summaries using abstractive approach (final summarization pass).

    Args:
        model: Summary model instance
        combined_text: Combined chunk summaries text
        chunk_summaries: List of chunk summaries (for fallback)
        max_length: Max summary length
        min_length: Min summary length
        prompt: Optional prompt
        model_max: Model's max position embeddings
        combined_tokens: Token count in combined text

    Returns:
        Final summary
    """
    final_max_length = int(
        min(
            max_length * FINAL_MAX_LENGTH_MULTIPLIER,
            model_max - MODEL_MAX_BUFFER,
            SAFE_MAX_LENGTH,
        )
    )

    logger.debug(
        f"Final summarization: {len(chunk_summaries)} chunks, "
        f"combined ~{combined_tokens} tokens, "
        f"using max_length={final_max_length} for final summary"
    )

    try:
        final_summary = model.summarize(
            combined_text,
            max_length=final_max_length,
            min_length=min_length,
            do_sample=False,
            prompt=prompt,
        )

        # Validate summary quality
        if len(final_summary) > len(combined_text) * SUMMARY_VALIDATION_THRESHOLD:
            logger.warning(
                f"Final summary length ({len(final_summary)} chars) is suspiciously close to "
                f"input length ({len(combined_text)} chars). Model may have failed to summarize. "
                "Returning summary as-is (abstractive path uses ALL summaries, no selection)."
            )
            # Don't select chunks - return what we got (even if suspicious)
            # Selection should only happen in extractive paths

        if len(final_summary) < min_length * MIN_SUMMARY_LENGTH_MULTIPLIER:
            logger.warning(
                f"Final summary seems too short ({len(final_summary)} chars). "
                "This might indicate summarization issues."
            )

        return final_summary
    except RuntimeError as e:
        error_msg = str(e).lower()
        if "invalid buffer size" in error_msg or "out of memory" in error_msg:
            # Abstractive path failed - re-raise to let caller decide
            # (they can fall back to extractive)
            logger.error(
                f"Abstractive summarization failed ({e}). "
                "Caller should fall back to extractive approach if needed. "
                "Abstractive paths must use ALL summaries, not a subset."
            )
        raise


def summarize_long_text(
    model: "SummaryModel",
    text: str,
    chunk_size: int = BART_MAX_POSITION_EMBEDDINGS,
    max_length: int = 150,
    min_length: int = 30,
    batch_size: Optional[int] = None,
    prompt: Optional[str] = None,
    use_word_chunking: bool = False,
    word_chunk_size: int = DEFAULT_WORD_CHUNK_SIZE,
    word_overlap: int = DEFAULT_WORD_OVERLAP,
    reduce_model: Optional["SummaryModel"] = None,
) -> str:
    """Summarize long text by chunking and combining summaries.

    This function implements a map-reduce workflow:
    1. Check if text fits without chunking (early exit)
    2. Prepare chunks (word-based or token-based)
    3. Map: Summarize each chunk (parallel or sequential)
    4. Reduce: Combine chunk summaries into final summary

    Args:
        model: Summary model instance
        text: Long input text (should be cleaned with clean_transcript first)
        chunk_size: Chunk size in tokens (from config, used as-is)
        max_length: Max summary length per chunk
        min_length: Min summary length per chunk
        batch_size: Batch size for parallel processing (CPU only)
        prompt: Optional instruction/prompt to prepend to guide summarization
        use_word_chunking: If True, use word-based chunking (recommended for BART/PEGASUS)
        word_chunk_size: Chunk size in words when use_word_chunking=True
            (MIN_WORD_CHUNK_SIZE-MAX_WORD_CHUNK_SIZE recommended)
        word_overlap: Overlap in words when use_word_chunking=True
            (MIN_WORD_OVERLAP-MAX_WORD_OVERLAP recommended)
        reduce_model: Optional separate model for reduce phase

    Returns:
        Combined summary
    """
    import time

    # Import preprocessing to avoid circular import
    from .. import preprocessing

    pipeline_start_time = time.time()

    # If no separate reduce model is provided, use the same model for map and reduce.
    if reduce_model is None:
        reduce_model = model

    # Use preprocessing module directly (avoid deprecation warning)
    cleaned_text = preprocessing.clean_for_summarization(text)
    if cleaned_text != text:
        removed_chars = len(text) - len(cleaned_text)
        removed_pct = (removed_chars / len(text) * 100) if len(text) else 0
        logger.debug(
            "[SPONSOR CLEANUP] Removed not clean segments before summarization: "
            f"{removed_chars:,} chars ({removed_pct:.1f}%)"
        )
        text = cleaned_text.strip()

    # === VALIDATION: Input metrics ===
    input_chars = len(text)
    input_words = len(text.split())
    if model.tokenizer:
        input_tokens = len(
            model.tokenizer.encode(text, add_special_tokens=False)  # type: ignore[attr-defined]
        )
    else:
        input_tokens = input_chars // CHARS_PER_TOKEN_ESTIMATE

    logger.debug(
        "[MAP-REDUCE VALIDATION] Input text: "
        f"{input_chars:,} chars, {input_words:,} words, ~{input_tokens:,} tokens"
    )
    logger.debug(
        "[MAP-REDUCE VALIDATION] Configuration: "
        f"max_length={max_length}, min_length={min_length}, "
        f"word_chunk_size={word_chunk_size if use_word_chunking else 'N/A'}, "
        f"word_overlap={word_overlap if use_word_chunking else 'N/A'}, "
        f"token_chunk_size={chunk_size}, "
        f"batch_size={batch_size if batch_size else 'N/A'}"
    )

    model_max_tokens = (
        getattr(model.model.config, "max_position_embeddings", BART_MAX_POSITION_EMBEDDINGS)
        if model.model and hasattr(model.model, "config")
        else BART_MAX_POSITION_EMBEDDINGS
    )
    requested_chunk_size = chunk_size
    chunk_size = max(1, min(chunk_size, model_max_tokens - MODEL_MAX_BUFFER))
    encoder_decoder_override = False
    if use_word_chunking:
        chunk_size = min(chunk_size, ENCODER_DECODER_TOKEN_CHUNK_SIZE)
        encoder_decoder_override = True

    logger.debug(
        "[MAP-REDUCE VALIDATION] Chunking strategy: "
        f"requested_chunk_size={requested_chunk_size} tokens, "
        f"model_max={model_max_tokens}, "
        f"effective_chunk_size={chunk_size} tokens, "
        f"encoder_decoder_override={'yes' if encoder_decoder_override else 'no'}"
    )

    # Step 1: Check if text fits without chunking (early exit)
    direct_summary = chunking.check_if_needs_chunking(
        model, text, chunk_size, max_length, min_length, prompt
    )
    if direct_summary is not None:
        output_chars = len(direct_summary)
        output_words = len(direct_summary.split())
        compression_ratio = input_chars / output_chars if output_chars > 0 else 0
        total_time = time.time() - pipeline_start_time
        logger.debug(
            "[MAP-REDUCE VALIDATION] Direct summary (no chunking): "
            f"output={output_chars:,} chars, {output_words:,} words, "
            f"compression={compression_ratio:.1f}x, time={total_time:.1f}s"
        )
        return direct_summary

    # Step 2: Prepare chunks
    chunks, chunk_size = chunking.prepare_chunks(
        model, text, chunk_size, use_word_chunking, word_chunk_size, word_overlap
    )

    # === VALIDATION: Chunking metrics ===
    chunk_sizes_chars = [len(chunk) for chunk in chunks]
    chunk_sizes_words = [len(chunk.split()) for chunk in chunks]
    if model.tokenizer:
        chunk_sizes_tokens = [
            len(
                model.tokenizer.encode(  # type: ignore[attr-defined]
                    chunk, add_special_tokens=False
                )
            )
            for chunk in chunks
        ]
    else:
        chunk_sizes_tokens = [c // CHARS_PER_TOKEN_ESTIMATE for c in chunk_sizes_chars]

    overlap_tokens = int(chunk_size * CHUNK_OVERLAP_RATIO)
    method_desc = "token-based (encoder-decoder override)" if use_word_chunking else "token-based"
    logger.debug(
        "[MAP-REDUCE VALIDATION] Chunking phase: "
        f"created {len(chunks)} chunks, "
        f"method={method_desc}, "
        f"chunk_size=tokens={chunk_size}, "
        f"overlap=tokens={overlap_tokens}"
    )
    logger.debug(
        "[MAP-REDUCE VALIDATION] Chunk size stats (words): "
        f"min={min(chunk_sizes_words)}, max={max(chunk_sizes_words)}, "
        f"avg={sum(chunk_sizes_words) // len(chunk_sizes_words)}"
    )
    if model.tokenizer:
        logger.debug(
            "[MAP-REDUCE VALIDATION] Chunk size stats (tokens): "
            f"min={min(chunk_sizes_tokens)}, max={max(chunk_sizes_tokens)}, "
            f"avg={sum(chunk_sizes_tokens) // len(chunk_sizes_tokens)}"
        )

    # Step 3: Map - Summarize each chunk
    map_start_time = time.time()
    chunk_summaries = summarize_chunks_map(
        model,
        chunks,
        max_length,
        min_length,
        prompt,
        batch_size,
        use_word_chunking,
        word_chunk_size,
        word_overlap,
        chunk_size,
    )
    map_time = time.time() - map_start_time

    # === VALIDATION: Map phase metrics ===
    map_output_chars = 0
    map_output_words = 0
    if chunk_summaries:
        summary_sizes_chars = [len(s) for s in chunk_summaries]
        summary_sizes_words = [len(s.split()) for s in chunk_summaries]
        map_output_chars = sum(summary_sizes_chars)
        map_output_words = sum(summary_sizes_words)
        map_compression_ratio = input_chars / map_output_chars if map_output_chars > 0 else 0

        logger.debug(
            "[MAP-REDUCE VALIDATION] Map phase: "
            f"processed {len(chunk_summaries)}/{len(chunks)} chunks, "
            f"time={map_time:.1f}s ({map_time/len(chunk_summaries):.2f}s/chunk), "
            f"output={map_output_chars:,} chars, {map_output_words:,} words, "
            f"compression={map_compression_ratio:.1f}x, "
            f"max_length={max_length}, min_length={min_length}"
        )
        logger.debug(
            "[MAP-REDUCE VALIDATION] Map output stats (words per chunk summary): "
            f"min={min(summary_sizes_words)}, max={max(summary_sizes_words)}, "
            f"avg={sum(summary_sizes_words) // len(summary_sizes_words)}"
        )
    else:
        logger.debug("[MAP-REDUCE VALIDATION] Map phase: No chunk summaries generated!")

    # Step 4: Reduce - Combine summaries into final result
    reduce_start_time = time.time()
    final_summary = combine_summaries_reduce(
        reduce_model, chunk_summaries, max_length, min_length, prompt
    )
    reduce_time = time.time() - reduce_start_time

    # === VALIDATION: Reduce phase and overall metrics ===
    final_chars = len(final_summary)
    final_words = len(final_summary.split())
    if model.tokenizer:
        final_tokens = len(
            model.tokenizer.encode(  # type: ignore[attr-defined]
                final_summary, add_special_tokens=False
            )
        )
    else:
        final_tokens = final_chars // CHARS_PER_TOKEN_ESTIMATE

    total_time = time.time() - pipeline_start_time
    overall_compression_ratio = input_chars / final_chars if final_chars > 0 else 0
    reduce_compression_ratio = (
        map_output_chars / final_chars if chunk_summaries and final_chars > 0 else 0
    )

    logger.debug(
        "[MAP-REDUCE VALIDATION] Reduce phase: "
        f"time={reduce_time:.1f}s, "
        f"input={map_output_chars:,} chars ({len(chunk_summaries)} summaries), "
        f"output={final_chars:,} chars, {final_words:,} words, ~{final_tokens:,} tokens, "
        f"compression={reduce_compression_ratio:.1f}x, "
        f"max_length={max_length}, min_length={min_length}"
    )
    logger.debug(
        "[MAP-REDUCE VALIDATION] Overall pipeline: "
        f"total_time={total_time:.1f}s "
        f"(map={map_time:.1f}s, reduce={reduce_time:.1f}s), "
        f"input={input_chars:,} chars -> output={final_chars:,} chars, "
        f"overall_compression={overall_compression_ratio:.1f}x, "
        f"chunks={len(chunks)}, model={model.model_name}, device={model.device}, "
        f"config: max_length={max_length}, min_length={min_length}, "
        f"word_chunk_size={word_chunk_size if use_word_chunking else 'N/A'}, "
        f"word_overlap={word_overlap if use_word_chunking else 'N/A'}"
    )

    return final_summary
