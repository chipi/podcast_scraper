"""Episode summarization using local transformer models.

This module provides functionality to generate concise summaries
from episode transcripts using local PyTorch transformer models.

Security Considerations:
- Model Loading: Uses Hugging Face transformers library to load pre-trained models
- Supply Chain Risk: Models are loaded from Hugging Face Hub (third-party source)
- Mitigation Strategies:
  1. TRUSTED_MODEL_SOURCES: Whitelist of verified model publishers
  2. Model Validation: _validate_model_source() warns about untrusted sources
  3. Revision Pinning: Optional revision parameter for reproducible builds
  4. User Choice: Warnings issued but custom models still allowed
- Recommendations:
  * Use default models (e.g., 'bart-large', 'fast') from trusted sources
  * Pin model revisions in production: SummaryModel(model, revision="abc123")
  * Review model source before using custom models
  * Keep transformers library updated for security patches

Note: Preprocessing functions (clean_transcript, remove_sponsor_blocks, etc.)
have been moved to preprocessing.py. Use those functions directly.

Low MI (radon): see docs/ci/CODE_QUALITY_TRENDS.md § Low-MI modules.
"""

import contextlib
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, cast, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    # Type hints only - these are not imported at runtime
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Pipeline


# Note: torch and transformers are imported lazily in methods that use them
# (e.g., _detect_device(), _load_model()) to avoid requiring ML dependencies
# at module import time. This allows unit tests to import this module without
# torch/transformers installed.

from ... import preprocessing
from ...preprocessing.profiles import apply_profile_with_stats

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _silence_hf_load_noise():
    """Silences Hugging Face 'from_pretrained' warnings that get printed to stderr/loggers.

    This is not 'more logging' — it prevents user-facing noise.
    """
    try:
        from transformers.utils import logging as hf_logging

        old_verbosity = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()  # block most HF warnings
    except ImportError:
        # transformers not available, skip silencing
        old_verbosity = None

    devnull = open(os.devnull, "w")
    with contextlib.redirect_stderr(devnull):
        try:
            yield
        finally:
            if old_verbosity is not None:
                try:
                    from transformers.utils import logging as hf_logging

                    hf_logging.set_verbosity(old_verbosity)
                except ImportError:
                    pass
            devnull.close()


def _load_pegasus_without_fake_warning(
    model_id: str,
    device: str,
    cache_dir: Optional[str] = None,
    revision: Optional[str] = None,
    local_files_only: bool = True,
):
    """Loads Pegasus and:
    - *allows* only the known/benign missing positional embedding keys
    - raises if anything else is missing/unexpected
    - silences the misleading stderr warning

    Args:
        model_id: HuggingFace model identifier (e.g., "google/pegasus-cnn_dailymail")
        device: Device to move model to (e.g., "cpu", "mps", "cuda")
        cache_dir: Optional cache directory
        revision: Optional model revision/commit SHA
        local_files_only: If True, only load from cache (no downloads)

    Returns:
        Tuple of (tokenizer, model)

    Raises:
        RuntimeError: If unexpected keys are missing or model config is invalid
    """
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    allowed_missing = {
        "model.encoder.embed_positions.weight",
        "model.decoder.embed_positions.weight",
    }

    # Prepare kwargs for from_pretrained
    # Security: trust_remote_code=False by default (Issue #379)
    # Pegasus models don't have safetensors files, so disable safetensors (like LED models)
    tokenizer_kwargs = {
        "local_files_only": local_files_only,
        "trust_remote_code": False,  # Security: don't execute remote code
        "use_safetensors": False,  # Pegasus doesn't have safetensors files
    }
    model_kwargs = {
        "local_files_only": local_files_only,
        "output_loading_info": True,  # Get loading info to validate
        "trust_remote_code": False,  # Security: don't execute remote code
        "use_safetensors": False,  # Pegasus doesn't have safetensors files
    }
    if cache_dir:
        tokenizer_kwargs["cache_dir"] = cache_dir  # type: ignore[assignment]
        model_kwargs["cache_dir"] = cache_dir  # type: ignore[assignment]
    if revision:
        tokenizer_kwargs["revision"] = revision  # type: ignore[assignment]
        model_kwargs["revision"] = revision  # type: ignore[assignment]

    # Load with silenced warnings
    with _silence_hf_load_noise():
        tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)  # nosec B615
        model, loading_info = AutoModelForSeq2SeqLM.from_pretrained(  # nosec B615
            model_id,
            **model_kwargs,
        )

    # Validate loading info
    missing = set(loading_info.get("missing_keys", []))
    unexpected = set(loading_info.get("unexpected_keys", []))
    mismatched = loading_info.get("mismatched_keys", [])

    # If anything important is missing, this is a real problem — fail fast.
    if missing and not missing.issubset(allowed_missing):
        raise RuntimeError(
            f"Pegasus load is NOT clean. Missing keys: {sorted(missing)}; "
            f"Unexpected: {sorted(unexpected)}; Mismatched: {mismatched}"
        )

    # Unexpected keys are usually also suspicious (version mismatch / wrong class).
    if unexpected:
        raise RuntimeError(
            f"Pegasus load has unexpected keys: {sorted(unexpected)}; mismatched={mismatched}"
        )

    # Optional sanity: ensure it's actually configured as static positions (expected for Pegasus).
    if getattr(model.config, "static_position_embeddings", None) is False:
        raise RuntimeError(
            "Pegasus config indicates non-static positional embeddings; "
            "this is unexpected for google/pegasus-cnn_dailymail."
        )

    # Move to device and set eval mode
    model = model.to(device)
    model.eval()

    # Log successful load with validation info
    if missing:
        logger.info(
            f"[PEGASUS LOAD] Model loaded successfully. "
            f"Expected missing keys (static positional embeddings): {sorted(missing)}"
        )
    else:
        logger.info("[PEGASUS LOAD] Model loaded successfully (all keys present)")

    return tokenizer, model


# Hugging Face cache directory (standard locations)
# Newer transformers versions use "hub"
HF_CACHE_BASE = Path.home() / ".cache" / "huggingface"
HF_CACHE_DIR = HF_CACHE_BASE / "hub"  # Default for newer transformers

# Model context window limits
BART_MAX_POSITION_EMBEDDINGS = 1024  # Standard BART/PEGASUS model limit
LED_MAX_CONTEXT_WINDOW = 16384  # LED (Longformer) model context window

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
# Minimum chunk size for MAP stage; smaller chunks are merged into previous (Issue #428)
MAP_CHUNK_MIN_TOKENS = 80

# Token limits for MAP phase chunk summaries (Issue #283)
# Increased from 80-160 to 150-300 to allow more detail preservation
# and reduce extractive behavior in BART/LED models
CHUNK_SUMMARY_MIN_TOKENS = 150  # Target lower bound for map summaries (was 80)
CHUNK_SUMMARY_MAX_TOKENS = 300  # Target upper bound for map summaries (was 160)
SECTION_SUMMARY_MIN_TOKENS = 150  # Target lower bound for hierarchical section summaries (was 80)
SECTION_SUMMARY_MAX_TOKENS = 300  # Target upper bound for hierarchical section summaries (was 160)
# Token limits for REDUCE phase final summary (Issue #283)
# Increased from 200-480 to 400-800 to produce more comprehensive summaries
FINAL_SUMMARY_MIN_TOKENS = 400  # Target lower bound for final reduce (was 200)
FINAL_SUMMARY_MAX_TOKENS = 800  # Target upper bound for final reduce (was 480)
# Sponsor removal patterns
SPONSOR_BLOCK_PATTERNS = [
    r"this episode is brought to you by.*?(?=\n\n|\Z)",
    r"today['’]s episode is sponsored by.*?(?=\n\n|\Z)",
    r"our sponsor(?:s)? (?:today|this week) (?:is|are).*?(?=\n\n|\Z)",
    r"thanks again to our (?:friends|sponsors) at.*?(?=\n\n|\Z)",
]
# Re-export OUTRO_BLOCK_PATTERNS from preprocessing for convenience
from ...preprocessing.core import OUTRO_BLOCK_PATTERNS  # noqa: F401

MAX_SPONSOR_BLOCK_CHARS = 3000  # max size of a single sponsor block
MAX_SPONSOR_REMOVAL_RATIO = 0.6  # never remove more than 60% of the text
REDUCE_PROMPT = (
    "You are summarizing a full podcast episode using multiple partial summaries as input.\n"
    "Your task is to produce a clear, accurate, and cohesive final summary.\n"
    "\n"
    "Follow these principles:\n"
    "\n"
    "CONTENT SELECTION\n"
    "- Focus on the most important ideas, insights, arguments, explanations, "
    "turning points, and decisions.\n"
    "- Prioritize what the speakers were trying to explain, what changed over "
    "time, what challenges existed, what solutions were discussed, and what "
    "strategies or lessons emerged.\n"
    "- Emphasize cause-and-effect, reasoning, motivations, and big takeaways.\n"
    "- Include only information that contributes to understanding the episode’s key themes.\n"
    "\n"
    "CONTENT FILTERING\n"
    "- Ignore sponsorships, promotions, discount codes, calls to action, and "
    "housekeeping notes.\n"
    "- Exclude generic intros/outros, chit-chat, fillers, and unrelated "
    "anecdotes unless essential to a key idea.\n"
    "- Do NOT include direct quotes, speaker names, banter, or dialogue-style "
    "formatting.\n"
    "- Do NOT attribute statements to specific individuals unless critical to meaning.\n"
    "\n"
    "STYLE\n"
    "- Write in a neutral, professional, narrative voice with well-structured "
    "paragraphs.\n"
    "- Use smooth transitions; avoid repetition, generic reflections, and "
    "unsupported conclusions.\n"
    "- Do NOT reference the structure of the source "
    "(e.g., 'in this section,' 'in these chunks,' 'later they said').\n"
    "\n"
    "OUTPUT LENGTH & QUALITY\n"
    "- Aim for a comprehensive, medium-length summary (2–4 paragraphs).\n"
    "- Ensure the summary reads as a complete, standalone explanation of the "
    "episode's core points."
)
REDUCE_PROMPT_SHORT = (
    "Summarize the following podcast episode into 2–4 paragraphs. "
    "Focus on the most important ideas, decisions, and lessons. "
    "Ignore sponsorships, promotions, intros, outros, and small talk."
)

# Token estimation
CHARS_PER_TOKEN_ESTIMATE = 4  # Rough estimate: 1 token ≈ 4 characters
CHARS_PER_TOKEN_FOR_LENGTH_CHECK = 8  # More conservative estimate for length validation

# Summarization thresholds
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
MINI_MAP_SECTION_MAX_LENGTH_MULTIPLIER = (
    1.35  # Allow section summaries to be longer than map chunks
)
MAX_HIERARCHICAL_PASSES = 4  # Maximum number of hierarchical chunk→summarize passes
SUMMARY_VALIDATION_THRESHOLD = 0.6  # Flag summary if length > 60% of input (likely failed)
REPETITIVE_SUMMARY_THRESHOLD = 0.8  # Flag summary if length > 80% of selected summaries
MAX_LENGTH_MULTIPLIER = 2  # Multiplier for safe max length calculation
FINAL_MAX_LENGTH_MULTIPLIER = 1.8  # Multiplier for final max length calculation
MODEL_MAX_BUFFER = 200  # Buffer to subtract from model max for safety
SAFE_MAX_LENGTH = 800  # Safe maximum length for final summarization (was 512, Issue #283)

# Chunk selection thresholds
FEW_CHUNKS_THRESHOLD = 3  # Use all chunks if <= this many
MEDIUM_CHUNKS_THRESHOLD = 10  # Use first/middle/last if <= this many
FALLBACK_CHUNKS_COUNT = 3  # Number of chunks to use for fallback

# Progress reporting
PROGRESS_LOG_INTERVAL = 5  # Log progress every N chunks (reduced from 20 for better visibility)
SECONDS_PER_CHUNK_ESTIMATE = 3  # Rough estimate for time calculation

# Parallel processing
MAX_PARALLEL_WORKERS = 4  # Maximum number of parallel workers for CPU processing

# Text processing thresholds
MIN_TEXT_LENGTH = 50  # Minimum text length for processing
LONG_TEXT_THRESHOLD = 50000  # Character threshold for long text (use chunking)
MIN_SENTENCE_LENGTH = 20  # Minimum sentence length for validation
MAX_REPETITIONS_THRESHOLD = 3  # Maximum repetitions before flagging as repetitive
MIN_SUMMARY_LENGTH_MULTIPLIER = 2  # Minimum summary should be at least 2x min_length

# Instruction leak protection
INSTRUCTION_LEAK_PATTERNS = [
    r"your task is to",
    r"aim for a comprehensive",
    r"write in a neutral",
    r"do not reference the structure",
    r"output length & quality",
    r"follow these principles",
    r"content selection",
    r"content filtering",
    r"style",
    r"summarize the following",
]

# Generation parameters - ALL DEFAULTS NOW IN CONFIG (config.py)
# These constants have been moved to Config to eliminate hardcoded values
# Import from config module: DEFAULT_MAP_NUM_BEAMS, DEFAULT_REDUCE_NUM_BEAMS, etc.
# See config.py for all default values

# Post-distill pruning patterns - ONLY remove KNOWN garbage after distillation
# IMPORTANT: Never do "importance pruning" before abstraction.
# BART/LED need messy, redundant, narrative content - they decide importance.
# These patterns are for AFTER DISTILL only, and are very conservative.

# Credit patterns that might leak through to final summary
# These are safe to remove because they're definitively not content
POST_DISTILL_CREDIT_PATTERNS = [
    r"\bproduced by\b",
    r"\bedited by\b",
    r"\bfact[- ]?checked by\b",
    r"\bengineering by\b",
    r"\bspecial thanks to\b",
]

# Rhetorical filler starts - sentences starting with these are low-value
RHETORICAL_FILLER_STARTS = [
    r"^if there'?s one (?:headline|thing|takeaway)",
    r"^(?:and )?that'?s (?:the|a) wrap",
    r"^(?:so )?what does (?:this|that|it) (?:all )?mean",
    r"^(?:in )?(?:the )?end,?\s",
    r"^(?:at )?the end of the day",
    r"^(?:all )?(?:in )?all,?\s",
    r"^(?:to )?sum(?:marize|ming)? (?:up|it all)",
]

# Structured joiner constants
MAX_BULLETS_PER_CHUNK = 12  # Cap bullets per chunk to prevent overflow (raised from 8)
MAX_BULLET_CHARS = 300  # Max characters per bullet (raised from 150 for longer summaries)
MIN_BULLET_CHARS = 10  # Min characters for a valid bullet
SENTENCE_SPLIT_THRESHOLD = 220  # Only split sentences if line > this many chars
MAX_CHUNK_CHARS = 2500  # Max total characters per chunk block (raised from 1000)
# Threshold for skipping bulletization - if combined text is small, just join with newlines
SKIP_BULLETIZATION_THRESHOLD = 4000  # If total map text < this, skip bulletization

# Pattern for stripping leading numbers from bullets
_LEADING_NUMBER_PATTERN = re.compile(r"^\d+[\.\)]\s*")

# Default model selection
# Note: BART models have 1024 token limit, requiring chunking for long texts
# PEGASUS models were trained directly for summarization, 1024 token limit
# LED (Longformer) models support up to 16k tokens, eliminating need for chunking
# Trusted model sources - these are verified, well-known models from reputable organizations
# Security: Using models from these sources reduces supply chain risks
# Users can override with custom models, but will receive security warnings
TRUSTED_MODEL_SOURCES = {
    "facebook",  # Meta AI (BART models)
    "google",  # Google Research (PEGASUS, T5 models)
    "sshleifer",  # Sam Shleifer (DistilBART - widely used)
    "allenai",  # Allen Institute for AI (LED models)
}

DEFAULT_SUMMARY_MODELS = {
    # BART-large (best quality, ~2GB memory), 1024 token limit
    "bart-large": "facebook/bart-large-cnn",
    # BART-base (smallest, lowest memory ~500MB), 1024 token limit
    "bart-small": "facebook/bart-base",
    # DistilBART (faster, lower memory ~300MB), 1024 token limit
    "fast": "sshleifer/distilbart-cnn-12-6",
    # PEGASUS-large (trained for summarization ~2.5GB), 1024 tokens
    "pegasus": "google/pegasus-large",
    # PEGASUS-CNN (production baseline, trained on CNN/DailyMail ~2.5GB), 1024 tokens
    "pegasus-cnn": "google/pegasus-cnn_dailymail",
    # PEGASUS-xsum (short summaries ~2.5GB), 1024 tokens
    "pegasus-xsum": "google/pegasus-xsum",
    # LED-large (long docs 16k tokens, ~2.5GB), NO chunking
    "long": "allenai/led-large-16384",
    # LED-large alias (explicit name for clarity)
    "long-large": "allenai/led-large-16384",
    # LED-base (long docs 16k tokens, ~1GB), NO chunking (production baseline reduce model)
    "long-fast": "allenai/led-base-16384",
}


def resolve_model_name(model_id: str) -> str:
    """Resolve model identifier to full HuggingFace model ID.

    Handles both aliases (e.g., "bart-small") and raw HuggingFace IDs (e.g., "facebook/bart-base").
    If the model_id contains "/", it's treated as a raw HF ID and passed through.

    Args:
        model_id: Model identifier (alias or raw HF ID)

    Returns:
        Full HuggingFace model ID

    Examples:
        >>> resolve_model_name("bart-small")
        "facebook/bart-base"
        >>> resolve_model_name("facebook/bart-base")
        "facebook/bart-base"
        >>> resolve_model_name("microsoft/DialoGPT-medium")
        "microsoft/DialoGPT-medium"
    """
    # NEW: raw HF model id passthrough (if contains "/", treat as raw ID)
    if "/" in model_id:
        return model_id

    # Alias-based resolution
    if model_id in DEFAULT_SUMMARY_MODELS:
        return DEFAULT_SUMMARY_MODELS[model_id]

    # If not found and doesn't contain "/", raise error
    raise ValueError(
        f"Unknown model id: {model_id}. "
        f"Available aliases: {list(DEFAULT_SUMMARY_MODELS.keys())}. "
        f"Or use a raw HuggingFace model ID (e.g., 'facebook/bart-base')."
    )


# Default prompt for summarization
# Important: Explicitly instruct model to avoid hallucinations and stick to source content
DEFAULT_SUMMARY_PROMPT = (
    "Summarize the following podcast episode transcript accurately. "
    "Focus on the main topics, key insights, and important discussions. "
    "Only include information that is explicitly stated in the transcript. "
    "Do not add, infer, or invent any information not present in the original text:"
)


def _validate_model_source(model_name: str) -> None:
    """Validate model source against allowlist (Issue #379).

    Security: This function enforces an allowlist of approved models to prevent
    config injection and supply chain attacks. Only models in the allowlist can be loaded.

    Args:
        model_name: HuggingFace model identifier

    Raises:
        ValueError: If model is not in the allowlist
    """
    from ...config_constants import ALLOWED_HUGGINGFACE_MODELS

    # Resolve alias to full model ID if needed
    resolved_name = resolve_model_name(model_name)

    if resolved_name not in ALLOWED_HUGGINGFACE_MODELS:
        raise ValueError(
            f"Model '{resolved_name}' (from '{model_name}') is not in the allowlist. "
            f"Allowed models: {sorted(ALLOWED_HUGGINGFACE_MODELS)}. "
            "This prevents config injection and supply chain attacks. "
            "To add a new model, update ALLOWED_HUGGINGFACE_MODELS in config_constants.py."
        )
    logger.debug(f"Model '{resolved_name}' validated against allowlist")


def _validate_model_source_legacy(model_name: str) -> None:
    """Legacy validation function (deprecated, kept for reference).

    Security: This function checks if a model comes from a known trusted source
    to mitigate supply chain risks. Custom models from unknown sources will
    trigger a warning but are still allowed (user choice).

    Note: Does not log sensitive information (model names, source identifiers)
    to avoid clear-text logging of potentially untrusted input.

    Args:
        model_name: Model identifier (e.g., "facebook/bart-large-cnn")

    Note:
        - Logs DEBUG for default trusted models (no sensitive info)
        - Logs WARNING for custom models from untrusted sources (generic message)
        - Does not block loading (respects user choice)
    """
    # Check if model is from a trusted source
    # Security: Avoid logging sensitive information (source names, model identifiers) in clear text
    if "/" in model_name:
        source = model_name.split("/")[0]
        if source in TRUSTED_MODEL_SOURCES:
            logger.debug("Loading model from verified trusted source")
        else:
            logger.warning(
                "⚠️  SECURITY NOTICE: Loading model from custom (untrusted) source.\n"
                "    This model is not from a pre-verified trusted source.\n"
                "    Only use models from sources you trust.\n"
                "    Consider using default models (e.g., 'bart-large', 'fast', 'pegasus') "
                "for better security."
            )
    else:
        # Local model or non-standard identifier
        logger.warning(
            "⚠️  SECURITY NOTICE: Loading model with non-standard identifier.\n"
            "    Unable to verify model source.\n"
            "    Only load models from sources you trust.\n"
            "    Use default model names for verified sources."
        )


def remove_sponsor_blocks(text: str) -> str:
    """Remove common sponsor block phrases from text (e.g. 'brought to you by', 'sponsored by').

    Strips up to the next blank line or 2000 chars after each matched phrase.
    Used to reduce noise in transcripts before summarization.

    Args:
        text: Raw transcript or summary text.

    Returns:
        Text with sponsor blocks removed.
    """
    lower = text.lower()
    cleaned = text
    for phrase in [
        "this episode is brought to you by",
        "today's episode is sponsored by",
        "today’s episode is sponsored by",
        "our sponsors today are",
    ]:
        idx = lower.find(phrase)
        if idx == -1:
            continue
        # Remove, say, up to the next blank line OR up to N chars
        end = cleaned.find("\n\n", idx)
        if end == -1 or end - idx > 2000:
            end = min(idx + 2000, len(cleaned))
        cleaned = cleaned[:idx] + cleaned[end:]
        lower = cleaned.lower()
    return cleaned


def select_summary_model(cfg) -> str:
    """Select MAP-phase model based on configuration.

    Defaults to Pegasus-CNN for production quality chunk summarization
    (baseline_ml_prod_authority_v1). This is the MAP model used to summarize
    individual chunks.

    Note: Tests typically use "facebook/bart-base" (smaller, faster) for speed,
    but production defaults to "google/pegasus-cnn_dailymail" (production baseline).

    Args:
        cfg: Configuration object with summary_model field

    Returns:
        Model identifier string (resolved from DEFAULT_SUMMARY_MODELS if key provided)
    """
    if cfg.summary_model:
        model_key = cast(str, cfg.summary_model)
        return resolve_model_name(model_key)

    # Default to Pegasus-CNN for MAP phase (production baseline: baseline_ml_prod_authority_v1)
    default_model = DEFAULT_SUMMARY_MODELS.get("pegasus-cnn")
    if not default_model:
        raise ValueError("DEFAULT_SUMMARY_MODELS['pegasus-cnn'] is not defined")
    return default_model


def select_reduce_model(cfg, _default_model_name: str) -> str:
    """Select reduce-phase model based on configuration.

    Defaults to LED-base for accurate, long-context final summarization
    (baseline_ml_prod_authority_v1). This hybrid approach is the production baseline:
    - MAP with Pegasus-CNN (production-optimized chunk summaries)
    - REDUCE with LED-base (accurate, handles long combined summaries without hallucination)

    If ``cfg.summary_reduce_model`` is not set, defaults to LED-base instead
    of falling back to the map model. This provides the production baseline quality by default.

    Args:
        cfg: Configuration object with summary_reduce_model field
        _default_model_name: Map model name (used only if reduce model explicitly set to same)

    Returns:
        Model identifier string (resolved from DEFAULT_SUMMARY_MODELS if key provided)
    """
    reduce_key = getattr(cfg, "summary_reduce_model", None)
    if not reduce_key:
        # Default to LED-base for reduce phase (production baseline: baseline_ml_prod_authority_v1)
        default_model = DEFAULT_SUMMARY_MODELS.get("long-fast")
        if not default_model:
            raise ValueError("DEFAULT_SUMMARY_MODELS['long-fast'] is not defined")
        return default_model

    reduce_key = cast(str, reduce_key)
    # Use resolve_model_name for consistent alias resolution and raw HF ID passthrough
    return resolve_model_name(reduce_key)


def _resolve_summarize_generation_params(
    is_reduce_phase: bool,
    is_distill_phase: bool,
    num_beams: Optional[int],
    no_repeat_ngram_size: Optional[int],
    length_penalty: Optional[float],
    early_stopping: Optional[bool],
    repetition_penalty: Optional[float],
) -> Dict[str, Any]:
    """Resolve generation params from config when not explicitly provided."""
    from podcast_scraper import config as config_module

    if num_beams is None:
        num_beams = (
            config_module.DEFAULT_DISTILL_NUM_BEAMS
            if is_distill_phase
            else (
                config_module.DEFAULT_REDUCE_NUM_BEAMS
                if is_reduce_phase
                else config_module.DEFAULT_MAP_NUM_BEAMS
            )
        )
    if no_repeat_ngram_size is None:
        no_repeat_ngram_size = (
            config_module.DEFAULT_DISTILL_NO_REPEAT_NGRAM_SIZE
            if is_distill_phase
            else (
                config_module.DEFAULT_REDUCE_NO_REPEAT_NGRAM_SIZE
                if is_reduce_phase
                else config_module.DEFAULT_MAP_NO_REPEAT_NGRAM_SIZE
            )
        )
    if length_penalty is None:
        length_penalty = (
            config_module.DEFAULT_DISTILL_LENGTH_PENALTY
            if is_distill_phase
            else (
                config_module.DEFAULT_REDUCE_LENGTH_PENALTY
                if is_reduce_phase
                else config_module.DEFAULT_MAP_LENGTH_PENALTY
            )
        )
        if is_reduce_phase:
            early_stopping = False
    elif early_stopping is None:
        early_stopping = config_module.DEFAULT_MAP_EARLY_STOPPING
    if repetition_penalty is None:
        repetition_penalty = (
            config_module.DEFAULT_DISTILL_REPETITION_PENALTY
            if is_distill_phase
            else (
                config_module.DEFAULT_REDUCE_REPETITION_PENALTY
                if is_reduce_phase
                else config_module.DEFAULT_MAP_REPETITION_PENALTY
            )
        )
    return {
        "num_beams": num_beams,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "length_penalty": length_penalty,
        "early_stopping": early_stopping,
        "repetition_penalty": repetition_penalty,
    }


def _warn_if_input_very_long(input_text: str) -> None:
    """Log a warning if input text is very long (risk of buffer errors on MPS)."""
    if len(input_text) > 100000:  # ~25k tokens, well above safe chunk size
        logger.warning(
            f"Text is very long ({len(input_text)} chars), consider using chunking. "
            "This may cause buffer size errors on MPS."
        )


def _resolve_effective_length_limits(
    max_new_tokens: Optional[int],
    min_new_tokens: Optional[int],
    max_length: int,
    min_length: int,
) -> Tuple[int, int]:
    """Return (effective_max_new_tokens, effective_min_new_tokens)."""
    effective_max = max_new_tokens if max_new_tokens is not None else max_length
    effective_min = min_new_tokens if min_new_tokens is not None else min_length
    return effective_max, effective_min


def _extract_summary_text_from_pipeline_result(result: Any) -> str:
    """Extract summary_text from pipeline result (list or dict)."""
    if isinstance(result, list) and len(result) > 0:
        return cast(str, result[0].get("summary_text", "")).strip()
    if isinstance(result, dict):
        return cast(str, result.get("summary_text", "")).strip()
    return ""


def _build_summarize_pipeline_kwargs(
    effective_truncation: bool,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    max_new_tokens: Optional[int],
    effective_max_new_tokens: int,
    min_new_tokens: Optional[int],
    effective_min_new_tokens: int,
    max_length: int,
    min_length: int,
    encoder_no_repeat_ngram_size: Optional[int],
    do_sample: bool,
    num_beams: int,
    length_penalty: float,
) -> Dict[str, Any]:
    """Build pipeline kwargs for summarization."""
    kwargs = {
        "truncation": effective_truncation,
        "repetition_penalty": repetition_penalty,
        "no_repeat_ngram_size": no_repeat_ngram_size,
    }
    if max_new_tokens is not None:
        kwargs["max_new_tokens"] = effective_max_new_tokens
    else:
        kwargs["max_length"] = max_length
    if min_new_tokens is not None:
        kwargs["min_new_tokens"] = effective_min_new_tokens
    else:
        kwargs["min_length"] = min_length
    if encoder_no_repeat_ngram_size is not None:
        kwargs["encoder_no_repeat_ngram_size"] = encoder_no_repeat_ngram_size
    if do_sample:
        kwargs["do_sample"] = True
    else:
        kwargs["num_beams"] = num_beams
        kwargs["length_penalty"] = length_penalty
        kwargs["early_stopping"] = True
    return kwargs


def _load_with_retry_summarizer(load_func, model_name: str, model_type: str = "model"):
    """Load model/tokenizer with retry and fallback to cache clearing on failure."""
    from requests.exceptions import (
        ConnectionError,
        HTTPError,
        RequestException,
        Timeout,
    )

    from ...utils.retry import retry_with_exponential_backoff

    retryable = (
        ConnectionError,
        HTTPError,
        Timeout,
        RequestException,
        OSError,
    )
    try:
        return retry_with_exponential_backoff(
            load_func,
            max_retries=3,
            initial_delay=1.0,
            max_delay=30.0,
            retryable_exceptions=retryable,
        )
    except Exception as e:
        logger.warning(
            "%s loading failed for %s: %s. Clearing cache and retrying once...",
            model_type.capitalize(),
            model_name,
            e,
        )
        try:
            from ...cache.manager import delete_transformers_model_cache

            deleted, freed_bytes = delete_transformers_model_cache(
                model_name, confirm=False, force=True
            )
            if deleted:
                logger.info(
                    "Cleared cache for %s (%.1f MB freed)",
                    model_name,
                    freed_bytes / (1024 * 1024),
                )
        except Exception as cache_error:
            logger.warning("Failed to clear cache for %s: %s", model_name, cache_error)
        logger.info("Retrying %s load for %s after cache clear...", model_type, model_name)
        try:
            return load_func()
        except Exception as retry_error:
            logger.error(
                "%s load failed again after cache clear: %s. "
                "Please run 'make preload-ml-models' to re-download the model.",
                model_type.capitalize(),
                retry_error,
            )
            raise


class SummaryModel:
    """Wrapper for local transformer summarization model."""

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        revision: Optional[str] = None,
    ):
        """Initialize summary model.

        Args:
            model_name: Hugging Face model identifier (e.g., "facebook/bart-large-cnn")
            device: Device to use ("cuda", "mps", "cpu", or None for auto-detection)
            cache_dir: Custom cache directory for model files
            revision: Specific model revision/commit hash for reproducibility and security.
                     Example: "main" or a specific commit hash like "abc123def456"
                     If None, uses the latest version (less secure but more convenient)
        """
        if not model_name:
            raise ValueError("model_name cannot be None or empty")
        self.model_name = model_name

        # Force pinned revision for models to avoid PR refs and ensure consistency
        # Use commit SHA from config_constants for reproducibility (Issue #379)
        model_lower = model_name.lower()
        pinned_revision = None
        model_type = None

        if "pegasus" in model_lower:
            from ...config_constants import PEGASUS_CNN_DAILYMAIL_REVISION

            pinned_revision = PEGASUS_CNN_DAILYMAIL_REVISION
            model_type = "PEGASUS"
        elif "led-base-16384" in model_lower or model_name == "allenai/led-base-16384":
            from ...config_constants import LED_BASE_16384_REVISION

            pinned_revision = LED_BASE_16384_REVISION
            model_type = "LED-BASE"
        elif "led-large-16384" in model_lower or model_name == "allenai/led-large-16384":
            from ...config_constants import LED_LARGE_16384_REVISION

            pinned_revision = LED_LARGE_16384_REVISION
            model_type = "LED-LARGE"

        if pinned_revision:
            if revision and revision != pinned_revision:
                logger.warning(
                    f"[{model_type}] Overriding revision '{revision}' with pinned revision "
                    f"'{pinned_revision}' for {model_type} model to ensure consistent, "
                    f"stable weights (avoiding PR refs)"
                )
            self.revision = pinned_revision
            # Log at ERROR when revision is not a SHA so unpinned use is visible (Issue #428)
            from ...config_constants import is_sha_revision

            if not is_sha_revision(pinned_revision):
                logger.error(
                    "[%s] Revision is not a pinned commit SHA: %r. Update %s_REVISION in "
                    "config_constants.py with a 40-char commit hash for reproducibility.",
                    model_type,
                    pinned_revision,
                    model_type,
                )
        else:
            self.revision = revision  # type: ignore[assignment]

        self.device = self._detect_device(device)

        # Security: Validate model source and sanitize inputs (Issue #379)
        _validate_model_source(model_name)
        from ...utils.path_validation import sanitize_model_name

        sanitized_model_name = sanitize_model_name(self.model_name)
        if sanitized_model_name != self.model_name:
            logger.warning(f"Model name sanitized: '{self.model_name}' -> '{sanitized_model_name}'")
            self.model_name = sanitized_model_name

        # Use provided cache_dir or get from cache_utils (consistent with preload script)
        # cache_utils.get_transformers_cache_dir() handles all priority logic:
        # 1. HF_HUB_CACHE env var (CI sets this explicitly)
        # 2. Local project cache (.cache/huggingface/hub/)
        # 3. huggingface_hub.constants.HF_HUB_CACHE
        # 4. Default fallback (~/.cache/huggingface/hub/)
        if cache_dir:
            # Validate cache path to prevent path traversal (Issue #379)
            from ...utils.path_validation import validate_cache_path

            try:
                validated_path = validate_cache_path(cache_dir)
                self.cache_dir = str(validated_path)
            except ValueError as e:
                logger.error(f"Invalid cache directory: {e}")
                raise
        else:
            try:
                from ...cache import get_transformers_cache_dir

                self.cache_dir = str(get_transformers_cache_dir())
            except Exception:
                # If cache_utils not available, use default
                self.cache_dir = str(HF_CACHE_DIR)
        # Type hints use TYPE_CHECKING imports above
        # Runtime imports happen lazily in _load_model()
        self.tokenizer: Optional["AutoTokenizer"] = None
        self.model: Optional["AutoModelForSeq2SeqLM"] = None
        self.pipeline: Optional["Pipeline"] = None
        self._batch_size: Optional[int] = None  # For parallel chunk processing (CPU only)
        # Threading lock to serialize tokenizer/model access (tokenizers are not thread-safe)
        # This prevents "Already borrowed" errors when multiple episodes process concurrently
        self._summarize_lock = threading.Lock()
        self._load_model()

    def _detect_device(self, device: Optional[str]) -> str:
        """Detect and return appropriate device.

        Supports CUDA (NVIDIA), MPS (Apple Silicon), and CPU fallback.

        Args:
            device: Optional device string ("cuda", "mps", "cpu", or None)

        Returns:
            Device string
        """
        # Lazy import: Only import torch when this method is called
        # This allows the module to be imported without ML dependencies installed
        import torch  # noqa: F401

        if device:
            return device

        # Check for Apple Silicon MPS backend first (M4 Pro)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"

        # Check for CUDA (NVIDIA GPUs)
        if torch.cuda.is_available():
            return "cuda"

        return "cpu"

    def _load_tokenizer_step(self) -> None:
        """Load tokenizer (skip for Pegasus; loaded with model)."""
        from transformers import AutoTokenizer  # noqa: F401

        model_lower = self.model_name.lower()
        if "pegasus" in model_lower:
            return
        tokenizer_kwargs = {
            "cache_dir": self.cache_dir,
            "local_files_only": True,
            "trust_remote_code": False,
            "use_safetensors": True,
        }
        if self.revision:
            tokenizer_kwargs["revision"] = self.revision
            logger.debug("Using pinned revision: %s", self.revision)
        self.tokenizer = _load_with_retry_summarizer(
            lambda: AutoTokenizer.from_pretrained(  # nosec B615
                self.model_name,
                **tokenizer_kwargs,
            ),
            self.model_name,
            "tokenizer",
        )

    def _load_model_move_to_device_and_pipeline(self) -> None:
        """Move model to device (with fallback) and create pipeline."""
        import contextlib
        import io

        from transformers import pipeline

        with contextlib.redirect_stdout(io.StringIO()):
            try:
                self.model = self.model.to(self.device)  # type: ignore[union-attr]
            except (RuntimeError, Exception) as e:
                error_msg = str(e).lower()
                if self.device in ("mps", "cuda") and (
                    "out of memory" in error_msg
                    or "invalid buffer size" in error_msg
                    or "not implemented" in error_msg
                    or "unsupported" in error_msg
                ):
                    logger.warning(
                        "Device fallback: %s failed (%s). Falling back to CPU.",
                        self.device,
                        e,
                    )
                    self.device = "cpu"
                    self.model = self.model.to("cpu")  # type: ignore[union-attr]
                else:
                    raise
        pipeline_device = 0 if self.device == "cuda" else "mps" if self.device == "mps" else -1
        self.pipeline = pipeline(  # type: ignore[call-overload]
            "summarization",
            model=self.model,
            tokenizer=self.tokenizer,
            device=pipeline_device,
        )

    def _load_model_pegasus_sanity_and_clear_config(self) -> None:
        """Run Pegasus sanity check and clear max_new_tokens from generation config."""
        import torch

        if "pegasus" in self.model_name.lower() and hasattr(self, "_pegasus_health_checks"):
            try:
                test_summary = self.summarize(
                    "This is a test sentence for model verification.",
                    max_length=20,
                    min_length=5,
                )
                generate_ok = bool(test_summary and test_summary.strip())
                self._pegasus_health_checks["generate_ok"] = generate_ok  # type: ignore
                if generate_ok:
                    parts = [
                        f"{k}={v}" for k, v in self._pegasus_health_checks.items() if v is not None
                    ]
                    try:
                        import transformers

                        parts.extend(
                            [
                                f"transformers={transformers.__version__}",
                                f"torch={torch.__version__}",
                                f"device={self.device}",
                            ]
                        )
                        if self.revision:
                            parts.append(f"revision={self.revision}")
                        logger.info("PEGASUS_OK %s", " ".join(parts))
                    except Exception:
                        pass
                else:
                    logger.warning(
                        "[PEGASUS MODEL VERIFICATION] Sanity check failed: "
                        "model returned empty summary"
                    )
            except Exception as e:
                self._pegasus_health_checks["generate_ok"] = False  # type: ignore
                logger.warning("[PEGASUS MODEL VERIFICATION] Sanity check error: %s", e)
        if self.pipeline is not None and getattr(self.pipeline, "model", None) is not None:
            model = self.pipeline.model
            if getattr(model, "generation_config", None) is not None:
                setattr(model.generation_config, "max_new_tokens", None)
            if getattr(model, "config", None) is not None and hasattr(
                model.config, "max_new_tokens"
            ):
                setattr(model.config, "max_new_tokens", None)

    def _load_model(self) -> None:
        """Load model and tokenizer from cache or download."""
        import os

        try:
            original_hf_disable = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
            try:
                device_info = (
                    f"{self.device} (Apple Silicon GPU)"
                    if self.device == "mps"
                    else (
                        f"{self.device} (NVIDIA GPU)"
                        if self.device == "cuda"
                        else f"{self.device} (CPU)"
                    )
                )
                logger.info(
                    "Loading summarization model: %s on %s",
                    self.model_name,
                    device_info,
                )
                logger.debug("Cache directory: %s", self.cache_dir)
                self._load_tokenizer_step()
                # Load model
                # Security: Revision pinning provides reproducibility and prevents
                # supply chain attacks. If revision is None, latest version is used
                # (less secure but more convenient).
                # ALWAYS use local_files_only=True - we never allow libraries to download.
                # All downloads must go through our centralized preload script logic.
                logger.debug("Loading model from cache...")
                model_lower = self.model_name.lower()
                # Pegasus models don't have safetensors files, so disable safetensors
                use_safetensors = "pegasus" not in model_lower
                model_kwargs = {
                    "cache_dir": self.cache_dir,
                    # Always use cache only - downloads via preload script
                    "local_files_only": True,
                    "trust_remote_code": False,  # Security: don't execute remote code (Issue #379)
                    # Disable safetensors for Pegasus (no safetensors files)
                    "use_safetensors": use_safetensors,
                }
                if self.revision:
                    model_kwargs["revision"] = self.revision
                    if "pegasus" in model_lower:
                        logger.info(
                            f"[PEGASUS LOAD] Using pinned revision for model: {self.revision}"
                        )
                    else:
                        logger.debug(f"Using pinned revision: {self.revision}")
                else:
                    # This should not happen for Pegasus (we force revision="main" above)
                    # but log a warning if it does
                    if "pegasus" in model_lower:
                        logger.warning(
                            "[PEGASUS LOAD] No revision specified - this should not happen. "
                            "Forcing revision='main' to avoid PR refs."
                        )
                        model_kwargs["revision"] = "main"
                        self.revision = "main"

                # Use model-specific classes for better compatibility
                # This prevents "weights not initialized" warnings that can occur
                # when using AutoModel with certain model architectures
                if "pegasus" in model_lower:
                    # Use specialized Pegasus loader that:
                    # - Silences misleading "newly initialized" warnings
                    # - Validates that only expected positional embedding keys are missing
                    # - Fails fast if anything unexpected is missing
                    logger.debug("Using specialized Pegasus loader (validates loading info)")

                    # Log transformers version before loading
                    try:
                        import transformers

                        transformers_version = transformers.__version__
                        logger.info(f"[PEGASUS LOAD] transformers version: {transformers_version}")
                    except Exception:
                        transformers_version = "unknown"

                    # Load using specialized function that validates and silences warnings
                    # Note: Tokenizer is already loaded above, but we'll reload it in the function
                    # to ensure consistency. The function returns both tokenizer and model.
                    # Wrap with retry for transient errors (Issue #379)
                    self.tokenizer, self.model = _load_with_retry_summarizer(
                        lambda: _load_pegasus_without_fake_warning(
                            model_id=self.model_name,
                            device=self.device,
                            cache_dir=self.cache_dir,
                            revision=self.revision,
                            local_files_only=True,
                        ),
                        self.model_name,
                        "Pegasus model",
                    )

                    # Structured health check: verify model is working correctly
                    health_checks = {}
                    if self.model and hasattr(self.model, "config"):
                        config = self.model.config
                        static_pos = getattr(config, "static_position_embeddings", None)
                        max_pos_emb = getattr(config, "max_position_embeddings", None)
                        d_model = getattr(config, "d_model", None)

                        health_checks["static_pos"] = static_pos is True  # type: ignore[assignment]
                        health_checks["max_pos_emb"] = max_pos_emb  # type: ignore[assignment]
                        health_checks["d_model"] = d_model  # type: ignore[assignment]

                        # Verify embed_positions shapes match expected config
                        encoder_shape_ok = False
                        decoder_shape_ok = False
                        try:
                            encoder_embed_pos = getattr(
                                self.model.model.encoder,  # type: ignore[union-attr]
                                "embed_positions",
                                None,
                            )
                            decoder_embed_pos = getattr(
                                self.model.model.decoder,  # type: ignore[union-attr]
                                "embed_positions",
                                None,
                            )
                            if encoder_embed_pos and hasattr(encoder_embed_pos, "weight"):
                                enc_shape = encoder_embed_pos.weight.shape
                                encoder_shape_ok = enc_shape == (max_pos_emb, d_model)
                                health_checks["encoder_shape"] = (
                                    f"{enc_shape[0]}x{enc_shape[1]}"  # type: ignore[assignment]
                                )
                            if decoder_embed_pos and hasattr(decoder_embed_pos, "weight"):
                                dec_shape = decoder_embed_pos.weight.shape
                                decoder_shape_ok = dec_shape == (max_pos_emb, d_model)
                                health_checks["decoder_shape"] = (
                                    f"{dec_shape[0]}x{dec_shape[1]}"  # type: ignore[assignment]
                                )
                        except (AttributeError, TypeError):
                            pass

                        health_checks["embed_pos_shape_ok"] = encoder_shape_ok and decoder_shape_ok

                    # Store health checks for later (to add generate_ok)
                    self._pegasus_health_checks = health_checks

                    # Note: Sanity check will run after pipeline is created (see below)
                elif "led" in model_lower or "longformer" in model_lower:
                    # Use LEDForConditionalGeneration for LED models
                    from transformers import LEDForConditionalGeneration

                    logger.debug("Using LEDForConditionalGeneration for LED model")
                    # Workaround: LED models with safetensors trigger API calls even with
                    # local_files_only=True. Disable safetensors for LED models to avoid this.
                    led_model_kwargs = model_kwargs.copy()
                    led_model_kwargs["use_safetensors"] = False
                    logger.debug(
                        "Disabling safetensors for LED model to avoid API calls "
                        "during model loading"
                    )
                    self.model = _load_with_retry_summarizer(
                        lambda: LEDForConditionalGeneration.from_pretrained(  # nosec B615
                            self.model_name,
                            **led_model_kwargs,
                        ),
                        self.model_name,
                        "LED model",
                    )
                elif "bart" in model_lower:
                    # Use BartForConditionalGeneration for BART models
                    from transformers import BartForConditionalGeneration

                    logger.debug("Using BartForConditionalGeneration for BART model")
                    self.model = _load_with_retry_summarizer(
                        lambda: BartForConditionalGeneration.from_pretrained(  # nosec B615
                            self.model_name,
                            **model_kwargs,
                        ),
                        self.model_name,
                        "BART model",
                    )
                else:
                    # Fallback to AutoModelForSeq2SeqLM for other models
                    logger.debug("Using AutoModelForSeq2SeqLM (auto-detection)")
                    self.model = _load_with_retry_summarizer(
                        lambda: AutoModelForSeq2SeqLM.from_pretrained(  # nosec B615
                            self.model_name,
                            **model_kwargs,
                        ),
                        self.model_name,
                        "AutoModel",
                    )
                logger.debug("Model loaded successfully (cached for future runs)")
            finally:
                if original_hf_disable is None:
                    os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
                else:
                    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = original_hf_disable
            self._load_model_move_to_device_and_pipeline()
            self._load_model_pegasus_sanity_and_clear_config()
            logger.debug("Successfully loaded model: %s", self.model_name)

        except Exception as e:
            logger.error(f"Failed to load summarization model: {e}")
            raise

    def _summarize_truncate_input(
        self,
        text: str,
        max_input_tokens: Optional[int],
        truncation: Optional[bool],
    ) -> str:
        """Truncate input to max_input_tokens if needed. Returns updated text."""
        if max_input_tokens is None or not self.tokenizer:
            return text
        effective_truncation = truncation if truncation is not None else True
        with self._summarize_lock:
            input_tokens = self.tokenizer.encode(  # type: ignore[attr-defined]
                text, add_special_tokens=False
            )
            if len(input_tokens) <= max_input_tokens:
                return text
            if effective_truncation:
                truncated = input_tokens[:max_input_tokens]
                out = self.tokenizer.decode(  # type: ignore[attr-defined]
                    truncated, skip_special_tokens=True
                )
                logger.debug(
                    "Truncated input from %d to %d tokens",
                    len(input_tokens),
                    max_input_tokens,
                )
                return cast(str, out)
            logger.warning(
                "Input exceeds max_input_tokens (%d > %d) but truncation=False.",
                len(input_tokens),
                max_input_tokens,
            )
        return text

    def _summarize_apply_t5_prefix(self, text: str) -> str:
        """Add T5 'summarize: ' prefix if model is T5 and text does not have it."""
        if not self.model or not hasattr(self.model, "config"):
            return text
        if not hasattr(self.model.config, "model_type"):
            return text
        if getattr(self.model.config, "model_type", None) != "t5":
            return text
        if text.startswith("summarize: "):
            return text
        logger.debug("Added T5 prefix 'summarize: ' to input text")
        return "summarize: " + text

    def summarize(
        self,
        text: str,
        max_length: int = 150,
        min_length: int = 30,
        do_sample: bool = False,
        prompt: Optional[str] = None,
        is_reduce_phase: bool = False,
        is_distill_phase: bool = False,
        # New explicit generation params (override phase-based defaults if provided)
        max_new_tokens: Optional[int] = None,
        min_new_tokens: Optional[int] = None,
        num_beams: Optional[int] = None,
        no_repeat_ngram_size: Optional[int] = None,
        length_penalty: Optional[float] = None,
        early_stopping: Optional[bool] = None,
        repetition_penalty: Optional[float] = None,
        encoder_no_repeat_ngram_size: Optional[int] = None,
        max_input_tokens: Optional[int] = None,
        truncation: Optional[bool] = None,
    ) -> str:
        """Generate summary of input text.

        Args:
            text: Input text to summarize
            max_length: Maximum length of summary in tokens (includes input+output).
                Used as fallback if max_new_tokens is not provided.
            min_length: Minimum length of summary in tokens (includes input+output).
                Used as fallback if min_new_tokens is not provided.
            do_sample: Whether to use sampling (False = deterministic)
            prompt: Optional instruction/prompt to prepend to guide
                summarization
            is_reduce_phase: If True, use REDUCE-specific generation params
                (if explicit params not provided)
            is_distill_phase: If True, use DISTILL-specific generation params
                (if explicit params not provided)
            max_new_tokens: Maximum number of new tokens to generate (output-only).
                Preferred over max_length as it only counts output tokens.
                If not provided, falls back to max_length.
            min_new_tokens: Minimum number of new tokens to generate (output-only).
                Preferred over min_length as it only counts output tokens.
                If not provided, falls back to min_length.
            num_beams: Number of beams for beam search
                (overrides phase-based defaults)
            no_repeat_ngram_size: Size of n-grams to prevent repeating
                (overrides phase-based defaults)
            length_penalty: Length penalty for beam search (overrides phase-based defaults)
            early_stopping: Stop generation when all beams agree (overrides phase-based defaults)
            max_input_tokens: Maximum input tokens (truncation limit)
            truncation: Whether to truncate input text that exceeds max_input_tokens

        Returns:
            Generated summary text
        """
        if not self.pipeline:
            raise RuntimeError("Model not loaded")

        # Handle empty or very short text
        if not text or len(text.strip()) < MIN_TEXT_LENGTH:
            return text.strip()

        # IMPORTANT: For BART/LED models, do NOT inject prompts into input text.
        # These models copy prompts verbatim into output, causing instruction leakage.
        # Only use prompts for instruction-following models (GPT, etc.)
        # For BART/LED, the model learns to summarize from training, not from prompts.
        input_text = text

        _warn_if_input_very_long(input_text)

        try:
            # Use max_length for summarization (correct parameter for this task)
            # Explicitly set max_new_tokens=None to prevent pipeline from using its default (256)
            # This eliminates the warning about both parameters being set
            # Add repetition_penalty and no_repeat_ngram_size to prevent hallucinations/repetition
            # Use beam search (num_beams) for better quality summaries instead of greedy decoding
            # LED models benefit from beam search for more coherent summaries

            gen = _resolve_summarize_generation_params(
                is_reduce_phase,
                is_distill_phase,
                num_beams,
                no_repeat_ngram_size,
                length_penalty,
                early_stopping,
                repetition_penalty,
            )
            num_beams = gen["num_beams"]
            no_repeat_ngram_size = gen["no_repeat_ngram_size"]
            length_penalty = gen["length_penalty"]
            early_stopping = gen["early_stopping"]
            repetition_penalty = gen["repetition_penalty"]

            effective_max_new_tokens, effective_min_new_tokens = _resolve_effective_length_limits(
                max_new_tokens, min_new_tokens, max_length, min_length
            )

            text = self._summarize_truncate_input(text, max_input_tokens, truncation)
            effective_truncation = truncation if truncation is not None else True
            input_text = self._summarize_apply_t5_prefix(text)

            pipeline_kwargs = _build_summarize_pipeline_kwargs(
                effective_truncation,
                repetition_penalty,
                no_repeat_ngram_size,
                max_new_tokens,
                effective_max_new_tokens,
                min_new_tokens,
                effective_min_new_tokens,
                max_length,
                min_length,
                encoder_no_repeat_ngram_size,
                do_sample,
                num_beams,
                length_penalty,
            )

            # Suppress transformers warnings about max_length configuration
            # These are expected for summarization tasks where we want shorter outputs
            import warnings

            with warnings.catch_warnings():
                # Filter max_length > input_length warnings
                # (should be rare now with dynamic adjustment)
                warnings.filterwarnings(
                    "ignore",
                    message=r".*max_length.*input_length.*",
                    category=UserWarning,
                )
                # Filter specific transformers warning about max_length
                warnings.filterwarnings(
                    "ignore",
                    message=r"Your max_length is set to \d+, but your input_length is only \d+.*",
                    category=UserWarning,
                )
                # Note: Thinc/spaCy FutureWarning about torch.cuda.amp.autocast is
                # filtered at CLI startup (see cli.py) to prevent user-facing noise
                # from version compatibility issues.
                # Tracked in Issue #416 - see docs/guides/DEPENDENCIES_GUIDE.md for details
                # Filter "Asking to truncate to max_length but no maximum length
                # is provided" warning
                warnings.filterwarnings(
                    "ignore",
                    message=r".*truncate.*max_length.*no maximum length.*",
                    category=UserWarning,
                )
                # Filter alternative wording of truncate warning
                warnings.filterwarnings(
                    "ignore",
                    message=r".*no maximum length.*Default to no truncation.*",
                    category=UserWarning,
                )
                # Serialize pipeline calls to prevent tokenizer "Already borrowed" errors
                # when multiple episodes process concurrently
                with self._summarize_lock:
                    result = self.pipeline(input_text, **pipeline_kwargs)

            summary_text = _extract_summary_text_from_pipeline_result(result)
            if not summary_text:
                return ""

            # Log raw generated length before post-processing (with lock for thread safety)
            raw_chars = len(summary_text)
            raw_words = len(summary_text.split()) if summary_text else 0
            raw_tokens = 0
            if summary_text and self.tokenizer:
                with self._summarize_lock:
                    raw_tokens = len(
                        self.tokenizer.encode(  # type: ignore[attr-defined]
                            summary_text,
                            add_special_tokens=False,
                            truncation=True,
                            max_length=100000,
                        )
                    )

            # Validate summary quality - detect instruction leaks and
            # repetitive/hallucinated content
            if summary_text:
                summary_text = _strip_instruction_leak(summary_text)
                validated_summary = _validate_and_fix_repetitive_summary(summary_text)

                # Log post-processed length and difference
                processed_chars = len(validated_summary)
                processed_words = len(validated_summary.split()) if validated_summary else 0
                chars_diff = raw_chars - processed_chars
                words_diff = raw_words - processed_words

                if chars_diff > 0 or words_diff > 0:
                    logger.debug(
                        f"[POST-PROCESS] Raw: {raw_chars} chars, {raw_words} words, "
                        f"{raw_tokens} tokens → Processed: {processed_chars} chars, "
                        f"{processed_words} words ({chars_diff} chars, {words_diff} words removed)"
                    )
                else:
                    logger.debug(
                        f"[POST-PROCESS] Raw: {raw_chars} chars, {raw_words} words, "
                        f"{raw_tokens} tokens → Processed: {processed_chars} chars, "
                        f"{processed_words} words (no trimming)"
                    )

                return cast(str, validated_summary)

            return summary_text

        except RuntimeError as e:
            error_msg = str(e).lower()
            # Handle MPS buffer size errors and CUDA OOM errors with device fallback (Issue #379)
            if "invalid buffer size" in error_msg or "out of memory" in error_msg:
                # Try device fallback if on MPS/CUDA
                if self.device in ("mps", "cuda"):
                    logger.warning(
                        f"Device fallback: {self.device} OOM during summarization ({e}). "
                        "Falling back to CPU and retrying..."
                    )
                    try:
                        # Move model to CPU
                        self.model = self.model.to("cpu")  # type: ignore[union-attr]
                        self.device = "cpu"
                        original_device_for_retry = self.device
                        logger.info(
                            f"Device fallback successful: model moved from "
                            f"{original_device_for_retry} to CPU. Retrying summarization..."
                        )
                        # Update pipeline device
                        from transformers import pipeline

                        pipeline_device = -1  # CPU
                        self.pipeline = pipeline(  # type: ignore[call-overload]
                            "summarization",
                            model=self.model,
                            tokenizer=self.tokenizer,
                            device=pipeline_device,
                        )
                        # Retry summarization on CPU
                        result = self.pipeline(
                            text, max_length=max_length, min_length=min_length
                        )  # type: ignore[call-overload]
                        return cast(str, result["summary_text"])  # type: ignore[index]
                    except Exception as fallback_error:
                        logger.error(
                            f"Device fallback to CPU also failed: {fallback_error}. "
                            "Text may be too long - use chunking with "
                            "summarize_long_text() instead."
                        )
                        return ""
                else:
                    # Already on CPU or other error
                    logger.error(
                        f"Buffer size error during summarization ({self.device}): {e}. "
                        "Text is too long - use chunking with summarize_long_text() instead."
                    )
                    return ""
            # Handle "Already borrowed" error from Rust tokenizer in parallel execution
            # This should not happen with the lock, but retry with lock and backoff if it does
            if "already borrowed" in error_msg:
                logger.warning(
                    f"Tokenizer threading error during summarization: {e}. "
                    "Retrying with lock and exponential backoff..."
                )
                # Retry with lock and exponential backoff (max 3 retries)
                max_retries = 3
                for attempt in range(max_retries):
                    wait_time = 0.1 * (2**attempt)  # Exponential backoff: 0.1s, 0.2s, 0.4s
                    time.sleep(wait_time)
                    try:
                        with self._summarize_lock:
                            # Rebuild pipeline kwargs (may have been modified)
                            retry_result = self.pipeline(input_text, **pipeline_kwargs)
                            if isinstance(retry_result, list) and len(retry_result) > 0:
                                summary_text = retry_result[0].get("summary_text", "")
                                return cast(str, summary_text).strip()
                            elif isinstance(retry_result, dict):
                                summary_text = retry_result.get("summary_text", "")
                                return cast(str, summary_text).strip()
                            else:
                                logger.error(
                                    f"Retry {attempt + 1}/{max_retries} returned invalid result"
                                )
                                if attempt == max_retries - 1:
                                    return ""
                    except RuntimeError as retry_error:
                        retry_error_msg = str(retry_error).lower()
                        if "already borrowed" in retry_error_msg and attempt < max_retries - 1:
                            logger.warning(
                                f"Retry {attempt + 1}/{max_retries} still got 'already borrowed', "
                                f"waiting {wait_time * 2:.2f}s before next retry..."
                            )
                            continue
                        else:
                            logger.error(f"Retry {attempt + 1}/{max_retries} failed: {retry_error}")
                            if attempt == max_retries - 1:
                                logger.error(
                                    "All retries exhausted. This should not happen with proper "
                                    "locking - there may be a deeper concurrency issue."
                                )
                                return ""
                    except Exception as retry_error:
                        logger.error(
                            f"Retry {attempt + 1}/{max_retries} raised unexpected error: "
                            f"{retry_error}"
                        )
                        if attempt == max_retries - 1:
                            return ""
                # If we get here, all retries failed
                logger.error("All retry attempts exhausted for tokenizer threading error")
                return ""
            raise
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return ""


def chunk_text_for_summarization(
    text: str,
    tokenizer: "AutoTokenizer",
    chunk_size: int,
    # Default token overlap (will be adjusted based on chunk_size)
    overlap: int = DEFAULT_TOKEN_OVERLAP,
    lock: Optional[threading.Lock] = None,
) -> List[str]:
    """Split long text into overlapping chunks.

    Args:
        text: Input text
        tokenizer: Tokenizer instance for accurate token counting
        chunk_size: Target chunk size in tokens
        overlap: Overlap between chunks in tokens
        lock: Optional threading lock to serialize tokenizer access (for thread safety)

    Returns:
        List of text chunks
    """
    # Tokenize to get accurate token counts
    # Use truncation=True to prevent warnings when text exceeds model max
    # (We're chunking anyway, so truncation here is just to avoid warnings)
    # Serialize tokenizer access if lock is provided (prevents "Already borrowed" errors)
    if lock:
        with lock:
            tokens = tokenizer.encode(  # type: ignore[attr-defined]
                text, add_special_tokens=False, truncation=True, max_length=100000
            )
    else:
        tokens = tokenizer.encode(  # type: ignore[attr-defined]
            text, add_special_tokens=False, truncation=True, max_length=100000
        )

    chunks = []
    start = 0
    total_tokens = len(tokens)

    while start < total_tokens:
        # Calculate chunk end (don't exceed total tokens)
        end = min(start + chunk_size, total_tokens)
        chunk_tokens = tokens[start:end]

        # Decode chunk tokens back to text (with lock if provided)
        if lock:
            with lock:
                chunk_text = tokenizer.decode(  # type: ignore[attr-defined]
                    chunk_tokens, skip_special_tokens=True
                )
        else:
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


def _chunk_by_tokens(
    text: str,
    tokenizer: "AutoTokenizer",
    max_tokens: int = 600,
    lock: Optional[threading.Lock] = None,
) -> List[str]:
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
    # Use truncation=True to prevent warnings when text exceeds model max
    # (We're chunking anyway, so truncation here is just to avoid warnings)
    # Serialize tokenizer access for thread safety
    if lock:
        with lock:
            ids = tokenizer.encode(  # type: ignore[attr-defined]
                text, add_special_tokens=False, truncation=True, max_length=100000
            )
    else:
        ids = tokenizer.encode(  # type: ignore[attr-defined]
            text, add_special_tokens=False, truncation=True, max_length=100000
        )
    chunks = []
    for i in range(0, len(ids), max_tokens):
        cs = ids[i : i + max_tokens]
        if lock:
            with lock:
                chunks.append(
                    tokenizer.decode(cs, skip_special_tokens=True)  # type: ignore[attr-defined]
                )
        else:
            chunks.append(
                tokenizer.decode(cs, skip_special_tokens=True)  # type: ignore[attr-defined]
            )
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


def _validate_and_fix_repetitive_summary(summary: str) -> str:
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


def _strip_instruction_leak(summary: str) -> str:
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


def _check_if_needs_chunking(
    model: SummaryModel,
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
    # Use truncation=True to prevent warnings when text exceeds model max
    # (This is just for counting, actual processing will chunk if needed)
    # Serialize tokenizer access for thread safety
    with model._summarize_lock:
        tokens = model.tokenizer.encode(  # type: ignore[attr-defined]
            text, add_special_tokens=False, truncation=True, max_length=100000
        )
    total_tokens = len(tokens)

    if total_tokens <= chunk_size:
        # Text fits in one chunk
        return model.summarize(text, max_length=max_length, min_length=min_length, prompt=prompt)

    return None


def _prepare_chunks(
    model: SummaryModel,
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

    # For encoder-decoder models, use 60-80 token overlap (as recommended)
    # For other models, use the default overlap ratio
    if use_word_chunking:
        # Encoder-decoder models: 60-80 token overlap for better performance
        overlap = max(60, min(80, int(chunk_size * 0.1)))  # 10% of chunk_size, clamped to 60-80
    else:
        overlap = max(1, int(chunk_size * CHUNK_OVERLAP_RATIO))

    chunks = chunk_text_for_summarization(
        text,
        model.tokenizer,
        chunk_size=chunk_size,
        overlap=overlap,
        lock=model._summarize_lock,  # Serialize tokenizer access for thread safety
    )

    total_words = len(text.split())
    # Use truncation=True to prevent warnings when text exceeds model max
    # (This is just for counting/logging, actual chunks are already created)
    # Serialize tokenizer access for thread safety
    with model._summarize_lock:
        total_tokens = len(
            model.tokenizer.encode(  # type: ignore[attr-defined]
                text, add_special_tokens=False, truncation=True, max_length=100000
            )
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


def _merge_tiny_chunks(
    model: SummaryModel,
    chunks: List[str],
    min_tokens: int = MAP_CHUNK_MIN_TOKENS,
) -> List[str]:
    """Merge chunks smaller than min_tokens into the previous chunk (Issue #428).

    Avoids HF warnings and wasted work for tiny tail chunks. Logs each merge.

    Args:
        model: Summary model (for tokenizer and lock).
        chunks: List of text chunks from _prepare_chunks.
        min_tokens: Minimum token count; chunks below this are merged.

    Returns:
        List of chunks with tiny chunks merged into the previous (or next if first).
    """
    if not chunks or not model.tokenizer:
        return chunks

    with model._summarize_lock:
        token_counts = [
            len(
                model.tokenizer.encode(  # type: ignore[attr-defined]
                    c, add_special_tokens=False, truncation=True, max_length=100000
                )
            )
            for c in chunks
        ]

    merged: List[str] = []
    i = 0
    while i < len(chunks):
        chunk = chunks[i]
        n = token_counts[i]
        if n >= min_tokens:
            merged.append(chunk)
            i += 1
            continue
        # Tiny chunk: merge into previous or next
        if merged:
            merged[-1] = merged[-1] + "\n\n" + chunk
            logger.debug("Merged tiny chunk (%d tokens) into previous (Issue #428).", n)
        else:
            # First chunk is tiny; merge into next if any
            if i + 1 < len(chunks):
                merged.append(chunk + "\n\n" + chunks[i + 1])
                logger.debug("Merged tiny first chunk (%d tokens) into next (Issue #428).", n)
                i += 2
                continue
            else:
                merged.append(chunk)
        i += 1
    return merged


def summarize_long_text(
    model: SummaryModel,
    text: str,
    chunk_size: int = BART_MAX_POSITION_EMBEDDINGS,
    max_length: int = 150,
    min_length: int = 30,
    batch_size: Optional[int] = None,
    prompt: Optional[str] = None,
    use_word_chunking: bool = False,
    word_chunk_size: int = DEFAULT_WORD_CHUNK_SIZE,
    word_overlap: int = DEFAULT_WORD_OVERLAP,
    reduce_model: Optional[SummaryModel] = None,
    return_intermediates: bool = False,
    # New explicit generation params for map stage
    map_max_new_tokens: Optional[int] = None,
    map_min_new_tokens: Optional[int] = None,
    map_num_beams: Optional[int] = None,
    map_no_repeat_ngram_size: Optional[int] = None,
    map_length_penalty: Optional[float] = None,
    map_early_stopping: Optional[bool] = None,
    map_repetition_penalty: Optional[float] = None,
    map_encoder_no_repeat_ngram_size: Optional[int] = None,
    map_max_input_tokens: Optional[int] = None,
    # New explicit generation params for reduce stage
    reduce_max_new_tokens: Optional[int] = None,
    reduce_min_new_tokens: Optional[int] = None,
    reduce_num_beams: Optional[int] = None,
    reduce_no_repeat_ngram_size: Optional[int] = None,
    reduce_length_penalty: Optional[float] = None,
    reduce_early_stopping: Optional[bool] = None,
    reduce_repetition_penalty: Optional[float] = None,
    reduce_encoder_no_repeat_ngram_size: Optional[int] = None,
    reduce_max_input_tokens: Optional[int] = None,
    truncation: Optional[bool] = None,
    preprocessing_profile: str = "cleaning_v4",  # Default to cleaning_v4
    # (matches production baseline)
    # Optional 2nd-pass distill parameters (Issue #387)
    enable_2nd_pass_distill: bool = False,
    transcript_text: Optional[str] = None,
    episode_description: Optional[str] = None,
) -> str | tuple[str, Dict[str, Any]]:
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

    Returns:
        Combined summary
    """
    import time

    pipeline_start_time = time.time()  # noqa: E501

    # If no separate reduce model is provided, use the same model for map and reduce.
    if reduce_model is None:
        reduce_model = model

    # Use preprocessing profile system (wired from experiment config)
    # Apply with statistics tracking for diagnostic logging
    cleaned_text, preprocess_stats = apply_profile_with_stats(text, preprocessing_profile)
    if cleaned_text != text:
        removed_chars = len(text) - len(cleaned_text)
        removed_pct = (removed_chars / len(text) * 100) if len(text) else 0

        # Log preprocessing profile and statistics
        logger.info(
            f"[PREPROCESSING] Profile: {preprocessing_profile}, "
            f"lines: {preprocess_stats['initial_lines']} → {preprocess_stats['final_lines']} "
            f"({preprocess_stats['lines_removed']} removed), "
            f"chars: {len(text):,} → {len(cleaned_text):,} "
            f"({removed_chars:,} removed, {removed_pct:.1f}%)"
        )

        # Log per-step line removal (if available)
        step_stats = [
            f"{k}: {v}" for k, v in preprocess_stats.items() if k.startswith("step_") and v > 0
        ]
        if step_stats:
            logger.info(f"[PREPROCESSING] Steps: {', '.join(step_stats)}")

        text = cleaned_text.strip()
    else:
        # Still log profile even if no changes
        logger.info(
            f"[PREPROCESSING] Profile: {preprocessing_profile}, "
            f"lines: {preprocess_stats['initial_lines']} (no changes)"
        )

    # === VALIDATION: Input metrics ===
    input_chars = len(text)
    input_words = len(text.split())
    if model.tokenizer:
        # Use truncation=True to prevent warnings when text exceeds model max
        # (This is just for counting/logging, actual processing will chunk if needed)
        # Serialize tokenizer access for thread safety
        with model._summarize_lock:
            input_tokens = len(
                model.tokenizer.encode(  # type: ignore[attr-defined]
                    text, add_special_tokens=False, truncation=True, max_length=100000
                )
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

    # For encoder-decoder models, use token chunking directly (don't convert from word chunking)
    # This provides more uniform chunk sizes and better performance
    encoder_decoder_override = False
    if use_word_chunking:
        # Use token chunking directly: 700 tokens with 60-80 token overlap (as recommended)
        chunk_size = min(chunk_size, 700)  # Optimal for encoder-decoder models
        encoder_decoder_override = True
        logger.debug(
            "Encoder-decoder model: Using token chunking directly "
            f"(chunk_size={chunk_size} tokens, recommended 700 tokens with 60-80 overlap)"
        )

    logger.debug(
        "[MAP-REDUCE VALIDATION] Chunking strategy: "
        f"requested_chunk_size={requested_chunk_size} tokens, "
        f"model_max={model_max_tokens}, "
        f"effective_chunk_size={chunk_size} tokens, "
        f"encoder_decoder_override={'yes' if encoder_decoder_override else 'no'}"
    )

    # Step 1: Check if text fits without chunking (early exit)
    direct_summary = _check_if_needs_chunking(
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
    chunks, chunk_size = _prepare_chunks(
        model, text, chunk_size, use_word_chunking, word_chunk_size, word_overlap
    )

    # Step 2b: Merge tiny chunks into previous to avoid HF warnings (Issue #428)
    chunks = _merge_tiny_chunks(model, chunks)

    # === VALIDATION: Chunking metrics ===
    chunk_sizes_chars = [len(chunk) for chunk in chunks]
    chunk_sizes_words = [len(chunk.split()) for chunk in chunks]
    if model.tokenizer:
        # Chunks are already within limits, but add truncation to prevent any warnings
        # Serialize tokenizer access for thread safety
        with model._summarize_lock:
            chunk_sizes_tokens = [
                len(
                    model.tokenizer.encode(  # type: ignore[attr-defined]
                        chunk, add_special_tokens=False, truncation=True, max_length=100000
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
    chunk_summaries = _summarize_chunks_map(
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
        # Pass map generation params
        map_max_new_tokens=map_max_new_tokens,
        map_min_new_tokens=map_min_new_tokens,
        map_num_beams=map_num_beams,
        map_no_repeat_ngram_size=map_no_repeat_ngram_size,
        map_length_penalty=map_length_penalty,
        map_early_stopping=map_early_stopping,
        map_repetition_penalty=map_repetition_penalty,
        map_encoder_no_repeat_ngram_size=map_encoder_no_repeat_ngram_size,
        map_max_input_tokens=map_max_input_tokens,
        truncation=truncation,
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
    # Calculate reduce input size (for logging/diagnostics)
    reduce_input_chars = 0
    if chunk_summaries:
        # Log map output before joining (to investigate shrink)
        map_output_total_chars = sum(len(s) for s in chunk_summaries)
        map_output_total_words = sum(len(s.split()) for s in chunk_summaries)
        logger.debug(
            f"[MAP-REDUCE VALIDATION] Map output before joining: "
            f"{map_output_total_chars:,} chars, {map_output_total_words:,} words, "
            f"{len(chunk_summaries)} summaries"
        )

        # Log first 80 chars of each summary for debugging
        for i, summary in enumerate(chunk_summaries[:4], 1):  # Log first 4
            preview = summary[:80].replace("\n", " ")
            logger.debug(f"[MAP-REDUCE VALIDATION] Map summary {i} preview: {preview}...")

        combined_text = _join_summaries_with_structure(chunk_summaries)
        reduce_input_chars = len(combined_text)

        # Log shrink if significant
        if map_output_total_chars > 0:
            shrink_ratio = reduce_input_chars / map_output_total_chars
            if shrink_ratio < 0.8:  # More than 20% shrink
                logger.debug(
                    f"[MAP-REDUCE VALIDATION] Map→Reduce shrink detected: "
                    f"{map_output_total_chars:,} chars → {reduce_input_chars:,} chars "
                    f"({shrink_ratio:.1%} retained, {1-shrink_ratio:.1%} lost in joining)"
                )

    reduce_start_time = time.time()
    final_summary = _combine_summaries_reduce(
        reduce_model,
        chunk_summaries,
        max_length,
        min_length,
        prompt,
        map_model=model,  # Pass MAP model for routing short inputs (Issue #380)
        # Pass reduce generation params
        reduce_max_new_tokens=reduce_max_new_tokens,
        reduce_min_new_tokens=reduce_min_new_tokens,
        reduce_num_beams=reduce_num_beams,
        reduce_no_repeat_ngram_size=reduce_no_repeat_ngram_size,
        reduce_length_penalty=reduce_length_penalty,
        reduce_early_stopping=reduce_early_stopping,
        reduce_repetition_penalty=reduce_repetition_penalty,
        reduce_encoder_no_repeat_ngram_size=reduce_encoder_no_repeat_ngram_size,
        reduce_max_input_tokens=reduce_max_input_tokens,
        truncation=truncation,
        # Pass 2nd-pass distill parameters
        enable_2nd_pass_distill=enable_2nd_pass_distill,
        transcript_text=transcript_text,
        episode_description=episode_description,
    )
    reduce_time = time.time() - reduce_start_time

    # === VALIDATION: Reduce phase and overall metrics ===
    final_chars = len(final_summary)
    final_words = len(final_summary.split())
    if model.tokenizer:
        # Final summary is already generated, but add truncation to prevent any warnings
        # Serialize tokenizer access for thread safety
        with model._summarize_lock:
            final_tokens = len(
                model.tokenizer.encode(  # type: ignore[attr-defined]
                    final_summary, add_special_tokens=False, truncation=True, max_length=100000
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

    # Prepare intermediate outputs if requested
    if return_intermediates:
        intermediates: Dict[str, Any] = {
            "map_summaries": (
                [{"chunk_id": i, "text": summary} for i, summary in enumerate(chunk_summaries)]
                if chunk_summaries
                else []
            ),
            "reduce_input_chars": reduce_input_chars,  # Calculated before reduce phase
        }
        return final_summary, intermediates

    return final_summary


def _summarize_chunks_map(
    model: SummaryModel,
    chunks: List[str],
    max_length: int,
    min_length: int,
    prompt: Optional[str],
    batch_size: Optional[int],
    use_word_chunking: bool,
    word_chunk_size: int,
    word_overlap: int,
    chunk_size: int,
    # New explicit generation params for map stage
    map_max_new_tokens: Optional[int] = None,
    map_min_new_tokens: Optional[int] = None,
    map_num_beams: Optional[int] = None,
    map_no_repeat_ngram_size: Optional[int] = None,
    map_length_penalty: Optional[float] = None,
    map_early_stopping: Optional[bool] = None,
    map_repetition_penalty: Optional[float] = None,
    map_encoder_no_repeat_ngram_size: Optional[int] = None,
    map_max_input_tokens: Optional[int] = None,
    truncation: Optional[bool] = None,
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
    total_chunks = len(chunks)
    import time

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

    # For MAP stage, use min_length/max_length instead of min_new_tokens/max_new_tokens
    # This is more reliable for encoder-decoder models like Pegasus
    # Target: 120-180 tokens per chunk summary (good ratio for 600-token input chunks)
    # User recommendation: min_length=120-160, max_length=220-260

    # Base max_length calculation
    if map_max_new_tokens is not None:
        base_chunk_max_length = min(chunk_size, map_max_new_tokens, CHUNK_SUMMARY_MAX_TOKENS)
    else:
        base_chunk_max_length = min(chunk_size, max_length, CHUNK_SUMMARY_MAX_TOKENS)

    # Dynamic reduction: when chunk input is small, reduce max_length to prevent warnings
    # This reduces "max_length > input_length" warnings and improves efficiency
    # We'll calculate this per-chunk, but set a reasonable default here
    # The actual dynamic adjustment happens in the chunk processing loop
    chunk_max_length = base_chunk_max_length

    if map_min_new_tokens is not None:
        # Convert min_new_tokens to min_length (approximate)
        chunk_min_length = min(chunk_max_length, max(map_min_new_tokens, CHUNK_SUMMARY_MIN_TOKENS))
    else:
        # For MAP stage, ignore reduce min_length parameter - use MAP-specific default
        # User recommendation: 120-160 tokens for MAP summaries
        # Use 120 as minimum (not the reduce min_length=220 which is too high for MAP)
        map_default_min = 120  # MAP-specific minimum (user recommendation)
        chunk_min_length = min(chunk_max_length, max(map_default_min, CHUNK_SUMMARY_MIN_TOKENS))

    # Ensure min_length is reasonable (not too high relative to max)
    # But don't cap too aggressively - we want 120+ tokens
    if chunk_min_length > chunk_max_length * 0.7:
        # If min is too close to max, set it to 60% of max (but not below 120)
        chunk_min_length = max(120, int(chunk_max_length * 0.6))

    logger.debug(
        f"[MAP-REDUCE CONFIG] Map stage: {total_chunks} chunks, chunk_size={chunk_size} tokens, "
        f"overlap={overlap} tokens, workers={max_workers}, "
        f"chunk_summary_range={chunk_min_length}-{chunk_max_length} tokens "
        f"(using min_length/max_length), "
        f"estimated time ~{estimated_minutes} minutes"
    )

    if max_workers > 1:
        # Parallel processing for CPU
        return _summarize_chunks_parallel(
            model,
            chunks,
            chunk_max_length,
            chunk_min_length,
            prompt,
            max_workers,
            start_time,
            # Pass map generation params
            map_max_new_tokens=map_max_new_tokens,
            map_min_new_tokens=map_min_new_tokens,
            map_num_beams=map_num_beams,
            map_no_repeat_ngram_size=map_no_repeat_ngram_size,
            map_length_penalty=map_length_penalty,
            map_early_stopping=map_early_stopping,
            map_repetition_penalty=map_repetition_penalty,
            map_encoder_no_repeat_ngram_size=map_encoder_no_repeat_ngram_size,
            map_max_input_tokens=map_max_input_tokens,
            truncation=truncation,
        )
    else:
        # Sequential processing (GPU or single worker)
        return _summarize_chunks_sequential(
            model,
            chunks,
            chunk_max_length,
            chunk_min_length,
            prompt,
            chunk_size,
            start_time,
            # Pass map generation params
            map_max_new_tokens=map_max_new_tokens,
            map_min_new_tokens=map_min_new_tokens,
            map_num_beams=map_num_beams,
            map_no_repeat_ngram_size=map_no_repeat_ngram_size,
            map_length_penalty=map_length_penalty,
            map_early_stopping=map_early_stopping,
            map_repetition_penalty=map_repetition_penalty,
            map_max_input_tokens=map_max_input_tokens,
            truncation=truncation,
        )


def _summarize_chunks_parallel(
    model: SummaryModel,
    chunks: List[str],
    chunk_max_length: int,
    chunk_min_length: int,
    prompt: Optional[str],
    max_workers: int,
    start_time: float,
    # New explicit generation params for map stage
    map_max_new_tokens: Optional[int] = None,
    map_min_new_tokens: Optional[int] = None,
    map_num_beams: Optional[int] = None,
    map_no_repeat_ngram_size: Optional[int] = None,
    map_length_penalty: Optional[float] = None,
    map_early_stopping: Optional[bool] = None,
    map_repetition_penalty: Optional[float] = None,
    map_encoder_no_repeat_ngram_size: Optional[int] = None,
    map_max_input_tokens: Optional[int] = None,
    truncation: Optional[bool] = None,
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
    from concurrent.futures import as_completed, ThreadPoolExecutor

    total_chunks = len(chunks)

    def _summarize_chunk(chunk_idx_and_text):
        chunk_idx, chunk_text = chunk_idx_and_text
        try:
            # Dynamic max_length/min_length adjustment: cap by input length to prevent warnings
            # Robust rule: max_length = min(configured_max, max(20, int(0.7 * in_len)))
            # This prevents the model from trying to generate longer-than-input outputs
            if model.tokenizer:
                # Serialize tokenizer access for thread safety
                with model._summarize_lock:
                    chunk_tokens = len(
                        model.tokenizer.encode(  # type: ignore[attr-defined]
                            chunk_text, add_special_tokens=False, truncation=True, max_length=100000
                        )
                    )
                # Cap max_length based on input: min(configured_max, max(20, 0.7 * input))
                dynamic_max_length = min(chunk_max_length, max(20, int(0.7 * chunk_tokens)))
                # Cap min_length based on input: min(configured_min, max(10, 0.3 * input))
                dynamic_min_length = min(chunk_min_length, max(10, int(0.3 * chunk_tokens)))
            else:
                dynamic_max_length = chunk_max_length
                dynamic_min_length = chunk_min_length

            # For MAP stage, use min_length/max_length instead of min_new_tokens/max_new_tokens
            return (
                chunk_idx,
                model.summarize(
                    chunk_text,
                    max_length=dynamic_max_length,
                    min_length=dynamic_min_length,
                    prompt=prompt,
                    # Pass other map generation params (but NOT max_new_tokens/min_new_tokens)
                    max_new_tokens=None,  # Use max_length instead
                    min_new_tokens=None,  # Use min_length instead
                    num_beams=map_num_beams,
                    no_repeat_ngram_size=map_no_repeat_ngram_size,
                    length_penalty=map_length_penalty,
                    early_stopping=map_early_stopping,
                    repetition_penalty=map_repetition_penalty,
                    encoder_no_repeat_ngram_size=map_encoder_no_repeat_ngram_size,
                    max_input_tokens=map_max_input_tokens,
                    truncation=truncation,
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


def _summarize_chunks_sequential(
    model: SummaryModel,
    chunks: List[str],
    chunk_max_length: int,
    chunk_min_length: int,
    prompt: Optional[str],
    chunk_size: int,
    start_time: float,
    # New explicit generation params for map stage
    map_max_new_tokens: Optional[int] = None,
    map_min_new_tokens: Optional[int] = None,
    map_num_beams: Optional[int] = None,
    map_no_repeat_ngram_size: Optional[int] = None,
    map_length_penalty: Optional[float] = None,
    map_early_stopping: Optional[bool] = None,
    map_repetition_penalty: Optional[float] = None,
    map_encoder_no_repeat_ngram_size: Optional[int] = None,
    map_max_input_tokens: Optional[int] = None,
    truncation: Optional[bool] = None,
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
            # Dynamic max_length/min_length adjustment: cap by input length to prevent warnings
            # Robust rule: max_length = min(configured_max, max(20, int(0.7 * in_len)))
            # This prevents the model from trying to generate longer-than-input outputs
            if model.tokenizer:
                # Serialize tokenizer access for thread safety
                with model._summarize_lock:
                    chunk_tokens = len(
                        model.tokenizer.encode(  # type: ignore[attr-defined]
                            chunk, add_special_tokens=False, truncation=True, max_length=100000
                        )
                    )
                # Cap max_length based on input: min(configured_max, max(20, 0.7 * input))
                dynamic_max_length = min(chunk_max_length, max(20, int(0.7 * chunk_tokens)))
                # Cap min_length based on input: min(configured_min, max(10, 0.3 * input))
                dynamic_min_length = min(chunk_min_length, max(10, int(0.3 * chunk_tokens)))
            else:
                dynamic_max_length = chunk_max_length
                dynamic_min_length = chunk_min_length

            # For MAP stage, use min_length/max_length instead of
            # min_new_tokens/max_new_tokens
            # This is more reliable for encoder-decoder models like Pegasus
            # Don't pass max_new_tokens/min_new_tokens - let model.summarize()
            # use max_length/min_length
            summary = model.summarize(
                chunk,
                max_length=dynamic_max_length,
                min_length=dynamic_min_length,
                prompt=prompt,
                # Pass other map generation params (but NOT max_new_tokens/min_new_tokens)
                max_new_tokens=None,  # Use max_length instead
                min_new_tokens=None,  # Use min_length instead
                num_beams=map_num_beams,
                no_repeat_ngram_size=map_no_repeat_ngram_size,
                length_penalty=map_length_penalty,
                early_stopping=map_early_stopping,
                repetition_penalty=map_repetition_penalty,
                encoder_no_repeat_ngram_size=map_encoder_no_repeat_ngram_size,
                max_input_tokens=map_max_input_tokens,
                truncation=truncation,
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


def _combine_summaries_reduce(
    model: SummaryModel,
    chunk_summaries: List[str],
    max_length: int,
    min_length: int,
    prompt: Optional[str],
    # New explicit generation params for reduce stage
    reduce_max_new_tokens: Optional[int] = None,
    reduce_min_new_tokens: Optional[int] = None,
    reduce_num_beams: Optional[int] = None,
    reduce_no_repeat_ngram_size: Optional[int] = None,
    reduce_length_penalty: Optional[float] = None,
    reduce_early_stopping: Optional[bool] = None,
    reduce_repetition_penalty: Optional[float] = None,
    reduce_encoder_no_repeat_ngram_size: Optional[int] = None,
    reduce_max_input_tokens: Optional[int] = None,
    truncation: Optional[bool] = None,
    map_model: Optional[SummaryModel] = None,  # MAP model for routing short inputs
    # Optional 2nd-pass distill parameters (Issue #387)
    enable_2nd_pass_distill: bool = False,
    transcript_text: Optional[str] = None,
    episode_description: Optional[str] = None,
) -> str:
    """Reduce step: Combine chunk summaries into final summary.

    Args:
        model: Summary model instance (REDUCE model, typically LED)
        chunk_summaries: List of chunk summaries to combine
        max_length: Max summary length
        min_length: Min summary length
        prompt: Optional prompt
        map_model: Optional MAP model (for routing short inputs to avoid LED padding overhead)

    Returns:
        Final combined summary
    """
    if not chunk_summaries:
        logger.warning("No chunk summaries generated, returning empty summary")
        return ""

    # Combine chunk summaries with clear structure
    combined_text = _join_summaries_with_structure(chunk_summaries)
    combined_chars = len(combined_text)
    combined_words = len(combined_text.split())
    if model.tokenizer:
        # Combined text is from chunk summaries, but add truncation to prevent warnings
        # (This is just for counting/logging, actual processing will handle truncation)
        # Serialize tokenizer access for thread safety
        with model._summarize_lock:
            combined_tokens = len(
                model.tokenizer.encode(  # type: ignore[attr-defined]
                    combined_text, add_special_tokens=False, truncation=True, max_length=100000
                )
            )
    else:
        combined_tokens = combined_chars // CHARS_PER_TOKEN_ESTIMATE

    # Performance optimization: Route short reduce inputs to MAP model to avoid
    # LED padding overhead (Issue #380)
    # LED models have attention_window=1024, so short inputs (e.g., 472 tokens)
    # get padded to 1024. This causes unnecessary overhead. For short inputs,
    # use MAP model (Pegasus) instead.
    LED_ATTENTION_WINDOW = 1024  # LED models pad to this window size
    SHORT_INPUT_THRESHOLD = LED_ATTENTION_WINDOW  # Route inputs < 1024 tokens to MAP model

    # Check if reduce model is LED (long-context model) and input is short
    reduce_model_max = (
        getattr(model.model.config, "max_position_embeddings", BART_MAX_POSITION_EMBEDDINGS)
        if model.model and hasattr(model.model, "config")
        else BART_MAX_POSITION_EMBEDDINGS
    )
    is_led_model = reduce_model_max >= LONG_CONTEXT_THRESHOLD
    is_short_input = combined_tokens < SHORT_INPUT_THRESHOLD

    # Route short inputs to MAP model if available and reduce model is LED
    effective_model = model
    if is_led_model and is_short_input and map_model is not None:
        logger.debug(
            f"[MAP-REDUCE PERFORMANCE] Routing short reduce input ({combined_tokens} tokens) "
            f"to MAP model ({map_model.model_name}) to avoid LED padding overhead "
            f"(reduce model {model.model_name} would pad to {LED_ATTENTION_WINDOW} tokens)"
        )
        effective_model = map_model
    elif is_led_model and is_short_input:
        logger.debug(
            f"[MAP-REDUCE PERFORMANCE] Short reduce input ({combined_tokens} tokens) "
            f"with LED model ({model.model_name}) will be padded to "
            f"{LED_ATTENTION_WINDOW} tokens (MAP model not available for routing)"
        )

    # Get model max length for decision making (use effective model)
    model_max = (
        getattr(
            effective_model.model.config, "max_position_embeddings", BART_MAX_POSITION_EMBEDDINGS
        )
        if effective_model.model and hasattr(effective_model.model, "config")
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
        selected = _select_key_summaries(chunk_summaries)
        logger.debug(
            "[MAP-REDUCE CONFIG] Extractive fallback: "
            f"summary_range={final_reduce_min_length}-{final_reduce_max_length} tokens"
        )

        return _combine_summaries_extractive(
            effective_model,
            selected,
            final_reduce_max_length,
            final_reduce_min_length,
            reduce_prompt,
            model_max,
            # Pass reduce generation params
            reduce_max_new_tokens=reduce_max_new_tokens,
            reduce_min_new_tokens=reduce_min_new_tokens,
            reduce_num_beams=reduce_num_beams,
            reduce_no_repeat_ngram_size=reduce_no_repeat_ngram_size,
            reduce_length_penalty=reduce_length_penalty,
            reduce_early_stopping=reduce_early_stopping,
            reduce_repetition_penalty=reduce_repetition_penalty,
            reduce_encoder_no_repeat_ngram_size=reduce_encoder_no_repeat_ngram_size,
            reduce_max_input_tokens=reduce_max_input_tokens,
            truncation=truncation,
        )

    if combined_tokens > single_pass_limit:
        return _combine_summaries_mini_map_reduce(
            effective_model,
            combined_text,
            chunk_summaries,
            final_reduce_max_length,
            final_reduce_min_length,
            reduce_prompt,
            combined_tokens,
            single_pass_limit,
            # Pass reduce generation params
            reduce_max_new_tokens=reduce_max_new_tokens,
            reduce_min_new_tokens=reduce_min_new_tokens,
            reduce_num_beams=reduce_num_beams,
            reduce_no_repeat_ngram_size=reduce_no_repeat_ngram_size,
            reduce_length_penalty=reduce_length_penalty,
            reduce_early_stopping=reduce_early_stopping,
            reduce_repetition_penalty=reduce_repetition_penalty,
            reduce_encoder_no_repeat_ngram_size=reduce_encoder_no_repeat_ngram_size,
            reduce_max_input_tokens=reduce_max_input_tokens,
            truncation=truncation,
        )

    # Single-pass abstractive reduce - use ALL summaries, no selection
    logger.debug(
        "[MAP-REDUCE CONFIG] Final reduce: "
        f"summary_range={final_reduce_min_length}-{final_reduce_max_length} tokens, "
        f"prompt={'REDUCE_PROMPT_SHORT' if prompt is None else 'custom'}"
    )
    try:
        return _combine_summaries_abstractive(
            effective_model,
            combined_text,
            chunk_summaries,
            final_reduce_max_length,
            final_reduce_min_length,
            reduce_prompt,
            model_max,
            combined_tokens,
            # Pass reduce generation params
            reduce_max_new_tokens=reduce_max_new_tokens,
            reduce_min_new_tokens=reduce_min_new_tokens,
            reduce_num_beams=reduce_num_beams,
            reduce_no_repeat_ngram_size=reduce_no_repeat_ngram_size,
            reduce_length_penalty=reduce_length_penalty,
            reduce_early_stopping=reduce_early_stopping,
            reduce_max_input_tokens=reduce_max_input_tokens,
            truncation=truncation,
            # Pass 2nd-pass distill parameters
            enable_2nd_pass_distill=enable_2nd_pass_distill,
            transcript_text=transcript_text,
            episode_description=episode_description,
        )
    except RuntimeError as e:
        error_msg = str(e).lower()
        if "invalid buffer size" in error_msg or "out of memory" in error_msg:
            # Abstractive failed - fall back to extractive (which does selection)
            logger.warning(
                "[MAP-REDUCE VALIDATION] Abstractive reduce failed, "
                "falling back to extractive approach"
            )
            return _combine_summaries_extractive(
                model,
                _select_key_summaries(chunk_summaries),
                final_reduce_max_length,
                final_reduce_min_length,
                reduce_prompt,
                model_max,
                # Pass reduce generation params
                reduce_max_new_tokens=reduce_max_new_tokens,
                reduce_min_new_tokens=reduce_min_new_tokens,
                reduce_num_beams=reduce_num_beams,
                reduce_no_repeat_ngram_size=reduce_no_repeat_ngram_size,
                reduce_length_penalty=reduce_length_penalty,
                reduce_early_stopping=reduce_early_stopping,
                reduce_max_input_tokens=reduce_max_input_tokens,
                truncation=truncation,
            )
        raise


def _select_key_summaries(chunk_summaries: List[str]) -> List[str]:
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


def _normalize_bullet(line: str) -> str:
    """Normalize a line to consistent bullet format.

    Strips leading numbering (1), 1., etc.) and existing bullet prefixes
    to ensure consistent formatting in reducer input.
    """
    line = line.strip()
    # Strip existing bullet prefixes
    if line.startswith(("-", "•", "*")):
        line = line[1:].strip()
    # Strip leading numbering like "1)", "1.", "2) "
    line = _LEADING_NUMBER_PATTERN.sub("", line)
    return line.strip()


def _join_summaries_with_structure(summaries: List[str]) -> str:
    """Join chunk summaries for BART/LED reduce phase.

    IMPORTANT: For BART/LED, we output ONLY plain bullets with neutral separators.
    NO instruction text, NO headers - these get copied verbatim by the model.

    The structure helps the model:
    1. See clear chunk boundaries (--- separators with blank lines)
    2. Treat content as "notes" not "sentences to copy"
    3. Avoid extractive copying of prose

    Uses "\n\n---\n\n" separators for LED models to provide better structure
    and prevent summaries from blending together, which can cause generation collapse.

    Args:
        summaries: List of chunk summary strings

    Returns:
        Structured text for reducer input (bullets only, no instructions)
    """
    # Calculate total size before processing
    total_chars = sum(len(s) for s in summaries)

    # If combined text is small, skip bulletization to avoid unnecessary compression
    # This prevents double-compression when we have headroom in the reduce phase
    if total_chars < SKIP_BULLETIZATION_THRESHOLD:
        # Just join with newlines and separators - no bulletization/truncation
        cleaned_summaries = []
        for s in summaries:
            s = s.strip()
            if not s:
                continue
            # Normalize Pegasus <n> markers to real newlines
            s = s.replace("<n>", "\n").replace("</n>", "")
            # Sanitize artifacts
            s = preprocessing.remove_summarization_artifacts(s)  # type: ignore[attr-defined]
            cleaned_summaries.append(s)
        # Join with separators but no bulletization
        return "\n\n---\n\n".join(cleaned_summaries).strip()

    # For larger texts, use bulletization with scaled caps
    blocks = []
    for i, s in enumerate(summaries, start=1):
        s = s.strip()
        if not s:
            continue

        # Normalize Pegasus <n> markers to real newlines BEFORE splitlines()
        # This is critical - Pegasus outputs "<n>" not "\n", so splitlines() won't work
        s = s.replace("<n>", "\n").replace("</n>", "")

        # Sanitize: remove any artifacts that might have leaked through
        s = preprocessing.remove_summarization_artifacts(s)  # type: ignore[attr-defined]

        # Convert prose to bullets (now splitlines() will work correctly)
        lines = [ln.strip() for ln in s.splitlines() if ln.strip()]

        # If single long line, split into sentences (but only if very long)
        if len(lines) == 1 and len(lines[0]) > SENTENCE_SPLIT_THRESHOLD:
            text = lines[0]
            # Crude sentence segmentation - only split on clear boundaries
            parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
            lines = parts[:MAX_BULLETS_PER_CHUNK]

        # Convert to bullets, cap length (with scaled caps based on number of chunks)
        # Scale caps based on number of chunks - more chunks = more headroom per chunk
        num_chunks = len(summaries)
        scaled_max_bullets = min(MAX_BULLETS_PER_CHUNK + (num_chunks - 1), 16)  # Scale up to 16 max
        scaled_max_chunk_chars = min(
            MAX_CHUNK_CHARS + (num_chunks - 1) * 200, 4000
        )  # Scale up to 4000 max

        bullet_lines = []
        chunk_total_chars = 0
        for ln in lines:
            ln = _normalize_bullet(ln)
            if len(ln) < MIN_BULLET_CHARS:
                continue
            if len(ln) > MAX_BULLET_CHARS:
                ln = ln[:MAX_BULLET_CHARS] + "..."
            if chunk_total_chars + len(ln) > scaled_max_chunk_chars:
                break
            bullet_lines.append(f"- {ln}")
            chunk_total_chars += len(ln)
            if len(bullet_lines) >= scaled_max_bullets:
                break

        if bullet_lines:
            # Use neutral separator - no headers (they get echoed)
            block = "\n".join(bullet_lines)
            blocks.append(block)

    # Join with clear separators for LED models
    # Using "\n\n---\n\n" instead of "\n===\n" provides better structure
    # for LED models, helping them distinguish chunk boundaries and
    # preventing summaries from blending together
    return "\n\n---\n\n".join(blocks).strip()


def _postprocess_ml_summary(text: str) -> str:
    """Clean up BART/LED output artifacts.

    BART/LED models can produce:
    - Leaked separators (=== or ---)
    - Bullet prefixes we don't want in final output
    - Repetition spirals
    - Minor formatting issues

    This function cleans the output into plain sentences.

    Args:
        text: Raw model output

    Returns:
        Clean, sentence-formatted summary
    """
    if not text:
        return text

    # Remove any leaked separators (both old === and new ---)
    text = text.replace("===", " ")
    text = text.replace("---", " ")
    text = re.sub(r"\s+", " ", text)

    # Handle inline bullets (.-) by converting to newlines first
    text = re.sub(r"\.\s*-\s*", ".\n", text)

    # Split into sentences
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    cleaned = []
    seen_starts = set()  # Track sentence starts to detect repetition

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        # Remove bullet prefixes
        if sent.startswith(("-", "•", "*")):
            sent = sent[1:].strip()

        # Skip very short fragments
        if len(sent) < 15:
            continue

        # Detect repetition spirals (same start repeated)
        start = sent[:30].lower() if len(sent) >= 30 else sent.lower()
        if start in seen_starts:
            continue
        seen_starts.add(start)

        # Capitalize first letter
        if sent and sent[0].islower():
            sent = sent[0].upper() + sent[1:]

        # Ensure ends with punctuation
        if sent and sent[-1] not in ".!?":
            sent = sent + "."

        cleaned.append(sent)

    result = " ".join(cleaned)

    # Final cleanup
    result = re.sub(r"\s+", " ", result)
    result = re.sub(r"\s+([.,!?])", r"\1", result)

    return result.strip()


def _dedupe_sentences(text: str, similarity_threshold: float = 0.7) -> str:
    """Remove near-duplicate sentences using character n-gram overlap.

    This is a safe, surface-level deduplication that removes:
    - Near-duplicate sentences (same idea, slightly different wording)
    - "she did X... she repeated Y..." patterns

    Uses character trigram Jaccard similarity - no semantics, no risk.

    Args:
        text: Text to deduplicate
        similarity_threshold: Jaccard similarity threshold (0.7 = 70% overlap)

    Returns:
        Text with near-duplicates removed
    """
    if not text:
        return text

    def get_trigrams(s: str) -> set:
        """Get character trigrams from a string."""
        s = s.lower()
        return {s[i : i + 3] for i in range(len(s) - 2)} if len(s) >= 3 else {s}

    def jaccard_similarity(s1: str, s2: str) -> float:
        """Calculate Jaccard similarity between two strings using trigrams."""
        t1 = get_trigrams(s1)
        t2 = get_trigrams(s2)
        if not t1 or not t2:
            return 0.0
        intersection = len(t1 & t2)
        union = len(t1 | t2)
        return intersection / union if union > 0 else 0.0

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    kept: list[str] = []

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        # Check if this sentence is too similar to any already kept
        is_duplicate = False
        for kept_sent in kept:
            if jaccard_similarity(sent, kept_sent) > similarity_threshold:
                # Keep the longer one (more informative)
                if len(sent) > len(kept_sent):
                    kept.remove(kept_sent)
                    kept.append(sent)
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(sent)

    return " ".join(kept)


def _prune_filler_sentences(text: str) -> str:
    """Remove ONLY known filler patterns after distillation.

    IMPORTANT: This is now VERY CONSERVATIVE.
    Only removes sentences that:
    1. Start with specific rhetorical filler patterns
    2. Contain credit patterns that leaked through

    NO semantic pruning (numbers, entities, etc.) - that was too aggressive
    and caused the model to collapse to credits.

    Args:
        text: Distilled summary text

    Returns:
        Text with known filler removed
    """
    if not text:
        return text

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    kept = []

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        sent_lower = sent.lower()

        # Check if sentence starts with rhetorical filler - remove
        is_filler_start = any(re.match(pat, sent_lower) for pat in RHETORICAL_FILLER_STARTS)
        if is_filler_start:
            logger.debug(f"[PRUNE] Removing filler sentence: {sent[:50]}...")
            continue

        # Check if sentence contains credit patterns - remove
        is_credit = any(re.search(pat, sent, re.IGNORECASE) for pat in POST_DISTILL_CREDIT_PATTERNS)
        if is_credit:
            logger.debug(f"[PRUNE] Removing credit sentence: {sent[:50]}...")
            continue

        # Keep everything else - no semantic pruning!
        kept.append(sent)

    result = " ".join(kept)

    # Safety: if we pruned too aggressively, return original
    if len(result) < len(text) * 0.5:  # More conservative threshold (was 0.3)
        logger.debug("[PRUNE] Pruning too aggressive, keeping original")
        return text

    return result


def _distill_final_summary_2nd_pass(
    model: "SummaryModel",
    summary_text: str,
    transcript_text: Optional[str] = None,
    episode_description: Optional[str] = None,
) -> str:
    """Apply optional 2nd-pass distillation with faithfulness prompt (Issue #387).

    This pass uses a prompt to guide the model to be faithful to the source
    and reduce hallucinations. Only effective with OpenAI provider
    (BART/LED models don't use prompts effectively).

    Args:
        model: The summary model (use reduce model for consistency)
        summary_text: The reduce phase output (or 1st distill output)
        transcript_text: Optional transcript text for faithfulness context
        episode_description: Optional episode description for faithfulness context

    Returns:
        More faithful summary
    """
    if not summary_text or not summary_text.strip():
        return summary_text

    logger.debug(f"[DISTILL-2ND] Input length: {len(summary_text)} chars")

    # Build faithfulness prompt
    faithfulness_prompt = (
        "Summarize the following text while being strictly faithful to the source. "
        "Only include information that is explicitly stated. "
        "Do not add, infer, or hallucinate any details not present in the source. "
        "Focus on accuracy and faithfulness over creativity."
    )

    # For OpenAI provider, we can use the prompt effectively
    # For BART/LED, prompts are less effective but we still try
    from podcast_scraper import config as config_module

    distill_max_tokens = config_module.DEFAULT_DISTILL_MAX_TOKENS
    distill_min_tokens = config_module.DEFAULT_DISTILL_MIN_TOKENS
    distill_num_beams = config_module.DEFAULT_DISTILL_NUM_BEAMS
    distill_no_repeat_ngram_size = config_module.DEFAULT_DISTILL_NO_REPEAT_NGRAM_SIZE
    distill_length_penalty = config_module.DEFAULT_DISTILL_LENGTH_PENALTY

    distilled = model.summarize(
        summary_text,
        max_length=distill_max_tokens,
        min_length=distill_min_tokens,
        do_sample=False,
        prompt=faithfulness_prompt,  # Use faithfulness prompt
        is_reduce_phase=False,
        is_distill_phase=True,
        max_new_tokens=distill_max_tokens,
        min_new_tokens=distill_min_tokens,
        num_beams=distill_num_beams,
        no_repeat_ngram_size=distill_no_repeat_ngram_size,
        length_penalty=distill_length_penalty,
        early_stopping=True,
    )

    # Post-process the distilled output
    distilled = _postprocess_ml_summary(distilled)

    # Remove near-duplicate sentences
    distilled = _dedupe_sentences(distilled)

    # Conservative post-distill pruning
    distilled = _prune_filler_sentences(distilled)

    logger.debug(f"[DISTILL-2ND] Output length: {len(distilled) if distilled else 0} chars")

    # If distillation produced something too short or empty, return original
    if not distilled or len(distilled) < 50:
        logger.warning(
            f"2nd-pass distillation produced too short output "
            f"({len(distilled) if distilled else 0} chars), returning pre-distill summary."
        )
        return _postprocess_ml_summary(summary_text)

    return distilled


def _distill_final_summary(model: "SummaryModel", summary_text: str) -> str:
    """Apply final distillation pass to tighten the summary.

    This pass:
    1. Runs another summarization with tight constraints
    2. Removes ONLY known garbage (credits, filler starts)

    IMPORTANT: No semantic pruning before distillation.
    BART/LED need messy content - they decide importance.

    Args:
        model: The summary model (use reduce model for consistency)
        summary_text: The reduce phase output

    Returns:
        Tighter summary
    """
    if not summary_text or not summary_text.strip():
        return summary_text

    # Log input length for debugging
    logger.debug(f"[DISTILL] Input length: {len(summary_text)} chars")

    # Step 1: Run distillation pass with tight constraints
    # Note: We don't use prompts for BART/LED (they get copied)
    # The tight decoding params do the work
    # NO pre-filtering - let the model see everything
    # All defaults come from Config (no hardcoded values)
    from podcast_scraper import config as config_module

    distill_max_tokens = config_module.DEFAULT_DISTILL_MAX_TOKENS
    distill_min_tokens = config_module.DEFAULT_DISTILL_MIN_TOKENS
    distill_num_beams = config_module.DEFAULT_DISTILL_NUM_BEAMS
    distill_no_repeat_ngram_size = config_module.DEFAULT_DISTILL_NO_REPEAT_NGRAM_SIZE
    distill_length_penalty = config_module.DEFAULT_DISTILL_LENGTH_PENALTY

    distilled = model.summarize(
        summary_text,  # Use original, no pre-filtering
        max_length=distill_max_tokens,
        min_length=distill_min_tokens,
        do_sample=False,
        is_reduce_phase=False,
        is_distill_phase=True,
        max_new_tokens=distill_max_tokens,
        min_new_tokens=distill_min_tokens,
        num_beams=distill_num_beams,
        no_repeat_ngram_size=distill_no_repeat_ngram_size,
        length_penalty=distill_length_penalty,
        early_stopping=True,  # Default for distill
    )

    # Step 2: Post-process the distilled output
    distilled = _postprocess_ml_summary(distilled)

    # Step 3: Remove near-duplicate sentences (surface similarity)
    # This is safe - no semantics, just character n-gram overlap
    distilled = _dedupe_sentences(distilled)

    # Step 4: CONSERVATIVE post-distill pruning
    # Only remove KNOWN garbage (credits, filler starts)
    distilled = _prune_filler_sentences(distilled)

    logger.debug(f"[DISTILL] Output length: {len(distilled) if distilled else 0} chars")

    # If distillation produced something too short or empty, return original
    if not distilled or len(distilled) < 50:
        logger.warning(
            f"Distillation produced too short output ({len(distilled) if distilled else 0} chars), "
            "returning pre-distill summary."
        )
        return _postprocess_ml_summary(summary_text)

    return distilled


def _combine_summaries_extractive(
    model: SummaryModel,
    selected_summaries: List[str],
    max_length: int,
    min_length: int,
    prompt: Optional[str],
    model_max: int,
    # New explicit generation params for reduce stage
    reduce_max_new_tokens: Optional[int] = None,
    reduce_min_new_tokens: Optional[int] = None,
    reduce_num_beams: Optional[int] = None,
    reduce_no_repeat_ngram_size: Optional[int] = None,
    reduce_length_penalty: Optional[float] = None,
    reduce_early_stopping: Optional[bool] = None,
    reduce_repetition_penalty: Optional[float] = None,
    reduce_encoder_no_repeat_ngram_size: Optional[int] = None,
    reduce_max_input_tokens: Optional[int] = None,
    truncation: Optional[bool] = None,
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

    final_summary = _join_summaries_with_structure(selected_summaries)

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
                is_reduce_phase=True,  # Use REDUCE-specific generation params
                # Pass reduce generation params
                max_new_tokens=reduce_max_new_tokens,
                min_new_tokens=reduce_min_new_tokens,
                num_beams=reduce_num_beams,
                no_repeat_ngram_size=reduce_no_repeat_ngram_size,
                length_penalty=reduce_length_penalty,
                early_stopping=reduce_early_stopping,
                max_input_tokens=reduce_max_input_tokens,
                truncation=truncation,
            )

            # Validate: if summary is suspiciously long, it might be hallucinating
            if len(final_summary) > len(selected_summaries) * REPETITIVE_SUMMARY_THRESHOLD:
                logger.warning(
                    "Final summary suspiciously long, using extractive summaries directly "
                    "(no further summarization to prevent hallucinations)"
                )
                return _join_summaries_with_structure(selected_summaries)

            return final_summary
        except Exception as e:
            logger.warning(f"Final summarization failed ({e}), using extractive summaries directly")
            return _join_summaries_with_structure(selected_summaries)
    else:
        logger.debug("Using extractive summaries directly (no further summarization)")
        return final_summary


def _combine_summaries_mini_map_reduce(
    model: SummaryModel,
    combined_text: str,
    chunk_summaries: List[str],
    max_length: int,
    min_length: int,
    prompt: Optional[str],
    combined_tokens: int,
    target_tokens: int,
    max_passes: int = MAX_HIERARCHICAL_PASSES,
    # New explicit generation params for reduce stage
    reduce_max_new_tokens: Optional[int] = None,
    reduce_min_new_tokens: Optional[int] = None,
    reduce_num_beams: Optional[int] = None,
    reduce_no_repeat_ngram_size: Optional[int] = None,
    reduce_length_penalty: Optional[float] = None,
    reduce_early_stopping: Optional[bool] = None,
    reduce_repetition_penalty: Optional[float] = None,
    reduce_encoder_no_repeat_ngram_size: Optional[int] = None,
    reduce_max_input_tokens: Optional[int] = None,
    truncation: Optional[bool] = None,
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
        mini_chunks = _chunk_by_tokens(
            current_text,
            model.tokenizer,
            max_tokens=mini_chunk_size_tokens,
            lock=model._summarize_lock,
        )

        # Validate chunk sizes to ensure they don't exceed model limit
        max_chunk_tokens = 0
        # Serialize tokenizer access for thread safety
        with model._summarize_lock:
            for i, chunk in enumerate(mini_chunks, 1):
                # Chunks are already within limits, but add truncation to prevent any warnings
                chunk_tokens = len(
                    model.tokenizer.encode(  # type: ignore[attr-defined]
                        chunk, add_special_tokens=False, truncation=True, max_length=100000
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
                    # Pass reduce generation params
                    max_new_tokens=reduce_max_new_tokens,
                    min_new_tokens=reduce_min_new_tokens,
                    num_beams=reduce_num_beams,
                    no_repeat_ngram_size=reduce_no_repeat_ngram_size,
                    length_penalty=reduce_length_penalty,
                    early_stopping=reduce_early_stopping,
                    repetition_penalty=reduce_repetition_penalty,
                    encoder_no_repeat_ngram_size=reduce_encoder_no_repeat_ngram_size,
                    max_input_tokens=reduce_max_input_tokens,
                    truncation=truncation,
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
            return _combine_summaries_extractive(
                model,
                _select_key_summaries(chunk_summaries),
                max_length,
                min_length,
                prompt,
                BART_MAX_POSITION_EMBEDDINGS,
            )

        # Step 3: Join summaries with newlines (preserves structure)
        current_text = _join_summaries_with_structure(section_summaries)
        current_chars = len(current_text)
        current_words = len(current_text.split())
        if model.tokenizer:
            # Current text is from section summaries, but add truncation to prevent warnings
            # Serialize tokenizer access for thread safety
            with model._summarize_lock:
                current_tokens = len(
                    model.tokenizer.encode(  # type: ignore[attr-defined]
                        current_text, add_special_tokens=False, truncation=True, max_length=100000
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
        return _combine_summaries_extractive(
            model,
            _select_key_summaries(chunk_summaries),
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

    final_summary = _combine_summaries_abstractive(
        model,
        current_text,
        last_section_summaries,
        max_length,
        min_length,
        prompt,
        model_max,
        current_tokens,
        # Pass reduce generation params
        reduce_max_new_tokens=reduce_max_new_tokens,
        reduce_min_new_tokens=reduce_min_new_tokens,
        reduce_num_beams=reduce_num_beams,
        reduce_no_repeat_ngram_size=reduce_no_repeat_ngram_size,
        reduce_length_penalty=reduce_length_penalty,
        reduce_early_stopping=reduce_early_stopping,
        reduce_repetition_penalty=reduce_repetition_penalty,
        reduce_encoder_no_repeat_ngram_size=reduce_encoder_no_repeat_ngram_size,
        reduce_max_input_tokens=reduce_max_input_tokens,
        truncation=truncation,
    )

    total_mini_time = time.time() - mini_map_start

    logger.debug(
        f"[MAP-REDUCE VALIDATION] ⚡ MINI MAP-REDUCE COMPLETE: "
        f"total_time={total_mini_time:.1f}s ({iteration} iteration(s)), "
        f"input={combined_tokens} tokens -> output={len(final_summary.split())} words"
    )

    return final_summary


def _combine_summaries_abstractive(
    model: SummaryModel,
    combined_text: str,
    chunk_summaries: List[str],
    max_length: int,
    min_length: int,
    prompt: Optional[str],
    model_max: int,
    combined_tokens: int,
    # New explicit generation params for reduce stage
    reduce_max_new_tokens: Optional[int] = None,
    reduce_min_new_tokens: Optional[int] = None,
    reduce_num_beams: Optional[int] = None,
    reduce_no_repeat_ngram_size: Optional[int] = None,
    reduce_length_penalty: Optional[float] = None,
    reduce_early_stopping: Optional[bool] = None,
    reduce_repetition_penalty: Optional[float] = None,
    reduce_encoder_no_repeat_ngram_size: Optional[int] = None,
    reduce_max_input_tokens: Optional[int] = None,
    truncation: Optional[bool] = None,
    # Optional 2nd-pass distill parameters (Issue #387)
    enable_2nd_pass_distill: bool = False,
    transcript_text: Optional[str] = None,
    episode_description: Optional[str] = None,
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
    # Dynamic capping to prevent expansion (Issue #283 fix)
    # The problem: min_new_tokens=220 forces expansion when input is only ~156 tokens
    # Solution: Cap both max_new_tokens and min_new_tokens based on input size

    # Calculate dynamic caps based on input size
    max_output_ratio = 0.45  # Target 45% of input tokens for output (prevents expansion)

    # Cap max_new_tokens: use smaller of config value or dynamic cap
    if reduce_max_new_tokens is None:
        # Fallback to max_length if max_new_tokens not provided
        base_max_new_tokens = max_length
    else:
        base_max_new_tokens = reduce_max_new_tokens

    # Dynamic cap: Use higher ratio when input is small (allows fuller rewrite without expansion)
    # For small inputs (< 250 tokens), allow 0.8-1.2 ratio (fuller summary)
    # For larger inputs, use 0.45 ratio (prevents expansion)
    if combined_tokens < 250:
        # Small input: allow more tokens for a fuller rewrite (won't expand wildly)
        small_input_ratio = min(1.2, max(0.8, max_output_ratio * 1.5))
        dynamic_max_new_tokens = max(120, int(combined_tokens * small_input_ratio))
        logger.debug(
            f"[MAP-REDUCE VALIDATION] Small reduce input ({combined_tokens} tokens), "
            f"using higher ratio ({small_input_ratio:.2f}) for fuller summary"
        )
    else:
        # Larger input: use standard ratio to prevent expansion
        dynamic_max_new_tokens = max(80, int(combined_tokens * max_output_ratio))
    effective_max_new_tokens = min(base_max_new_tokens, dynamic_max_new_tokens)

    # Dynamic min_new_tokens: 0 (no minimum) or small value, never force expansion
    # IMPORTANT: For reduce stage, min_new_tokens should ALWAYS be 0 to prevent forced expansion
    # Reduce stage should never be forced to produce a minimum length - let the model decide
    # the optimal summary length based on content, not arbitrary minimums
    # Map stage can keep a floor (e.g., 60-100 tokens) for consistency, but reduce should be 0
    effective_min_new_tokens = 0
    if reduce_min_new_tokens is not None and reduce_min_new_tokens > 0:
        logger.debug(
            f"[MAP-REDUCE] Reduce stage: ignoring min_new_tokens={reduce_min_new_tokens}, "
            f"using 0 to prevent forced expansion (reduce should never be forced to expand)"
        )

    # Also cap max_length for backward compatibility (though max_new_tokens takes precedence)
    base_max_length = int(
        min(
            max_length * FINAL_MAX_LENGTH_MULTIPLIER,
            model_max - MODEL_MAX_BUFFER,
            SAFE_MAX_LENGTH,
        )
    )
    input_based_max_length = int(combined_tokens * max_output_ratio)
    final_max_length = max(min_length, min(base_max_length, input_based_max_length))

    # Log if we capped any parameters - include policy name and thresholds for future debugging
    max_capped = effective_max_new_tokens < base_max_new_tokens
    min_capped = effective_min_new_tokens < (
        reduce_min_new_tokens if reduce_min_new_tokens is not None else float("inf")
    )

    # Determine policy name based on input size
    if combined_tokens < 250:
        policy_name = "cap_v2_small_input"
        policy_threshold = f"ratio={small_input_ratio:.2f}, min_floor=0 when input < 250 tokens"
    else:
        policy_name = "cap_v2_standard"
        policy_threshold = f"ratio={max_output_ratio}, min_floor=0 when input >= 250 tokens"

    if max_capped or min_capped:
        logger.info(
            f"[MAP-REDUCE VALIDATION] Policy: {policy_name} ({policy_threshold}) - "
            f"Capping reduce params to prevent expansion: "
            f"max_new_tokens={base_max_new_tokens}→{effective_max_new_tokens}, "
            f"min_new_tokens={reduce_min_new_tokens or 0}→{effective_min_new_tokens} "
            f"(input={combined_tokens} tokens)"
        )
    else:
        # Log policy even when not capping (for transparency)
        logger.debug(
            f"[MAP-REDUCE VALIDATION] Policy: {policy_name} ({policy_threshold}) - "
            f"No capping needed: max_new_tokens={effective_max_new_tokens}, "
            f"min_new_tokens={effective_min_new_tokens} (input={combined_tokens} tokens)"
        )

    logger.debug(
        f"Final summarization: {len(chunk_summaries)} chunks, "
        f"combined ~{combined_tokens} tokens, "
        f"using max_new_tokens={effective_max_new_tokens}, "
        f"min_new_tokens={effective_min_new_tokens}"
    )

    try:
        logger.debug(f"[REDUCE] Input length: {len(combined_text)} chars")

        final_summary = model.summarize(
            combined_text,
            max_length=final_max_length,
            min_length=min_length,
            do_sample=False,
            prompt=prompt,
            is_reduce_phase=True,  # Use REDUCE-specific generation params
            # Pass dynamically capped generation params
            max_new_tokens=effective_max_new_tokens,
            min_new_tokens=effective_min_new_tokens,
            num_beams=reduce_num_beams,
            no_repeat_ngram_size=reduce_no_repeat_ngram_size,
            length_penalty=reduce_length_penalty,
            early_stopping=reduce_early_stopping,
            repetition_penalty=reduce_repetition_penalty,
            encoder_no_repeat_ngram_size=reduce_encoder_no_repeat_ngram_size,
            max_input_tokens=reduce_max_input_tokens,
            truncation=truncation,
        )

        logger.debug(f"[REDUCE] Output length: {len(final_summary) if final_summary else 0} chars")

        # Post-process the REDUCE output
        final_summary = _postprocess_ml_summary(final_summary)

        # HARD GUARDRAIL: Detect expansion and retry or fall back to map-only
        # If output is within 95% of input, treat as failure and retry with smaller cap
        # If retry still fails, skip reduce and use map-only (join chunk summaries)
        EXPANSION_THRESHOLD = 0.95  # If output > input * 0.95, treat as expansion
        RETRY_MAX_OUTPUT_RATIO = 0.30  # Retry with 30% cap (more aggressive)

        output_chars = len(final_summary) if final_summary else 0
        input_chars = len(combined_text)
        used_map_only = False  # Track if we fell back to map-only

        if output_chars > input_chars * EXPANSION_THRESHOLD:
            logger.warning(
                f"[GUARDRAIL] Reduce output ({output_chars} chars) exceeds expansion "
                f"threshold ({input_chars * EXPANSION_THRESHOLD:.0f} chars, "
                f"{EXPANSION_THRESHOLD*100:.0f}% of input). "
                f"Retrying with smaller cap ({RETRY_MAX_OUTPUT_RATIO*100:.0f}% of input)..."
            )

            # Retry with more aggressive capping (30% instead of 45%)
            retry_dynamic_max_new_tokens = max(60, int(combined_tokens * RETRY_MAX_OUTPUT_RATIO))
            retry_effective_max_new_tokens = min(
                effective_max_new_tokens, retry_dynamic_max_new_tokens
            )
            retry_effective_min_new_tokens = 0  # Force no minimum on retry

            logger.debug(
                f"[GUARDRAIL] Retry params: "
                f"max_new_tokens={effective_max_new_tokens}→{retry_effective_max_new_tokens}, "
                f"min_new_tokens={effective_min_new_tokens}→{retry_effective_min_new_tokens}"
            )

            try:
                retry_final_summary = model.summarize(
                    combined_text,
                    max_length=final_max_length,
                    min_length=min_length,
                    do_sample=False,
                    prompt=prompt,
                    is_reduce_phase=True,
                    max_new_tokens=retry_effective_max_new_tokens,
                    min_new_tokens=retry_effective_min_new_tokens,
                    num_beams=reduce_num_beams,
                    no_repeat_ngram_size=reduce_no_repeat_ngram_size,
                    length_penalty=reduce_length_penalty,
                    early_stopping=reduce_early_stopping,
                    repetition_penalty=reduce_repetition_penalty,
                    encoder_no_repeat_ngram_size=reduce_encoder_no_repeat_ngram_size,
                    max_input_tokens=reduce_max_input_tokens,
                    truncation=truncation,
                )
                retry_final_summary = _postprocess_ml_summary(retry_final_summary)
                retry_output_chars = len(retry_final_summary) if retry_final_summary else 0

                # Check if retry succeeded (output < 95% of input)
                if retry_output_chars <= input_chars * EXPANSION_THRESHOLD:
                    logger.info(
                        f"[GUARDRAIL] Retry succeeded: output ({retry_output_chars} chars) "
                        f"is now within threshold ({input_chars * EXPANSION_THRESHOLD:.0f} chars)"
                    )
                    final_summary = retry_final_summary
                    output_chars = retry_output_chars
                else:
                    # Retry also failed - fall back to map-only (join chunk summaries)
                    logger.warning(
                        f"[GUARDRAIL] Retry failed: output ({retry_output_chars} chars) "
                        f"still exceeds threshold. "
                        f"Falling back to map-only (joining chunk summaries without "
                        f"reduce phase)."
                    )
                    # Join chunk summaries directly (map-only, no reduce)
                    final_summary = _join_summaries_with_structure(chunk_summaries)
                    used_map_only = True
                    logger.info(
                        f"[GUARDRAIL] Using map-only result: {len(final_summary)} chars "
                        f"(from {len(chunk_summaries)} chunk summaries)"
                    )
            except Exception as retry_error:
                # Retry failed with error - fall back to map-only
                logger.warning(
                    f"[GUARDRAIL] Retry failed with error: {retry_error}. "
                    "Falling back to map-only (joining chunk summaries without reduce phase)."
                )
                final_summary = _join_summaries_with_structure(chunk_summaries)
                used_map_only = True
                logger.info(
                    f"[GUARDRAIL] Using map-only result: {len(final_summary)} chars "
                    f"(from {len(chunk_summaries)} chunk summaries)"
                )

        # Apply DISTILL phase conditionally - only if output is long enough to benefit
        # When reduce input is already short (~150-200 tokens), distilling again often expands
        # Skip DISTILL if we used map-only fallback (already concise)
        if not used_map_only:
            DISTILL_THRESHOLD_CHARS = 1200  # Only distill if output is above this threshold
            if len(final_summary) > DISTILL_THRESHOLD_CHARS:
                logger.debug(
                    f"[DISTILL] Reduce output ({len(final_summary)} chars) exceeds threshold "
                    f"({DISTILL_THRESHOLD_CHARS} chars), applying distillation"
                )
                final_summary = _distill_final_summary(model, final_summary)

                # Apply optional 2nd-pass distill with faithfulness prompt (Issue #387)
                if enable_2nd_pass_distill:
                    logger.debug(
                        "[DISTILL-2ND] Applying 2nd-pass distillation with faithfulness prompt"
                    )
                    final_summary = _distill_final_summary_2nd_pass(
                        model,
                        final_summary,
                        transcript_text,
                        episode_description,
                    )
            else:
                logger.debug(
                    f"[DISTILL] Skipping distillation - reduce output ({len(final_summary)} chars) "
                    f"is already concise (threshold: {DISTILL_THRESHOLD_CHARS} chars)"
                )

        # Validate summary quality (Issue #283 follow-up)
        # Check for expansion (summary longer than input) - this is a critical issue
        if len(final_summary) > len(combined_text):
            logger.warning(
                f"Final summary ({len(final_summary)} chars) is LONGER than input "
                f"({len(combined_text)} chars). Model expanded instead of summarizing. "
                "This may indicate model issues or input already being very concise."
            )
        # Check for poor compression (summary close to input length)
        elif len(final_summary) > len(combined_text) * SUMMARY_VALIDATION_THRESHOLD:
            compression_ratio = len(combined_text) / len(final_summary)
            logger.debug(
                f"Final summary length ({len(final_summary)} chars) is close to "
                f"input length ({len(combined_text)} chars, compression={compression_ratio:.2f}x). "
                f"Acceptable for LED models (threshold={SUMMARY_VALIDATION_THRESHOLD})."
            )

        # Smarter "too short" warning: only warn if output is suspiciously short
        # AND input was large enough to expect more content
        # Don't warn on short-format shows or when input was already concise
        output_words = len(final_summary.split()) if final_summary else 0
        input_words = len(combined_text.split()) if combined_text else 0
        if output_words < 80 and input_words > 1200 and len(final_summary) < 400:
            logger.warning(
                f"Final summary seems too short ({len(final_summary)} chars, {output_words} words) "
                f"for large input ({input_words} words). This might indicate summarization issues."
            )
        elif output_words < 50 and input_words > 2000:
            # Very short output for very large input - definitely suspicious
            logger.warning(
                f"Final summary is very short ({len(final_summary)} chars, "
                f"{output_words} words) for very large input ({input_words} words). "
                f"This might indicate summarization issues."
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


def safe_summarize(
    model: SummaryModel,
    text: str,
    max_length: int = 150,
    prompt: Optional[str] = None,
) -> str:
    """Safely generate summary with error handling.

    Args:
        model: Summary model instance
        text: Input text
        max_length: Maximum summary length

    Returns:
        Summary text, or empty string on failure
    """
    try:
        return model.summarize(text, max_length=max_length, prompt=prompt)
    except Exception as e:
        # Handle both CUDA and MPS out-of-memory errors
        # Check error message string (torch not needed for this check)
        if isinstance(e, RuntimeError) and (
            "out of memory" in str(e).lower() or "mps" in str(e).lower()
        ):
            logger.error(f"Device out of memory during summarization ({model.device}): {e}")
            # Fallback: use CPU or smaller model
            return ""
        logger.error(f"Summarization error: {e}")
        return ""


def optimize_model_memory(model: SummaryModel) -> None:
    """Optimize model for memory efficiency.

    Supports both CUDA (NVIDIA) and MPS (Apple Silicon) backends.

    Args:
        model: Summary model instance
    """
    if model.model is None:
        return

    if model.device == "cuda":
        # Enable gradient checkpointing (trades compute for memory)
        if hasattr(model.model, "gradient_checkpointing_enable"):
            model.model.gradient_checkpointing_enable()  # type: ignore[attr-defined]

        # Use half precision (FP16) to reduce memory
        model.model = model.model.half()  # type: ignore[attr-defined,assignment]

        # Lazy import torch for CUDA operations
        import torch  # noqa: F401

        # Clear cache
        torch.cuda.empty_cache()
    elif model.device == "mps":
        # Lazy import torch for MPS operations
        import torch  # noqa: F401

        # Apple Silicon MPS backend
        # MPS supports FP16 natively, but may have different optimizations
        # For M4 Pro with 48GB, memory is less constrained, but still optimize
        if hasattr(model.model, "gradient_checkpointing_enable"):
            model.model.gradient_checkpointing_enable()  # type: ignore[attr-defined]

        # MPS may benefit from FP16, but test performance impact
        # model.model = model.model.half()  # Test if needed

        # Clear MPS cache if available
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()


def unload_model(model: Optional[SummaryModel]) -> None:
    """Unload model to free memory and clean up resources.

    Args:
        model: Summary model instance, or None (no-op if None)
    """
    if model is None:
        return

    # Lazy import torch for model cleanup and cache clearing (only if available)
    # Unit tests run without ML dependencies, so torch may not be installed
    import gc  # noqa: F401

    try:
        import torch  # noqa: F401

        # Move model to CPU before deletion to avoid teardown hazards (Issue #390)
        # This prevents bus errors on macOS during interpreter shutdown
        if model.model is not None:
            # Move model to CPU before deletion
            try:
                model.model = model.model.to("cpu")  # type: ignore[assignment,attr-defined]
            except Exception:
                # Ignore errors (model might already be on CPU or in unexpected state)
                pass

        # Clear device-specific cache (wrap in try-except to handle any torch errors)
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                if hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
        except Exception:
            # Ignore any errors from torch operations (e.g., device not available, etc.)
            # This ensures cleanup doesn't fail even if torch is in an unexpected state
            pass
    except ImportError:
        # torch not available (e.g., in unit tests without ML dependencies)
        # This is fine - we'll just do basic garbage collection
        pass

    # Delete model components to release memory
    if model.model:
        del model.model
    if model.tokenizer:
        del model.tokenizer
    if model.pipeline:
        del model.pipeline

    model.model = None
    model.tokenizer = None
    model.pipeline = None

    # Force garbage collection to clean up any remaining references
    # This helps release memory and clean up threads that might be holding references
    # Note: In test environments, gc.collect() can sometimes hang due to finalizers,
    # so we skip it if we detect we're in a test environment
    import os

    # Skip gc.collect() in test environments to avoid hangs
    # The test fixture cleanup_ml_resources_after_test will handle GC
    if os.environ.get("PYTEST_CURRENT_TEST") is None:
        gc.collect()


def get_cache_size(cache_dir: Optional[str] = None) -> int:
    """Get total size of Hugging Face model cache in bytes.

    Args:
        cache_dir: Cache directory path (defaults to standard HF cache)

    Returns:
        Total cache size in bytes
    """
    if cache_dir:
        cache_path = Path(cache_dir)
    else:
        # Use consistent cache path resolution (respects HF_HUB_CACHE env var)
        try:
            from ...cache import get_transformers_cache_dir

            cache_path = get_transformers_cache_dir()
        except Exception:
            cache_path = HF_CACHE_DIR

    if not cache_path.exists():
        return 0

    total_size = 0
    try:
        for item in cache_path.rglob("*"):
            if item.is_file():
                total_size += item.stat().st_size
    except (OSError, PermissionError) as e:
        # Partial results are acceptable for cache info display
        # Permission errors might occur in shared environments
        logger.debug(f"Could not fully calculate cache size (partial result): {e}")

    return total_size


def format_cache_size(size_bytes: int) -> str:
    """Format cache size in human-readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    size: float = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def prune_cache(cache_dir: Optional[str] = None, dry_run: bool = False) -> int:
    """Prune Hugging Face model cache by removing all cached models.

    Args:
        cache_dir: Cache directory path (defaults to standard HF cache)
        dry_run: If True, only report what would be deleted without deleting

    Returns:
        Number of files that would be/were deleted
    """
    if cache_dir:
        cache_path = Path(cache_dir)
        # Security check: ensure cache directory is within safe locations
        # Only allow deletion within user's home directory or standard cache locations
        # (but not the home directory or ~/.cache themselves to prevent accidental deletion)
        try:
            resolved_path = cache_path.resolve()
            home = Path.home()
            cache_root = home / ".cache"
            safe_roots = {home, cache_root}
            # Security check: path must be within a safe root AND not be the safe root itself
            # Explicitly exclude home directory and ~/.cache themselves
            is_safe = (
                any(
                    resolved_path.is_relative_to(root) and resolved_path != root
                    for root in safe_roots
                )
                and resolved_path != cache_root
            )
            if not is_safe:
                raise ValueError(
                    f"Cache directory {resolved_path} is outside safe locations "
                    f"(home directory or ~/.cache) or is a protected root directory. "
                    f"Refusing to delete for security."
                )
        except (OSError, RuntimeError) as e:
            raise ValueError(f"Invalid cache directory path: {e}") from e
    else:
        # Use consistent cache path resolution (respects HF_HUB_CACHE env var)
        try:
            from ...cache import get_transformers_cache_dir

            cache_path = get_transformers_cache_dir()
        except Exception:
            cache_path = HF_CACHE_DIR

    if not cache_path.exists():
        logger.info("Cache directory does not exist: %s", cache_path)
        return 0

    deleted_count = 0
    total_size = 0

    try:
        for item in cache_path.rglob("*"):
            if item.is_file():
                file_size = item.stat().st_size
                total_size += file_size
                if not dry_run:
                    try:
                        item.unlink()
                        deleted_count += 1
                    except (OSError, PermissionError) as e:
                        logger.debug("Failed to delete %s: %s", item, e)
                else:
                    deleted_count += 1
            elif item.is_dir() and not any(item.iterdir()):
                # Remove empty directories
                if not dry_run:
                    try:
                        item.rmdir()
                    except (OSError, PermissionError) as e:
                        # Best-effort cleanup of empty directories; failures are acceptable
                        logger.debug(f"Could not remove empty directory {item}: {e}")

        size_str = format_cache_size(total_size)
        if dry_run:
            logger.info(
                "Would delete %d files (%s) from cache: %s",
                deleted_count,
                size_str,
                cache_path,
            )
        else:
            logger.info(
                "Deleted %d files (%s) from cache: %s",
                deleted_count,
                size_str,
                cache_path,
            )
    except (OSError, PermissionError) as e:
        logger.error("Error pruning cache: %s", e)

    return deleted_count
