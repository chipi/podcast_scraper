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
  * Use default models (e.g., 'bart-large', 'bart-small', 'long', 'long-fast') from trusted sources
  * Pin model revisions in production: SummaryModel(model, revision="abc123")
  * Review model source before using custom models
  * Keep transformers library updated for security patches

Note: Preprocessing functions (clean_transcript, remove_sponsor_blocks, etc.)
have been moved to preprocessing.py in Stage 1. Wrapper functions remain here
for backward compatibility but will be deprecated in a future release.
"""

import logging
import warnings
from pathlib import Path
from typing import cast, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    # Type hints only - these are not imported at runtime
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Pipeline


# Note: torch and transformers are imported lazily in methods that use them
# (e.g., _detect_device(), _load_model()) to avoid requiring ML dependencies
# at module import time. This allows unit tests to import this module without
# torch/transformers installed.

from . import preprocessing

# Import from refactored modules
from .summarization import chunking, map_reduce

logger = logging.getLogger(__name__)

# Hugging Face cache directory (standard locations)
# Newer transformers versions use "hub", older versions use "transformers"
HF_CACHE_BASE = Path.home() / ".cache" / "huggingface"
HF_CACHE_DIR = HF_CACHE_BASE / "hub"  # Default for newer transformers
HF_CACHE_DIR_LEGACY = HF_CACHE_BASE / "transformers"  # Fallback for older versions

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
CHUNK_SUMMARY_MIN_TOKENS = 80  # Target lower bound for map summaries
CHUNK_SUMMARY_MAX_TOKENS = 160  # Target upper bound for map summaries
SECTION_SUMMARY_MIN_TOKENS = 80  # Target lower bound for hierarchical section summaries
SECTION_SUMMARY_MAX_TOKENS = 160  # Target upper bound for hierarchical section summaries
FINAL_SUMMARY_MIN_TOKENS = 200  # Target lower bound for final reduce
FINAL_SUMMARY_MAX_TOKENS = (
    480  # Target upper bound for final reduce (slightly higher for more detail)
)
# Sponsor removal patterns
SPONSOR_BLOCK_PATTERNS = [
    r"this episode is brought to you by.*?(?=\n\n|\Z)",
    r"today['’]s episode is sponsored by.*?(?=\n\n|\Z)",
    r"our sponsor(?:s)? (?:today|this week) (?:is|are).*?(?=\n\n|\Z)",
    r"thanks again to our (?:friends|sponsors) at.*?(?=\n\n|\Z)",
]
# Re-export OUTRO_BLOCK_PATTERNS from preprocessing for backward compatibility
OUTRO_BLOCK_PATTERNS = preprocessing.OUTRO_BLOCK_PATTERNS
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
SAFE_MAX_LENGTH = 512  # Safe maximum length for final summarization

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

# Generation parameters
REPETITION_PENALTY = 1.3  # Penalty for repetition to prevent hallucinations
NO_REPEAT_NGRAM_SIZE = 4  # Prevent n-gram repetition
NUM_BEAMS = 4  # Number of beams for beam search (balance of quality and speed)
LENGTH_PENALTY = (
    1.0  # Length penalty for beam search (1.0 = no penalty, encourage longer summaries)
)

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

# Import model ID constants (config-driven, no hardcoded values)
from .config_constants import (
    SUMMARY_MODEL_BART_BASE,
    SUMMARY_MODEL_BART_LARGE_CNN,
    SUMMARY_MODEL_LED_BASE_16384,
    SUMMARY_MODEL_LED_LARGE_16384,
)

# Model aliases - only include models that are actually tested and used
# Note: Other models (distilbart, pegasus) can still be used via direct model IDs
# but are not included as aliases since they haven't been tested/validated
DEFAULT_SUMMARY_MODELS = {
    # BART-large (best quality, ~2GB memory), 1024 token limit
    # Production default for MAP phase
    "bart-large": SUMMARY_MODEL_BART_LARGE_CNN,
    # BART-base (smallest, lowest memory ~500MB), 1024 token limit
    # Test default for MAP phase
    "bart-small": SUMMARY_MODEL_BART_BASE,
    # LED-large (long docs 16k tokens, ~2.5GB), NO chunking
    # Production default for REDUCE phase
    "long": SUMMARY_MODEL_LED_LARGE_16384,
    # LED-base (long docs 16k tokens, ~1GB), NO chunking
    # Test default for REDUCE phase
    "long-fast": SUMMARY_MODEL_LED_BASE_16384,
}

# Default prompt for summarization
# Important: Explicitly instruct model to avoid hallucinations and stick to source content
DEFAULT_SUMMARY_PROMPT = (
    "Summarize the following podcast episode transcript accurately. "
    "Focus on the main topics, key insights, and important discussions. "
    "Only include information that is explicitly stated in the transcript. "
    "Do not add, infer, or invent any information not present in the original text:"
)


def _validate_model_source(model_name: str) -> None:
    """Validate model source and warn if not from trusted sources.

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
                "    Consider using default models "
                "(e.g., 'bart-large', 'bart-small', 'long', 'long-fast') "
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


# Wrapper functions for backward compatibility (Stage 1)
# These functions delegate to preprocessing module and will be deprecated in a future release.


def clean_transcript(
    text: str,
    remove_timestamps: bool = True,
    normalize_speakers: bool = True,
    collapse_blank_lines: bool = True,
    remove_fillers: bool = False,
) -> str:
    """Clean podcast transcript for better summarization quality.

    .. deprecated:: 2.5.0
        Use :func:`podcast_scraper.preprocessing.clean_transcript` instead.

    This is a wrapper function that delegates to the preprocessing module.
    It is kept for backward compatibility but will be removed in a future release.

    Args:
        text: Raw transcript text
        remove_timestamps: Whether to remove timestamp patterns like [00:12:34]
        normalize_speakers: Whether to remove generic speaker tags (preserves actual names)
        collapse_blank_lines: Whether to collapse multiple blank lines into single line
        remove_fillers: Whether to remove common English filler words (disabled by default)

    Returns:
        Cleaned transcript text
    """
    warnings.warn(
        "summarizer.clean_transcript() is deprecated. "
        "Use podcast_scraper.preprocessing.clean_transcript() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return preprocessing.clean_transcript(
        text,
        remove_timestamps=remove_timestamps,
        normalize_speakers=normalize_speakers,
        collapse_blank_lines=collapse_blank_lines,
        remove_fillers=remove_fillers,
    )


def remove_sponsor_blocks(text: str) -> str:
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


def remove_outro_blocks(text: str) -> str:
    """Remove outro/closing blocks from transcript.

    .. deprecated:: 2.5.0
        Use :func:`podcast_scraper.preprocessing.remove_outro_blocks` instead.

    This is a wrapper function that delegates to the preprocessing module.
    It is kept for backward compatibility but will be removed in a future release.

    Args:
        text: Transcript text potentially containing outro blocks

    Returns:
        Text with outro blocks removed
    """
    warnings.warn(
        "summarizer.remove_outro_blocks() is deprecated. "
        "Use podcast_scraper.preprocessing.remove_outro_blocks() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return preprocessing.remove_outro_blocks(text)


def clean_for_summarization(text: str) -> str:
    """High-level cleaner for BOTH:
      - offline .cleaned.txt generation
      - runtime summarization (if you want consistency)

    .. deprecated:: 2.5.0
        Use :func:`podcast_scraper.preprocessing.clean_for_summarization` instead.

    This is a wrapper function that delegates to the preprocessing module.
    It is kept for backward compatibility but will be removed in a future release.

    Args:
        text: Raw transcript text

    Returns:
        Fully cleaned transcript text ready for summarization
    """
    warnings.warn(
        "summarizer.clean_for_summarization() is deprecated. "
        "Use podcast_scraper.preprocessing.clean_for_summarization() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return preprocessing.clean_for_summarization(text)


def select_summary_model(cfg) -> str:
    """Select MAP-phase model based on configuration.

    Defaults to BART-large for production quality chunk summarization.
    This is the MAP model used to summarize individual chunks.

    Note: Tests typically use "facebook/bart-base" (smaller, faster) for speed,
    but production defaults to "facebook/bart-large-cnn" (larger, better quality).

    Args:
        cfg: Configuration object with summary_model field

    Returns:
        Model identifier string (resolved from DEFAULT_SUMMARY_MODELS if key provided)

    Raises:
        ValueError: If an unsupported model is provided (not in DEFAULT_SUMMARY_MODELS)
    """
    if cfg.summary_model:
        model_key = cast(str, cfg.summary_model)
        # Check if it's a key in DEFAULT_SUMMARY_MODELS
        # (e.g., "bart-large", "bart-small", "long", "long-fast")
        if model_key in DEFAULT_SUMMARY_MODELS:
            return DEFAULT_SUMMARY_MODELS[model_key]
        # Reject non-alias models - only tested models are supported
        supported_models = ", ".join(DEFAULT_SUMMARY_MODELS.keys())
        raise ValueError(
            f"Unsupported model: {model_key}. "
            f"Only tested models are supported. "
            f"Please use one of the supported models: {supported_models}"
        )

    # Default to BART-large for MAP phase (production quality chunk summarization)
    default_model = DEFAULT_SUMMARY_MODELS.get("bart-large")
    if not default_model:
        raise ValueError("DEFAULT_SUMMARY_MODELS['bart-large'] is not defined")
    return default_model


def select_reduce_model(cfg, default_model_name: str) -> str:
    """Select reduce-phase model based on configuration.

    Defaults to LED-large for accurate, long-context final summarization.
    This hybrid approach is widely used in production:
    - MAP with BART (fast, efficient chunk summaries)
    - REDUCE with LED-large (slower but accurate, handles long combined
      summaries without hallucination)

    If ``cfg.summary_reduce_model`` is not set, defaults to LED-large instead
    of falling back to the map model. This provides the best quality by default.

    Args:
        cfg: Configuration object with summary_reduce_model field
        default_model_name: Map model name (used only if reduce model explicitly set to same)

    Returns:
        Model identifier string (resolved from DEFAULT_SUMMARY_MODELS if key provided)

    Raises:
        ValueError: If an unsupported model is provided (not in DEFAULT_SUMMARY_MODELS)
    """
    reduce_key = getattr(cfg, "summary_reduce_model", None)
    if not reduce_key:
        # Default to LED-large for reduce phase (best quality, long-context)
        default_model = DEFAULT_SUMMARY_MODELS.get("long")
        if not default_model:
            raise ValueError("DEFAULT_SUMMARY_MODELS['long'] is not defined")
        return default_model

    reduce_key = cast(str, reduce_key)
    # Only allow keys from DEFAULT_SUMMARY_MODELS
    # (e.g., "long", "long-fast", "bart-large", "bart-small")
    if reduce_key in DEFAULT_SUMMARY_MODELS:
        return DEFAULT_SUMMARY_MODELS[reduce_key]
    # Reject non-alias models - only tested models are supported
    supported_models = ", ".join(DEFAULT_SUMMARY_MODELS.keys())
    raise ValueError(
        f"Unsupported reduce model: {reduce_key}. "
        f"Only tested models are supported. "
        f"Please use one of the supported models: {supported_models}"
    )


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
        self.revision = revision
        self.device = self._detect_device(device)

        # Security: Validate model source before loading
        _validate_model_source(model_name)
        # Use provided cache_dir or get from cache_utils (consistent with preload script)
        # cache_utils.get_transformers_cache_dir() handles all priority logic:
        # 1. HF_HUB_CACHE env var (CI sets this explicitly)
        # 2. Local project cache (.cache/huggingface/hub/)
        # 3. huggingface_hub.constants.HF_HUB_CACHE
        # 4. Default fallback (~/.cache/huggingface/hub/)
        if cache_dir:
            self.cache_dir = cache_dir
        else:
            try:
                from .cache_utils import get_transformers_cache_dir

                self.cache_dir = str(get_transformers_cache_dir())
            except Exception:
                # If cache_utils not available, use default
                if HF_CACHE_DIR.exists() or not HF_CACHE_DIR_LEGACY.exists():
                    self.cache_dir = str(HF_CACHE_DIR)
                else:
                    self.cache_dir = str(HF_CACHE_DIR_LEGACY)
        # Type hints use TYPE_CHECKING imports above
        # Runtime imports happen lazily in _load_model()
        self.tokenizer: Optional["AutoTokenizer"] = None
        self.model: Optional["AutoModelForSeq2SeqLM"] = None
        self.pipeline: Optional["Pipeline"] = None
        self._batch_size: Optional[int] = None  # For parallel chunk processing (CPU only)
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

    def _load_model(self) -> None:
        """Load model and tokenizer from cache or download."""
        # Lazy import: Only import transformers when this method is called
        # This allows the module to be imported without ML dependencies installed
        import os

        import torch  # noqa: F401
        from transformers import (  # noqa: F401
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            pipeline,
        )

        try:
            # Disable progress bars to avoid misleading "Downloading" messages
            # when loading from cache. This is especially important in test environments
            # where network is blocked and progress bars can be confusing.
            # Set environment variable to suppress Hugging Face Hub progress bars
            original_hf_disable = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

            try:
                # Log device detection details for debugging
                device_info = f"{self.device}"
                if self.device == "mps":
                    device_info += " (Apple Silicon GPU)"
                elif self.device == "cuda":
                    device_info += " (NVIDIA GPU)"
                else:
                    device_info += " (CPU)"
                logger.info(f"Loading summarization model: {self.model_name} on {device_info}")
                logger.debug(f"Cache directory: {self.cache_dir}")

                # Load tokenizer
                # Security: Revision pinning provides reproducibility and prevents
                # supply chain attacks. If revision is None, latest version is used
                # (less secure but more convenient). Use local_files_only=True to
                # prevent network access when models are cached
                # This is critical for tests where network is blocked (pytest-socket)
                logger.debug("Loading tokenizer (will use cache if available)...")
                tokenizer_kwargs = {
                    "cache_dir": self.cache_dir,
                    "local_files_only": True,  # Prevent network access - use cache only
                }
                if self.revision:
                    tokenizer_kwargs["revision"] = self.revision
                    logger.debug(f"Using pinned revision: {self.revision}")
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
                        self.model_name,
                        **tokenizer_kwargs,
                    )
                except (OSError, FileNotFoundError) as e:
                    # Model not cached - provide helpful error message
                    # Note: Only supported models reach this point (unsupported models
                    # are rejected in select_summary_model/select_reduce_model)
                    error_msg = (
                        f"Model {self.model_name} not found in cache. "
                        f"Pre-cache it using: "
                        f"python -m transformers-cli download {self.model_name}"
                    )
                    raise FileNotFoundError(error_msg) from e

                # Load model
                # Security: Revision pinning provides reproducibility and prevents
                # supply chain attacks. If revision is None, latest version is used
                # (less secure but more convenient). Use local_files_only=True to
                # prevent network access when models are cached
                # This is critical for tests where network is blocked (pytest-socket)
                logger.debug("Loading model (will use cache if available)...")
                model_kwargs = {
                    "cache_dir": self.cache_dir,
                    # Prevent network access - use cache only
                    "local_files_only": True,
                }
                if self.revision:
                    model_kwargs["revision"] = self.revision
                try:
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(  # nosec B615
                        self.model_name,
                        **model_kwargs,
                    )
                except (OSError, FileNotFoundError) as e:
                    # Model not cached - provide helpful error message
                    error_msg = (
                        f"Model {self.model_name} not found in cache. "
                        f"Pre-cache it using: "
                        f"python -m transformers-cli download {self.model_name}"
                    )
                    raise FileNotFoundError(error_msg) from e
                logger.debug("Model loaded successfully (cached for future runs)")
            finally:
                # Restore original environment variable
                if original_hf_disable is None:
                    os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
                else:
                    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = original_hf_disable

            # Move model to device
            self.model = self.model.to(self.device)  # type: ignore[union-attr]

            # Create pipeline for easy inference
            # Map device to pipeline device parameter:
            # - "cuda" -> 0 (first CUDA device)
            # - "mps" -> "mps" (Apple Silicon)
            # - "cpu" -> -1 (CPU)
            pipeline_device = 0 if self.device == "cuda" else "mps" if self.device == "mps" else -1
            self.pipeline = pipeline(  # type: ignore[call-overload]
                "summarization",
                model=self.model,
                tokenizer=self.tokenizer,
                device=pipeline_device,
            )

            # Remove default max_new_tokens from pipeline/model generation config
            # to prevent warnings
            # The pipeline/model may default to max_new_tokens=256, but we use max_length instead
            if hasattr(self.pipeline, "model") and self.pipeline.model is not None:
                if (
                    hasattr(self.pipeline.model, "generation_config")
                    and self.pipeline.model.generation_config is not None
                ):
                    # Set to None to disable (we'll use max_length instead)
                    setattr(self.pipeline.model.generation_config, "max_new_tokens", None)
                # Also check model.config directly
                if (
                    hasattr(self.pipeline.model, "config")
                    and self.pipeline.model.config is not None
                ):
                    if hasattr(self.pipeline.model.config, "max_new_tokens"):
                        setattr(self.pipeline.model.config, "max_new_tokens", None)

            logger.debug(f"Successfully loaded model: {self.model_name}")

        except Exception as e:
            logger.error(f"Failed to load summarization model: {e}")
            raise

    def summarize(
        self,
        text: str,
        max_length: int = 150,
        min_length: int = 30,
        do_sample: bool = False,
        prompt: Optional[str] = None,
    ) -> str:
        """Generate summary of input text.

        Args:
            text: Input text to summarize
            max_length: Maximum length of summary
            min_length: Minimum length of summary
            do_sample: Whether to use sampling (False = deterministic)
            prompt: Optional instruction/prompt to prepend to guide summarization

        Returns:
            Generated summary text
        """
        if not self.pipeline:
            raise RuntimeError("Model not loaded")

        # Handle empty or very short text
        if not text or len(text.strip()) < map_reduce.MIN_TEXT_LENGTH:
            return text.strip()

        # Note: Summarization pipelines don't support prompts in the input text
        # Prepending prompts causes them to leak into the summary output
        # The prompt is logged for debugging but not used in the input
        # Summarization models are trained to summarize without explicit prompts
        # OLDinput_text = text
        if prompt:
            # Inject prompt directly into the summarization text
            # input_text = prompt + "\n\n" + text
            input_text = f"Instruction: {prompt}\n\nTranscript:\n{text}"
        else:
            input_text = text

        # Prompt logging removed to reduce log noise (called once per chunk)
        # Custom prompts are logged at the start of summarize_long_text if needed

        # Check input length to prevent buffer size errors
        # MPS has buffer size limits, so we need to ensure text isn't too long
        # Rough estimate: 1 token ≈ 4 characters, so 1024 tokens ≈ 4096 chars
        # Add safety margin: warn if text is very long
        if len(input_text) > 100000:  # ~25k tokens, well above safe chunk size
            logger.warning(
                f"Text is very long ({len(input_text)} chars), consider using chunking. "
                "This may cause buffer size errors on MPS."
            )

        try:
            # Use max_length for summarization (correct parameter for this task)
            # Explicitly set max_new_tokens=None to prevent pipeline from using its default (256)
            # This eliminates the warning about both parameters being set
            # Add repetition_penalty and no_repeat_ngram_size to prevent hallucinations/repetition
            # Use beam search (num_beams) for better quality summaries instead of greedy decoding
            # LED models benefit from beam search for more coherent summaries
            pipeline_kwargs = {
                "max_length": max_length,
                "max_new_tokens": None,  # Explicitly disable to prevent warning
                "min_length": min_length,
                "truncation": True,
                # Penalize repetition to prevent hallucinations
                "repetition_penalty": REPETITION_PENALTY,
                "no_repeat_ngram_size": NO_REPEAT_NGRAM_SIZE,  # Prevent n-gram repetition
            }

            # Use beam search for better quality (LED models work better with beam search)
            # Only use do_sample if explicitly requested, otherwise use beam search
            if do_sample:
                pipeline_kwargs["do_sample"] = True
            else:
                # Beam search produces better summaries than greedy decoding
                pipeline_kwargs["num_beams"] = NUM_BEAMS  # Balance of quality and speed
                pipeline_kwargs["length_penalty"] = (
                    LENGTH_PENALTY  # Encourage longer, more detailed summaries
                )
                pipeline_kwargs["early_stopping"] = True  # Stop when all beams agree

            result = self.pipeline(input_text, **pipeline_kwargs)

            # Pipeline returns list of dicts with 'summary_text' key
            if isinstance(result, list) and len(result) > 0:
                summary_text = result[0].get("summary_text", "")
                summary_text = cast(str, summary_text).strip()
            elif isinstance(result, dict):
                summary_text = result.get("summary_text", "")
                summary_text = cast(str, summary_text).strip()
            else:
                return ""

            # Validate summary quality - detect instruction leaks and
            # repetitive/hallucinated content
            if summary_text:
                summary_text = map_reduce.strip_instruction_leak(summary_text)
                validated_summary = map_reduce.validate_and_fix_repetitive_summary(summary_text)
                return cast(str, validated_summary)

            return summary_text

        except RuntimeError as e:
            error_msg = str(e).lower()
            # Handle MPS buffer size errors and CUDA OOM errors
            if "invalid buffer size" in error_msg or "out of memory" in error_msg:
                logger.error(
                    f"Buffer size error during summarization ({self.device}): {e}. "
                    "Text is too long - use chunking with summarize_long_text() instead."
                )
                return ""
            raise
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return ""


# Backward compatibility wrapper for chunking functions
def chunk_text_for_summarization(
    text: str,
    tokenizer: "AutoTokenizer",
    chunk_size: int,
    overlap: int = chunking.DEFAULT_TOKEN_OVERLAP,
) -> List[str]:
    """Split long text into overlapping chunks.

    .. deprecated:: 2.5.0
        Use :func:`podcast_scraper.summarization.chunking.chunk_text_for_summarization` instead.

    This is a wrapper function that delegates to the chunking module.
    It is kept for backward compatibility but will be removed in a future release.

    Args:
        text: Input text
        tokenizer: Tokenizer instance for accurate token counting
        chunk_size: Target chunk size in tokens
        overlap: Overlap between chunks in tokens

    Returns:
        List of text chunks
    """
    return chunking.chunk_text_for_summarization(text, tokenizer, chunk_size, overlap)


# Backward compatibility wrapper
def _chunk_by_tokens(text: str, tokenizer: "AutoTokenizer", max_tokens: int = 600) -> List[str]:
    """Simple token-based chunking without overlap (for mini map-reduce).

    .. deprecated:: 2.5.0
        Use :func:`podcast_scraper.summarization.chunking.chunk_by_tokens` instead.

    This is a wrapper function that delegates to the chunking module.
    It is kept for backward compatibility but will be removed in a future release.

    Args:
        text: Input text to chunk
        tokenizer: Tokenizer instance for encoding/decoding
        max_tokens: Maximum tokens per chunk (default: 600, safe for BART's 1024 limit)

    Returns:
        List of text chunks, each guaranteed to be <= max_tokens
    """
    return chunking.chunk_by_tokens(text, tokenizer, max_tokens)


# Backward compatibility wrapper
def chunk_text_words(
    text: str,
    chunk_size: int = chunking.DEFAULT_WORD_CHUNK_SIZE,
    overlap: int = chunking.DEFAULT_WORD_OVERLAP,
) -> List[str]:
    """Split long text into overlapping chunks using word-based approximation.

    .. deprecated:: 2.5.0
        Use :func:`podcast_scraper.summarization.chunking.chunk_text_words` instead.

    This is a wrapper function that delegates to the chunking module.
    It is kept for backward compatibility but will be removed in a future release.

    Args:
        text: Input text
        chunk_size: Target chunk size in words
        overlap: Overlap between chunks in words

    Returns:
        List of text chunks
    """
    return chunking.chunk_text_words(text, chunk_size, overlap)


# Backward compatibility wrapper
# Backward compatibility wrapper
def _validate_and_fix_repetitive_summary(summary: str) -> str:
    """Detect and fix repetitive/hallucinated summaries.

    .. deprecated:: 2.5.0
        Use :func:`podcast_scraper.summarization.map_reduce.\
validate_and_fix_repetitive_summary` instead.

    This is a wrapper function that delegates to the map_reduce module.
    It is kept for backward compatibility but will be removed in a future release.

    Args:
        summary: Generated summary text

    Returns:
        Fixed summary (or original if no issues detected)
    """
    return map_reduce.validate_and_fix_repetitive_summary(summary)


# Backward compatibility wrapper
def _strip_instruction_leak(summary: str) -> str:
    """Remove sentences that look like leaked instructions from prompts.

    .. deprecated:: 2.5.0
        Use :func:`podcast_scraper.summarization.map_reduce.strip_instruction_leak` instead.

    This is a wrapper function that delegates to the map_reduce module.
    It is kept for backward compatibility but will be removed in a future release.

    Args:
        summary: Summary text potentially containing instruction leaks

    Returns:
        Cleaned summary text
    """
    return map_reduce.strip_instruction_leak(summary)


# Backward compatibility wrapper
def _check_if_needs_chunking(
    model: SummaryModel,
    text: str,
    chunk_size: int,
    max_length: int,
    min_length: int,
    prompt: Optional[str],
) -> Optional[str]:
    """Check if text can be summarized without chunking.

    .. deprecated:: 2.5.0
        Use :func:`podcast_scraper.summarization.chunking.check_if_needs_chunking` instead.

    This is a wrapper function that delegates to the chunking module.
    It is kept for backward compatibility but will be removed in a future release.

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
    return chunking.check_if_needs_chunking(model, text, chunk_size, max_length, min_length, prompt)


# Backward compatibility wrapper
def _prepare_chunks(
    model: SummaryModel,
    text: str,
    chunk_size: int,
    use_word_chunking: bool,
    word_chunk_size: int,
    word_overlap: int,
) -> Tuple[List[str], int]:
    """Prepare text chunks for summarization using token-based chunking.

    .. deprecated:: 2.5.0
        Use :func:`podcast_scraper.summarization.chunking.prepare_chunks` instead.

    This is a wrapper function that delegates to the chunking module.
    It is kept for backward compatibility but will be removed in a future release.

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
    return chunking.prepare_chunks(
        model, text, chunk_size, use_word_chunking, word_chunk_size, word_overlap
    )


def summarize_long_text(
    model: SummaryModel,
    text: str,
    chunk_size: int = map_reduce.BART_MAX_POSITION_EMBEDDINGS,
    max_length: int = 150,
    min_length: int = 30,
    batch_size: Optional[int] = None,
    prompt: Optional[str] = None,
    use_word_chunking: bool = False,
    word_chunk_size: int = chunking.DEFAULT_WORD_CHUNK_SIZE,
    word_overlap: int = chunking.DEFAULT_WORD_OVERLAP,
    reduce_model: Optional[SummaryModel] = None,
) -> str:
    """Summarize long text by chunking and combining summaries.

    .. deprecated:: 2.5.0
        Use :func:`podcast_scraper.summarization.map_reduce.summarize_long_text` instead.

    This is a wrapper function that delegates to the map_reduce module.
    It is kept for backward compatibility but will be removed in a future release.

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
        word_overlap: Overlap in words when use_word_chunking=True
        reduce_model: Optional separate model for reduce phase

    Returns:
        Combined summary
    """
    return map_reduce.summarize_long_text(
        model,
        text,
        chunk_size=chunk_size,
        max_length=max_length,
        min_length=min_length,
        batch_size=batch_size,
        prompt=prompt,
        use_word_chunking=use_word_chunking,
        word_chunk_size=word_chunk_size,
        word_overlap=word_overlap,
        reduce_model=reduce_model,
    )


# Backward compatibility wrapper
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
) -> List[str]:
    """Map step: Summarize each chunk (parallel or sequential).

    .. deprecated:: 2.5.0
        Use :func:`podcast_scraper.summarization.map_reduce.summarize_chunks_map` instead.

    This is a wrapper function that delegates to the map_reduce module.
    It is kept for backward compatibility but will be removed in a future release.

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
    return map_reduce.summarize_chunks_map(
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


# Backward compatibility wrapper
def _summarize_chunks_parallel(
    model: SummaryModel,
    chunks: List[str],
    chunk_max_length: int,
    chunk_min_length: int,
    prompt: Optional[str],
    max_workers: int,
    start_time: float,
) -> List[str]:
    """Summarize chunks in parallel (CPU only).

    .. deprecated:: 2.5.0
        Use :func:`podcast_scraper.summarization.map_reduce.summarize_chunks_parallel` instead.

    This is a wrapper function that delegates to the map_reduce module.
    It is kept for backward compatibility but will be removed in a future release.

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
    return map_reduce.summarize_chunks_parallel(
        model, chunks, chunk_max_length, chunk_min_length, prompt, max_workers, start_time
    )


# Backward compatibility wrapper
def _summarize_chunks_sequential(
    model: SummaryModel,
    chunks: List[str],
    chunk_max_length: int,
    chunk_min_length: int,
    prompt: Optional[str],
    chunk_size: int,
    start_time: float,
) -> List[str]:
    """Summarize chunks sequentially (GPU or single worker).

    .. deprecated:: 2.5.0
        Use :func:`podcast_scraper.summarization.map_reduce.summarize_chunks_sequential` instead.

    This is a wrapper function that delegates to the map_reduce module.
    It is kept for backward compatibility but will be removed in a future release.

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
    return map_reduce.summarize_chunks_sequential(
        model, chunks, chunk_max_length, chunk_min_length, prompt, chunk_size, start_time
    )


# Backward compatibility wrapper
def _combine_summaries_reduce(
    model: SummaryModel,
    chunk_summaries: List[str],
    max_length: int,
    min_length: int,
    prompt: Optional[str],
) -> str:
    """Reduce step: Combine chunk summaries into final summary.

    .. deprecated:: 2.5.0
        Use :func:`podcast_scraper.summarization.map_reduce.combine_summaries_reduce` instead.

    This is a wrapper function that delegates to the map_reduce module.
    It is kept for backward compatibility but will be removed in a future release.

    Args:
        model: Summary model instance
        chunk_summaries: List of chunk summaries to combine
        max_length: Max summary length
        min_length: Min summary length
        prompt: Optional prompt

    Returns:
        Final combined summary
    """
    return map_reduce.combine_summaries_reduce(
        model, chunk_summaries, max_length, min_length, prompt
    )


# Backward compatibility wrapper
# Backward compatibility wrapper
def _select_key_summaries(chunk_summaries: List[str]) -> List[str]:
    """Select representative chunk summaries for extractive approach.

    .. deprecated:: 2.5.0
        Use :func:`podcast_scraper.summarization.map_reduce.select_key_summaries` instead.

    This is a wrapper function that delegates to the map_reduce module.
    It is kept for backward compatibility but will be removed in a future release.

    Args:
        chunk_summaries: List of all chunk summaries

    Returns:
        Selected subset of summaries (representative chunks)
    """
    return map_reduce.select_key_summaries(chunk_summaries)


# Backward compatibility wrapper
def _join_summaries_with_structure(summaries: List[str]) -> str:
    """Join summaries with structural separation to preserve semantics.

    .. deprecated:: 2.5.0
        Use :func:`podcast_scraper.summarization.map_reduce.join_summaries_with_structure` instead.

    This is a wrapper function that delegates to the map_reduce module.
    It is kept for backward compatibility but will be removed in a future release.

    Args:
        summaries: List of summaries to join

    Returns:
        Joined summaries with structural separation
    """
    return map_reduce.join_summaries_with_structure(summaries)


# Backward compatibility wrapper
def _combine_summaries_extractive(
    model: SummaryModel,
    selected_summaries: List[str],
    max_length: int,
    min_length: int,
    prompt: Optional[str],
    model_max: int,
) -> str:
    """Combine summaries using extractive approach (select representative chunks).

    .. deprecated:: 2.5.0
        Use :func:`podcast_scraper.summarization.map_reduce.combine_summaries_extractive` instead.

    This is a wrapper function that delegates to the map_reduce module.
    It is kept for backward compatibility but will be removed in a future release.

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
    return map_reduce.combine_summaries_extractive(
        model, selected_summaries, max_length, min_length, prompt, model_max
    )


# Backward compatibility wrapper
def _combine_summaries_mini_map_reduce(
    model: SummaryModel,
    combined_text: str,
    chunk_summaries: List[str],
    max_length: int,
    min_length: int,
    prompt: Optional[str],
    combined_tokens: int,
    target_tokens: int,
    max_passes: int = map_reduce.MAX_HIERARCHICAL_PASSES,
) -> str:
    """Combine summaries using iterative mini map-reduce approach (fully abstractive).

    .. deprecated:: 2.5.0
        Use :func:`podcast_scraper.summarization.map_reduce.\
combine_summaries_mini_map_reduce` instead.

    This is a wrapper function that delegates to the map_reduce module.
    It is kept for backward compatibility but will be removed in a future release.

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
    return map_reduce.combine_summaries_mini_map_reduce(
        model,
        combined_text,
        chunk_summaries,
        max_length,
        min_length,
        prompt,
        combined_tokens,
        target_tokens,
        max_passes,
    )


# Backward compatibility wrapper
def _combine_summaries_abstractive(
    model: SummaryModel,
    combined_text: str,
    chunk_summaries: List[str],
    max_length: int,
    min_length: int,
    prompt: Optional[str],
    model_max: int,
    combined_tokens: int,
) -> str:
    """Combine summaries using abstractive approach (final summarization pass).

    .. deprecated:: 2.5.0
        Use :func:`podcast_scraper.summarization.map_reduce.combine_summaries_abstractive` instead.

    This is a wrapper function that delegates to the map_reduce module.
    It is kept for backward compatibility but will be removed in a future release.

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
    return map_reduce.combine_summaries_abstractive(
        model,
        combined_text,
        chunk_summaries,
        max_length,
        min_length,
        prompt,
        model_max,
        combined_tokens,
    )


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

    # Lazy import torch for cache clearing (only if available)
    # Unit tests run without ML dependencies, so torch may not be installed
    import gc  # noqa: F401

    try:
        import torch  # noqa: F401

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
            from .cache_utils import get_transformers_cache_dir

            cache_path = get_transformers_cache_dir()
        except Exception:
            cache_path = HF_CACHE_DIR if HF_CACHE_DIR.exists() else HF_CACHE_DIR_LEGACY

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
            from .cache_utils import get_transformers_cache_dir

            cache_path = get_transformers_cache_dir()
        except Exception:
            cache_path = HF_CACHE_DIR if HF_CACHE_DIR.exists() else HF_CACHE_DIR_LEGACY

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
