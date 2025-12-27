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
have been moved to preprocessing.py in Stage 1. Wrapper functions remain here
for backward compatibility but will be deprecated in a future release.
"""

import logging
import re
import warnings
from pathlib import Path
from typing import cast, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    # Type hints only - these are not imported at runtime
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Pipeline

# IMPORTANT: Set warning filters BEFORE importing transformers
# Suppress transformers max_new_tokens/max_length warning globally
# This warning appears for every chunk but is harmless - we're using max_length correctly
# Suppress it at module level to avoid spam (one warning per chunk)
# Use broader filters to catch all variations of this warning
warnings.filterwarnings(
    "ignore",
    message=".*max_new_tokens.*max_length.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*max_length.*max_new_tokens.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*Both.*max_new_tokens.*max_length.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*Both.*max_length.*max_new_tokens.*",
    category=UserWarning,
)

# Note: torch and transformers are imported lazily in methods that use them
# (e.g., _detect_device(), _load_model()) to avoid requiring ML dependencies
# at module import time. This allows unit tests to import this module without
# torch/transformers installed.

from . import preprocessing

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

DEFAULT_SUMMARY_MODELS = {
    # BART-large (best quality, ~2GB memory), 1024 token limit
    "bart-large": "facebook/bart-large-cnn",
    # BART-base (smallest, lowest memory ~500MB), 1024 token limit
    "bart-small": "facebook/bart-base",
    # DistilBART (faster, lower memory ~300MB), 1024 token limit
    "fast": "sshleifer/distilbart-cnn-12-6",
    # PEGASUS-large (trained for summarization ~2.5GB), 1024 tokens
    "pegasus": "google/pegasus-large",
    # PEGASUS-xsum (short summaries ~2.5GB), 1024 tokens
    "pegasus-xsum": "google/pegasus-xsum",
    # LED-large (long docs 16k tokens, ~2.5GB), NO chunking
    "long": "allenai/led-large-16384",
    # LED-base (long docs 16k tokens, ~1GB), NO chunking
    "long-fast": "allenai/led-base-16384",
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

    Defaults to BART-large for fast, efficient chunk summarization.
    This is the MAP model used to summarize individual chunks.

    Args:
        cfg: Configuration object with summary_model field

    Returns:
        Model identifier string (resolved from DEFAULT_SUMMARY_MODELS if key provided)
    """
    if cfg.summary_model:
        model_key = cast(str, cfg.summary_model)
        # Check if it's a key in DEFAULT_SUMMARY_MODELS
        # (e.g., "bart-large", "bart-small", "long-fast", "pegasus")
        if model_key in DEFAULT_SUMMARY_MODELS:
            return DEFAULT_SUMMARY_MODELS[model_key]
        # Otherwise, assume it's a direct model identifier (e.g., "facebook/bart-large-cnn")
        return model_key

    # Default to BART-large for MAP phase (fast, efficient chunk summarization)
    return DEFAULT_SUMMARY_MODELS["bart-large"]


def select_reduce_model(cfg, default_model_name: str) -> str:
    """Select reduce-phase model based on configuration.

    Defaults to LED (long-fast) for accurate, long-context final summarization.
    This hybrid approach is widely used in production:
    - MAP with BART (fast, efficient chunk summaries)
    - REDUCE with LED (slower but accurate, handles long combined summaries without hallucination)

    If ``cfg.summary_reduce_model`` is not set, defaults to LED (long-fast) instead
    of falling back to the map model. This provides the best quality by default.

    Args:
        cfg: Configuration object with summary_reduce_model field
        default_model_name: Map model name (used only if reduce model explicitly set to same)

    Returns:
        Model identifier string (resolved from DEFAULT_SUMMARY_MODELS if key provided)
    """
    reduce_key = getattr(cfg, "summary_reduce_model", None)
    if not reduce_key:
        # Default to LED for reduce phase (best quality, long-context)
        return DEFAULT_SUMMARY_MODELS["long-fast"]

    reduce_key = cast(str, reduce_key)
    # Allow using keys from DEFAULT_SUMMARY_MODELS (e.g., "long-fast", "long", "bart-large")
    if reduce_key in DEFAULT_SUMMARY_MODELS:
        return DEFAULT_SUMMARY_MODELS[reduce_key]
    # Otherwise assume it's a direct model identifier
    return reduce_key


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
        self.model_name = model_name
        self.revision = revision
        self.device = self._detect_device(device)

        # Security: Validate model source before loading
        _validate_model_source(model_name)
        # Use provided cache_dir or default to standard Hugging Face cache location
        # Transformers will automatically use this directory for caching
        if cache_dir:
            self.cache_dir = cache_dir
        else:
            # Prefer newer cache location, fallback to legacy if it exists
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
        import torch  # noqa: F401
        from transformers import (  # noqa: F401
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            Pipeline,
            pipeline,
        )

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
            # Security: Revision pinning provides reproducibility and prevents supply chain attacks
            # If revision is None, latest version is used (less secure but more convenient)
            logger.debug("Loading tokenizer (will use cache if available)...")
            tokenizer_kwargs = {
                "cache_dir": self.cache_dir,
            }
            if self.revision:
                tokenizer_kwargs["revision"] = self.revision
                logger.debug(f"Using pinned revision: {self.revision}")
            self.tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
                self.model_name,
                **tokenizer_kwargs,
            )

            # Load model
            # Security: Revision pinning provides reproducibility and prevents supply chain attacks
            # If revision is None, latest version is used (less secure but more convenient)
            logger.debug("Loading model (will use cache if available)...")
            model_kwargs = {
                "cache_dir": self.cache_dir,
            }
            if self.revision:
                model_kwargs["revision"] = self.revision
            self.model = AutoModelForSeq2SeqLM.from_pretrained(  # nosec B615
                self.model_name,
                **model_kwargs,
            )
            logger.debug("Model loaded successfully (cached for future runs)")

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
        if not text or len(text.strip()) < MIN_TEXT_LENGTH:
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
                summary_text = _strip_instruction_leak(summary_text)
                validated_summary = _validate_and_fix_repetitive_summary(summary_text)
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


def _chunk_by_tokens(text: str, tokenizer: "AutoTokenizer", max_tokens: int = 600) -> List[str]:
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
    tokens = model.tokenizer.encode(text, add_special_tokens=False)  # type: ignore[attr-defined]
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

    Returns:
        Combined summary
    """
    import time

    pipeline_start_time = time.time()  # noqa: E501

    # If no separate reduce model is provided, use the same model for map and reduce.
    if reduce_model is None:
        reduce_model = model

    cleaned_text = clean_for_summarization(text)
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
    final_summary = _combine_summaries_reduce(
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
        return _summarize_chunks_parallel(
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
        return _summarize_chunks_sequential(
            model,
            chunks,
            chunk_max_length,
            chunk_min_length,
            prompt,
            chunk_size,
            start_time,
        )


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


def _combine_summaries_reduce(
    model: SummaryModel,
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
    combined_text = _join_summaries_with_structure(chunk_summaries)
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
        selected = _select_key_summaries(chunk_summaries)
        logger.debug(
            "[MAP-REDUCE CONFIG] Extractive fallback: "
            f"summary_range={final_reduce_min_length}-{final_reduce_max_length} tokens"
        )

        return _combine_summaries_extractive(
            model,
            selected,
            final_reduce_max_length,
            final_reduce_min_length,
            reduce_prompt,
            model_max,
        )

    if combined_tokens > single_pass_limit:
        return _combine_summaries_mini_map_reduce(
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
        return _combine_summaries_abstractive(
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
            return _combine_summaries_extractive(
                model,
                _select_key_summaries(chunk_summaries),
                final_reduce_max_length,
                final_reduce_min_length,
                reduce_prompt,
                model_max,
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


def _join_summaries_with_structure(summaries: List[str]) -> str:
    """Join summaries with structural separation to preserve semantics."""
    return "\n\n".join(summaries)


def _combine_summaries_extractive(
    model: SummaryModel,
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
        # Lazy import torch for error handling
        import torch  # noqa: F401

        # Handle both CUDA and MPS out-of-memory errors
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
    """Unload model to free memory.

    Args:
        model: Summary model instance, or None (no-op if None)
    """
    if model is None:
        return

    if model.model:
        del model.model
    if model.tokenizer:
        del model.tokenizer
    if model.pipeline:
        del model.pipeline

    model.model = None
    model.tokenizer = None
    model.pipeline = None

    # Lazy import torch for cache clearing
    import torch  # noqa: F401

    # Clear device-specific cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()


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
