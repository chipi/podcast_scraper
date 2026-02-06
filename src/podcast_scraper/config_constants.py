"""Configuration constants for podcast_scraper.

This module contains all configuration constants that were previously defined
in config.py. Extracted to reduce config.py size and improve maintainability.

All constants are re-exported from config.py for convenience.
"""

import os

# General defaults
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_NUM_SPEAKERS = 2
DEFAULT_SCREENPLAY_GAP_SECONDS = 1.25
DEFAULT_TIMEOUT_SECONDS = 20
# Timeout defaults for ML operations (Issue #379)
DEFAULT_TRANSCRIPTION_TIMEOUT_SECONDS = 1800  # 30 minutes
DEFAULT_SUMMARIZATION_TIMEOUT_SECONDS = 600  # 10 minutes
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/119.0 Safari/537.36"
)
DEFAULT_WORKERS = max(1, min(8, os.cpu_count() or 4))
DEFAULT_LANGUAGE = "en"

# Cache directory defaults
DEFAULT_PREPROCESSING_CACHE_DIR = ".cache/preprocessing"

# Validation ranges (for summary word chunking)
# These are used in Config validators and field descriptions
RECOMMENDED_WORD_CHUNK_SIZE_MIN = 800
RECOMMENDED_WORD_CHUNK_SIZE_MAX = 1200
RECOMMENDED_WORD_OVERLAP_MIN = 100
RECOMMENDED_WORD_OVERLAP_MAX = 200

# Default file extensions
DEFAULT_MEDIA_EXTENSION = ".bin"
DEFAULT_TRANSCRIPT_EXTENSION = ".txt"

# Speaker detection defaults
# DEFAULT_NER_MODEL is now set via _get_default_ner_model() in config.py
# to support dev/prod distinction (TEST_DEFAULT_NER_MODEL vs PROD_DEFAULT_NER_MODEL)
DEFAULT_MAX_DETECTED_NAMES = 4
MIN_NUM_SPEAKERS = 1
MIN_TIMEOUT_SECONDS = 1

# Summary model identifiers (full Hugging Face model IDs)
# These are used in alias dictionary and throughout the codebase
SUMMARY_MODEL_BART_LARGE_CNN = "facebook/bart-large-cnn"
SUMMARY_MODEL_BART_BASE = "facebook/bart-base"
SUMMARY_MODEL_DISTILBART_CNN = "sshleifer/distilbart-cnn-12-6"
SUMMARY_MODEL_PEGASUS_LARGE = "google/pegasus-large"
SUMMARY_MODEL_PEGASUS_XSUM = "google/pegasus-xsum"
SUMMARY_MODEL_LED_LARGE_16384 = "allenai/led-large-16384"
SUMMARY_MODEL_LED_BASE_16384 = "allenai/led-base-16384"

# Test defaults (smaller, faster models for CI/local dev)
# These are used in tests for speed, while production uses quality models
# See docs/guides/TESTING_GUIDE.md for details
TEST_DEFAULT_WHISPER_MODEL = "tiny.en"  # Smallest, fastest English-only model
# Test defaults use aliases (not direct model IDs) since summarizer.py only accepts aliases
TEST_DEFAULT_SUMMARY_MODEL = "bart-small"  # Maps to facebook/bart-base (~500MB, fast)
TEST_DEFAULT_SUMMARY_REDUCE_MODEL = "long-fast"  # Maps to allenai/led-base-16384 (fast)
# spaCy NER model defaults (dev/prod distinction)
# Dev: Small, fast model for CI/local dev (~50MB, ~200ms/episode)
# Prod: Transformer-based, higher quality for production (~500MB, ~450ms/episode)
TEST_DEFAULT_NER_MODEL = "en_core_web_sm"  # Dev: Small, fast
PROD_DEFAULT_NER_MODEL = "en_core_web_trf"  # Prod: Transformer-based, higher quality

# Production defaults (quality models for production use)
# These are used in production deployments and nightly-only tests
# Aligned with baseline_ml_prod_authority_v1 (Pegasus-CNN → LED-base)
PROD_DEFAULT_WHISPER_MODEL = "base.en"  # Better quality than tiny.en, English-only
PROD_DEFAULT_SUMMARY_MODEL = (
    "google/pegasus-cnn_dailymail"  # Production baseline: Pegasus-CNN for map phase
)
PROD_DEFAULT_SUMMARY_REDUCE_MODEL = (
    SUMMARY_MODEL_LED_BASE_16384  # Production baseline: LED-base for reduce phase
)

# Model revision pinning (for reproducibility and security)
# Pin to specific commit SHAs instead of "main" to avoid PR refs and ensure stable weights
# To find the latest commit SHA for a model, check the model's HuggingFace page or use:
#   from huggingface_hub import HfApi
#   api = HfApi()
#   model_info = api.model_info("google/pegasus-cnn_dailymail", revision="main")
#   commit_hash = model_info.sha
# Last updated: 2025-01-XX (commit SHA from main branch)
PEGASUS_CNN_DAILYMAIL_REVISION = (
    "40d588fdab0cc077b80d950b300bf66ad3c75b92"  # Pinned commit SHA for reproducibility
)
# LED model revisions (Issue #379)
# Pinned commit SHAs for reproducibility (updated 2026-01-XX)
# To find latest commit SHA:
#   from huggingface_hub import HfApi
#   api = HfApi()
#   model_info = api.model_info("allenai/led-base-16384", revision="main")
#   commit_hash = model_info.sha
LED_BASE_16384_REVISION = (
    "38335783885b338d93791936c54bb4be46bebed9"  # Pinned commit SHA for reproducibility
)
LED_LARGE_16384_REVISION = "main"  # Placeholder - update with actual commit SHA when needed

# OpenAI model defaults (Issue #191)
# Test defaults: cheapest models for dev/testing (minimize API costs)
# Production defaults: best quality/cost balance
#
# For current pricing, see: https://openai.com/pricing
TEST_DEFAULT_OPENAI_TRANSCRIPTION_MODEL = "whisper-1"  # Only OpenAI option
TEST_DEFAULT_OPENAI_SPEAKER_MODEL = "gpt-4o-mini"  # Cheap, fast for dev/testing
TEST_DEFAULT_OPENAI_SUMMARY_MODEL = "gpt-4o-mini"  # Cheap, fast for dev/testing
PROD_DEFAULT_OPENAI_TRANSCRIPTION_MODEL = "whisper-1"  # Only OpenAI option
PROD_DEFAULT_OPENAI_SPEAKER_MODEL = "gpt-4o-mini"  # Cost-effective for production
PROD_DEFAULT_OPENAI_SUMMARY_MODEL = "gpt-4o"  # Higher quality for production

# Gemini model defaults (Issue #194)
# Test defaults: free tier models for dev/testing (gemini-2.0-flash)
# Production defaults: best quality models (gemini-1.5-pro with 2M context)
#
# For current pricing, see: https://ai.google.dev/pricing
TEST_DEFAULT_GEMINI_TRANSCRIPTION_MODEL = "gemini-2.0-flash"  # Free tier, fast
TEST_DEFAULT_GEMINI_SPEAKER_MODEL = "gemini-2.0-flash"  # Free tier, fast
TEST_DEFAULT_GEMINI_SUMMARY_MODEL = "gemini-2.0-flash"  # Free tier, fast
PROD_DEFAULT_GEMINI_TRANSCRIPTION_MODEL = "gemini-1.5-pro"  # Best quality, 2M context
PROD_DEFAULT_GEMINI_SPEAKER_MODEL = "gemini-1.5-pro"  # Best quality
PROD_DEFAULT_GEMINI_SUMMARY_MODEL = "gemini-1.5-pro"  # Best quality, 2M context

# Anthropic model defaults (Issue #106)
# Test defaults: cheaper models for dev/testing
# Production defaults: best quality models (claude-3-5-sonnet with 200K context)
#
# For current pricing, see: https://www.anthropic.com/pricing
# Note: Anthropic does NOT support native audio transcription (no audio API)
# Note: claude-3-5-haiku-20241022 is deprecated (EOL: 2026-02-19), using latest versions
TEST_DEFAULT_ANTHROPIC_TRANSCRIPTION_MODEL = (
    "claude-3-5-sonnet-20241022"  # Placeholder (not used - no audio support)
)
TEST_DEFAULT_ANTHROPIC_SPEAKER_MODEL = "claude-3-5-haiku-latest"  # Latest version, cheaper, fast
TEST_DEFAULT_ANTHROPIC_SUMMARY_MODEL = "claude-3-5-haiku-latest"  # Latest version, cheaper, fast
PROD_DEFAULT_ANTHROPIC_TRANSCRIPTION_MODEL = (
    "claude-3-5-sonnet-20241022"  # Placeholder (not used - no audio support)
)
PROD_DEFAULT_ANTHROPIC_SPEAKER_MODEL = "claude-3-5-sonnet-20241022"  # Best quality
PROD_DEFAULT_ANTHROPIC_SUMMARY_MODEL = "claude-3-5-sonnet-20241022"  # Best quality, 200K context

# Mistral model defaults (Issue #106)
# Test defaults: cheapest models for dev/testing (minimize API costs)
# Production defaults: best quality/cost balance
#
# For current pricing, see: https://docs.mistral.ai/pricing/
TEST_DEFAULT_MISTRAL_TRANSCRIPTION_MODEL = "voxtral-mini-latest"  # Only option
PROD_DEFAULT_MISTRAL_TRANSCRIPTION_MODEL = "voxtral-mini-latest"  # Only option
TEST_DEFAULT_MISTRAL_SPEAKER_MODEL = "mistral-small-latest"  # Cheapest text
PROD_DEFAULT_MISTRAL_SPEAKER_MODEL = "mistral-large-latest"  # Best quality
TEST_DEFAULT_MISTRAL_SUMMARY_MODEL = "mistral-small-latest"  # Cheapest text
PROD_DEFAULT_MISTRAL_SUMMARY_MODEL = "mistral-large-latest"  # Best quality, 256k context


# DeepSeek model defaults (Issue #107)
# Test defaults: cheapest models for dev/testing (deepseek-chat - extremely cheap)
# Production defaults: same model (deepseek-chat - still extremely cheap, 95% cheaper than OpenAI)
#
# For current pricing, see: https://platform.deepseek.com/pricing
# Key advantage: 95% cheaper than OpenAI for text processing
# Note: DeepSeek does NOT support transcription (no audio API)
TEST_DEFAULT_DEEPSEEK_SPEAKER_MODEL = "deepseek-chat"  # Extremely cheap, same for test/prod
TEST_DEFAULT_DEEPSEEK_SUMMARY_MODEL = "deepseek-chat"  # Extremely cheap, same for test/prod
PROD_DEFAULT_DEEPSEEK_SPEAKER_MODEL = "deepseek-chat"  # Same model, still very cheap
PROD_DEFAULT_DEEPSEEK_SUMMARY_MODEL = "deepseek-chat"  # Same model, still very cheap

# Ollama model defaults (Issue #196)
# Test defaults: smaller, faster models for dev/testing (llama3.2:latest)
# Production defaults: best quality models (llama3.3:latest)
#
# Key advantage: Fully offline, zero cost, complete privacy
# Note: Ollama does NOT support transcription (no audio API)
# Models must be pulled before use: ollama pull llama3.3
TEST_DEFAULT_OLLAMA_SPEAKER_MODEL = "llama3.2:latest"  # Smaller, faster for testing
PROD_DEFAULT_OLLAMA_SPEAKER_MODEL = "llama3.3:latest"  # Best quality
TEST_DEFAULT_OLLAMA_SUMMARY_MODEL = "llama3.2:latest"  # Smaller, faster for testing
PROD_DEFAULT_OLLAMA_SUMMARY_MODEL = "llama3.3:latest"  # Best quality, 128k context

# Grok (xAI) model defaults (Issue #1095)
# Test defaults: beta model for dev/testing (grok-beta)
# Production defaults: production model (grok-2)
#
# For current pricing, see: https://console.x.ai or https://docs.x.ai
# Key advantage: Real-time information access via X/Twitter integration
# Note: Grok does NOT support transcription (no audio API)
# Note: Model names should be verified with your xAI API access
TEST_DEFAULT_GROK_SPEAKER_MODEL = "grok-beta"  # Beta model for development
TEST_DEFAULT_GROK_SUMMARY_MODEL = "grok-beta"  # Beta model for development
PROD_DEFAULT_GROK_SPEAKER_MODEL = "grok-2"  # Production model, best quality
PROD_DEFAULT_GROK_SUMMARY_MODEL = "grok-2"  # Production model, best quality

# Validation constants
VALID_WHISPER_MODELS = (
    "tiny",
    "base",
    "small",
    "medium",
    "large",
    "large-v2",
    "large-v3",
    "tiny.en",
    "base.en",
    "small.en",
    "medium.en",
    "large.en",
)

VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
MAX_RUN_ID_LENGTH = 100
MAX_METADATA_SUBDIRECTORY_LENGTH = 255

# Whisper model fallback chains (for graceful degradation when models unavailable)
# These define the order of fallback models from largest to smallest
# English-only models (.en suffix) are preferred for English language
FALLBACK_WHISPER_MODELS_EN = ["tiny.en", "base.en", "small.en", "medium.en", "large.en"]
FALLBACK_WHISPER_MODELS_MULTILINGUAL = ["tiny", "base", "small", "medium", "large"]

# Models that should get .en suffix automatically for English
WHISPER_MODELS_WITH_EN_VARIANT = ("tiny", "base", "small", "medium")

# Summarization defaults
DEFAULT_SUMMARY_BATCH_SIZE = 1
# Maximum parallel workers for episode summarization (memory-bound)
# Lower values reduce memory usage but may slow down processing
# Production: Higher values for better throughput
# Tests/Dev: Lower values to reduce memory footprint
DEFAULT_SUMMARY_MAX_WORKERS_CPU = 4  # Production default for CPU
DEFAULT_SUMMARY_MAX_WORKERS_CPU_TEST = (
    1  # Test/dev default for CPU (reduces memory - sequential processing)
)
DEFAULT_SUMMARY_MAX_WORKERS_GPU = 2  # Production default for GPU
DEFAULT_SUMMARY_MAX_WORKERS_GPU_TEST = 1  # Test/dev default for GPU (reduces memory)
DEFAULT_SUMMARY_CHUNK_SIZE = (
    2048  # Default token chunk size (BART models support up to 1024, but larger chunks work safely)
)
DEFAULT_SUMMARY_WORD_CHUNK_SIZE = (
    900  # Per SUMMARY_REVIEW.md: 800-1200 words recommended for encoder-decoder models
)
DEFAULT_SUMMARY_WORD_OVERLAP = 150  # Per SUMMARY_REVIEW.md: 100-200 words recommended

# Allowed HuggingFace models (Issue #379)
# Security: This allowlist prevents config injection and supply chain attacks.
# Only models in this list can be loaded by the summarizer.
# To add a new model, update this list and ensure it's from a trusted source.
ALLOWED_HUGGINGFACE_MODELS = frozenset(
    [
        # Facebook models
        "facebook/bart-large-cnn",
        "facebook/bart-base",
        # Google models
        "google/pegasus-large",
        "google/pegasus-cnn_dailymail",
        "google/pegasus-xsum",
        # AllenAI models
        "allenai/led-large-16384",
        "allenai/led-base-16384",
        # SSHleifer models
        "sshleifer/distilbart-cnn-12-6",
    ]
)
