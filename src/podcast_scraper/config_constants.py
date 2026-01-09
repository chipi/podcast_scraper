"""Configuration constants for podcast_scraper.

This module contains all configuration constants that were previously defined
in config.py. Extracted to reduce config.py size and improve maintainability.

All constants are re-exported from config.py for backward compatibility.
"""

import os

# General defaults
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_NUM_SPEAKERS = 2
DEFAULT_SCREENPLAY_GAP_SECONDS = 1.25
DEFAULT_TIMEOUT_SECONDS = 20
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/119.0 Safari/537.36"
)
DEFAULT_WORKERS = max(1, min(8, os.cpu_count() or 4))
DEFAULT_LANGUAGE = "en"

# Speaker detection defaults
DEFAULT_NER_MODEL = "en_core_web_sm"
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
# Note: TEST_DEFAULT_NER_MODEL uses DEFAULT_NER_MODEL ("en_core_web_sm")
# - same for tests and production

# Production defaults (quality models for production use)
# These are used in production deployments and nightly-only tests
PROD_DEFAULT_WHISPER_MODEL = "base.en"  # Better quality than tiny.en, English-only
PROD_DEFAULT_SUMMARY_MODEL = (
    SUMMARY_MODEL_BART_LARGE_CNN  # Large, ~2GB, best quality for production
)
PROD_DEFAULT_SUMMARY_REDUCE_MODEL = (
    SUMMARY_MODEL_LED_LARGE_16384  # Large, ~2.5GB, production quality for long-context
)

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
DEFAULT_SUMMARY_MAX_LENGTH = 160  # Per SUMMARY_REVIEW.md: chunk summaries should be ~160 tokens
DEFAULT_SUMMARY_MIN_LENGTH = (
    60  # Per SUMMARY_REVIEW.md: chunk summaries should be at least 60 tokens
)
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
