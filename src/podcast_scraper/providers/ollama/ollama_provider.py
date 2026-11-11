"""Unified Ollama provider for speaker detection and summarization.

This module provides a single OllamaProvider class that implements two protocols:
- SpeakerDetector (using Ollama chat API via OpenAI SDK)
- SummarizationProvider (using Ollama chat API via OpenAI SDK)

This unified approach matches the pattern of OpenAI providers, where a single
provider type handles multiple capabilities using shared API client.

Key insight: Ollama uses an OpenAI-compatible API, so we reuse the OpenAI SDK
with a custom base_url. No new dependency required.

Key advantages:
- Fully offline - no internet required
- Zero cost - no per-token pricing
- Complete privacy - data never leaves local machine

Note: Ollama does NOT support transcription (no audio API).
Note: Model names are normalized to ensure correct format (e.g., '3.1:7b' -> 'llama3.1:7b').
"""

from __future__ import annotations

import logging
from typing import Any, cast, Dict, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from ...models import Episode
    from ..capabilities import ProviderCapabilities
else:
    from ... import models

    Episode = models.Episode  # type: ignore[assignment]

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

from ... import config, config_constants
from ...cleaning.base import TranscriptCleaningProcessor
from ...utils.cleaning_max_tokens import (
    clamp_cleaning_max_tokens,
    estimate_cleaning_output_tokens,
    OLLAMA_CLEANING_MAX_TOKENS,
)
from ...utils.log_redaction import format_exception_for_log
from ...utils.provider_metadata import warn_if_truncated
from ...utils.timeout_config import get_http_timeout
from ...workflow import metrics
from .. import guardrails as _guardrails
from ..common.summary_use_case import (
    build_use_case_prompts,
    reply_budget,
    SummaryUseCase,
    use_case_from_params,
    use_case_prompt_metadata,
    use_case_stage,
)
from ..common.transcript_cache import openai_style_messages as _openai_style_messages

logger = logging.getLogger(__name__)


# Default speaker names when detection fails
from ._ollama_provider_calls import (  # noqa: F401 - re-exported: moved there with S1
    _flatten_json_speaker_names,
    _ollama_openai_chat_extra_kwargs,
)
from .ollama_provider_gi_naming import (  # noqa: F401 - re-exported: moved there with S1
    _CHARS_PER_TOKEN,
    _ENTAILMENT_FALLBACK,
    _EXTRACT_QUOTE_RESERVE_TOKENS,
    _json_object_from_response,
    _render_entailment_prompt,
    _render_extract_quote_prompt,
    _transcript_budget_chars,
    OllamaProviderGiNamingMethods,
)

# Error messages
OLLAMA_NOT_RUNNING_ERROR = """
Ollama server is not running. Please start it with:

    ollama serve

Or install Ollama from: https://ollama.ai
"""

MODEL_NOT_FOUND_ERROR_TEMPLATE = """
Model '{model}' is not available in Ollama. Install it with:

    ollama pull {model}

Available models can be listed with:

    ollama list
"""


def _ollama_native_api_root(openai_compat_base: str) -> str:
    """Return Ollama server root for ``/api/*`` paths (strip OpenAI ``/v1`` only).

    ``str.rstrip("/v1")`` must not be used: it removes any trailing characters in
    ``{'/', 'v', '1'}``, which corrupts hosts/ports ending in those characters
    (e.g. ``http://127.0.0.1:51201/v1`` becomes ``http://127.0.0.1:5120``).
    """
    u = openai_compat_base.rstrip("/")
    if u.endswith("/v1"):
        return u[:-3].rstrip("/")
    return u


class OllamaProvider(OllamaProviderGiNamingMethods):
    """Unified Ollama provider: SpeakerDetector and SummarizationProvider (no transcription).

    Uses Ollama chat API via OpenAI SDK. All capabilities share the same client.
    """

    cleaning_processor: TranscriptCleaningProcessor  # Type annotation for mypy

    def __init__(self, cfg: config.Config):
        """Initialize unified Ollama provider.

        Args:
            cfg: Configuration object with settings for both capabilities

        Raises:
            ValueError: If Ollama server is not running or model is not available
            ImportError: If openai or httpx packages are not installed
            ConnectionError: If Ollama server is not accessible
        """
        if OpenAI is None:
            raise ImportError(
                "openai package required for Ollama provider. "
                "Install the project (OpenAI SDK is a core dependency), e.g. pip install -e ."
            )

        if httpx is None:
            raise ImportError(
                "httpx package required for Ollama provider (for health checks). "
                "Install with: pip install -e '.[llm]' (httpx is in the llm extra)"
            )

        self.cfg = cfg
        # RFC-115: relocate the transcript to a cacheable leading system block (default on).
        self._cache_transcript_prefix = bool(getattr(cfg, "cache_transcript_prefix", True))

        # Set up transcript cleaning processor based on strategy (Issue #418)
        from ... import stage_hooks

        cleaning_strategy = getattr(cfg, "transcript_cleaning_strategy", "hybrid")
        self.cleaning_processor = stage_hooks.transcript_cleaner(cleaning_strategy)

        # Cleaning model settings (smaller model for cost efficiency)
        cleaning_model_raw = getattr(cfg, "ollama_cleaning_model", "llama3.1:8b")
        self.cleaning_model = self._normalize_model_name(cleaning_model_raw)
        logger.info(
            "Ollama cleaning model configured: '%s' -> '%s'",
            cleaning_model_raw,
            self.cleaning_model,
        )
        self.cleaning_temperature = getattr(cfg, "ollama_cleaning_temperature", 0.2)

        # Validate Ollama server is running
        base_url = cfg.ollama_api_base or "http://localhost:11434/v1"
        self._validate_ollama_running(base_url)

        # Suppress verbose OpenAI SDK debug logs (same as OpenAI provider)
        root_logger = logging.getLogger()
        root_level = root_logger.level if root_logger.level else logging.INFO
        if root_level <= logging.DEBUG:
            openai_loggers = [
                "openai",
                "openai._base_client",
                "openai.api_resources",
                "httpx",
                "httpcore",
            ]
            for logger_name in openai_loggers:
                openai_logger = logging.getLogger(logger_name)
                openai_logger.setLevel(logging.WARNING)

        # Create OpenAI client with Ollama base_url
        # Ollama doesn't require API key, but OpenAI SDK requires one (use dummy)
        client_kwargs: dict[str, Any] = {
            "api_key": "ollama",  # Ollama ignores API key, but SDK requires one
            "base_url": base_url,
        }

        # Configure HTTP timeouts with separate connect/read timeouts
        # Use ollama_timeout for read timeout (local Ollama can be slower)
        ollama_timeout = getattr(cfg, "ollama_timeout", 120)
        client_kwargs["timeout"] = get_http_timeout(cfg, read_timeout=float(ollama_timeout))

        self.client = OpenAI(**client_kwargs)

        # Speaker detection settings
        speaker_model_raw = getattr(cfg, "ollama_speaker_model", "llama3.1:8b")
        self.speaker_model = self._normalize_model_name(speaker_model_raw)
        logger.info(
            "Ollama speaker model configured: '%s' -> '%s'",
            speaker_model_raw,
            self.speaker_model,
        )
        self.speaker_temperature = getattr(cfg, "ollama_temperature", 0.3)

        # Summarization settings
        summary_model_raw = getattr(cfg, "ollama_summary_model", "llama3.1:8b")
        self.summary_model = self._normalize_model_name(summary_model_raw)
        logger.info(
            "Ollama summary model configured: '%s' -> '%s'",
            summary_model_raw,
            self.summary_model,
        )
        self.summary_temperature = getattr(cfg, "ollama_temperature", 0.3)
        # Ollama 0.19.0 tiers num_ctx by VRAM (48GB → 32k default). But silent
        # truncation still happens when prompt+output exceed the set limit, so we
        # set explicitly. 32768 is the research-recommended safe default on 48GB.
        # gemma2 is structurally capped at 8192 regardless; Ollama silently clamps.
        self.summary_num_ctx: int = int(getattr(cfg, "ollama_num_ctx", 32768) or 32768)
        # The window this deployment SERVES is `num_ctx` — the value we send on every request —
        # not what the model could support in principle (#2050).
        #
        # This was 128000 ("modern Ollama models support 128k"), which is a statement about the
        # model, not about the server we are talking to. Once transcript budgets started deriving
        # from `max_context_tokens`, that 4x overstatement became dangerous HERE specifically,
        # because Ollama does not reject an over-long prompt — it TRUNCATES SILENTLY and returns
        # 200 (see the num_ctx note in this module's docstring). And ollama is prod's summary
        # failover tier (`summary_fallback_providers: [ollama]`), so the path is:
        #
        #   vLLM 400s on a long episode -> fail over to ollama -> ollama silently drops most of
        #   the transcript -> a "successful" reply is recorded as a valid extraction.
        #
        # Silently degraded corpus data is worse than a missing episode: a gap is measurable and
        # backfillable, bad data is neither. At num_ctx=32768 the bundled-quote budget is 83,865
        # chars; the 128000 figure made it 383,846.
        self.max_context_tokens = (
            int(getattr(cfg, "ollama_max_context_tokens", 0) or 0) or self.summary_num_ctx
        )

        # Initialization state
        self._speaker_detection_initialized = False
        self._summarization_initialized = False

        # Mark provider as thread-safe (API clients can be shared across threads)
        self._requires_separate_instances = False

    @staticmethod
    def get_pricing(model: str, capability: str) -> Dict[str, float]:
        """Read pricing from ``config/pricing_assumptions.yaml`` (#651).

        Ollama runs locally at zero API cost — YAML rows are all 0.0 but the
        aggregate still flows through the same code path as billable providers
        for consistency.
        """
        from podcast_scraper.pricing_assumptions import (
            get_loaded_table,
            lookup_external_pricing,
        )

        table, _ = get_loaded_table("config/pricing_assumptions.yaml")
        if not table:
            return {"input_cost_per_1m_tokens": 0.0, "output_cost_per_1m_tokens": 0.0}
        ext = lookup_external_pricing(table, "ollama", capability, model)
        return (
            dict(ext)
            if ext
            else {"input_cost_per_1m_tokens": 0.0, "output_cost_per_1m_tokens": 0.0}
        )

    def _normalize_model_name(self, model: str) -> str:
        """Normalize Ollama model name to ensure correct format.

        Handles cases where users specify shortened names like "3.1:7b"
        instead of the full name "llama3.1:7b". Ollama requires exact model names.

        Args:
            model: Model name from config (may be shortened)

        Returns:
            Normalized model name (e.g., "3.1:7b" -> "llama3.1:7b")
        """
        if not model:
            return model

        # If model name starts with a digit, it's likely a shortened format
        # Common patterns: "3.1:7b", "3.2:latest", "3.3:8b"
        if model[0].isdigit():
            normalized = f"llama{model}"
            logger.warning(
                "Normalizing Ollama model name: '%s' -> '%s'. "
                "Ollama requires exact model names. If this is incorrect, "
                "specify the full name in your config (e.g., 'llama3.1:7b').",
                model,
                normalized,
            )
            return normalized

        # Warn about :latest tags - they can resolve to different model sizes
        # and may load larger models than expected (e.g., 70B instead of 7B)
        model_lower = model.lower()
        if ":latest" in model_lower:
            logger.warning(
                "⚠️  WARNING: Model name uses ':latest' tag: '%s'. "
                "The ':latest' tag can resolve to different model sizes "
                "(e.g., 70B instead of 7B). "
                "For predictable behavior, use a specific model tag like "
                "'llama3.1:7b' or 'llama3.1:8b'. "
                "Check 'ollama list' to see what size model ':latest' points to.",
                model,
            )

        # Also handle cases like "llama3.1:70b" when user wants "llama3.1:7b"
        # This is a common typo - check if it looks like a 70b when they might mean 7b
        if ":70b" in model_lower and ":7b" not in model_lower:
            logger.error(
                "⚠️  WARNING: Model name contains ':70b' - did you mean ':7b'? "
                "Ollama will load the 70B model which is much larger and slower. "
                "Current model name: '%s'. If you want the 7B model, change it to '%s'",
                model,
                model.replace("70b", "7b").replace("70B", "7b"),
            )

        return model

    def _model_name_to_prompt_dir(self, model: str) -> str:
        """Convert Ollama model name to prompt directory name.

        Converts model names like "llama3.1:8b" to directory names like "llama3.1_8b"
        for use in prompt paths (e.g., "ollama/llama3.1_8b/ner/system_ner_v1").

        Args:
            model: Normalized model name (e.g., "llama3.1:8b", "mistral:7b", "mistral-nemo:12b",
                "mistral-small3.2", "qwen2.5:7b", "qwen2.5:32b", "qwen3.5:9b", "qwen3.5:35b-a3b")

        Returns:
            Directory name for prompts (e.g., "llama3.1_8b", "mistral_7b", "mistral-nemo_12b",
                "mistral-small3.2", "qwen2.5_7b", "qwen2.5_32b", "qwen3.5_9b", "qwen3.5_35b-a3b")
        """
        if not model:
            return ""
        # Replace colons with underscores for directory names (keep dots)
        # "llama3.1:8b" -> "llama3.1_8b"
        # "qwen2.5:7b" -> "qwen2.5_7b", "qwen2.5:32b" -> "qwen2.5_32b"
        # "qwen3.5:9b" -> "qwen3.5_9b", "qwen3.5:35b-a3b" -> "qwen3.5_35b-a3b"
        # "mistral:7b" -> "mistral_7b"
        # "mistral-nemo:12b" -> "mistral-nemo_12b"
        # "mistral-small3.2" (no colon) -> "mistral-small3.2"
        # "mistral-small3.2:latest" -> "mistral-small3.2_latest"
        # "phi3:mini" -> "phi3_mini"
        # "gemma2:9b" -> "gemma2_9b"
        return model.replace(":", "_")

    def _get_model_specific_prompt_path(
        self, model: str, task: str, prompt_file: str, fallback: str
    ) -> str:
        """Get model-specific prompt path with fallback to generic prompt.

        Tries to load model-specific prompt first (e.g., "ollama/llama3.1_8b/ner/system_ner_v1"),
        falls back to generic prompt (e.g., "ollama/ner/system_ner_v1") if model-specific
        prompt does not exist.

        Args:
            model: Normalized model name (e.g., "llama3.1:8b")
            task: Task type (e.g., "ner", "summarization")
            prompt_file: Prompt filename without extension (e.g., "system_ner_v1")
            fallback: Fallback prompt path (e.g., "ollama/ner/system_ner_v1")

        Returns:
            Prompt path to use (model-specific if available, otherwise fallback)
        """
        if not model:
            return fallback

        # Convert model name to directory name
        model_dir = self._model_name_to_prompt_dir(model)
        if not model_dir:
            return fallback

        # Try model-specific prompt path
        model_specific_path = f"ollama/{model_dir}/{task}/{prompt_file}"

        try:
            from ...prompts.store import get_prompt_dir, PromptNotFoundError

            prompt_dir = get_prompt_dir()
            prompt_file_path = prompt_dir / f"{model_specific_path}.j2"
            if prompt_file_path.exists():
                logger.debug(
                    "Using model-specific prompt: %s (model: %s)", model_specific_path, model
                )
                return model_specific_path
        except (PromptNotFoundError, Exception):
            # If check fails or file doesn't exist, fall back to generic prompt
            pass

        # Fallback to generic prompt
        logger.debug(
            "Model-specific prompt not found: %s, using fallback: %s (model: %s)",
            model_specific_path,
            fallback,
            model,
        )
        return fallback

    def _validate_ollama_running(self, base_url: str) -> None:
        """Validate that Ollama server is running and accessible.

        Args:
            base_url: Ollama API base URL

        Raises:
            ConnectionError: If Ollama server is not running
        """
        try:
            health_url = _ollama_native_api_root(base_url) + "/api/version"
            response = httpx.get(health_url, timeout=5.0)
            response.raise_for_status()
        except Exception as exc:
            # Catch all httpx exceptions (ConnectError, TimeoutException, RequestError, etc.)
            # httpx exceptions inherit from httpx.HTTPError, but we catch Exception
            # to be safe across different httpx versions
            # Check if it's an httpx error by checking the exception type name or module
            exc_type_name = type(exc).__name__
            exc_module = getattr(type(exc), "__module__", "")
            # Check for httpx errors - either by module name or exception type name
            is_httpx_error = False

            # Try isinstance check first (works when httpx is not mocked)
            # Get the real httpx module from sys.modules if available
            # This works even when httpx is mocked in the provider module
            import sys

            real_httpx_module = sys.modules.get("httpx")
            # Check if real_httpx_module is actually the real httpx module (not a mock)
            # Real httpx module has __file__ attribute, mocks don't
            if (
                real_httpx_module is not None
                and hasattr(real_httpx_module, "__file__")
                and hasattr(real_httpx_module, "ConnectError")
            ):
                try:
                    is_httpx_error = isinstance(
                        exc,
                        (
                            real_httpx_module.ConnectError,
                            real_httpx_module.TimeoutException,
                            real_httpx_module.RequestError,
                            real_httpx_module.HTTPError,
                        ),
                    )
                except (TypeError, AttributeError):
                    # isinstance() failed (likely because httpx is mocked)
                    pass

            # Fallback to string-based checks if isinstance didn't work
            # This handles cases where httpx is mocked
            # Real httpx exceptions have module "httpx", not "unittest.mock"
            if not is_httpx_error:
                is_httpx_error = exc_module == "httpx" or exc_type_name in (
                    "ConnectError",
                    "TimeoutException",
                    "RequestError",
                    "HTTPError",
                )
            if is_httpx_error:
                raise ConnectionError(OLLAMA_NOT_RUNNING_ERROR) from exc
            # If it's not an httpx error, re-raise the original exception
            raise

    def _validate_model_available(self, model: str) -> None:
        """Validate that model is available in Ollama.

        Args:
            model: Model name to check

        Raises:
            ValueError: If model is not available
        """
        logger.debug("Validating Ollama model availability: %s", model)
        try:
            base_url = self.cfg.ollama_api_base or "http://localhost:11434/v1"
            health_url = _ollama_native_api_root(base_url) + "/api/tags"
            response = httpx.get(health_url, timeout=10.0)
            response.raise_for_status()
            data = response.json()
            available_models = [m.get("name", "") for m in data.get("models", [])]

            if model not in available_models:
                logger.error(
                    "Model '%s' not found in Ollama. Available models: %s",
                    model,
                    ", ".join(sorted(available_models)) if available_models else "(none)",
                )
                # Check for similar model names (fuzzy matching suggestion)
                model_lower = model.lower()
                similar_models = [
                    m
                    for m in available_models
                    if model_lower in m.lower() or m.lower() in model_lower
                ]
                if similar_models:
                    logger.error(
                        "Similar models found (did you mean one of these?): %s",
                        ", ".join(similar_models),
                    )
                error_msg = MODEL_NOT_FOUND_ERROR_TEMPLATE.format(model=model)
                raise ValueError(error_msg)
            logger.info(
                "Model '%s' validated successfully (available in Ollama). "
                "Available models at validation time: %s",
                model,
                ", ".join(sorted(available_models)) if available_models else "(none)",
            )
        except Exception as exc:
            # Catch all httpx exceptions (RequestError, ConnectError, etc.)
            # httpx exceptions inherit from httpx.HTTPError, but we catch Exception
            # to be safe across different httpx versions
            if "httpx" in type(exc).__module__:
                logger.warning(
                    "Could not validate model availability: %s", format_exception_for_log(exc)
                )
                # Don't fail - model might still work, just warn
            else:
                raise

    def initialize(self) -> None:
        """Initialize all Ollama capabilities.

        For Ollama API, initialization validates models are available.
        This method is idempotent and can be called multiple times safely.
        """
        # Initialize speaker detection if enabled
        if self.cfg.auto_speakers and not self._speaker_detection_initialized:
            self._initialize_speaker_detection()

        # ALWAYS initialized — this capability also serves extract_kg_graph / generate_insights /
        # clean_transcript / the evidence stack, so gating it on the generate_summaries STAGE flag
        # starved KG-only runs and lazily-built failover tiers (#1720). For Ollama this validates
        # the summary model against the local server even when the summary stage is off — a
        # KG-only run needs that model available exactly as much as a summary run does.
        if not self._summarization_initialized:
            self._initialize_summarization()

    def warmup(self, timeout_s: int = 600) -> None:
        """Warm up Ollama models by loading them into memory.

        This method performs a small "ping" request to each model that will be used,
        forcing Ollama to load the model weights into memory. This prevents the first
        real request from waiting for model loading, which can take several minutes
        for large models.

        Note: To keep models loaded during your pipeline run, set the OLLAMA_KEEP_ALIVE
        environment variable on the Ollama server side (e.g., `export OLLAMA_KEEP_ALIVE=30m`).
        This prevents models from unloading between requests.

        Args:
            timeout_s: Timeout in seconds for warmup requests (default: 600s = 10 minutes).
                       First model load can take a while, so use a long timeout.

        Raises:
            RuntimeError: If warmup fails for any model
        """
        base_url = self.cfg.ollama_api_base or "http://localhost:11434/v1"
        generate_url = _ollama_native_api_root(base_url) + "/api/generate"

        models_to_warm = set()

        # Collect models that need warming
        if self.cfg.auto_speakers and self._speaker_detection_initialized:
            models_to_warm.add(self.speaker_model)
        if self.cfg.generate_summaries and self._summarization_initialized:
            models_to_warm.add(self.summary_model)

        if not models_to_warm:
            logger.debug("No Ollama models to warm up")
            return

        logger.info(
            f"Warming up Ollama models: {', '.join(sorted(models_to_warm))} "
            f"(timeout: {timeout_s}s)"
        )

        for model in models_to_warm:
            try:
                logger.debug(f"Warming up model: {model}")
                # Use httpx directly for warmup (simpler than OpenAI SDK)
                response = httpx.post(
                    generate_url,
                    json={
                        "model": model,
                        "prompt": "ping",
                        "stream": False,
                        "options": {"num_predict": 1},  # Generate only 1 token
                    },
                    timeout=timeout_s,
                )
                response.raise_for_status()
                logger.debug(f"Model {model} warmed up successfully")
            except Exception as exc:
                logger.warning(f"Failed to warm up model {model}: {format_exception_for_log(exc)}")
                # Don't fail - model might still work, just warn
                # The first real request will trigger loading anyway

    def wait_until_ready(self, max_wait_s: int = 600, poll_interval_s: float = 2.0) -> bool:
        """Wait until Ollama models are ready (loaded and responsive).

        This method polls Ollama with small warmup requests until they succeed,
        indicating that models are loaded and ready for use. This prevents the
        pipeline from starting before models are ready.

        Args:
            max_wait_s: Maximum time to wait in seconds (default: 600s = 10 minutes)
            poll_interval_s: Time between poll attempts in seconds (default: 2.0s)

        Returns:
            True if models are ready, False if timeout exceeded

        Raises:
            RuntimeError: If Ollama server becomes unavailable during wait
        """
        import time

        base_url = self.cfg.ollama_api_base or "http://localhost:11434/v1"
        generate_url = _ollama_native_api_root(base_url) + "/api/generate"

        models_to_check = set()

        # Collect models that need checking
        if self.cfg.auto_speakers and self._speaker_detection_initialized:
            models_to_check.add(self.speaker_model)
        if self.cfg.generate_summaries and self._summarization_initialized:
            models_to_check.add(self.summary_model)

        if not models_to_check:
            logger.debug("No Ollama models to check readiness for")
            return True

        logger.info(
            f"Waiting for Ollama models to be ready: {', '.join(sorted(models_to_check))} "
            f"(max wait: {max_wait_s}s, poll interval: {poll_interval_s}s)"
        )

        start_time = time.time()
        ready_models: set[str] = set()

        while len(ready_models) < len(models_to_check):
            elapsed = time.time() - start_time
            if elapsed > max_wait_s:
                logger.error(
                    f"Timeout waiting for Ollama models to be ready. "
                    f"Ready: {len(ready_models)}/{len(models_to_check)} "
                    f"after {elapsed:.1f}s"
                )
                return False

            for model in models_to_check - ready_models:
                try:
                    # Try a small warmup request
                    response = httpx.post(
                        generate_url,
                        json={
                            "model": model,
                            "prompt": "ping",
                            "stream": False,
                            "options": {"num_predict": 1},
                        },
                        timeout=30.0,  # Short timeout for polling
                    )
                    response.raise_for_status()
                    ready_models.add(model)
                    logger.debug(f"Model {model} is ready (elapsed: {elapsed:.1f}s)")
                except Exception as exc:
                    # Model not ready yet, continue polling
                    logger.debug(f"Model {model} not ready yet: {format_exception_for_log(exc)}")

            # If not all models are ready, wait before next poll
            if len(ready_models) < len(models_to_check):
                time.sleep(poll_interval_s)

        elapsed = time.time() - start_time
        logger.info(
            f"All Ollama models ready: {', '.join(sorted(models_to_check))} "
            f"(took {elapsed:.1f}s)"
        )
        return True

    def _initialize_speaker_detection(self) -> None:
        """Initialize speaker detection capability."""
        logger.debug("Initializing Ollama speaker detection (model: %s)", self.speaker_model)
        # Validate model is available
        self._validate_model_available(self.speaker_model)
        self._speaker_detection_initialized = True
        logger.debug("Ollama speaker detection initialized successfully")

    def _initialize_summarization(self) -> None:
        """Initialize summarization capability."""
        logger.debug("Initializing Ollama summarization (model: %s)", self.summary_model)
        # Validate model is available
        self._validate_model_available(self.summary_model)
        self._summarization_initialized = True
        logger.debug("Ollama summarization initialized successfully")

    # ============================================================================
    # SpeakerDetector Protocol Implementation
    # ============================================================================

    def analyze_patterns(
        self,
        episodes: list[Episode],  # type: ignore[valid-type]
        known_hosts: Set[str],
    ) -> dict[str, object] | None:
        """Analyze patterns across multiple episodes (optional).

        For Ollama provider, pattern analysis is not implemented.
        Returns None to use local pattern analysis logic.
        """
        return None

    # ============================================================================
    # SummarizationProvider Protocol Implementation
    # ============================================================================

    def summarize(
        self,
        text: str,
        episode_title: Optional[str] = None,
        episode_description: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        pipeline_metrics: metrics.Metrics | None = None,
        call_metrics: Any | None = None,  # ProviderCallMetrics from utils.provider_metrics
    ) -> Dict[str, Any]:
        """Summarize text using Ollama API.

        The transcript is clipped to what num_ctx actually holds (#2050). This docstring used
        to claim a 128k window and send the transcript whole — Ollama does not reject an
        over-long prompt, it truncates and returns 200, so the claim cost the tail of every long
        episode on the failover tier.
        No chunking needed for most podcast transcripts.

        Key advantage: Fully offline, zero cost, complete privacy.

        Args:
            text: Transcript text to summarize
            episode_title: Optional episode title
            episode_description: Optional episode description
            params: Optional parameters dict with max_length, min_length, etc.
            pipeline_metrics: Optional metrics tracker

        Returns:
            Dictionary with summary results:
            {
                "summary": str,
                "summary_short": Optional[str],
                "metadata": {...}
            }

        Raises:
            ValueError: If summarization fails
            RuntimeError: If provider is not initialized
        """
        if not self._summarization_initialized:
            raise RuntimeError(
                "OllamaProvider summarization not initialized. Call initialize() first."
            )

        # Extract parameters with defaults from config
        max_length = (
            (params.get("max_length") if params else None)
            or self.cfg.summary_reduce_params.get("max_new_tokens")
            or 800
        )
        min_length = (
            (params.get("min_length") if params else None)
            or self.cfg.summary_reduce_params.get("min_new_tokens")
            or 100
        )
        custom_prompt = params.get("prompt") if params else None
        # A summary other than the episode's (a board's) runs this same flow with its
        # own prompts, stage and budget; absent, this is the episode summary exactly as before.
        use_case = use_case_from_params(params)
        max_length = reply_budget(use_case, max_length)

        # Ollama-specific reduce params (from ollama_reduce_params in hybrid config)
        # Fall back to summary_temperature / OpenAI-compat defaults when not set
        effective_temperature = (
            getattr(self.cfg, "ollama_reduce_temperature", None) or self.summary_temperature
        )
        effective_top_p: Optional[float] = getattr(self.cfg, "ollama_reduce_top_p", None)
        effective_frequency_penalty: Optional[float] = getattr(
            self.cfg, "ollama_reduce_frequency_penalty", None
        )

        logger.debug(
            "Summarizing text via Ollama API (model: %s, max_tokens: %d)",
            self.summary_model,
            max_length,
        )

        try:
            # Build prompts using prompt_store
            (
                system_prompt,
                user_prompt,
                system_prompt_name,
                user_prompt_name,
                paragraphs_min,
                paragraphs_max,
            ) = self._build_summarization_prompts(
                text,
                episode_title,
                episode_description,
                max_length,
                min_length,
                custom_prompt,
                use_case=use_case,
            )

            # Track retries and rate limits
            from ...utils.provider_metrics import (
                _safe_openai_retryable,
                ProviderCallMetrics,
                retry_with_metrics,
            )

            if call_metrics is None:
                call_metrics = ProviderCallMetrics()
            call_metrics.set_provider_name("ollama")

            def _make_api_call():
                logger.info(
                    "Calling Ollama API for summarization with model: '%s' "
                    "(exact name being sent to Ollama)",
                    self.summary_model,
                )
                optional_kwargs: Dict[str, Any] = {}
                if effective_top_p is not None:
                    optional_kwargs["top_p"] = effective_top_p
                if effective_frequency_penalty is not None:
                    optional_kwargs["frequency_penalty"] = effective_frequency_penalty
                return self.client.chat.completions.create(
                    model=self.summary_model,
                    messages=_openai_style_messages(  # RFC-115 transcript-first
                        text, system_prompt, user_prompt, enabled=self._cache_transcript_prefix
                    ),
                    temperature=effective_temperature,
                    max_tokens=max_length,
                    **optional_kwargs,
                    **_ollama_openai_chat_extra_kwargs(
                        self.summary_model, num_ctx=self.summary_num_ctx
                    ),
                )

            try:
                response = retry_with_metrics(
                    _make_api_call,
                    max_retries=3,
                    initial_delay=1.0,
                    max_delay=30.0,
                    retryable_exceptions=_safe_openai_retryable(),
                    metrics=call_metrics,
                    error_context="ollama_local",
                )
            except Exception:
                call_metrics.finalize()
                raise

            call_metrics.finalize()

            warn_if_truncated(
                response.choices[0].finish_reason,
                "ollama",
                "summarize",
            )

            summary = response.choices[0].message.content

            # Token + cost capture up-front so cost emits in both branches.
            input_tokens = None
            output_tokens = None
            if hasattr(response, "usage") and response.usage:
                prompt_tokens_val = getattr(response.usage, "prompt_tokens", None)
                completion_tokens_val = getattr(response.usage, "completion_tokens", None)
                input_tokens = (
                    int(prompt_tokens_val) if isinstance(prompt_tokens_val, (int, float)) else 0
                )
                output_tokens = (
                    int(completion_tokens_val)
                    if isinstance(completion_tokens_val, (int, float))
                    else 0
                )
                if input_tokens > 0 or output_tokens > 0:
                    call_metrics.set_tokens(input_tokens, output_tokens)

            cost: Optional[float] = None
            if input_tokens is not None:
                from ...workflow.helpers import calculate_provider_cost

                cost = calculate_provider_cost(
                    cfg=self.cfg,
                    provider_type="ollama",
                    capability="summarization",
                    model=self.summary_model,
                    prompt_tokens=input_tokens,
                    completion_tokens=output_tokens,
                )

            def _record_cost(*, triggered_guardrail: bool = False) -> None:
                if input_tokens is None:
                    return
                from ...utils.provider_metrics import record_provider_call_cost

                record_provider_call_cost(
                    call_metrics,
                    cost,
                    cfg=self.cfg,
                    provider_type="ollama",
                    capability="summarization",
                    model=self.summary_model,
                    prompt_tokens=input_tokens,
                    completion_tokens=output_tokens,
                    triggered_guardrail=triggered_guardrail,
                    stage=use_case_stage(use_case),
                )

            # Response-shape guardrail (ADR-099/100). Cost emitted in BOTH
            # branches (Ollama is free locally but the field still flows so
            # cost-rollup pivots work consistently across providers).
            try:
                _guardrails.check_chat_response(summary, service="ollama")
            except _guardrails.GuardrailViolation:
                _record_cost(triggered_guardrail=True)
                raise

            logger.debug(
                "Ollama summarization completed: %d characters",
                len(summary),
            )

            _record_cost()

            # Track LLM call metrics if available (aggregate tracking)
            if (
                pipeline_metrics is not None
                and input_tokens is not None
                and output_tokens is not None
            ):
                pipeline_metrics.record_llm_summarization_call(
                    input_tokens, output_tokens, cost_usd=cost
                )

            # Get prompt metadata for tracking
            from ...prompts.store import get_prompt_metadata

            prompt_metadata = {}
            if system_prompt_name:
                prompt_metadata["system"] = get_prompt_metadata(system_prompt_name)
            if user_prompt_name == "custom":
                # Inline custom prompt (e.g. from hybrid reduce); no template file
                prompt_metadata["user"] = {
                    "name": "custom",
                    "file": None,
                    "sha256": None,
                }
            else:
                user_params = {
                    "transcript": text[:100] + "..." if len(text) > 100 else text,
                    "title": episode_title or "",
                    "paragraphs_min": paragraphs_min,
                    "paragraphs_max": paragraphs_max,
                }
                user_params.update(self.cfg.summary_prompt_params)
                prompt_metadata["user"] = get_prompt_metadata(user_prompt_name, params=user_params)

            # A use case records its own templates; the episode summary keeps the above.
            prompt_metadata = use_case_prompt_metadata("ollama", use_case) or prompt_metadata
            return {
                "summary": summary,
                "summary_short": None,  # Ollama doesn't generate short summaries separately
                "metadata": {
                    "model": self.summary_model,
                    "provider": "ollama",
                    "max_length": max_length,
                    "min_length": min_length,
                    "prompts": prompt_metadata,
                },
            }

        except _guardrails.GuardrailViolation:
            # ADR-100: propagate the raw violation so FallbackAwareSummarizationProvider
            # can route to the degradation policy's fallback. Wrapping into
            # ProviderRuntimeError would hide the type from the fallback layer.
            raise
        except Exception as exc:
            logger.error("Ollama API error in summarization: %s", format_exception_for_log(exc))
            from podcast_scraper.exceptions import ProviderRuntimeError

            raise ProviderRuntimeError(
                message=f"Ollama summarization failed: {format_exception_for_log(exc)}",
                provider="OllamaProvider/Summarization",
            ) from exc

    def _build_summarization_prompts(
        self,
        text: str,
        episode_title: Optional[str],
        episode_description: Optional[str],
        max_length: int,
        min_length: int,
        custom_prompt: Optional[str],
        use_case: Optional[SummaryUseCase] = None,
    ) -> tuple[str, str, Optional[str], str, int, int]:
        """Build system and user prompts for summarization using prompt_store."""

        # #2050 / advisor M2: Ollama TRUNCATES SILENTLY rather than returning a 400 (see the
        # num_ctx note in this module's docstring), and it is prod's summary failover
        # (`summary_fallback_providers: [ollama]`). The base class clips in `summarize()`, but
        # Ollama overrides `summarize()`, so that fix never applied here — the exact path the
        # #2050 commit claimed to have closed. This builder is reached only by `summarize`;
        # `summarize_bundled` renders its own prompt and is clipped separately at its own call.
        #
        # The gap widened when the DGX went to 64k: the episode gate now admits ~197k chars
        # against an Ollama window of num_ctx=32768, roughly 2x what it can hold.
        _budget = config_constants.transcript_budget_chars(
            getattr(self, "max_context_tokens", None), response_tokens=int(max_length)
        )
        if _budget is not None and len(text or "") > _budget:
            logger.warning(
                "ollama: summarization transcript %d chars exceeds the %d-char budget for a "
                "%s-token window; clipping (#2050). Ollama truncates silently rather than "
                "rejecting, so without this the tail was dropped with no error.",
                len(text),
                _budget,
                getattr(self, "max_context_tokens", None),
            )
            text = text[:_budget]
        from ...prompts.store import render_prompt

        if use_case is not None:  # a use case's own prompts
            system, user, system_name, user_name = build_use_case_prompts("ollama", use_case)
            return system, user, system_name, user_name, 0, 0

        # Try model-specific prompts first, fallback to generic
        if self.cfg.ollama_summary_system_prompt:
            system_prompt_name = self.cfg.ollama_summary_system_prompt
        else:
            system_prompt_name = self._get_model_specific_prompt_path(
                self.summary_model,
                "summarization",
                "system_v1",
                "ollama/summarization/system_v1",
            )

        if self.cfg.ollama_summary_user_prompt:
            user_prompt_name = self.cfg.ollama_summary_user_prompt
        else:
            user_prompt_name = self._get_model_specific_prompt_path(
                self.summary_model,
                "summarization",
                "long_v1",
                "ollama/summarization/long_v1",
            )

        system_prompt = render_prompt(system_prompt_name)

        paragraphs_min = max(1, min_length // 100)
        paragraphs_max = max(paragraphs_min, max_length // 100)

        if custom_prompt:
            user_prompt = custom_prompt.replace("{{ transcript }}", text)
            if episode_title:
                user_prompt = user_prompt.replace("{{ title }}", episode_title)
            user_prompt_name = "custom"
        else:
            template_params = {
                "transcript": text,
                "title": episode_title or "",
                "paragraphs_min": paragraphs_min,
                "paragraphs_max": paragraphs_max,
            }
            template_params.update(self.cfg.summary_prompt_params)
            user_prompt = render_prompt(user_prompt_name, **template_params)

        return (
            system_prompt,
            user_prompt,
            system_prompt_name,
            user_prompt_name,
            paragraphs_min,
            paragraphs_max,
        )

    # ============================================================================
    # Cleanup Methods
    # ============================================================================

    def cleanup(self) -> None:
        """Cleanup all provider resources (no-op for API provider)."""
        self._speaker_detection_initialized = False
        self._summarization_initialized = False

    def clear_cache(self) -> None:
        """Clear cache (no-op for API provider)."""
        pass

    def clean_transcript(self, text: str, pipeline_metrics: Optional[Any] = None) -> str:
        """Clean transcript using LLM for semantic filtering.

        Args:
            text: Transcript text to clean (should already be pattern-cleaned)

        Returns:
            Cleaned transcript text

        Raises:
            RuntimeError: If provider is not initialized or cleaning fails
        """
        if not self._summarization_initialized:
            raise RuntimeError("OllamaProvider not initialized. Call initialize() first.")

        from ...prompts.store import render_prompt

        # Build cleaning prompt using prompt_store
        prompt_name = "ollama/cleaning/v1"
        user_prompt = render_prompt(prompt_name, transcript=text)

        # Use system prompt (OpenAI-compatible pattern)
        system_prompt = (
            "You are a transcript cleaning assistant. "
            "Remove sponsors, ads, intros, outros, and meta-commentary. "
            "Preserve all substantive content and speaker information. "
            "Return only the cleaned text, no explanations."
        )

        logger.debug(
            "Cleaning transcript via Ollama API (model: %s, text length: %d chars)",
            self.cleaning_model,
            len(text),
        )

        try:
            # Track retries and rate limits
            from ...utils.provider_metrics import (
                _safe_openai_retryable,
                ProviderCallMetrics,
                retry_with_metrics,
            )

            call_metrics = ProviderCallMetrics()
            call_metrics.set_provider_name("ollama")

            def _make_api_call():
                logger.info(
                    "Calling Ollama API for cleaning with model: '%s' "
                    "(exact name being sent to Ollama)",
                    self.cleaning_model,
                )
                return self.client.chat.completions.create(
                    model=self.cleaning_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.cleaning_temperature,
                    max_tokens=clamp_cleaning_max_tokens(
                        estimate_cleaning_output_tokens(len(text.split())),
                        OLLAMA_CLEANING_MAX_TOKENS,
                    ),
                    **_ollama_openai_chat_extra_kwargs(self.cleaning_model),
                )

            try:
                response = retry_with_metrics(
                    _make_api_call,
                    max_retries=3,
                    initial_delay=1.0,
                    max_delay=30.0,
                    retryable_exceptions=_safe_openai_retryable(),
                    metrics=call_metrics,
                    error_context="ollama_local",
                )
            except Exception:
                call_metrics.finalize()
                raise

            call_metrics.finalize()

            cleaned = response.choices[0].message.content

            # Cost capture up-front (ADR-100 cost-attribution in both branches).
            in_tok_cl = None
            out_tok_cl = None
            if hasattr(response, "usage") and response.usage:
                pt = getattr(response.usage, "prompt_tokens", None)
                ct = getattr(response.usage, "completion_tokens", None)
                in_tok_cl = int(pt) if isinstance(pt, (int, float)) else None
                out_tok_cl = int(ct) if isinstance(ct, (int, float)) else None
            cleaning_cost: Optional[float] = None
            if in_tok_cl is not None and out_tok_cl is not None:
                from ...workflow.helpers import calculate_provider_cost

                cleaning_cost = calculate_provider_cost(
                    cfg=self.cfg,
                    provider_type="ollama",
                    capability="cleaning",
                    model=self.cleaning_model,
                    prompt_tokens=in_tok_cl,
                    completion_tokens=out_tok_cl,
                )

            def _emit_cleaning_cost(*, triggered_guardrail: bool = False) -> None:
                if in_tok_cl is None or out_tok_cl is None:
                    return
                try:
                    from ...workflow.cost_monitoring import emit_llm_cost_event

                    emit_llm_cost_event(
                        self.cfg,
                        provider="ollama",
                        stage="cleaning",
                        model=self.cleaning_model,
                        estimated_cost_usd=float(cleaning_cost or 0.0),
                        prompt_tokens=in_tok_cl,
                        completion_tokens=out_tok_cl,
                        triggered_guardrail=triggered_guardrail,
                    )
                except Exception:  # noqa: BLE001
                    pass

            if not cleaned:
                logger.warning("Ollama API returned empty cleaned text, using original")
                _emit_cleaning_cost()
                return text
            # ADR-100: cleaning catch-and-degrade; cost emitted in both branches.
            try:
                _guardrails.check_chat_response(cleaned, service="ollama")
            except _guardrails.GuardrailViolation:
                _emit_cleaning_cost(triggered_guardrail=True)
                logger.warning(
                    "Ollama cleaning output failed guardrail; " "returning original transcript text"
                )
                return text

            _emit_cleaning_cost()
            logger.debug("Ollama cleaning completed: %d -> %d chars", len(text), len(cleaned))
            return cast(str, cleaned)

        except _guardrails.GuardrailViolation:
            # Defensive outer catch — preserve contract if a future change
            # adds a guardrail call outside the inline block.
            logger.warning(
                "Ollama cleaning output failed guardrail (outer); "
                "returning original transcript text"
            )
            return text
        except Exception as exc:
            logger.error("Ollama API error in cleaning: %s", format_exception_for_log(exc))
            from podcast_scraper.exceptions import ProviderRuntimeError

            # Handle Ollama-specific error types
            error_msg = str(exc).lower()
            if "connection" in error_msg or "refused" in error_msg:
                raise ProviderRuntimeError(
                    message=f"Ollama server connection failed: {format_exception_for_log(exc)}",
                    provider="OllamaProvider/Cleaning",
                    suggestion="Ensure Ollama server is running at the configured base URL",
                ) from exc
            else:
                raise ProviderRuntimeError(
                    message=f"Ollama cleaning failed: {format_exception_for_log(exc)}",
                    provider="OllamaProvider/Cleaning",
                ) from exc

    def complete_text(
        self, prompt: str, max_tokens: int = 800, stage: str = "speaker_detection"
    ) -> str:
        """One prompt in, one JSON answer out — the generic call ADR-110's resolver needs.

        `detect_speakers` is asked who speaks BEFORE the audio is downloaded, so it can only answer
        from show notes and returns the people an episode is ABOUT (#876). The resolver asks the
        same model the same question AFTER diarization, with each voice's own words in hand — and
        that needs a plain completion, not a task method whose prompt is baked in.
        """
        if not self._summarization_initialized:
            raise RuntimeError("Ollama summarization not initialized for complete_text")

        response = self.client.chat.completions.create(
            model=self.summary_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=int(max_tokens),
            **_ollama_openai_chat_extra_kwargs(self.summary_model, num_ctx=self.summary_num_ctx),
        )
        from ...utils.provider_metrics import record_response_cost

        record_response_cost(
            self.cfg,
            response,
            provider_type="ollama",
            capability="speaker_detection",
            stage=stage,
            model=self.summary_model,
        )
        content = (response.choices[0].message.content or "").strip()
        _guardrails.check_chat_response(content, service="ollama")
        return content

    def extract_kg_graph(
        self,
        text: str,
        episode_title: Optional[str] = None,
        max_topics: int = 5,
        max_entities: int = 15,
        params: Optional[Dict[str, Any]] = None,
        pipeline_metrics: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """Extract topics and entities as JSON (KG layer). Returns None on failure."""
        if not self._summarization_initialized:
            raise RuntimeError(
                "OllamaProvider summarization not initialized. Call initialize() first."
            )
        from ...kg.llm_extract import (
            build_kg_transcript_system_prompt,
            build_kg_user_prompt,
            KG_RESPONSE_TOKENS,
            parse_kg_graph_response,
            resolve_kg_model_id,
            truncate_transcript_for_kg,
        )

        max_topics = min(max(1, max_topics), 20)
        max_entities = min(max(1, max_entities), 50)
        # #2050: derive the KG clip from this deployment's window, not a shared 120,000 literal
        # that fit no model in particular. Char-derived rather than tokenizer-counted: the KG path
        # accounts for 0 of the 565 measured context rejections, so a /tokenize round trip per
        # call is not justified here.
        text_slice = truncate_transcript_for_kg(
            text or "",
            limit=config_constants.transcript_budget_chars(
                getattr(self, "max_context_tokens", None),
                response_tokens=KG_RESPONSE_TOKENS,
            ),
        )
        if not text_slice.strip():
            return None
        model = resolve_kg_model_id(self, params)
        user_prompt = build_kg_user_prompt(
            text_slice,
            episode_title or "",
            max_topics,
            max_entities,
            prompt_version=(params or {}).get("kg_prompt_version", "v4"),
            ner_entity_hints=(params or {}).get("ner_entity_hints"),
        )
        system_msg = build_kg_transcript_system_prompt(max_topics, max_entities)
        try:
            from ...utils.provider_metrics import (
                _safe_openai_retryable,
                retry_with_metrics,
            )

            def _make_api_call():
                return self.client.chat.completions.create(
                    model=model,
                    messages=_openai_style_messages(  # RFC-115 transcript-first
                        text_slice, system_msg, user_prompt, enabled=self._cache_transcript_prefix
                    ),
                    temperature=0.1,
                    max_tokens=2048,
                    **_ollama_openai_chat_extra_kwargs(model),
                )

            response = retry_with_metrics(
                _make_api_call,
                max_retries=3,
                initial_delay=1.0,
                max_delay=30.0,
                retryable_exceptions=_safe_openai_retryable(),
                error_context="ollama_local",
            )
            from ...utils.provider_metrics import record_response_cost

            record_response_cost(
                self.cfg,
                response,
                provider_type="ollama",
                capability="kg",
                stage="kg",
                model=model,
            )
            raw = (response.choices[0].message.content or "").strip()
            return parse_kg_graph_response(raw, max_topics=max_topics, max_entities=max_entities)
        except Exception as e:
            logger.debug("Ollama extract_kg_graph failed: %s", e, exc_info=True)
            return None

    @property
    def is_initialized(self) -> bool:
        """Check if provider is initialized (any component)."""
        return self._speaker_detection_initialized or self._summarization_initialized

    def get_capabilities(self) -> ProviderCapabilities:
        """Get provider capabilities.

        Returns:
            ProviderCapabilities object describing Ollama provider capabilities
        """
        from ..capabilities import ProviderCapabilities  # noqa: PLC0415

        return ProviderCapabilities(
            supports_transcription=False,  # Ollama doesn't support audio transcription
            supports_speaker_detection=True,
            supports_summarization=True,
            supports_semantic_cleaning=True,  # Ollama supports LLM-based cleaning
            supports_audio_input=False,  # Ollama doesn't accept audio files
            supports_json_mode=True,  # Ollama supports JSON mode
            max_context_tokens=self.max_context_tokens,
            supports_tool_calls=True,  # Ollama supports function calling
            supports_system_prompt=True,  # Ollama supports system prompts
            supports_streaming=True,  # Ollama API supports streaming
            provider_name="ollama",
            supports_gi_segment_timing=False,
        )
