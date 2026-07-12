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

import json
import logging
from typing import Any, cast, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

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
from ...cleaning import PatternBasedCleaner
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

logger = logging.getLogger(__name__)

# The old inline wording, kept only as the fallback if the template is missing. Do not tune here —
# tune the template, which is what the calibration harness measures.
_ENTAILMENT_FALLBACK = (
    "You rate how much the premise supports the hypothesis. "
    "Reply with ONLY a number between 0 and 1 (0=not at all, 1=fully supports)."
)


def _json_object_from_response(text: str) -> str:
    """Pull the JSON object out of a chat reply.

    qwen3.5 reasons before answering, and the reasoning is full of digits and braces. Scoping past
    any ``</think>`` block first is what stops us parsing the model's scratch work as the answer.
    """
    body = text or ""
    marker = body.rfind("</think>")
    if marker != -1:
        body = body[marker + len("</think>") :]
    start = body.find("{")
    end = body.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object in response")
    return body[start : end + 1]


# Roughly 3.5 characters per token for English prose. Used only to keep the transcript inside the
# context window we were actually configured with, instead of a hardcoded guess.
_CHARS_PER_TOKEN = 3.5
# Headroom for the instructions, the insight, and the model's own output.
_EXTRACT_QUOTE_RESERVE_TOKENS = 2000


def _transcript_budget_chars(num_ctx: int) -> int:
    """How much transcript actually fits in the configured context window.

    The old code truncated to a hardcoded 50 000 characters. Real episodes here are 67 000-78 000,
    so **the last third of every episode was invisible to quote extraction** — an insight drawn
    from it could never be grounded, no matter how good the model or the gate. With num_ctx=32768
    the true budget is ~107 000 characters, and nothing needs truncating at all.
    """
    usable = max(0, int(num_ctx) - _EXTRACT_QUOTE_RESERVE_TOKENS)
    return int(usable * _CHARS_PER_TOKEN)


def _render_extract_quote_prompt(transcript: str, insight: str, num_ctx: int) -> "tuple[str, str]":
    """``(system, user)`` for GIL quote extraction, from the ollama prompt template.

    The old inline prompt had three defects, each silently costing evidence: it truncated the
    transcript to 50 000 chars (below our episode length); its system and user messages disagreed
    about the output shape (``{"quotes": [...]}`` vs "quote_text only"); and it embedded a copyable
    example string, which is exactly what local models reproduce verbatim (#1179).
    """
    from ...prompts.store import render_prompt

    budget = _transcript_budget_chars(num_ctx)
    text = transcript.strip()
    if len(text) > budget:
        logger.warning(
            "transcript %d chars exceeds the %d-char context budget; quote extraction will not "
            "see the tail of this episode",
            len(text),
            budget,
        )
        text = text[:budget]

    try:
        return "", render_prompt(
            "ollama/evidence/extract_quote/v1",
            transcript=text,
            insight=insight.strip(),
        )
    except Exception as exc:  # noqa: BLE001 — a missing template must not break grounding
        logger.warning("ollama extract_quote template unavailable (%s); using inline fallback", exc)
        return (
            "Extract short verbatim quotes from the transcript that support the insight. "
            'Reply with ONLY a JSON object with a single key "quotes" (a list of exact strings).',
            f"Transcript:\n{text}\n\nInsight: {insight.strip()}",
        )


def _render_entailment_prompt(premise: str, hypothesis: str) -> "tuple[str, str]":
    """``(system, user)`` for the GIL entailment gate, from the ollama prompt template.

    The wording IS the gate. Strict textual entailment ("does the premise support the hypothesis")
    is not the question the pipeline means — a quote can be excellent evidence for an insight
    without logically entailing it — and asking it strictly cost 60% of the evidence a trusted
    annotator had accepted (#1179). Keeping this in a template is what lets it be calibrated.
    """
    from ...prompts.store import render_prompt

    try:
        rendered = render_prompt(
            "ollama/evidence/entailment/v1",
            premise=premise.strip(),
            hypothesis=hypothesis.strip(),
        )
        return "", rendered
    except Exception as exc:  # noqa: BLE001 — a missing template must not break grounding
        logger.warning("ollama entailment template unavailable (%s); using inline fallback", exc)
        return (
            _ENTAILMENT_FALLBACK,
            f"Premise: {premise.strip()}\n\nHypothesis: {hypothesis.strip()}",
        )


def _ollama_openai_chat_extra_kwargs(model: str, num_ctx: Optional[int] = None) -> Dict[str, Any]:
    """Extra kwargs for Ollama's OpenAI-compatible ``/v1/chat/completions``.

    Two Ollama-specific concerns handled here:

    1. **Context window (num_ctx)** — Ollama defaults to 2048 tokens, which silently
       truncates any prompt longer than that. For summarisation we need the full
       transcript, so we pass ``num_ctx`` explicitly via ``extra_body``. Models have
       different maximums (e.g. gemma2 = 8k, qwen3.5 = 256k); pick a value large
       enough for your expected prompt length but within the model's range.

    2. **Qwen 3.x chain-of-thought** — Qwen 3.x reasoning models (3.5, 3.6, and
       newer variants in the qwen3 family) emit a separate ``reasoning`` /
       ``thinking`` channel by default. Without ``reasoning_effort: none``,
       ``message.content`` can stay empty while the model consumes
       ``max_tokens`` on thinking tokens (finish_reason ``length``).
       Observed on 2026-06-08 sweep: qwen3.6:latest produced 0-token content
       for every episode until we added it here. See:
       https://docs.ollama.com/capabilities/thinking
    """
    m = (model or "").lower()
    extra_body: Dict[str, Any] = {}
    if num_ctx is not None:
        # Ollama's OpenAI-compat endpoint accepts native Ollama options via extra_body.
        extra_body["options"] = {"num_ctx": int(num_ctx)}
    # Match any qwen3.x reasoning model — qwen3.5, qwen3.6, qwen3-coder, etc.
    if any(tag in m for tag in ("qwen3.5", "qwen3.6", "qwen3-", "qwen3:")):
        extra_body["reasoning_effort"] = "none"
    return {"extra_body": extra_body} if extra_body else {}


def _flatten_json_speaker_names(value: Any) -> List[str]:
    """Flatten nested JSON name lists from Ollama JSON speaker responses."""
    if value is None:
        return []
    if isinstance(value, str):
        t = value.strip()
        return [t] if t else []
    if isinstance(value, (list, tuple)):
        out: List[str] = []
        for v in value:
            out.extend(_flatten_json_speaker_names(v))
        return out
    t = str(value).strip()
    return [t] if t else []


# Default speaker names when detection fails
from ..ml.speaker_detection import DEFAULT_SPEAKER_NAMES

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


class OllamaProvider:
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

        # Set up transcript cleaning processor based on strategy (Issue #418)
        from ...cleaning import HybridCleaner, LLMBasedCleaner

        cleaning_strategy = getattr(cfg, "transcript_cleaning_strategy", "hybrid")
        if cleaning_strategy == "pattern":
            self.cleaning_processor = PatternBasedCleaner()  # type: ignore[assignment]
        elif cleaning_strategy == "llm":
            self.cleaning_processor = LLMBasedCleaner()  # type: ignore[assignment]
        else:  # hybrid (default)
            self.cleaning_processor = HybridCleaner()  # type: ignore[assignment]

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
        # Modern Ollama models support 128k context window
        self.max_context_tokens = 128000  # Conservative estimate

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

        # Initialize summarization if enabled
        if self.cfg.generate_summaries and not self._summarization_initialized:
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

    def detect_hosts(
        self,
        feed_title: str | None,
        feed_description: str | None,
        feed_authors: list[str] | None = None,
    ) -> Set[str]:
        """Detect host names from feed-level metadata using Ollama API.

        Args:
            feed_title: Feed title (can be None)
            feed_description: Optional feed description
            feed_authors: Optional list of author names from RSS feed (preferred source)

        Returns:
            Set of detected host names
        """
        if not self._speaker_detection_initialized:
            raise RuntimeError(
                "OllamaProvider speaker detection not initialized. Call initialize() first."
            )

        # Prefer RSS author tags if available (like OpenAI)
        if feed_authors:
            return set(feed_authors)

        # Otherwise, use Ollama API to detect hosts from feed metadata
        if not feed_title:
            return set()

        try:
            # Use detect_speakers with empty known_hosts to detect hosts
            speakers, detected_hosts, _, _ = self.detect_speakers(
                episode_title=feed_title,
                episode_description=feed_description,
                known_hosts=set(),
            )
            return detected_hosts
        except Exception as exc:
            logger.warning(
                "Failed to detect hosts from feed metadata: %s", format_exception_for_log(exc)
            )
            return set()

    def detect_speakers(
        self,
        episode_title: str,
        episode_description: str | None,
        known_hosts: Set[str],
        pipeline_metrics: metrics.Metrics | None = None,
    ) -> Tuple[list[str], Set[str], bool, bool]:
        """Detect speaker names from episode metadata using Ollama API.

        Args:
            episode_title: Episode title
            episode_description: Optional episode description
            known_hosts: Set of known host names (for context)
            pipeline_metrics: Optional metrics tracker

        Returns:
            Tuple of:
            - List of detected speaker names (hosts + guests)
            - Set of detected host names (subset of known_hosts)
            - Success flag (True if detection succeeded)
            - used_defaults: True if default names were returned (e.g. on failure)

        Raises:
            ValueError: If detection fails
            RuntimeError: If provider is not initialized
        """
        # If auto_speakers is disabled, return defaults without requiring initialization
        if not self.cfg.auto_speakers:
            logger.debug("Auto-speakers disabled, detection failed")
            return DEFAULT_SPEAKER_NAMES.copy(), set(), False, True

        if not self._speaker_detection_initialized:
            raise RuntimeError(
                "OllamaProvider speaker detection not initialized. Call initialize() first."
            )

        logger.debug("Detecting speakers via Ollama API for episode: %s", episode_title[:50])

        try:
            # Build prompt using prompt_store
            user_prompt = self._build_speaker_detection_prompt(
                episode_title, episode_description, known_hosts
            )

            # Get system prompt from prompt_store
            from ...prompts.store import render_prompt

            # Try model-specific prompt first, fallback to generic
            if self.cfg.ollama_speaker_system_prompt:
                system_prompt_name = self.cfg.ollama_speaker_system_prompt
            else:
                system_prompt_name = self._get_model_specific_prompt_path(
                    self.speaker_model,
                    "ner",
                    "system_ner_v1",
                    "ollama/ner/system_ner_v1",
                )
            system_prompt = render_prompt(system_prompt_name)

            # Call Ollama API (OpenAI-compatible format) with retry
            from ...utils.provider_metrics import retry_with_metrics

            logger.info(
                "Calling Ollama API for speaker detection with model: '%s' "
                "(exact name being sent to Ollama)",
                self.speaker_model,
            )

            response = retry_with_metrics(
                lambda: self.client.chat.completions.create(
                    model=self.speaker_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.speaker_temperature,
                    max_tokens=300,
                    response_format={"type": "json_object"},
                    **_ollama_openai_chat_extra_kwargs(self.speaker_model),
                ),
                max_retries=2,
                initial_delay=1.0,
                max_delay=30.0,
                retryable_exceptions=(Exception,),
            )

            response_text = response.choices[0].message.content
            if not response_text:
                logger.warning("Ollama API returned empty response")
                return DEFAULT_SPEAKER_NAMES.copy(), set(), False, True

            # Parse JSON response
            speakers, detected_hosts, success = self._parse_speakers_from_response(
                response_text, known_hosts
            )

            logger.debug(
                "Ollama speaker detection completed: %d speakers, %d hosts, success=%s",
                len(speakers),
                len(detected_hosts),
                success,
            )

            # Track LLM call metrics if available (Ollama doesn't report usage, but track anyway)
            if pipeline_metrics is not None and hasattr(response, "usage"):
                input_tokens = response.usage.prompt_tokens if response.usage else 0
                output_tokens = response.usage.completion_tokens if response.usage else 0
                sd_cost: Optional[float] = None
                if input_tokens > 0 or output_tokens > 0:
                    from ...workflow.helpers import calculate_provider_cost

                    sd_cost = calculate_provider_cost(
                        cfg=self.cfg,
                        provider_type="ollama",
                        capability="speaker_detection",
                        model=self.speaker_model,
                        prompt_tokens=int(input_tokens),
                        completion_tokens=int(output_tokens),
                    )
                pipeline_metrics.record_llm_speaker_detection_call(
                    input_tokens, output_tokens, cost_usd=sd_cost
                )

            return speakers, detected_hosts, success, False

        except json.JSONDecodeError as exc:
            logger.error(
                "Failed to parse Ollama API JSON response: %s", format_exception_for_log(exc)
            )
            return DEFAULT_SPEAKER_NAMES.copy(), set(), False, True
        except Exception as exc:
            logger.error("Ollama API error in speaker detection: %s", format_exception_for_log(exc))
            from podcast_scraper.exceptions import ProviderRuntimeError

            raise ProviderRuntimeError(
                message=f"Ollama speaker detection failed: {format_exception_for_log(exc)}",
                provider="OllamaProvider/SpeakerDetection",
            ) from exc

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

    def _build_speaker_detection_prompt(
        self, episode_title: str, episode_description: str | None, known_hosts: Set[str]
    ) -> str:
        """Build user prompt for speaker detection using prompt_store."""
        from ...prompts.store import render_prompt

        # Try model-specific prompt first, fallback to generic
        if self.cfg.ollama_speaker_user_prompt:
            user_prompt_name = self.cfg.ollama_speaker_user_prompt
        else:
            user_prompt_name = self._get_model_specific_prompt_path(
                self.speaker_model,
                "ner",
                "guest_host_v1",
                "ollama/ner/guest_host_v1",
            )
        template_params = {
            "episode_title": episode_title,
            "episode_description": episode_description or "",
            "known_hosts": ", ".join(sorted(known_hosts)) if known_hosts else "",
        }
        template_params.update(self.cfg.ner_prompt_params)
        user_prompt = render_prompt(user_prompt_name, **template_params)
        return user_prompt

    def _parse_speakers_from_response(
        self, response_text: str, known_hosts: Set[str]
    ) -> Tuple[list[str], Set[str], bool]:
        """Parse speaker names from Ollama API response."""
        try:
            data = json.loads(response_text)
            if isinstance(data, dict):
                speakers_raw = data.get("speakers", [])
                hosts_flat = _flatten_json_speaker_names(data.get("hosts", []))
                hosts = set(hosts_flat)
                guests_flat = _flatten_json_speaker_names(data.get("guests", []))
                if speakers_raw:
                    all_speakers = _flatten_json_speaker_names(speakers_raw)
                else:
                    all_speakers = list(hosts) + guests_flat
                return all_speakers, hosts, True
        except json.JSONDecodeError:
            if response_text.strip().startswith("{"):
                return DEFAULT_SPEAKER_NAMES.copy(), set(), False
            pass

        # Fallback: parse from plain text
        speakers = []
        for line in response_text.strip().split("\n"):
            for name in line.split(","):
                name = name.strip().strip("-").strip("*").strip()
                if name and len(name) > 1:
                    speakers.append(name)
        detected_hosts = set(s for s in speakers if s in known_hosts)
        return speakers, detected_hosts, len(speakers) > 0

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

        Can handle full transcripts directly due to large context window (128k tokens).
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
                text, episode_title, episode_description, max_length, min_length, custom_prompt
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
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
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

    def summarize_bundled(
        self,
        text: str,
        episode_title: Optional[str] = None,
        episode_description: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        pipeline_metrics: metrics.Metrics | None = None,
        call_metrics: Any | None = None,
    ) -> Dict[str, Any]:
        """One completion: semantic transcript clean + JSON title/summary/bullets (Issue #477).

        Returns the same ``summary`` shape as :meth:`summarize` (JSON string
        with ``title``, ``summary``, and ``bullets``).
        """
        if not self._summarization_initialized:
            raise RuntimeError(
                "OllamaProvider summarization not initialized. " "Call initialize() first."
            )

        from ...prompts.store import get_prompt_metadata, render_prompt
        from ...utils.provider_metrics import (
            _safe_openai_retryable,
            ProviderCallMetrics,
            retry_with_metrics,
        )

        max_out = int(getattr(self.cfg, "llm_bundled_max_output_tokens", 16384) or 16384)

        tmpl_kwargs = dict(self.cfg.summary_prompt_params or {})
        system_prompt = render_prompt(
            "ollama/summarization/bundled_clean_summary_system_v1",
            **tmpl_kwargs,
        )
        user_prompt = render_prompt(
            "ollama/summarization/bundled_clean_summary_user_v1",
            transcript=text,
            title=episode_title or "",
            **tmpl_kwargs,
        )

        if call_metrics is None:
            call_metrics = ProviderCallMetrics()
        call_metrics.set_provider_name("ollama")

        def _make_api_call() -> Any:
            return self.client.chat.completions.create(
                model=self.summary_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.summary_temperature,
                max_tokens=max_out,
                response_format={"type": "json_object"},
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
            "summarize_bundled",
        )
        raw = (response.choices[0].message.content or "").strip()

        # Token + cost capture up-front so cost emits in both branches (ADR-100).
        input_tokens = None
        output_tokens = None
        if hasattr(response, "usage") and response.usage:
            pt = getattr(response.usage, "prompt_tokens", None)
            ct = getattr(response.usage, "completion_tokens", None)
            input_tokens = int(pt) if isinstance(pt, (int, float)) else 0
            output_tokens = int(ct) if isinstance(ct, (int, float)) else 0
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
            )

        if pipeline_metrics is not None and input_tokens is not None and output_tokens is not None:
            pipeline_metrics.record_llm_bundled_clean_summary_call(input_tokens, output_tokens)

        # Response-shape guardrail (ADR-099/100, #999/#1003).
        # Note: this fires BEFORE the JSON-parse block below; record the
        # guardrail violation as a Path D parse-failure kind so a sweep that
        # trips the guardrail still surfaces in metrics_report.md.
        try:
            _guardrails.check_chat_response(raw, service="ollama")
        except _guardrails.GuardrailViolation:
            if pipeline_metrics is not None and hasattr(
                pipeline_metrics, "record_llm_bundled_parse_failure"
            ):
                pipeline_metrics.record_llm_bundled_parse_failure("guardrail_violation")
            _record_cost(triggered_guardrail=True)
            raise

        # #912 Path D: bump the bundled-parse-failure counter on every parse
        # failure (per-kind) so future autoresearch sweeps can't silently
        # crown a flaky bundled candidate. Guards with hasattr/getattr so
        # production callers without an eval-side ``pipeline_metrics`` are
        # unaffected (Metrics only flows in from the autoresearch harness).
        def _record_parse_failure(kind: str) -> None:
            if pipeline_metrics is not None and hasattr(
                pipeline_metrics, "record_llm_bundled_parse_failure"
            ):
                pipeline_metrics.record_llm_bundled_parse_failure(kind)

        try:
            data = json.loads(raw, strict=False)
        except json.JSONDecodeError as exc:
            _record_parse_failure("not_valid_json")
            raise ValueError(f"Bundled response is not valid JSON: {exc}") from exc

        if not isinstance(data, dict):
            _record_parse_failure("not_an_object")
            raise ValueError("Bundled JSON must be an object")
        summary_prose = data.get("summary")
        bullets = data.get("bullets")
        if not isinstance(summary_prose, str) or not summary_prose.strip():
            _record_parse_failure("missing_summary")
            raise ValueError("Bundled JSON missing non-empty summary string")
        if not isinstance(bullets, list) or not bullets:
            _record_parse_failure("missing_bullets")
            raise ValueError("Bundled JSON missing non-empty bullets list")

        _record_cost()

        prompt_metadata = {
            "system": get_prompt_metadata(
                "ollama/summarization/" "bundled_clean_summary_system_v1",
                params=tmpl_kwargs,
            ),
            "user": get_prompt_metadata(
                "ollama/summarization/" "bundled_clean_summary_user_v1",
                params={
                    **tmpl_kwargs,
                    "transcript": (text[:100] + "..." if len(text) > 100 else text),
                },
            ),
        }

        return {
            "summary": raw,
            "summary_short": None,
            "metadata": {
                "model": self.summary_model,
                "provider": "ollama",
                "bundled": True,
                "max_output_tokens": max_out,
                "prompts": prompt_metadata,
            },
        }

    def _build_summarization_prompts(
        self,
        text: str,
        episode_title: Optional[str],
        episode_description: Optional[str],
        max_length: int,
        min_length: int,
        custom_prompt: Optional[str],
    ) -> tuple[str, str, Optional[str], str, int, int]:
        """Build system and user prompts for summarization using prompt_store."""
        from ...prompts.store import render_prompt

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

    def generate_insights(
        self,
        text: str,
        episode_title: Optional[str] = None,
        max_insights: int = 5,
        params: Optional[Dict[str, Any]] = None,
        pipeline_metrics: Optional[Any] = None,
    ) -> List[str]:
        """Generate a list of short insight statements from transcript (GIL).

        Uses ollama/insight_extraction/v2 prompt; parses response as one insight per line.
        Returns empty list on failure so GIL can fall back to stub.
        """
        if not self._summarization_initialized:
            logger.warning("Ollama summarization not initialized for generate_insights")
            return []

        from ...prompts.store import render_prompt

        max_insights = max(1, min(int(max_insights), config_constants.GI_MAX_INSIGHTS_CEILING))
        insight_max_tokens = max(
            config_constants.GI_INSIGHT_TOKENS_FLOOR,
            max_insights * config_constants.GI_INSIGHT_TOKENS_EACH,
        )
        text_slice = (text or "").strip()
        if len(text_slice) > 120000:
            text_slice = text_slice[:120000] + "\n\n[Transcript truncated.]"

        try:
            user_prompt = render_prompt(
                "ollama/insight_extraction/v2",
                transcript=text_slice,
                title=episode_title or "",
                max_insights=max_insights,
            )
            system_prompt = (
                "Output only the list of key takeaways, one per line. "
                "No numbering, bullets, or extra text."
            )
            response = self.client.chat.completions.create(
                model=self.summary_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=insight_max_tokens,
                **_ollama_openai_chat_extra_kwargs(self.summary_model),
            )
            content = (response.choices[0].message.content or "").strip()

            # ADR-100 cost-attribution: emit llm_cost in both branches.
            in_tok_gi = None
            out_tok_gi = None
            if hasattr(response, "usage") and response.usage:
                pt = getattr(response.usage, "prompt_tokens", None)
                ct = getattr(response.usage, "completion_tokens", None)
                in_tok_gi = int(pt) if isinstance(pt, (int, float)) else None
                out_tok_gi = int(ct) if isinstance(ct, (int, float)) else None
            gi_cost: Optional[float] = None
            if in_tok_gi is not None and out_tok_gi is not None:
                from ...workflow.helpers import calculate_provider_cost

                gi_cost = calculate_provider_cost(
                    cfg=self.cfg,
                    provider_type="ollama",
                    capability="gi",
                    model=self.summary_model,
                    prompt_tokens=in_tok_gi,
                    completion_tokens=out_tok_gi,
                )

            def _emit_gi_cost(*, triggered_guardrail: bool = False) -> None:
                if in_tok_gi is None or out_tok_gi is None:
                    return
                try:
                    from ...workflow.cost_monitoring import emit_llm_cost_event

                    emit_llm_cost_event(
                        self.cfg,
                        provider="ollama",
                        stage="gi",
                        model=self.summary_model,
                        estimated_cost_usd=float(gi_cost or 0.0),
                        prompt_tokens=in_tok_gi,
                        completion_tokens=out_tok_gi,
                        triggered_guardrail=triggered_guardrail,
                    )
                except Exception:  # noqa: BLE001
                    pass

            # ADR-100: GI is fail-up. Cost emitted in BOTH branches.
            try:
                _guardrails.check_chat_response(content, service="ollama")
            except _guardrails.GuardrailViolation:
                _emit_gi_cost(triggered_guardrail=True)
                raise
            _emit_gi_cost()
            lines = [
                line.strip()
                for line in content.splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
            cleaned = []
            for line in lines:
                s = line.strip()
                if not s:
                    continue
                if len(s) >= 2 and s[0].isdigit() and s[1] in ".)":
                    s = s[2:].strip()
                if s.startswith("- ") or s.startswith("* "):
                    s = s[2:].strip()
                if s:
                    cleaned.append(s)
            if len(cleaned) > max_insights:
                # Overproduction is a signal, not a detail to swallow. Truncating silently is
                # what hid the fact that the model was returning 300+ lines and we were keeping
                # 50 — which read as "it obediently returned exactly the cap".
                logger.warning(
                    "generate_insights: model returned %d insights for a ceiling of %d; "
                    "keeping the first %d. The prompt is not constraining the count.",
                    len(cleaned),
                    max_insights,
                    max_insights,
                )
            return cleaned[:max_insights]
        except _guardrails.GuardrailViolation:
            # ADR-100: GI is fail-up. Propagate so FallbackAware routes.
            raise
        except Exception as e:
            logger.debug("Ollama generate_insights failed: %s", e, exc_info=True)
            return []

    def classify_insights(self, insights: List[str]) -> List[int]:
        """Tier each insight 0-3 for the value gate (see ``gi.value_gate``).

        One call for the whole episode. Raises on failure — the gate fails open and owns the
        decision to keep everything, so this must not paper over a bad response.
        """
        if not insights:
            return []
        if not self._summarization_initialized:
            raise RuntimeError("Ollama summarization not initialized for classify_insights")

        import json as _json

        from ...prompts.store import render_prompt

        listing = "\n".join(f"i{idx}: {text}" for idx, text in enumerate(insights))
        user_prompt = render_prompt("ollama/insight_value_gate/v1", insights=listing)
        response = self.client.chat.completions.create(
            model=self.summary_model,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0.0,
            max_tokens=max(
                config_constants.GI_INSIGHT_TOKENS_FLOOR,
                len(insights) * config_constants.GI_VALUE_GATE_TOKENS_EACH,
            ),
            **_ollama_openai_chat_extra_kwargs(self.summary_model),
        )
        content = (response.choices[0].message.content or "").strip()
        _guardrails.check_chat_response(content, service="ollama")
        tiers = _json.loads(_json_object_from_response(content))
        # Preserve input order; a missing id keeps the insight (tier 3) rather than dropping it.
        return [int(tiers.get(f"i{idx}", 3)) for idx in range(len(insights))]

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
            logger.warning("Ollama summarization not initialized for extract_kg_graph")
            return None
        from ...kg.llm_extract import (
            build_kg_transcript_system_prompt,
            build_kg_user_prompt,
            parse_kg_graph_response,
            resolve_kg_model_id,
            truncate_transcript_for_kg,
        )

        max_topics = min(max(1, max_topics), 20)
        max_entities = min(max(1, max_entities), 50)
        text_slice = truncate_transcript_for_kg(text or "")
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
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_prompt},
                    ],
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
            raw = (response.choices[0].message.content or "").strip()
            return parse_kg_graph_response(raw, max_topics=max_topics, max_entities=max_entities)
        except Exception as e:
            logger.debug("Ollama extract_kg_graph failed: %s", e, exc_info=True)
            return None

    def extract_quotes(
        self,
        transcript: str,
        insight_text: str,
        **kwargs: Any,
    ) -> List[Any]:
        """Extract candidate quote span that supports the insight (GIL QA via LLM)."""
        if not self._summarization_initialized or not (transcript and insight_text):
            return []
        import json

        from ...gi.grounding import QuoteCandidate, resolve_llm_quote_span

        system, user = _render_extract_quote_prompt(transcript, insight_text, self.summary_num_ctx)
        try:
            from ...utils.provider_metrics import (
                _safe_openai_retryable,
                apply_gil_evidence_llm_call_metrics,
                merge_gil_evidence_call_metrics_on_failure,
                openai_compatible_chat_usage_tokens,
                ProviderCallMetrics,
                retry_with_metrics,
            )

            call_metrics = ProviderCallMetrics()
            call_metrics.set_provider_name("ollama")
            pm = kwargs.get("pipeline_metrics")

            def _make_api_call():
                return self.client.chat.completions.create(
                    model=self.summary_model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.0,
                    max_tokens=config_constants.GI_QUOTE_RESPONSE_TOKENS,
                    **_ollama_openai_chat_extra_kwargs(self.summary_model),
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
                merge_gil_evidence_call_metrics_on_failure(call_metrics, pm)
                raise
            in_tok, out_tok = openai_compatible_chat_usage_tokens(response)
            apply_gil_evidence_llm_call_metrics(
                call_metrics,
                pm,
                in_tok,
                out_tok,
                cfg=self.cfg,
                provider_type="ollama",
                model=self.summary_model,
                stage="extract_quotes",
            )
            content = (response.choices[0].message.content or "").strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            obj = json.loads(content)
            quotes_raw = obj.get("quotes") or []
            if isinstance(quotes_raw, str):
                quotes_raw = [quotes_raw]
            # Backward compat: fall back to single quote_text
            if not quotes_raw:
                qt = (obj.get("quote_text") or "").strip()
                if qt:
                    quotes_raw = [qt]
            results_q: list = []
            for qt_str in quotes_raw:
                qt_clean = str(qt_str).strip()
                if not qt_clean:
                    continue
                resolved = resolve_llm_quote_span(transcript, qt_clean)
                if resolved is None:
                    continue
                r_start, r_end, r_verbatim = resolved
                results_q.append(
                    QuoteCandidate(
                        char_start=r_start,
                        char_end=r_end,
                        text=r_verbatim,
                        qa_score=1.0,
                    )
                )
            # Deduplicate: LLMs sometimes return the same quote multiple times
            seen_texts: set = set()
            deduped: list = []
            for q in results_q:
                if q.text not in seen_texts:
                    seen_texts.add(q.text)
                    deduped.append(q)
            results_q = deduped
            return results_q
        except Exception as e:
            logger.debug("Ollama extract_quotes failed: %s", e, exc_info=True)
            return []

    def score_entailment(
        self,
        premise: str,
        hypothesis: str,
        **kwargs: Any,
    ) -> float:
        """Score entailment of hypothesis given premise (GIL NLI via LLM). 0–1.

        The prompt comes from ``ollama/evidence/entailment`` rather than being hardcoded here.
        That is not tidying: the wording *is* the gate. Asked for strict textual entailment (the
        old inline wording), qwen3.5:35b accepted only 40% of the evidence a trusted annotator had
        accepted, and the corpus grounded 13.3% of its insights against the cloud's 91.3%. Asked
        the question the pipeline actually means — "is this quote EVIDENCE for this insight?" — it
        accepts 95%. The template is calibrated against gemini's own judgements
        (scripts/eval/score/entailment_calibration_v1.py); a hardcoded string cannot be.
        """
        if not self._summarization_initialized or not (premise and hypothesis):
            return 0.0

        system, user = _render_entailment_prompt(premise, hypothesis)
        try:
            from ...utils.provider_metrics import (
                _safe_openai_retryable,
                apply_gil_evidence_llm_call_metrics,
                merge_gil_evidence_call_metrics_on_failure,
                openai_compatible_chat_usage_tokens,
                ProviderCallMetrics,
                retry_with_metrics,
            )

            call_metrics = ProviderCallMetrics()
            call_metrics.set_provider_name("ollama")
            pm = kwargs.get("pipeline_metrics")

            def _make_api_call():
                return self.client.chat.completions.create(
                    model=self.summary_model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.0,
                    max_tokens=10,
                    **_ollama_openai_chat_extra_kwargs(self.summary_model),
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
                merge_gil_evidence_call_metrics_on_failure(call_metrics, pm)
                raise
            in_tok, out_tok = openai_compatible_chat_usage_tokens(response)
            apply_gil_evidence_llm_call_metrics(
                call_metrics,
                pm,
                in_tok,
                out_tok,
                cfg=self.cfg,
                provider_type="ollama",
                model=self.summary_model,
                stage="score_entailment",
            )
            content = (response.choices[0].message.content or "0").strip()
            for part in content.replace(",", " ").split():
                try:
                    v = float(part)
                    return max(0.0, min(1.0, v))
                except ValueError:
                    continue
            return 0.0
        except Exception as e:
            logger.debug("Ollama score_entailment failed: %s", e, exc_info=True)
            return 0.0

    def extract_quotes_bundled(
        self,
        transcript: str,
        insight_texts: List[str],
        **kwargs: Any,
    ) -> Dict[int, List[Any]]:
        """Bundle ``extract_quotes`` (#698 Layer A) — Ollama (OpenAI-compat client)."""
        if not self._summarization_initialized or not transcript:
            return {idx: [] for idx in range(len(insight_texts))}
        if not insight_texts:
            return {}

        from ...gi.grounding import QuoteCandidate, resolve_llm_quote_span
        from ...providers.common.bundle_extract_parser import (
            BundleExtractParseError,
            parse_bundled_extract_response,
        )
        from ...providers.common.bundled_prompts import (
            extract_quotes_bundled_max_tokens,
            EXTRACT_QUOTES_BUNDLED_SYSTEM,
            extract_quotes_bundled_user,
            transcript_clip,
        )
        from ...utils.provider_metrics import (
            _safe_openai_retryable,
            apply_gil_evidence_llm_call_metrics,
            merge_gil_evidence_call_metrics_on_failure,
            openai_compatible_chat_usage_tokens,
            ProviderCallMetrics,
            retry_with_metrics,
        )

        system = EXTRACT_QUOTES_BUNDLED_SYSTEM
        user = extract_quotes_bundled_user(transcript_clip(transcript), insight_texts)
        call_metrics = ProviderCallMetrics()
        call_metrics.set_provider_name("ollama")
        pm = kwargs.get("pipeline_metrics")
        max_out = extract_quotes_bundled_max_tokens(len(insight_texts))

        def _make_api_call() -> Any:
            return self.client.chat.completions.create(
                model=self.summary_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
                max_tokens=max_out,
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
            )
        except Exception:
            merge_gil_evidence_call_metrics_on_failure(call_metrics, pm)
            raise
        in_tok, out_tok = openai_compatible_chat_usage_tokens(response)
        apply_gil_evidence_llm_call_metrics(
            call_metrics,
            pm,
            in_tok,
            out_tok,
            cfg=self.cfg,
            provider_type="ollama",
            model=self.summary_model,
            stage="extract_quotes",
        )

        content = (response.choices[0].message.content or "").strip()
        try:
            parsed = parse_bundled_extract_response(content, expected_count=len(insight_texts))
        except BundleExtractParseError as exc:
            logger.debug("Ollama extract_quotes_bundled parse failed: %s", exc)
            raise

        out: Dict[int, List[Any]] = {}
        for idx in range(len(insight_texts)):
            quote_strings = parsed.get(idx, [])
            seen: set = set()
            candidates: List[Any] = []
            for qt_str in quote_strings:
                qt_clean = str(qt_str).strip()
                if not qt_clean:
                    continue
                resolved = resolve_llm_quote_span(transcript, qt_clean)
                if resolved is None:
                    continue
                r_start, r_end, r_verbatim = resolved
                if r_verbatim in seen:
                    continue
                seen.add(r_verbatim)
                candidates.append(
                    QuoteCandidate(
                        char_start=r_start,
                        char_end=r_end,
                        text=r_verbatim,
                        qa_score=1.0,
                    )
                )
            out[idx] = candidates
        return out

    def score_entailment_bundled(
        self,
        pairs: List[Tuple[str, str]],
        chunk_size: int = 15,
        **kwargs: Any,
    ) -> Dict[int, float]:
        """Bundle ``score_entailment`` (#698 Layer B) — Ollama."""
        if not self._summarization_initialized or not pairs:
            return {}
        chunk_size = max(1, int(chunk_size))
        out: Dict[int, float] = {}
        pm = kwargs.get("pipeline_metrics")
        for chunk_start in range(0, len(pairs), chunk_size):
            chunk = pairs[chunk_start : chunk_start + chunk_size]
            chunk_scores = self._score_entailment_bundled_chunk(
                chunk_pairs=chunk, pipeline_metrics=pm
            )
            for local_idx, score in chunk_scores.items():
                out[chunk_start + local_idx] = score
            if pm is not None and hasattr(pm, "gi_evidence_score_entailment_bundled_pairs_total"):
                pm.gi_evidence_score_entailment_bundled_pairs_total += len(chunk)
        return out

    def _score_entailment_bundled_chunk(
        self,
        chunk_pairs: List[Tuple[str, str]],
        pipeline_metrics: Optional[Any],
    ) -> Dict[int, float]:
        """One bundled NLI Ollama call."""
        from ...providers.common.bundle_nli_parser import (
            BundleNliParseError,
            parse_bundled_nli_response,
        )
        from ...providers.common.bundled_prompts import (
            score_entailment_bundled_max_tokens,
            SCORE_ENTAILMENT_BUNDLED_SYSTEM,
            score_entailment_bundled_user,
        )
        from ...utils.provider_metrics import (
            _safe_openai_retryable,
            apply_gil_evidence_llm_call_metrics,
            merge_gil_evidence_call_metrics_on_failure,
            openai_compatible_chat_usage_tokens,
            ProviderCallMetrics,
            retry_with_metrics,
        )

        system = SCORE_ENTAILMENT_BUNDLED_SYSTEM
        user = score_entailment_bundled_user(chunk_pairs)
        call_metrics = ProviderCallMetrics()
        call_metrics.set_provider_name("ollama")
        max_out = score_entailment_bundled_max_tokens(len(chunk_pairs))

        def _make_api_call() -> Any:
            return self.client.chat.completions.create(
                model=self.summary_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
                max_tokens=max_out,
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
            )
        except Exception:
            merge_gil_evidence_call_metrics_on_failure(call_metrics, pipeline_metrics)
            raise
        in_tok, out_tok = openai_compatible_chat_usage_tokens(response)
        apply_gil_evidence_llm_call_metrics(
            call_metrics,
            pipeline_metrics,
            in_tok,
            out_tok,
            cfg=self.cfg,
            provider_type="ollama",
            model=self.summary_model,
            stage="score_entailment",
        )

        content = (response.choices[0].message.content or "").strip()
        try:
            return parse_bundled_nli_response(content, expected_count=len(chunk_pairs))
        except BundleNliParseError as exc:
            logger.debug("Ollama score_entailment_bundled parse failed: %s", exc)
            raise

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
