"""Unified OpenAI provider for transcription, speaker detection, and summarization.

This module provides a single OpenAIProvider class that implements all three protocols:
- TranscriptionProvider (using Whisper API)
- SpeakerDetector (using GPT API)
- SummarizationProvider (using GPT API)

This unified approach matches the pattern of ML providers, where a single
provider type handles multiple capabilities using shared API client.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, cast, Dict, List, Optional, Set, TYPE_CHECKING

from ... import config, config_constants, models

if TYPE_CHECKING:
    from ...models import Episode
else:
    Episode = models.Episode  # type: ignore[assignment]
from ...cleaning.base import TranscriptCleaningProcessor
from ...utils.cleaning_max_tokens import (
    _CLEANING_WORD_FACTOR,
    clamp_cleaning_max_tokens,
    estimate_cleaning_output_tokens,
    OPENAI_CLEANING_MAX_TOKENS,
)
from ...utils.log_redaction import format_exception_for_log
from ...utils.provider_metadata import (
    extract_region_from_endpoint,
    log_provider_metadata,
    validate_api_key_format,
    warn_if_truncated,
)
from ...utils.timeout_config import get_openai_client_timeout, get_single_chat_call_timeout
from ...workflow import metrics
from .. import guardrails as _guardrails, insight_salvage as _insight_salvage
from ..capabilities import ProviderCapabilities

logger = logging.getLogger(__name__)


#: Vendor-published context windows, used ONLY as the fallback when nothing declared one and the
#: server cannot be asked (#2050).
#:
#: This is not the global constant that was deleted. That one applied ONE model's window to EVERY
#: provider — a 32,768 DGX serving flag clipping a 1M-token Gemini. These are per-namespace figures
#: the vendor publishes for the models on that endpoint, which is knowledge, not a guess.
#:
#: Two namespaces are deliberately absent, and ``None`` for them means "do not bound":
#:   ``vllm``    — self-hosted; the window is whatever ``--max-model-len`` was set to, which no
#:                 published figure can tell us. Discovered from ``GET /v1/models`` instead.
#:   ``litellm`` — a gateway in front of arbitrary upstream models; there is no single window.
#:                 Declare ``litellm_max_context_tokens`` per profile if the upstream is pinned.
#:
#: Prefer declaring the real served window over relying on these: a budget derived from an
#: overstated window is a prompt the server rejects, or — on a backend that truncates silently —
#: a corpus artifact built from half a transcript.
_PUBLISHED_CONTEXT_TOKENS: Dict[str, int] = {
    "openai": 128_000,
    "qwen": 131_072,
    "groq": 131_072,
}


def _kg_reply_snippet(raw: str, head: int = 300, tail: int = 200) -> str:
    """A bounded, log-safe view of a model reply: whole if short, else head + tail.

    Enough to tell a truncated JSON, a prose answer, a fenced block or an empty reply apart, without
    writing a multi-kilobyte reply into every failure line.
    """
    if len(raw) <= head + tail:
        return repr(raw)
    return f"{raw[:head]!r} … [{len(raw) - head - tail} chars] … {raw[-tail:]!r}"


def _bump_metric(metrics: Optional[Any], name: str, amount: int = 1) -> None:
    """Increment a counter on the pipeline metrics object; never raise on telemetry."""
    if metrics is None or not amount:
        return
    try:
        setattr(metrics, name, getattr(metrics, name, 0) + amount)
    except Exception:  # noqa: BLE001 - telemetry must never break a run
        pass


#: Smallest output budget worth sending. Below this a reply cannot be a usable JSON object or
#: a line of insights, so clamping to it is dishonest — better to let the call fail loudly than
#: to "succeed" with a budget that guarantees a truncated, unparsable answer.
_MIN_USEFUL_OUTPUT_TOKENS = 256

#: "maximum context length is 32768 tokens" — vLLM and the OpenAI API both phrase it this way.
_CONTEXT_LIMIT_RE = re.compile(r"maximum context length is\s+(\d+)\s+tokens", re.IGNORECASE)
#: "your prompt contains at least 30721 input tokens"
_PROMPT_TOKENS_RE = re.compile(r"prompt contains at least\s+(\d+)\s+", re.IGNORECASE)


def _learn_context_limit_from_error(msg: str) -> Optional[int]:
    """The model's context limit, parsed out of a 400 that complained about it (#1893)."""
    m = _CONTEXT_LIMIT_RE.search(msg or "")
    if not m:
        return None
    try:
        n = int(m.group(1))
    except (TypeError, ValueError):  # pragma: no cover - regex guarantees digits
        return None
    return n if n > 0 else None


def _context_clamp_token_budget(
    kwargs: Dict[str, Any], context_limit: Optional[int], prompt_hint: str = ""
) -> bool:
    """Shrink the request's output budget so prompt + output fits *context_limit*.

    Returns True when a clamp was applied (the caller may retry), False otherwise.

    The prompt size is taken from the server's own error text when retrying, and otherwise
    ESTIMATED from the message characters. The estimate is deliberately pessimistic (3.5
    chars/token — real English is nearer 4): over-estimating shrinks the reply a little, while
    under-estimating sends a request that 400s, and one of those two costs an entire episode's
    stage while the other costs a few tokens of headroom.

    Refuses to clamp below ``_MIN_USEFUL_OUTPUT_TOKENS``. If the prompt alone leaves less room
    than that, no output budget makes the request valid — the input has to shrink instead, which
    is a caller decision (transcript clipping / smaller batches), not something to paper over
    here. That case logs and leaves the request untouched so it fails with the server's own
    explicit message rather than a mystery empty reply.
    """
    if not context_limit or context_limit <= 0:
        return False
    key = "max_completion_tokens" if "max_completion_tokens" in kwargs else "max_tokens"
    requested = kwargs.get(key)
    if not isinstance(requested, int) or requested <= 0:
        return False

    prompt_tokens: Optional[int] = None
    m = _PROMPT_TOKENS_RE.search(prompt_hint or "")
    if m:
        try:
            prompt_tokens = int(m.group(1))
        except (TypeError, ValueError):  # pragma: no cover
            prompt_tokens = None
    if prompt_tokens is None:
        chars = sum(len(str(msg.get("content") or "")) for msg in kwargs.get("messages") or [])
        prompt_tokens = int(chars / 3.5) if chars else 0

    room = context_limit - prompt_tokens
    if room >= requested:
        return False
    if room < _MIN_USEFUL_OUTPUT_TOKENS:
        logger.error(
            "context limit %d cannot fit this request: the prompt alone is ~%d tokens, leaving "
            "%d for the reply. Clamping the output budget cannot help — the INPUT must shrink "
            "(smaller transcript clip or a smaller batch). Sending unchanged so the failure is "
            "the server's explicit message.",
            context_limit,
            prompt_tokens,
            room,
        )
        return False
    logger.warning(
        "clamping output budget %d -> %d so prompt (~%d) + output fits the %d-token context.",
        requested,
        room,
        prompt_tokens,
        context_limit,
    )
    kwargs[key] = room
    return True


# Protocol types imported for type hints (used in docstrings and type annotations)
# from ..speaker_detectors.base import SpeakerDetector  # noqa: F401
# from ..summarization.base import SummarizationProvider  # noqa: F401
# from ..transcription.base import TranscriptionProvider  # noqa: F401

# Use canonical default from speaker_detection (Issue #428: typed placeholder, not "Guest")

# Pricing for OpenAI models lives in ``config/pricing_assumptions.yaml`` (#651).
# See ``get_pricing()`` below — it delegates to the YAML loader so there is a
# single source of truth. CI guard
# (``scripts/validate/check_profile_pricing_coverage.py``) fails PRs that
# reference a model without a matching YAML row.


# Models that reject any explicit temperature other than the default (1). gpt-5.5 does;
# gpt-5.4-mini
# gpt-5.4-mini does not. Seeded here, grown at runtime from the API error (see _chat_create),
# so a future model that fixes temperature self-heals after one failed call, not a crash.
_TEMPERATURE_FIXED_MODELS = frozenset({"gpt-5.5", "gpt-5.5-pro"})


from ..common.summary_use_case import (
    build_use_case_prompts,
    reply_budget,
    SummaryUseCase,
    use_case_from_params,
    use_case_prompt_metadata,
    use_case_stage,
)
from ..common.token_budget import fit_text_to_token_budget

# RFC-115: transcript-prefix caching lives in the shared, provider-family-agnostic module so every
# provider (base siblings + grok/mistral/ollama/anthropic/gemini) relocates the transcript
# identically.
from ..common.transcript_cache import openai_style_messages as _openai_style_messages
from ._openai_provider_calls import (  # noqa: F401 - re-exported: moved there with S1
    _openai_chat_usage_tokens,
    _openai_response_cost_usd,
    _repetition_signal,
)
from .openai_provider_gi_naming import (  # noqa: F401 - re-exported: moved there with S1
    _as_known_host,
    _BUNDLE_CAPTURE_DIR,
    _BUNDLE_CAPTURE_MAX_CHARS,
    _BUNDLE_KEY_RE,
    _capture_bundle_failure,
    _clip_speaker_description,
    _DECODING_LOOP_MIN_REPEATS,
    _fold_name,
    _is_decoding_loop,
    _JSON_STRING_RE,
    _record_openai_summarization_call,
    _salvage_closed_quotes,
    _SPEAKER_DESCRIPTION_MAX_CHARS,
    _tail_degeneracy,
    OpenAICompatibleProviderGiNamingMethods,
)


class OpenAICompatibleProvider(OpenAICompatibleProviderGiNamingMethods):
    """OpenAI-compatible transport shared by OpenAIProvider (OpenAI-native) and VLLMProvider
    (DGX-local open models). They are SIBLINGS, not parent/child — vLLM serves a wide family of
    non-OpenAI models, so it is not modelled as "an OpenAI thing" (ADR-147). Identity is set by
    the ``_CONFIG_NS`` / ``_TELEMETRY_PROVIDER`` / ``_PROVIDER_LABEL`` class attrs and the
    ``_authenticate`` / ``_resolve_api_key`` hooks, so a sibling changes behaviour by overriding
    those, not by copying method bodies.

    Implements TranscriptionProvider, SpeakerDetector, and SummarizationProvider over the OpenAI
    wire API. All capabilities share one client, initialized once and reused.
    """  # noqa: E501

    # --- provider identity (parameterized so a sibling like VLLMProvider can share this
    # OpenAI-compatible transport without being an "OpenAI thing"; see ADR-147). Reads of
    # config fields use the ``{_CONFIG_NS}_*`` prefix and telemetry uses ``_TELEMETRY_PROVIDER``,
    # so a subclass changes identity by overriding these three, not by copying method bodies. ---
    _CONFIG_NS: str = "openai"  # config-field prefix: {ns}_summary_model, {ns}_api_base, …
    _TELEMETRY_PROVIDER: str = "openai"  # provider/service label in metrics + guardrails
    # The prompt folder a use case's logical prompt names resolve under; the family's
    # summarization prompts live under openai/ too, whatever the sibling.
    _PROMPT_NS: str = "openai"
    _PROVIDER_LABEL: str = "OpenAI"  # human label for log_provider_metadata
    # Output-token cap for transcript cleaning. OpenAI's per-model output cap is conservative;
    # gateway subclasses (litellm) route to models that support more, so they raise this. A cap
    # too low truncates cleaning (finish_reason=length) and the guardrail discards the result.
    _CLEANING_MAX_TOKENS_CAP: int = OPENAI_CLEANING_MAX_TOKENS
    # Default API endpoint used when no ``{ns}_api_base`` is configured. ``None`` = fall back to the
    # OpenAI SDK default (api.openai.com). A vendor sibling (deepseek) sets its own vendor endpoint
    # so the "direct, no gateway" path needs zero config; pointing ``{ns}_api_base`` at a LiteLLM
    # gateway still overrides it, so the same class serves both direct and via-gateway. ADR-147.
    _DEFAULT_API_BASE: Optional[str] = None

    cleaning_processor: TranscriptCleaningProcessor  # Type annotation for mypy

    def _resolve_api_key(self, cfg: "config.Config") -> Optional[str]:
        """Bearer for the OpenAI-compatible client. Base = the ``{ns}_api_key`` field
        (OpenAI-native). VLLMProvider overrides to make the bearer optional — a local vLLM served
        without auth accepts any bearer, so a dummy is supplied when none is configured. ADR-147."""
        return getattr(cfg, f"{self._CONFIG_NS}_api_key", None)

    def _authenticate(self, cfg: "config.Config") -> None:
        """Validate credentials at construction. Base = OpenAI-native: a key is REQUIRED and must
        look like an OpenAI key. VLLMProvider overrides this to a no-op (a local vLLM bearer is
        optional). ADR-147."""
        api_key = getattr(cfg, f"{self._CONFIG_NS}_api_key", None)
        if not api_key:
            raise ValueError(
                "OpenAI API key required for OpenAI provider. "
                "Set OPENAI_API_KEY environment variable or openai_api_key in config."
            )
        is_valid, _ = validate_api_key_format(
            api_key,
            self._PROVIDER_LABEL,
            expected_prefixes=["sk-", "sk-proj-"],
        )
        if not is_valid:
            # Do not log validation detail: CodeQL taints any message from this API-key path.
            logger.warning(
                "OpenAI API key validation failed (missing, too short, or wrong prefix); "
                "credentials are never logged."
            )

    # Join keys for the LiteLLM gateway's Langfuse callback (session_id = run, trace_metadata =
    # run / episode / OTel trace). Only providers whose server takes a nested ``metadata`` turn
    # this on — the DGX vLLM and the gateway do (verified 2026-10-14); OpenAI's own ``metadata``
    # takes flat strings only.
    _SENDS_JOIN_METADATA: bool = False

    def __init__(self, cfg: config.Config):
        """Initialize unified OpenAI provider.

        Args:
            cfg: Configuration object with settings for all three capabilities

        Raises:
            ValueError: If OpenAI API key is not provided
            ImportError: If openai package is not installed (core dependency)
        """
        ns = self._CONFIG_NS  # config-field prefix; "openai" here, overridden by siblings (ADR-147)
        # Lazy import to allow unit tests without openai installed (Issue #405)
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package is required for OpenAI provider. "
                "Install the project (OpenAI SDK is a core dependency), e.g. pip install -e ."
            ) from exc

        # Credentials: the base validates an OpenAI-native key (required + sk- prefix). A sibling
        # like VLLMProvider overrides _authenticate (bearer optional for a local vLLM). ADR-147.
        self._authenticate(cfg)

        self.cfg = cfg

        # Set up transcript cleaning processor based on strategy (Issue #418)
        from ... import stage_hooks

        cleaning_strategy = getattr(cfg, "transcript_cleaning_strategy", "hybrid")
        self.cleaning_processor = stage_hooks.transcript_cleaner(cleaning_strategy)

        # Cleaning model settings (config default: gpt-4o-mini; override for cheaper tiers)
        self.cleaning_model = getattr(
            cfg, f"{ns}_cleaning_model", config.DEFAULT_OPENAI_CLEANING_MODEL
        )
        self.cleaning_temperature = getattr(cfg, f"{ns}_cleaning_temperature", 0.2)

        # Suppress verbose OpenAI SDK debug logs (they're too long and clutter the output)
        # Set OpenAI SDK loggers to WARNING level when root logger is DEBUG
        # This keeps our debug logs visible while hiding OpenAI's verbose request/response logs
        root_logger = logging.getLogger()
        root_level = root_logger.level if root_logger.level else logging.INFO
        if root_level <= logging.DEBUG:
            openai_loggers = [
                "openai",
                "openai._base_client",
                "openai.api_resources",
                "httpx",
                "httpcore",
                "httpcore.connection",
                "httpcore.http11",
            ]
            for logger_name in openai_loggers:
                openai_logger = logging.getLogger(logger_name)
                openai_logger.setLevel(logging.WARNING)

        # Support custom base_url for E2E testing with mock servers, and a per-sibling default
        # endpoint (_DEFAULT_API_BASE) so a vendor sibling talks direct with no config (ADR-147).
        client_kwargs: dict[str, Any] = {"api_key": self._resolve_api_key(cfg)}
        _api_base = getattr(cfg, f"{ns}_api_base", None) or self._DEFAULT_API_BASE
        if _api_base:
            client_kwargs["base_url"] = _api_base

        # Read timeout must cover Whisper and long chat calls (cleaning, summarize);
        # cfg.timeout alone is often too low (RSS/download tuning).
        client_kwargs["timeout"] = get_openai_client_timeout(cfg)

        # #1852: the client-level read timeout above is the MAX of the summarization and
        # transcription timeouts (one client serves Whisper + chat), so for CHAT it is >= the 1200s
        # summarization deadline — a single chat call can sit on a silent socket past the soft
        # deadline (the "DEADLINE EXCEEDED ... STILL RUNNING" alert). Bound each chat request with a
        # PER-REQUEST timeout instead (see _chat_create). Kept a separate value so Whisper keeps the
        # longer client window; per-CALL, so a legitimately long AGGREGATE stage (summary+GI+KG, an
        # accepted overrun-is-success case) is not hard-aborted.
        #
        # #1894: that per-request value used to be the deadline ITSELF, which is not a bound at
        # all in practice — one stuck call could consume the entire budget, and the only symptom
        # was the deadline alert after the fact, unable to say which of ~40 calls was at fault.
        # Now a FRACTION of the deadline, so a hang surfaces in minutes and the RFC-106 ladder
        # still has budget left to fail over. See get_single_chat_call_timeout.
        self._chat_request_timeout = get_single_chat_call_timeout(cfg)

        self.client = OpenAI(**client_kwargs)

        # vLLM / OpenAI-compatible: inject a fixed extra_body into every
        # chat.completions.create call (#960). Required for Qwen3.5/3.6 family
        # served by vLLM, which needs ``chat_template_kwargs={enable_thinking:
        # False}`` to produce a clean summary instead of leaking reasoning prose.
        # Wrapping the client method here avoids fanning the kwarg through 17
        # call sites scattered across the provider.
        # #2051: the vendor's researched sampling knob, routed from the registry rather than
        # hand-copied into a profile. `presence_penalty` is the documented mitigation for the
        # endless repetition behind #2053; it sat declared-but-unread for the life of the DGX
        # deployment because `summary_extra` had no consumer.
        #
        # `setdefault` semantics, NOT override: a call site that has a reason to pass its own
        # penalty keeps it. Nothing does today, but a provider-wide default that silently wins
        # over an explicit argument is the same class of bug as the one being fixed.
        _presence_penalty = getattr(cfg, f"{ns}_presence_penalty", None)
        _fixed_extra_body = getattr(cfg, f"{ns}_extra_body", None)
        if _fixed_extra_body or _presence_penalty is not None:
            _orig_chat_create = self.client.chat.completions.create
            _frozen_extra = dict(_fixed_extra_body or {})
            _frozen_penalty = _presence_penalty

            def _chat_create_with_vendor_defaults(**kwargs: Any) -> Any:
                if _frozen_extra:
                    merged = dict(kwargs.get("extra_body") or {})
                    merged.update(_frozen_extra)
                    kwargs["extra_body"] = merged
                if _frozen_penalty is not None:
                    kwargs.setdefault("presence_penalty", _frozen_penalty)
                return _orig_chat_create(**kwargs)

            _patched = _chat_create_with_vendor_defaults
            self.client.chat.completions.create = _patched  # type: ignore[method-assign]
            if _frozen_penalty is not None:
                logger.info(
                    "%s: presence_penalty=%s applied to every chat call from the registry (#2051)",
                    self._PROVIDER_LABEL,
                    _frozen_penalty,
                )

        # Log non-sensitive provider metadata (for debugging)
        # Extract region from base_url if possible
        region = extract_region_from_endpoint(_api_base)
        log_provider_metadata(
            provider_name=self._PROVIDER_LABEL,
            organization=getattr(cfg, f"{ns}_organization", None),
            base_url=_api_base,
            region=region,
        )

        # Transcription settings
        self.transcription_model = getattr(cfg, f"{ns}_transcription_model", "whisper-1")

        # Speaker detection settings
        self.speaker_model = getattr(cfg, f"{ns}_speaker_model", "gpt-4o-mini")
        self.speaker_temperature = getattr(cfg, f"{ns}_temperature", 0.3)
        _detection_t = getattr(cfg, "speaker_detection_temperature", None)
        if isinstance(_detection_t, (int, float)) and not isinstance(_detection_t, bool):
            self.speaker_temperature = float(_detection_t)

        # Summarization settings
        self.summary_model = getattr(cfg, f"{ns}_summary_model", "gpt-4o-mini")
        _insight_override = getattr(cfg, f"{ns}_insight_model", None)
        if isinstance(_insight_override, str) and _insight_override.strip():
            self.insight_model = _insight_override.strip()
        else:
            self.insight_model = self.summary_model
        self.summary_temperature = getattr(cfg, f"{ns}_temperature", 0.3)
        _seed = getattr(cfg, f"{ns}_summary_seed", None)
        self.summary_seed: Optional[int] = int(_seed) if _seed is not None else None
        # The served context window, in tokens.
        #
        # 128,000 is the OpenAI-native default and is WRONG for every other deployment on this
        # transport — the DGX vLLM serves 32,768 (#2050). It is a starting value, not a fact:
        # subclasses that can ask their server override it (VLLMProvider reads `max_model_len`
        # off `GET /v1/models`), and `_context_limits` corrects it again from any 400 that names
        # a real limit. Budget from `transcript_budget_chars()`, never from this attribute
        # directly, so the correction is always applied.
        self.max_context_tokens: Optional[int] = int(
            getattr(cfg, f"{ns}_max_context_tokens", 0) or 0
        ) or _PUBLISHED_CONTEXT_TOKENS.get(ns)

        # Initialization state
        self._transcription_initialized = False
        self._speaker_detection_initialized = False
        self._summarization_initialized = False

        # Mark provider as thread-safe (API clients can be shared across threads)
        # API providers handle rate limiting internally, so parallelism isn't needed
        self._requires_separate_instances = False

        # RFC-115: relocate the transcript to the leading system block so it prefix-caches across
        # an episode's LLM stages (default on). Off falls back to the exact legacy layout.
        self._cache_transcript_prefix = bool(getattr(cfg, "cache_transcript_prefix", True))

        # Models that reject any explicit ``temperature`` other than the default (1). gpt-5.4-mini
        # still accepts temperature=0, but gpt-5.5 rejects it with a 400 "temperature does not
        # support 0 ... only the default (1)". Seeded and grown at runtime — see _chat_create.
        self._temp_fixed_at_default: Set[str] = set(_TEMPERATURE_FIXED_MODELS)
        # #1893: model -> served context limit, LEARNED from a 400 rather than configured.
        # Same self-healing shape as _temp_fixed_at_default above: the first call for a
        # model may pay a retry, every later one is clamped up front. A hand-maintained
        # per-model table would go stale silently the moment a model is swapped behind an
        # alias, and the server already knows the true number.
        self._context_limits: Dict[str, int] = {}
        # Cleared the first time a server rejects the join metadata, so it is not sent again.
        self._join_metadata_ok = self._SENDS_JOIN_METADATA

    def _attach_join_metadata(self, kwargs: Dict[str, Any], stage: Optional[str] = None) -> None:
        """Add the Langfuse join keys to ``extra_body.metadata`` (gateway → Langfuse).

        LiteLLM's Langfuse logger pops these out of the request metadata: ``session_id`` (the run),
        ``trace_id`` (the OTel trace id, so a VictoriaTraces id opens the Langfuse trace),
        ``trace_name``, ``generation_name`` (the stage) and ``trace_metadata``. Without
        ``trace_id`` it keys the trace on the session, and one run became one trace.
        Never overwrites a key the caller set; adds nothing when nothing is known.
        """
        if not self._join_metadata_ok:
            return
        from ...utils import correlation

        run_id = correlation.get_run_id()
        episode_id = correlation.get_episode_id()
        raw_trace = correlation.current_trace_id()
        trace_id = raw_trace if raw_trace and raw_trace != "-" else None
        trace_md = {
            k: v
            for k, v in (
                ("run_id", run_id),
                ("episode_id", episode_id),
                ("otel_trace_id", trace_id),
            )
            if v
        }
        if not trace_md and not stage:
            return
        body = dict(kwargs.get("extra_body") or {})
        md = dict(body.get("metadata") or {})
        if run_id:
            md.setdefault("session_id", run_id)
        if trace_id:
            md.setdefault("trace_id", trace_id)
            md.setdefault("trace_name", f"episode {episode_id}" if episode_id else run_id or stage)
        elif run_id:
            # No OTel span yet (run-level calls): the trace is keyed on the session; unnamed, it
            # reads "litellm-acompletion" in Langfuse.
            md.setdefault("trace_name", f"run {run_id}")
        if stage:
            md.setdefault("generation_name", stage)
        if trace_md:
            md.setdefault("trace_metadata", trace_md)
        # LiteLLM maps any ``trace_<key>`` onto the trace: the release, as GlitchTip has it.
        release = os.environ.get("PODCAST_RELEASE", "").strip()
        if release:
            md.setdefault("trace_release", release)
        body["metadata"] = md
        kwargs["extra_body"] = body

    def _chat_create(self, **kwargs: Any) -> Any:
        """``chat.completions.create`` that survives models which fixed ``temperature`` at default.

        Every OpenAI call routes through here. If the model is known to allow only the default
        temperature, the parameter is dropped up front. If a model we did not know about rejects a
        non-default temperature — a 400 that names ``temperature`` — the model is recorded and the
        call retried once without it. So gpt-5.5 (and whatever comes next) self-heals instead of
        crashing the run at episode 1, and only the FIRST call per model pays the retry.
        """
        # #1852: bound every chat request with a per-request transport timeout tied to the
        # summarization deadline, so a single call cannot sit on a silent socket past it (the shared
        # client read timeout is max(summarization, transcription) = >= the deadline). setdefault so
        # an explicit per-call value still wins; kwargs is mutated once so the temperature-retry
        # below carries the same timeout.
        # ``_lf_stage``: the call's stage, only for the Langfuse generation name; never sent.
        stage = kwargs.pop("_lf_stage", None)
        kwargs.setdefault("timeout", self._chat_request_timeout)
        model = kwargs.get("model")
        if model in self._temp_fixed_at_default:
            kwargs.pop("temperature", None)
        # #1893: clamp the OUTPUT budget against a context limit we have learned for this model,
        # before the call. Prevention rather than reaction — see _learn_context_limit_from_error.
        _context_clamp_token_budget(kwargs, self._context_limits.get(str(model or "")))
        self._attach_join_metadata(kwargs, stage)
        try:
            return self.client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 — inspect, re-raise unless it is a known case
            msg = str(exc).lower()
            # A server that validates `metadata` as flat strings (older vLLM) 400s the join keys:
            # drop them for this provider and retry, so telemetry can never fail an LLM call.
            if "metadata" in msg and "metadata" in (kwargs.get("extra_body") or {}):
                self._join_metadata_ok = False
                kwargs["extra_body"] = {
                    k: v for k, v in kwargs["extra_body"].items() if k != "metadata"
                }
                if not kwargs["extra_body"]:
                    kwargs.pop("extra_body")
                logger.warning(
                    "%s: server rejected the Langfuse join metadata; sending calls without it",
                    self._PROVIDER_LABEL,
                )
                return self._chat_create(**kwargs)
            # #1893: "maximum context length is N tokens ... requested M output tokens and your
            # prompt contains at least P input tokens". Production hit this 56 times in a 13
            # SECOND burst — every episode in flight failing the same way, because nothing in
            # the pipeline checked that prompt + output fits before sending.
            #
            # The limit is learned from the server's own message rather than configured. A
            # hand-maintained per-model context table is exactly the kind of thing that goes
            # stale silently the moment a model is swapped behind an alias, and vLLM already
            # tells us the true number — it just tells us by failing. Learn it once, clamp
            # every subsequent call for that model, and retry this one.
            learned = _learn_context_limit_from_error(msg)
            if learned:
                # LEARNING and RETRYING are separate concerns, and conflating them was a bug
                # (#1893). This used to retry only when `learned` DIFFERED from the cached value,
                # so the FIRST 400 for a model recovered and every subsequent one fell straight
                # through to `raise`. Against 565 rejections over 30 days the recovery could fire
                # at most once per process.
                #
                # It is exactly backwards. Once the limit is known, a 400 means the up-front clamp
                # UNDER-counted this particular prompt — and the server's message carries its exact
                # token count, so that is when a retry is most likely to succeed, not least.
                newly_learned = (
                    isinstance(model, str) and self._context_limits.get(model) != learned
                )
                if isinstance(model, str):
                    self._context_limits[model] = learned
                if newly_learned:
                    logger.warning(
                        "%s: context limit for %s learned as %d from a 400 (prompt + requested "
                        "output exceeded it). Clamping the output budget and retrying; later "
                        "calls for this model are clamped up front.",
                        self._PROVIDER_LABEL,
                        model,
                        learned,
                    )
                # `prompt_hint=msg` makes the clamp use the server's EXACT prompt size instead of
                # our 3.5 chars/token estimate — the one place the budget is measured, not guessed.
                if _context_clamp_token_budget(kwargs, learned, prompt_hint=msg):
                    if not newly_learned:
                        logger.warning(
                            "%s: 400 on %s despite a known %d-token limit — the up-front estimate "
                            "under-counted this prompt. Re-clamping from the server's exact count "
                            "and retrying.",
                            self._PROVIDER_LABEL,
                            model,
                            learned,
                        )
                    return self.client.chat.completions.create(**kwargs)
            if (
                "temperature" in kwargs
                and "temperature" in msg
                and (
                    "does not support" in msg
                    or "only the default" in msg
                    or "unsupported_value" in msg
                )
            ):
                if isinstance(model, str):
                    self._temp_fixed_at_default.add(model)
                kwargs.pop("temperature", None)
                return self.client.chat.completions.create(**kwargs)
            raise

    def count_tokens(self, text: str) -> Optional[int]:
        """Exact token count from the backend's tokenizer, or ``None`` if it has none (#2050).

        ``None`` is not a failure — most backends expose no tokenizer, and the caller falls back
        to a pessimistic character estimate. Overridden by VLLMProvider, which has ``/tokenize``.
        """
        return None

    def served_context_tokens(self) -> Optional[int]:
        """The best available figure for this deployment's context window, in tokens.

        Three sources, most-authoritative first (#2050):

        1. ``_context_limits[model]`` — learned from a 400 that named the real limit. The server
           said this, so nothing beats it.
        2. ``max_context_tokens`` — declared: discovered from ``GET /v1/models`` where the backend
           advertises it (VLLMProvider), else configured, else the OpenAI-native default.

        Before this existed the declared value was the hardcoded 128,000 on this class, the learned
        value started empty and was consulted only inside the retry path, and neither was ever used
        to size a prompt. So the transport knew the real window and budgeted from a constant anyway.
        """
        learned = self._context_limits.get(self.summary_model) if self.summary_model else None
        if isinstance(learned, int) and learned > 0:
            return learned
        return self.max_context_tokens

    def clip_transcript_to_budget(
        self,
        text: str,
        *,
        response_tokens: int,
        instruction_tokens: int = config_constants.GI_QUOTE_INSTRUCTION_TOKEN_RESERVE,
        label: str = "transcript",
    ) -> str:
        """Clip *text* to what this deployment can actually hold — COUNTED where possible (#2050).

        Prefers the backend's own tokenizer over any chars-per-token constant. That rate varies
        with content (3.48 on real corpus transcripts vs 4.44 on the simpler-vocabulary fixtures),
        and every context-overflow bug in this repo came from a constant being fractionally wrong
        in the optimistic direction — 508 rejections in 30 days, each over by exactly one token.

        Returns *text* unchanged when the served window is unknown: not knowing is self-correcting
        via the server's 400, whereas guessing is not.
        """
        window = self.served_context_tokens()
        if not window:
            return text
        max_prompt_tokens = window - response_tokens - instruction_tokens
        return fit_text_to_token_budget(
            text,
            max_prompt_tokens,
            self.count_tokens,
            label=label,
        )

    def transcript_budget_chars(
        self,
        *,
        response_tokens: int,
        instruction_tokens: int = config_constants.GI_QUOTE_INSTRUCTION_TOKEN_RESERVE,
    ) -> Optional[int]:
        """Chars of transcript this deployment can actually hold for a call of this shape.

        Every transcript clip in this provider goes through here, so raising the served window
        raises all of them at once and none of them can drift apart again (#2050).
        """
        return config_constants.transcript_budget_chars(
            self.served_context_tokens(),
            response_tokens=response_tokens,
            instruction_tokens=instruction_tokens,
        )

    def _token_kwarg(self, n: int, model: Optional[str] = None) -> Dict[str, Any]:
        """The right token-limit kwarg for this model, as a dict to spread into a create() call.

        The o1/o3/gpt-5 series RENAMED ``max_tokens`` to ``max_completion_tokens`` and hard-reject
        the old name with a 400. The summarization path already knew this, but the EVIDENCE calls
        (quote extraction, entailment) still sent ``max_tokens`` — so on gpt-5 every grounding call
        400'd, fell back to staged, 400'd again, and the whole arm produced 0 grounded quotes while
        looking like it had "completed". One helper, used at every call site, so the rename cannot
        be honoured in one place and forgotten in another. ``model`` defaults to the summary model;
        the cleaning path passes its own.
        """
        name = model or self.summary_model
        if name.startswith(("o1", "o3", "gpt-5")):
            return {"max_completion_tokens": n}
        return {"max_tokens": n}

    def _emit_stage_cost(
        self,
        *,
        stage: str,
        capability: str,
        model: str,
        response: Any,
        triggered_guardrail: bool = False,
    ) -> None:
        """Emit the ``llm_cost`` event for one chat call (ADR-100 cost-attribution).

        2026-08-24 prod audit: speaker_detection / evidence (quote + entailment) /
        kg / insight-classification calls emitted NO cost events — 64 of the 101
        gateway calls in a single repair run (86% of the tokens) were invisible to
        cost telemetry, so the pipeline's self-reported LLM spend disagreed with
        the gateway ledger. Emission was a per-call-site ritual that only the
        summarize / gi / cleaning paths performed; the blind sites now call this
        one helper. Prefers the upstream's actual cost (``usage.cost`` /
        ``_hidden_params.response_cost``) over the local pricing-table estimate.
        """
        in_tok, out_tok = _openai_chat_usage_tokens(response)
        if in_tok is None or out_tok is None:
            return
        try:
            from ...workflow.cost_monitoring import emit_llm_cost_event
            from ...workflow.helpers import calculate_provider_cost

            cost = _openai_response_cost_usd(response) or calculate_provider_cost(
                cfg=self.cfg,
                provider_type=self._TELEMETRY_PROVIDER,
                capability=capability,
                model=model,
                prompt_tokens=int(in_tok),
                completion_tokens=int(out_tok),
            )
            emit_llm_cost_event(
                self.cfg,
                provider=self._TELEMETRY_PROVIDER,
                stage=stage,
                model=model,
                estimated_cost_usd=float(cost or 0.0),
                prompt_tokens=int(in_tok),
                completion_tokens=int(out_tok),
                triggered_guardrail=triggered_guardrail,
                served_model=getattr(response, "model", None),
                response=response,
            )
        except Exception:  # noqa: BLE001 - telemetry must never break the call
            pass

    def _build_stage_messages(
        self,
        *,
        transcript: str,
        system_prompt: str,
        user_prompt: str,
    ) -> List[Dict[str, str]]:
        """RFC-115: assemble chat messages for a transcript-bearing LLM stage.

        When ``cache_transcript_prefix`` is on (default) and the transcript appears verbatim in the
        user prompt, relocate it to the LEADING block of the system prompt so it forms a stable,
        stage-invariant prefix the provider caches at ~0.1x price across an episode's stages. The
        content is only reordered — the model still sees the same transcript + instructions + task.

        With the flag off — or when the transcript is not found verbatim in the user prompt (a
        custom prompt, or a stage that truncated the transcript before rendering) — this returns
        today's exact legacy layout, so it is always a safe no-op fallback that never drops content.

        Auto-cache providers (openai/deepseek/qwen/litellm/vllm) need no extra API fields — the
        layout alone enables caching. Anthropic (explicit ``cache_control``) and Gemini (native
        ``cached_content``) are handled in their own providers in later RFC-115 phases.
        """
        return _openai_style_messages(
            transcript,
            system_prompt,
            user_prompt,
            enabled=bool(getattr(self, "_cache_transcript_prefix", True)),
        )

    @classmethod
    def get_pricing(cls, model: str, capability: str) -> Dict[str, float]:
        """Read pricing from ``config/pricing_assumptions.yaml`` (#651).

        YAML is the single source of truth; this thin wrapper is kept for API
        stability (test fixtures, tooling). Production cost calc goes through
        :func:`podcast_scraper.workflow.helpers._get_provider_pricing`.

        Namespaced by ``_TELEMETRY_PROVIDER`` (ADR-147) so a sibling
        (litellm/vllm/deepseek) reads its own pricing rows, never ``openai``'s.
        Returns the rate dict for ``(provider, capability, model)`` or ``{}``
        when no matching row exists.
        """
        from podcast_scraper.pricing_assumptions import (
            get_loaded_table,
            lookup_external_pricing,
        )

        table, _ = get_loaded_table("config/pricing_assumptions.yaml")
        if not table:
            return {}
        ext = lookup_external_pricing(table, cls._TELEMETRY_PROVIDER, capability, model)
        return dict(ext) if ext else {}

    def initialize(self) -> None:
        """Initialize all OpenAI capabilities.

        For OpenAI API, initialization is a no-op but we track it for consistency.
        This method is idempotent and can be called multiple times safely.

        Note:
            All capabilities share the same OpenAI client, so initialization
            is lightweight and doesn't require separate setup for each capability.
        """
        # Initialize transcription if enabled
        if self.cfg.transcribe_missing and not self._transcription_initialized:
            self._initialize_transcription()

        # Initialize speaker detection if enabled
        if self.cfg.auto_speakers and not self._speaker_detection_initialized:
            self._initialize_speaker_detection()

        # ALWAYS initialized — this capability also serves extract_kg_graph / generate_insights /
        # clean_transcript / the evidence stack, so gating it on the generate_summaries STAGE flag
        # starved KG-only runs and lazily-built failover tiers (#1720).
        if not self._summarization_initialized:
            self._initialize_summarization()

    def _initialize_transcription(self) -> None:
        """Initialize transcription capability."""
        logger.debug("Initializing OpenAI transcription (model: %s)", self.transcription_model)
        self._transcription_initialized = True
        logger.debug("OpenAI transcription initialized successfully")

    def _initialize_speaker_detection(self) -> None:
        """Initialize speaker detection capability."""
        logger.debug("Initializing OpenAI speaker detection (model: %s)", self.speaker_model)
        self._speaker_detection_initialized = True
        logger.debug("OpenAI speaker detection initialized successfully")

    def _initialize_summarization(self) -> None:
        """Initialize summarization capability."""
        logger.debug("Initializing OpenAI summarization (model: %s)", self.summary_model)
        self._summarization_initialized = True
        logger.debug("OpenAI summarization initialized successfully")

    # ============================================================================
    # TranscriptionProvider Protocol Implementation
    # ============================================================================

    def transcribe(self, audio_path: str, language: str | None = None) -> str:
        """Transcribe audio file to text using OpenAI Whisper API.

        Args:
            audio_path: Path to audio file
            language: Optional language code (e.g., "en", "fr").
                     If provided (not None), uses that language.
                     If not provided (default None), uses cfg.language if available.
                     If explicitly passed as None, auto-detects (ignores cfg.language).
                     Note: To explicitly request auto-detect, pass None. To use cfg.language,
                     omit the argument.

        Returns:
            Transcribed text as string

        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If transcription fails or API key is invalid
            RuntimeError: If provider is not initialized
        """
        if not self._transcription_initialized:
            raise RuntimeError(
                "OpenAIProvider transcription not initialized. Call initialize() first."
            )

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Use provided language or fall back to config
        # We can't perfectly distinguish "not provided" from "explicitly None" with default args
        # So we use a simple rule: if language is provided (not None), use it.
        # If language is None, use cfg.language if available, otherwise auto-detect.
        # Note: To explicitly request auto-detect when cfg.language is set, the caller
        # would need to pass a special value, but for backward compatibility we use None.
        # The test case where language=None is explicitly passed and cfg.language is set
        # is ambiguous - we choose to use cfg.language in this case for consistency.
        if language is not None:
            effective_language = language
        elif hasattr(self.cfg, "language") and self.cfg.language is not None:
            # Use config language when parameter is None (default or explicitly passed)
            effective_language = self.cfg.language
        else:
            # No language specified - auto-detect
            effective_language = None

        logger.debug(
            "Transcribing audio file via OpenAI API: %s (language: %s)",
            audio_path,
            effective_language or "auto",
        )

        try:
            from ...utils.provider_metrics import (
                _safe_openai_retryable,
                retry_with_metrics,
            )

            def _make_transcribe_call():
                with open(audio_path, "rb") as audio_file:
                    if effective_language is not None:
                        return self.client.audio.transcriptions.create(
                            model=self.transcription_model,
                            file=audio_file,
                            language=effective_language,
                            response_format="verbose_json",
                        )
                    else:
                        return self.client.audio.transcriptions.create(
                            model=self.transcription_model,
                            file=audio_file,
                            response_format="verbose_json",
                        )

            transcript = retry_with_metrics(
                _make_transcribe_call,
                max_retries=2,
                initial_delay=1.0,
                max_delay=30.0,
                retryable_exceptions=_safe_openai_retryable(),
            )

            # verbose_json returns a Transcription object with `.text` (and `.duration`,
            # `.segments`, etc.). We return just the text here; cost tracking in
            # transcribe_with_segments uses the full response.
            if hasattr(transcript, "text"):
                text_attr = getattr(transcript, "text", None)
                text = text_attr if isinstance(text_attr, str) else ""
            elif isinstance(transcript, dict):
                text = str(transcript.get("text", ""))
            elif isinstance(transcript, str):
                text = transcript
            else:
                text = str(transcript)

            logger.debug(
                "OpenAI transcription completed: %d characters",
                len(text) if text else 0,
            )

            return text

        except Exception as exc:
            # Use the concrete sibling's label/class so a Groq (or any sibling) transcription
            # failure is not misreported as an OpenAI one — the transcribe path is inherited.
            logger.error(
                "%s Whisper API error: %s", self._PROVIDER_LABEL, format_exception_for_log(exc)
            )
            from podcast_scraper.exceptions import ProviderRuntimeError

            raise ProviderRuntimeError(
                message=(
                    f"{self._PROVIDER_LABEL} transcription failed: "
                    f"{format_exception_for_log(exc)}"
                ),
                provider=f"{type(self).__name__}/Transcription",
            ) from exc

    def transcribe_with_segments(
        self,
        audio_path: str,
        language: str | None = None,
        pipeline_metrics: metrics.Metrics | None = None,
        episode_duration_seconds: int | None = None,
        call_metrics: Any | None = None,  # ProviderCallMetrics from utils.provider_metrics
    ) -> tuple[dict[str, object], float]:
        """Transcribe audio file and return full result with segments.

        Returns the complete OpenAI transcription result including segments
        and timestamps for screenplay formatting.

        Args:
            audio_path: Path to audio file
            language: Optional language code (e.g., "en", "fr").
                     If None, uses cfg.language or auto-detects

        Returns:
            Tuple of (result_dict, elapsed_time) where result_dict contains:
            - "text": Full transcribed text
            - "segments": List of segment dicts with start, end, text
            - Other OpenAI API metadata
        """
        if not self._transcription_initialized:
            raise RuntimeError(
                "OpenAIProvider transcription not initialized. Call initialize() first."
            )

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Use provided language or fall back to config
        effective_language = language if language is not None else (self.cfg.language or None)

        logger.debug(
            "Transcribing audio file with segments via OpenAI API: %s (language: %s)",
            audio_path,
            effective_language or "auto",
        )

        start_time = time.time()
        try:
            # Track retries and rate limits
            from ...utils.provider_metrics import ProviderCallMetrics

            if call_metrics is None:
                call_metrics = ProviderCallMetrics()
            call_metrics.set_provider_name(self._TELEMETRY_PROVIDER)
            call_metrics.set_breaker_config_from_cfg(self.cfg)

            # Wrap API call with retry tracking
            from ...utils.provider_metrics import (
                _safe_openai_retryable,
                retry_with_metrics,
            )

            def _make_api_call():
                with open(audio_path, "rb") as audio_file:
                    # verbose_json + word timestamps: whisper's segment-level times drift on
                    # long audio; word-level times are accurate and we reset segment bounds
                    # from them below (#1173).
                    kwargs: dict[str, Any] = {
                        "model": self.transcription_model,
                        "file": audio_file,
                        "response_format": "verbose_json",
                        "timestamp_granularities": ["word", "segment"],
                    }
                    if effective_language is not None:
                        kwargs["language"] = effective_language
                    return self.client.audio.transcriptions.create(**kwargs)

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
                call_metrics.finalize()
                raise

            elapsed = time.time() - start_time
            call_metrics.finalize()

            # Extract token counts (OpenAI Whisper API doesn't return tokens,
            # but track audio minutes)
            # Get audio duration from episode metadata, response, or estimate from file size
            audio_minutes = 0.0
            if episode_duration_seconds is not None:
                # Handle Mock objects from tests by checking type
                if isinstance(episode_duration_seconds, (int, float)):
                    audio_minutes = float(episode_duration_seconds) / 60.0
            elif hasattr(response, "duration") and response.duration:
                duration_val = getattr(response, "duration", None)
                if isinstance(duration_val, (int, float)):
                    audio_minutes = float(duration_val) / 60.0
            else:
                # Fallback only fires when neither episode_duration_seconds nor
                # response.duration is available (rare with verbose_json). Scale
                # by the configured bitrate: at 128 kbps ≈ 1 MB/min, but low-bitrate
                # preprocessing (speech_optimal_v1 uses 32 kbps → ~0.25 MB/min) would
                # under-report ~4× if we assumed 128. Default to 128 when unset
                # (matches pre-preprocessing podcast MP3s).
                try:
                    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
                    bitrate_kbps_cfg = getattr(self.cfg, "preprocessing_mp3_bitrate_kbps", None)
                    bitrate_kbps = float(bitrate_kbps_cfg) if bitrate_kbps_cfg else 128.0
                    audio_minutes = file_size_mb * (128.0 / bitrate_kbps)
                except OSError:
                    pass

            # Calculate cost for transcription (per minute pricing)
            if audio_minutes > 0:
                # Compute cost first so the same value can flow into both
                # call_metrics (per-episode) and pipeline_metrics (per-stage).
                from ...utils.provider_metrics import (
                    record_provider_call_cost,
                    record_transcription_cost_to_pipeline,
                )
                from ...workflow.helpers import calculate_provider_cost

                record_provider_call_cost(
                    call_metrics,
                    calculate_provider_cost(
                        cfg=self.cfg,
                        provider_type=self._TELEMETRY_PROVIDER,
                        capability="transcription",
                        model=self.transcription_model,
                        audio_minutes=audio_minutes,
                    ),
                    cfg=self.cfg,
                    provider_type=self._TELEMETRY_PROVIDER,
                    capability="transcription",
                    model=self.transcription_model,
                    audio_minutes=audio_minutes,
                )
                record_transcription_cost_to_pipeline(
                    pipeline_metrics, call_metrics, audio_minutes, call_metrics.estimated_cost
                )

            # OpenAI API returns a Transcription object with text and segments
            # when verbose_json is used. Convert to dict format matching Whisper output.
            # Response structure: response.text (str), response.segments (List[Segment])
            # Each Segment has: start (float), end (float), text (str)
            segments = []
            if hasattr(response, "segments") and response.segments:
                for seg in response.segments:
                    # Handle both dict-like and object-like segment structures
                    if isinstance(seg, dict):
                        segments.append(
                            {
                                "start": float(seg.get("start", 0.0)),
                                "end": float(seg.get("end", 0.0)),
                                "text": seg.get("text", ""),
                            }
                        )
                    else:
                        # Object-like structure (OpenAI SDK TranscriptionSegment)
                        segments.append(
                            {
                                "start": float(getattr(seg, "start", 0.0)),
                                "end": float(getattr(seg, "end", 0.0)),
                                "text": getattr(seg, "text", ""),
                            }
                        )

            # Reset each segment's start/end from the accurate word-level times — whisper's
            # segment-level timestamps drift on long audio (up to tens of seconds), word-level
            # do not (#1173). No-op when the model/response carries no words.
            raw_words = getattr(response, "words", None)
            if raw_words:
                from podcast_scraper.transcription.word_timestamps import (
                    apply_word_timestamps,
                    word_dicts,
                )

                segments = apply_word_timestamps(segments, word_dicts(raw_words))

            result_dict: dict[str, object] = {
                "text": response.text if hasattr(response, "text") else str(response),
                "segments": segments,
            }

            segments_list = result_dict.get("segments", [])
            segment_count = len(segments_list) if isinstance(segments_list, list) else 0
            logger.debug(
                "OpenAI transcription with segments completed in %.2fs (%d segments)",
                elapsed,
                segment_count,
            )

            return result_dict, elapsed

        except Exception as exc:
            elapsed = time.time() - start_time
            # Label from the concrete sibling (Groq/etc.), not a hardcoded "OpenAI" (inherited).
            label = self._PROVIDER_LABEL
            provider_tag = f"{type(self).__name__}/Transcription"
            logger.error("%s Whisper API error: %s", label, format_exception_for_log(exc))
            from podcast_scraper.exceptions import ProviderAuthError, ProviderRuntimeError

            # Handle OpenAI-compatible error types
            error_msg = str(exc).lower()
            exc_type_name = type(exc).__name__
            if (
                "api key" in error_msg
                or "authentication" in error_msg
                or "permission" in error_msg
                or "401" in error_msg
                or "unauthorized" in error_msg
                or exc_type_name == "AuthenticationError"
            ):
                raise ProviderAuthError(
                    message=f"{label} authentication failed: {format_exception_for_log(exc)}",
                    provider=provider_tag,
                    suggestion=(
                        "Check your API key environment variable or config setting. "
                        "Verify the key is valid and has not expired."
                    ),
                ) from exc
            elif "quota" in error_msg or "rate limit" in error_msg or "429" in error_msg:
                raise ProviderRuntimeError(
                    message=f"{label} rate limit exceeded: {format_exception_for_log(exc)}",
                    provider=provider_tag,
                    suggestion="Wait before retrying or check your API quota",
                ) from exc
            else:
                raise ProviderRuntimeError(
                    message=f"{label} transcription failed: {format_exception_for_log(exc)}",
                    provider=provider_tag,
                ) from exc

    # ============================================================================
    # SpeakerDetector Protocol Implementation
    # ============================================================================

    def analyze_patterns(
        self,
        episodes: list[Episode],  # type: ignore[valid-type]
        known_hosts: Set[str],
    ) -> dict[str, object] | None:
        """Analyze patterns across multiple episodes (optional).

        For OpenAI provider, pattern analysis is not implemented.
        Returns None to use local pattern analysis logic.

        Args:
            episodes: List of episodes to analyze
            known_hosts: Set of known host names

        Returns:
            None (uses local pattern analysis)
        """
        # OpenAI provider doesn't implement pattern analysis
        # Return None to use local logic
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
        """Summarize text using OpenAI GPT API.

        The transcript is clipped to what THIS deployment's window can hold (#2050). This method
        used to be unbounded, on the docstring's own reasoning that it "can handle full
        transcripts directly due to large context window (128k+ tokens)" — an OpenAI-native
        assumption that is simply false on the DGX, which serves 32,768. Of the six sites that
        build a prompt from a transcript, this was the ONLY one with no limit at all, which makes
        it the prime suspect for the 565 server-side context rejections measured over 30 days.

        Args:
            text: Transcript text to summarize
            episode_title: Optional episode title
            episode_description: Optional episode description
            params: Optional parameters dict with:
                - max_length: Maximum summary length in tokens
                    (default from summary_reduce_params.max_new_tokens)
                - min_length: Minimum summary length in tokens
                    (default from config)
                - prompt: Optional custom prompt (overrides default)

        Returns:
            Dictionary with summary results:
            {
                "summary": str,
                "summary_short": Optional[str],
                "metadata": {
                    "model": str,
                    "provider": "openai",
                    ...
                }
            }

        Raises:
            ValueError: If summarization fails
            RuntimeError: If provider is not initialized
        """
        if not self._summarization_initialized:
            raise RuntimeError(
                "OpenAIProvider summarization not initialized. Call initialize() first."
            )

        # Extract parameters with defaults from config
        # OpenAI doesn't have map/reduce, so use reduce params for final summary
        max_length = (
            (params.get("max_length") if params else None)
            or self.cfg.summary_reduce_params.get("max_new_tokens")
            or 800
        )
        # Plain-text summary path — bypass the structured-JSON floor that
        # was silently clamping max_tokens up to 4096 even for plain text
        # (#1016 Phase 2b). Bundled / structured-JSON callsites below
        # keep the default ``structured=True`` and the floor still applies.
        from ..common.output_tokens import cloud_structured_max_output_tokens

        max_length = cloud_structured_max_output_tokens(self.cfg, max_length, structured=False)
        min_length = (
            (params.get("min_length") if params else None)
            or self.cfg.summary_reduce_params.get("min_new_tokens")
            or 100
        )
        custom_prompt = params.get("prompt") if params else None
        # A summary other than the episode's (a board's) runs this same flow with its own
        # prompts, stage and budget; absent, this is the episode summary exactly as before.
        use_case = use_case_from_params(params)
        max_length = reply_budget(
            use_case,
            max_length,
            lambda n: cloud_structured_max_output_tokens(self.cfg, n, structured=False),
        )

        # #2050: bound the transcript against the served window. `max_length` is the reply budget
        # and is only known here, which is why the clip lives at this point and not at the caller.
        # A None budget means the window is not known — send it, and let a 400 name the real limit
        # for the retry path to clamp against. Never guess a window here.
        text = self.clip_transcript_to_budget(
            text, response_tokens=int(max_length), label="summarization transcript"
        )

        logger.debug(
            "Summarizing text via OpenAI API (model: %s, max_tokens: %d)",
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
            from ...utils.provider_metrics import ProviderCallMetrics

            if call_metrics is None:
                call_metrics = ProviderCallMetrics()
            call_metrics.set_provider_name(self._TELEMETRY_PROVIDER)
            call_metrics.set_breaker_config_from_cfg(self.cfg)

            # Wrap API call with retry tracking
            from ...utils.provider_metrics import (
                _safe_openai_retryable,
                retry_with_metrics,
            )

            # Token-limit kwarg via the single helper (o1/o3/gpt-5 rename + provider caps).
            _token_kwarg = self._token_kwarg(max_length)

            # RFC-115 Phase 1: relocate the transcript to a leading, cacheable system block
            # (default on). ``text`` is the cleaned transcript rendered verbatim into user_prompt.
            messages = self._build_stage_messages(
                transcript=text,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )

            def _make_api_call():
                kwargs: Dict[str, Any] = {
                    "model": self.summary_model,
                    "messages": messages,
                    "temperature": self.summary_temperature,
                    **_token_kwarg,
                }
                if self.summary_seed is not None:
                    kwargs["seed"] = self.summary_seed
                return self._chat_create(
                    **kwargs, _lf_stage=use_case_stage(use_case) or "summarization"
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
                call_metrics.finalize()
                raise

            call_metrics.finalize()

            warn_if_truncated(
                response.choices[0].finish_reason,
                "openai",
                "summarize",
            )

            summary = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason

            # Extract token counts up-front so the cost event can fire on
            # both the happy path and the GuardrailViolation path
            # (ADR-100 paid-but-rejected cost-attribution).
            input_tokens = None
            output_tokens = None
            if hasattr(response, "usage") and response.usage:
                prompt_tokens = getattr(response.usage, "prompt_tokens", None)
                completion_tokens = getattr(response.usage, "completion_tokens", None)
                # Handle Mock objects from tests by checking type
                input_tokens = int(prompt_tokens) if isinstance(prompt_tokens, (int, float)) else 0
                output_tokens = (
                    int(completion_tokens) if isinstance(completion_tokens, (int, float)) else 0
                )
                if input_tokens > 0 or output_tokens > 0:
                    call_metrics.set_tokens(input_tokens, output_tokens)

            cost: Optional[float] = None
            if input_tokens is not None:
                from ...workflow.helpers import calculate_provider_cost

                cost = calculate_provider_cost(
                    cfg=self.cfg,
                    provider_type=self._TELEMETRY_PROVIDER,
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
                    provider_type=self._TELEMETRY_PROVIDER,
                    capability="summarization",
                    model=self.summary_model,
                    prompt_tokens=input_tokens,
                    completion_tokens=output_tokens,
                    triggered_guardrail=triggered_guardrail,
                    response=response,  # RFC-115: surfaces prefix-cache read tokens in llm_cost
                    stage=use_case_stage(use_case),
                )

            # Response-shape guardrail (ADR-100, #1003): empty / thinking-prose /
            # finish_reason=length. Cost event fires in BOTH branches so the
            # operator can pivot on paid-but-rejected spend; the raw
            # GuardrailViolation propagates so FallbackAware can route the
            # configured degradation_policy.fallback_provider_on_failure.
            try:
                _guardrails.check_chat_response(
                    summary, service=self._TELEMETRY_PROVIDER, finish_reason=finish_reason
                )
            except _guardrails.GuardrailViolation:
                _record_cost(triggered_guardrail=True)
                raise

            logger.debug(
                "OpenAI summarization completed: %d characters",
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
            user_params = {
                "transcript": (
                    text[:100] + "..." if len(text) > 100 else text
                ),  # Truncate for metadata
                "title": episode_title or "",
                "paragraphs_min": paragraphs_min,
                "paragraphs_max": paragraphs_max,
            }
            user_params.update(self.cfg.summary_prompt_params)
            prompt_metadata["user"] = get_prompt_metadata(user_prompt_name, params=user_params)
            # A use case records its own templates; the episode summary keeps the above.
            prompt_metadata = use_case_prompt_metadata(self._PROMPT_NS, use_case) or prompt_metadata

            return {
                "summary": summary,
                # OpenAI provider doesn't generate short summaries separately
                "summary_short": None,
                "metadata": {
                    "model": self.summary_model,
                    "provider": "openai",
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
            logger.error("OpenAI API error in summarization: %s", format_exception_for_log(exc))
            from podcast_scraper.exceptions import ProviderAuthError, ProviderRuntimeError

            # Handle OpenAI-specific error types
            error_msg = str(exc).lower()
            exc_type_name = type(exc).__name__
            if (
                "api key" in error_msg
                or "authentication" in error_msg
                or "permission" in error_msg
                or "401" in error_msg
                or "unauthorized" in error_msg
                or exc_type_name == "AuthenticationError"
            ):
                raise ProviderAuthError(
                    message=f"OpenAI authentication failed: {format_exception_for_log(exc)}",
                    provider=f"{type(self).__name__}/Summarization",
                    suggestion=(
                        "Check your OPENAI_API_KEY environment variable or config setting. "
                        "Verify the key is valid and has not expired."
                    ),
                ) from exc
            elif "quota" in error_msg or "rate limit" in error_msg or "429" in error_msg:
                raise ProviderRuntimeError(
                    message=f"OpenAI rate limit exceeded: {format_exception_for_log(exc)}",
                    provider=f"{type(self).__name__}/Summarization",
                    suggestion="Wait before retrying or check your API quota",
                ) from exc
            else:
                raise ProviderRuntimeError(
                    message=f"OpenAI summarization failed: {format_exception_for_log(exc)}",
                    provider=f"{type(self).__name__}/Summarization",
                ) from exc

    def complete_text(
        self, prompt: str, max_tokens: int = 800, stage: str = "speaker_detection"
    ) -> str:
        """One prompt in, one JSON answer out — the generic call ADR-110's resolver needs.

        ``max_tokens`` defaults to the resolver's 800; transcript naming asks for one answer per
        voice of the whole episode and passes more. ``stage`` labels the call's cost and trace.

        `detect_speakers` is asked who speaks BEFORE the audio is downloaded, so it can only answer
        from show notes and returns the people an episode is ABOUT (that is how Elon Musk became a
        speaker, #876). The resolver asks the same model the same question AFTER diarization, with
        each voice's own words in hand — and that needs a plain completion, not a task method whose
        prompt is baked in.

        A provider WITHOUT this silently resolves nobody: the pipeline degrades instead of failing,
        and the model gets blamed for a method we never wrote.
        """
        if not self._summarization_initialized:
            raise RuntimeError("OpenAI summarization not initialized for complete_text")

        def _call() -> str:
            response = self._chat_create(
                model=self.summary_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                **self._token_kwarg(max_tokens),
                response_format={"type": "json_object"},
                _lf_stage=stage,
            )
            # 2026-08-24 cost audit: ADR-110 resolver completions emitted no llm_cost
            # event. Inside _call so a re-roll's second call is also counted.
            self._emit_stage_cost(
                stage=stage,
                capability="speaker_detection",
                model=self.summary_model,
                response=response,
            )
            return (response.choices[0].message.content or "").strip()

        def _validate(content: str) -> None:
            _guardrails.check_chat_response(
                content, service=self._TELEMETRY_PROVIDER, expect_json=True
            )

        # ADR-148: one bounded in-place re-roll on a transient invalid JSON response before the
        # guardrail fallover fires. A persistently-bad response still raises GuardrailViolation.
        content = _guardrails.structured_call_with_reroll(
            _call, _validate, service=self._TELEMETRY_PROVIDER, max_reroll=1
        )
        return _insight_salvage.strip_json_fence(content)

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
                "OpenAIProvider summarization not initialized. Call initialize() first."
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
        # Kept outside the try so a failure can say WHAT the model sent back (see the except below).
        raw: Optional[str] = None
        finish_reason: Optional[str] = None
        try:
            from ...utils.provider_metrics import (
                _safe_openai_retryable,
                retry_with_metrics,
            )

            def _make_api_call():
                return self._chat_create(
                    model=model,
                    # RFC-115: transcript-first (text_slice is embedded verbatim in user_prompt),
                    # so KG reuses the same cached transcript prefix as the other episode stages.
                    messages=self._build_stage_messages(
                        transcript=text_slice, system_prompt=system_msg, user_prompt=user_prompt
                    ),
                    temperature=0.1,
                    **self._token_kwarg(2048),
                    _lf_stage="kg",
                )

            def _attempt() -> tuple[str, Any, Optional[Dict[str, Any]]]:
                response = retry_with_metrics(
                    _make_api_call,
                    max_retries=3,
                    initial_delay=1.0,
                    max_delay=30.0,
                    retryable_exceptions=_safe_openai_retryable(),
                )
                in_tok, out_tok = _openai_chat_usage_tokens(response)
                pm = pipeline_metrics
                if (
                    pm is not None
                    and in_tok is not None
                    and out_tok is not None
                    and hasattr(pm, "record_llm_kg_call")
                ):
                    from ...workflow.helpers import calculate_provider_cost

                    kg_cost = _openai_response_cost_usd(response) or calculate_provider_cost(
                        cfg=self.cfg,
                        provider_type=self._TELEMETRY_PROVIDER,
                        capability="summarization",
                        model=model,
                        prompt_tokens=int(in_tok),
                        completion_tokens=int(out_tok),
                    )
                    pm.record_llm_kg_call(in_tok, out_tok, cost_usd=kg_cost)
                # 2026-08-24 cost audit: KG extraction fed the metrics aggregate above but
                # emitted no llm_cost event — invisible in per-call cost telemetry.
                self._emit_stage_cost(stage="kg", capability="kg", model=model, response=response)
                choice = response.choices[0]
                reply = (choice.message.content or "").strip()
                return (
                    reply,
                    getattr(choice, "finish_reason", None),
                    parse_kg_graph_response(
                        reply, max_topics=max_topics, max_entities=max_entities
                    ),
                )

            raw, finish_reason, parsed = _attempt()
            # A reply that is not valid JSON gets ONE fresh attempt: a single dropped "},{" in a
            # complete 6 KB reply cost "How to Run an Institution in an Anti-Institution Era" its
            # whole KG (2026-10-03), and such replies have extracted cleanly when re-asked.
            if parsed is None and raw:
                logger.warning(
                    "kg: %s reply was not usable JSON; retrying the extraction once", model
                )
                raw, finish_reason, parsed = _attempt()
            if not parsed or not (parsed.get("topics") or parsed.get("entities")):
                # A reply arrived and yielded nothing. Downstream this only reads as "no topics",
                # so say here what the model actually returned.
                logger.warning(
                    "kg: %s returned no usable topics/entities "
                    "(finish_reason=%s, reply %d chars): %s",
                    model,
                    finish_reason,
                    len(raw),
                    _kg_reply_snippet(raw),
                )
            return parsed
        except Exception as e:
            # WARNING, not DEBUG (2026-09-29): three episodes failed KG extraction twice in prod and
            # the only trace anywhere was "produced no topics/entities" — this handler had eaten the
            # reason at DEBUG. The same episodes extracted cleanly 3/3 in isolation, so the cause
            # lived in exactly the detail this line used to hide. Behaviour is unchanged (still
            # None, so the fallback chain and "no topics" path run as before); only visibility is.
            logger.warning(
                "kg: extract_kg_graph failed on %s (%s: %s; finish_reason=%s): reply=%s",
                model,
                type(e).__name__,
                e,
                finish_reason,
                "<none received>" if raw is None else _kg_reply_snippet(raw),
            )
            logger.debug("OpenAI extract_kg_graph traceback", exc_info=True)
            return None

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
        """Build system and user prompts for summarization using prompt_store.

        Args:
            text: Transcript text to summarize
            episode_title: Optional episode title
            episode_description: Optional episode description
            max_length: Maximum summary length in tokens
            min_length: Minimum summary length in tokens
            custom_prompt: Optional custom prompt (overrides default)
            use_case: A summary other than the episode's: its own prompts

        Returns:
            Tuple of (system_prompt, user_prompt, system_prompt_name,
            user_prompt_name, paragraphs_min, paragraphs_max)
        """
        from ...prompts.store import render_prompt

        if use_case is not None:
            system, user, system_name, user_name = build_use_case_prompts(self._PROMPT_NS, use_case)
            return system, user, system_name, user_name, 0, 0

        # Use prompt_store to load versioned prompt templates
        system_prompt_name = (
            getattr(self.cfg, f"{self._CONFIG_NS}_summary_system_prompt", None)
            or "openai/summarization/system_v1"
        )
        user_prompt_name: str = getattr(self.cfg, f"{self._CONFIG_NS}_summary_user_prompt")

        # Render system prompt
        system_prompt = render_prompt(system_prompt_name)

        # Estimate paragraphs: roughly 100 tokens per paragraph
        paragraphs_min = max(1, min_length // 100)
        paragraphs_max = max(paragraphs_min, max_length // 100)

        # Render user prompt
        if custom_prompt:
            # Use custom prompt if provided
            user_prompt = custom_prompt.replace("{{ transcript }}", text)
            if episode_title:
                user_prompt = user_prompt.replace("{{ title }}", episode_title)
            # For custom prompts, use a placeholder name
            user_prompt_name = "custom"
        else:
            # Use prompt_store template
            # Merge config params with template params
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
        """Cleanup all provider resources (no-op for API provider).

        This method releases any resources held by the provider.
        For OpenAI API, there are no local resources to clean up.
        It may be called multiple times safely (idempotent).
        """
        # No resources to clean up for API provider
        # But we can mark as uninitialized
        self._transcription_initialized = False
        self._speaker_detection_initialized = False
        self._summarization_initialized = False

    def clear_cache(self) -> None:
        """Clear cache (no-op for API provider).

        OpenAI provider doesn't use cached models, so this is a no-op.
        """
        # No cache to clear for API provider
        pass

    # ============================================================================

    @property
    def is_initialized(self) -> bool:
        """Check if provider is initialized (any component).

        Returns:
            True if any component is initialized, False otherwise
        """
        return (
            self._transcription_initialized
            or self._speaker_detection_initialized
            or self._summarization_initialized
        )

    def _clean_transcript_chunked(
        self,
        text: str,
        pipeline_metrics: Optional[Any] = None,
    ) -> str:
        """PA1: clean a long transcript in line-aligned chunks that each fit the model output cap.

        A single cleaning call on a long transcript truncates (finish_reason=length) and the
        guardrail discards the whole result, so cleaning silently no-ops. Splitting on line
        boundaries (one speaker-turn per line in screenplay format) keeps speaker labels intact;
        each chunk is sized so ``words * factor`` stays under ~85% of the cap, then re-cleaned via
        the normal single-call path (recursion terminates: each chunk is under the cap).
        """
        max_words = max(1, int((self._CLEANING_MAX_TOKENS_CAP * 0.85) / _CLEANING_WORD_FACTOR))
        lines = text.split("\n")
        chunks: list[str] = []
        cur: list[str] = []
        cur_words = 0
        for line in lines:
            w = len(line.split())
            if cur and cur_words + w > max_words:
                chunks.append("\n".join(cur))
                cur, cur_words = [], 0
            cur.append(line)
            cur_words += w
        if cur:
            chunks.append("\n".join(cur))
        logger.info(
            "Cleaning long transcript in %d chunks (%d words > ~%d/chunk cap) to avoid truncation",
            len(chunks),
            len(text.split()),
            max_words,
        )
        cleaned_parts = [
            self.clean_transcript(chunk, pipeline_metrics=pipeline_metrics)
            for chunk in chunks
            if chunk.strip()
        ]
        return "\n".join(part for part in cleaned_parts if part)

    def clean_transcript(
        self,
        text: str,
        pipeline_metrics: Optional[Any] = None,
    ) -> str:
        """Clean transcript using LLM for semantic filtering.

        Args:
            text: Transcript text to clean (should already be pattern-cleaned)
            pipeline_metrics: Optional pipeline metrics for LLM usage accounting

        Returns:
            Cleaned transcript text

        Raises:
            RuntimeError: If provider is not initialized or cleaning fails
        """
        if not self._summarization_initialized:
            raise RuntimeError("OpenAIProvider not initialized. Call initialize() first.")

        # PA1: cleaning output is ~the FULL input length, but a single call's output is bounded by
        # the model cap (deepseek 8192). Any transcript over ~cap/1.6 words would truncate
        # (finish_reason=length) and the guardrail discards the WHOLE result — cleaning then
        # silently no-ops on long episodes (43/88 in the 100-ep run). Split into line-aligned
        # chunks that each fit, clean each, rejoin. Screenplay lines are one speaker-turn, so line
        # splits preserve speaker labels.
        if estimate_cleaning_output_tokens(len(text.split())) > self._CLEANING_MAX_TOKENS_CAP and (
            "\n" in text.strip()
        ):
            return self._clean_transcript_chunked(text, pipeline_metrics)

        from ...prompts.store import render_prompt

        # Build cleaning prompt using prompt_store
        prompt_name = "openai/cleaning/v1"
        user_prompt = render_prompt(prompt_name, transcript=text)

        # Use system prompt (optional, can be empty for cleaning)
        system_prompt = (
            "You are a transcript cleaning assistant. "
            "Remove sponsors, ads, intros, outros, and meta-commentary. "
            "Preserve all substantive content and speaker information. "
            "Return only the cleaned text, no explanations."
        )

        logger.debug(
            "Cleaning transcript via OpenAI API (model: %s, text length: %d chars)",
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
            call_metrics.set_provider_name(self._TELEMETRY_PROVIDER)
            call_metrics.set_breaker_config_from_cfg(self.cfg)

            def _make_api_call():
                return self._chat_create(
                    model=self.cleaning_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.cleaning_temperature,
                    **self._token_kwarg(
                        clamp_cleaning_max_tokens(
                            estimate_cleaning_output_tokens(len(text.split())),
                            self._CLEANING_MAX_TOKENS_CAP,
                        ),
                        model=self.cleaning_model,
                    ),
                    _lf_stage="cleaning",
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
                call_metrics.finalize()
                raise

            call_metrics.finalize()

            cleaned = response.choices[0].message.content
            # Token + cost capture up-front so cost emits on both happy path
            # and the GuardrailViolation catch-and-degrade path (ADR-100).
            in_tok, out_tok = _openai_chat_usage_tokens(response)
            cleaning_cost: Optional[float] = None
            if in_tok is not None and out_tok is not None:
                from ...workflow.helpers import calculate_provider_cost

                cleaning_cost = _openai_response_cost_usd(response) or calculate_provider_cost(
                    cfg=self.cfg,
                    provider_type=self._TELEMETRY_PROVIDER,
                    capability="cleaning",
                    model=self.cleaning_model,
                    prompt_tokens=int(in_tok),
                    completion_tokens=int(out_tok),
                )
                if pipeline_metrics is not None and hasattr(
                    pipeline_metrics, "record_llm_cleaning_call"
                ):
                    pipeline_metrics.record_llm_cleaning_call(
                        in_tok, out_tok, cost_usd=cleaning_cost
                    )

            def _emit_cleaning_cost(*, triggered_guardrail: bool = False) -> None:
                if in_tok is None or out_tok is None:
                    return
                try:
                    from ...workflow.cost_monitoring import emit_llm_cost_event

                    emit_llm_cost_event(
                        self.cfg,
                        provider=self._TELEMETRY_PROVIDER,
                        stage="cleaning",
                        model=self.cleaning_model,
                        estimated_cost_usd=float(cleaning_cost or 0.0),
                        prompt_tokens=int(in_tok),
                        completion_tokens=int(out_tok),
                        triggered_guardrail=triggered_guardrail,
                        served_model=getattr(response, "model", None),
                        response=response,
                    )
                except Exception:  # noqa: BLE001 - defensive
                    pass

            # Response-shape guardrail (ADR-100): cleaning catch-and-degrade.
            # Empty content is the gracefully-handled path (return text below);
            # the guardrail only fires on thinking-prose / finish_reason=length.
            if cleaned:
                try:
                    _guardrails.check_chat_response(
                        cleaned,
                        service=self._TELEMETRY_PROVIDER,
                        finish_reason=response.choices[0].finish_reason,
                    )
                except _guardrails.GuardrailViolation:
                    _emit_cleaning_cost(triggered_guardrail=True)
                    logger.warning(
                        "OpenAI cleaning output failed guardrail; "
                        "returning original transcript text"
                    )
                    return text
            if not cleaned:
                logger.warning("OpenAI API returned empty cleaned text, using original")
                _emit_cleaning_cost()
                return text

            _emit_cleaning_cost()
            logger.debug("OpenAI cleaning completed: %d -> %d chars", len(text), len(cleaned))
            return cast(str, cleaned)

        except _guardrails.GuardrailViolation:
            # ADR-100 per-stage policy: cleaning degrades gracefully — a
            # guardrail-tripping cleaned response means we serve the original
            # transcript text rather than fail the run. Distinct from summarize
            # (fail-up) and GI/KG (fail-up).
            logger.warning(
                "OpenAI cleaning output failed guardrail; returning original transcript text"
            )
            return text
        except Exception as exc:
            logger.error("OpenAI API error in cleaning: %s", format_exception_for_log(exc))
            from podcast_scraper.exceptions import ProviderAuthError, ProviderRuntimeError

            # Handle OpenAI-specific error types
            error_msg = str(exc).lower()
            exc_type_name = type(exc).__name__
            if (
                "api key" in error_msg
                or "authentication" in error_msg
                or "permission" in error_msg
                or "401" in error_msg
                or "unauthorized" in error_msg
                or exc_type_name == "AuthenticationError"
            ):
                raise ProviderAuthError(
                    message=f"OpenAI authentication failed: {format_exception_for_log(exc)}",
                    provider=f"{type(self).__name__}/Cleaning",
                    suggestion=(
                        "Check your OPENAI_API_KEY environment variable or config setting. "
                        "Verify the key is valid and has not expired."
                    ),
                ) from exc
            elif "quota" in error_msg or "rate limit" in error_msg or "429" in error_msg:
                raise ProviderRuntimeError(
                    message=f"OpenAI rate limit exceeded: {format_exception_for_log(exc)}",
                    provider=f"{type(self).__name__}/Cleaning",
                    suggestion="Wait before retrying or check your API quota",
                ) from exc
            else:
                raise ProviderRuntimeError(
                    message=f"OpenAI cleaning failed: {format_exception_for_log(exc)}",
                    provider=f"{type(self).__name__}/Cleaning",
                ) from exc

    def get_capabilities(self) -> ProviderCapabilities:
        """Get provider capabilities.

        Returns:
            ProviderCapabilities object describing OpenAI provider capabilities
        """
        return ProviderCapabilities(
            supports_transcription=True,
            supports_speaker_detection=True,
            supports_summarization=True,
            supports_semantic_cleaning=True,  # OpenAI supports LLM-based cleaning
            supports_audio_input=True,  # Whisper API accepts audio files
            supports_json_mode=True,  # GPT models support JSON mode
            # 0 means "not declared" in the capability report — the same thing DeepgramProvider
            # reports for a backend with no chat window at all. It is a DESCRIPTION, not a budget:
            # nothing sizes a prompt from here (#2050).
            max_context_tokens=self.max_context_tokens or 0,
            supports_tool_calls=True,  # GPT models support function calling
            supports_system_prompt=True,  # GPT models support system prompts
            supports_streaming=True,  # OpenAI API supports streaming
            provider_name=self._TELEMETRY_PROVIDER,
            supports_gi_segment_timing=True,
        )


class OpenAIProvider(OpenAICompatibleProvider):
    """OpenAI-native provider (GPT + Whisper).

    Inherits the OpenAI-compatible transport unchanged: the base's default identity
    (``_CONFIG_NS = "openai"``) and OpenAI-native auth (required ``sk-`` key) are exactly what this
    class needs, so no overrides. Kept as a distinct class so the provider registry, factories, and
    every existing ``OpenAIProvider`` import resolve, and so ADR-147's "``openai`` is reserved for
    OpenAI-native models" is explicit in the type system.
    """
