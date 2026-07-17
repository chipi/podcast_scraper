"""Factory for creating transcription providers.

This module provides a factory function to create transcription
providers based on configuration or experiment-style parameters.

Supports two modes:
1. Config-based: Pass a Config object
2. Experiment-based: Pass provider_type and params dict
"""

from __future__ import annotations

import os
from typing import Any, cast, Dict, Literal, Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from podcast_scraper import config
    from podcast_scraper.providers.params import TranscriptionParams
    from podcast_scraper.transcription.base import TranscriptionProvider
else:
    from podcast_scraper import config
    from podcast_scraper.providers.params import TranscriptionParams
    from podcast_scraper.transcription.base import TranscriptionProvider

from podcast_scraper.utils.protocol_verification import verify_protocol_compliance


def _transcription_fallback_tiers(cfg: "config.Config") -> list[str]:
    """The ordered transcription failover ladder for ``cfg`` (RFC-105 / #1198).

    Prefers the registry-emitted plural ``transcription_fallback_providers``. Falls back to the
    legacy singular ``transcription_fallback_provider`` as a one-element chain (full back-compat:
    an existing DGX-primary profile keeps its single cloud fallback with no profile churn).
    """
    plural = [
        str(p).strip()
        for p in (getattr(cfg, "transcription_fallback_providers", None) or [])
        if str(p).strip()
    ]
    if plural:
        return plural
    singular = (getattr(cfg, "transcription_fallback_provider", None) or "").strip()
    return [singular] if singular else []


def _build_chain_tier(cfg: "config.Config", provider_type: str) -> TranscriptionProvider:
    """Build one chain tier: the same provider ``create_transcription_provider`` would build for
    ``provider_type``, but with fallback-wrapping suppressed so tiers do not recursively re-wrap."""
    data = cfg.model_dump()
    data["transcription_provider"] = provider_type
    sub = config.Config.model_validate(data)
    return create_transcription_provider(sub, _wrap_fallback=False)


def create_transcription_provider(  # noqa: C901
    cfg_or_provider_type: Union[config.Config, str],
    params: Optional[Union[TranscriptionParams, Dict[str, Any]]] = None,
    *,
    _wrap_fallback: bool = True,
) -> TranscriptionProvider:
    """Create a transcription provider based on configuration or experiment params.

    Supports two modes:
    1. Config-based: Pass a Config object
    2. Experiment-based: Pass provider_type string and params dict/object

    Args:
        cfg_or_provider_type: Either a Config object or provider type string
            ("whisper", "openai", "gemini", "mistral", "deepgram", etc.)
        params: Optional parameters dict or TranscriptionParams object (experiment mode only)

    Returns:
        TranscriptionProvider instance

    Raises:
        ValueError: If provider type is not supported or params are invalid
        TypeError: If params are provided but cfg_or_provider_type is a Config object

    Note:
        Returns MLProvider for "whisper" provider type (unified ML provider).
        Returns OpenAIProvider for "openai" provider type (unified OpenAI provider).
        Reuses preloaded MLProvider instance if available (from early preloading).

    Example (Config-based):
        >>> from podcast_scraper import Config
        >>> cfg = Config(rss_url="...", transcription_provider="whisper")
        >>> provider = create_transcription_provider(cfg)

    Example (Experiment-based, new):
        >>> from podcast_scraper.providers.params import TranscriptionParams
        >>> params = TranscriptionParams(
        ...     model_name="base.en",
        ...     device="mps",
        ...     language="en"
        ... )
        >>> provider = create_transcription_provider("whisper", params)
    """
    # Determine mode: Config-based or experiment-based
    if isinstance(cfg_or_provider_type, config.Config):
        # Config-based mode
        if params is not None:
            raise TypeError(
                "Cannot provide params when using Config-based mode. "
                "Use experiment mode: create_transcription_provider(provider_type, params)"
            )
        cfg = cfg_or_provider_type
        provider_type = cfg.transcription_provider
        experiment_mode = False

        # RFC-105 (#1198): if the profile declares a failover ladder, wrap the primary + each tier
        # in a FallbackChainTranscriptionProvider. Suppressed via ``_wrap_fallback=False`` while
        # assembling the individual tiers, so a tier never recursively re-wraps itself.
        if _wrap_fallback:
            tiers = _transcription_fallback_tiers(cfg)
            if tiers:
                from ..providers.resilience.fallback import (
                    FallbackChainTranscriptionProvider,
                )

                chain: list[tuple[str, TranscriptionProvider]] = [
                    (provider_type, _build_chain_tier(cfg, provider_type))
                ]
                for tier_name in tiers:
                    chain.append((tier_name, _build_chain_tier(cfg, tier_name)))
                wrapped = cast(TranscriptionProvider, FallbackChainTranscriptionProvider(chain))
                verify_protocol_compliance(wrapped, TranscriptionProvider, "TranscriptionProvider")
                return wrapped
    else:
        # Experiment-based mode
        provider_type_str = str(cfg_or_provider_type)
        # Type narrowing: validate it's one of the allowed values
        if provider_type_str not in (
            "whisper",
            "openai",
            "gemini",
            "mistral",
            "deepgram",
            "tailnet_dgx_whisper",
        ):
            raise ValueError(f"Invalid provider type: {provider_type_str}")

        provider_type_value = cast(
            Literal["whisper", "openai", "gemini", "mistral", "deepgram", "tailnet_dgx_whisper"],
            provider_type_str,
        )
        experiment_mode = True
        provider_type = provider_type_value

    # Convert params to TranscriptionParams if needed
    if experiment_mode:
        if params is None:
            # Use default params if none provided
            # Use test default model for experiment mode without params
            params = TranscriptionParams(model_name=config.TEST_DEFAULT_WHISPER_MODEL)
        elif isinstance(params, dict):
            params = TranscriptionParams(**params)
        elif not isinstance(params, TranscriptionParams):
            raise TypeError(f"params must be TranscriptionParams or dict, got {type(params)}")

    if provider_type == "whisper":
        # Check for preloaded MLProvider instance (from early preloading)
        # Only reuse if in Config mode (experiment mode should create fresh instances)
        if not experiment_mode:
            try:
                from ..workflow import _preloaded_ml_provider

                if _preloaded_ml_provider is not None:
                    provider = cast(TranscriptionProvider, _preloaded_ml_provider)
                    # Runtime protocol verification (dev-mode only)
                    verify_protocol_compliance(
                        provider, TranscriptionProvider, "TranscriptionProvider"
                    )
                    return provider
            except (ImportError, AttributeError):
                # workflow module not available (e.g., in tests), create new instance
                pass

        # Create new instance
        from ..providers.ml.ml_provider import MLProvider

        if experiment_mode:
            # Create a minimal Config from params for experiment mode
            from ..config import Config

            # After conversion above, params is guaranteed to be TranscriptionParams
            assert isinstance(params, TranscriptionParams)
            cfg = Config(
                rss="",  # Dummy, not used for transcription (use alias)
                transcription_provider="whisper",
                whisper_model=params.model_name if params.model_name else "base.en",
                whisper_device=params.device,
                language=params.language or "en",
            )
            provider = MLProvider(cfg)
        else:
            provider = MLProvider(cfg)

        # Runtime protocol verification (dev-mode only)
        verify_protocol_compliance(provider, TranscriptionProvider, "TranscriptionProvider")
        return provider
    elif provider_type == "openai":
        from ..providers.openai.openai_provider import OpenAIProvider

        if experiment_mode:
            # Create a minimal Config from params for experiment mode
            from ..config import Config

            # After conversion above, params is guaranteed to be TranscriptionParams
            assert isinstance(params, TranscriptionParams)
            cfg = Config(
                rss="",  # Dummy, not used for transcription (use alias)
                transcription_provider="openai",
                openai_transcription_model=params.model_name if params.model_name else "whisper-1",
                openai_api_key=os.getenv("OPENAI_API_KEY"),  # Load from env
            )
            provider = OpenAIProvider(cfg)
        else:
            provider = OpenAIProvider(cfg)

        # Runtime protocol verification (dev-mode only)
        verify_protocol_compliance(provider, TranscriptionProvider, "TranscriptionProvider")
        return provider
    elif provider_type == "gemini":
        from ..providers.gemini.gemini_provider import GeminiProvider

        if experiment_mode:
            # Create a minimal Config from params for experiment mode
            from ..config import Config

            # After conversion above, params is guaranteed to be TranscriptionParams
            assert isinstance(params, TranscriptionParams)
            cfg = Config(
                rss="",  # Dummy, not used for transcription (use alias)
                transcription_provider="gemini",
                gemini_transcription_model=(
                    params.model_name if params.model_name else "gemini-2.5-flash-lite"
                ),
                gemini_api_key=os.getenv("GEMINI_API_KEY"),  # Load from env
            )
            provider = GeminiProvider(cfg)
        else:
            provider = GeminiProvider(cfg)

        # Runtime protocol verification (dev-mode only)
        verify_protocol_compliance(provider, TranscriptionProvider, "TranscriptionProvider")
        return provider
    elif provider_type == "mistral":
        from ..providers.mistral.mistral_provider import MistralProvider

        if experiment_mode:
            # Create a minimal Config from params for experiment mode
            from ..config import Config

            # After conversion above, params is guaranteed to be TranscriptionParams
            assert isinstance(params, TranscriptionParams)
            cfg = Config(
                rss="",  # Dummy, not used for transcription (use alias)
                transcription_provider="mistral",
                mistral_transcription_model=(
                    params.model_name if params.model_name else "voxtral-mini-latest"
                ),
                mistral_api_key=os.getenv("MISTRAL_API_KEY"),  # Load from env
            )
            provider = MistralProvider(cfg)
        else:
            provider = MistralProvider(cfg)

        # Runtime protocol verification (dev-mode only)
        verify_protocol_compliance(provider, TranscriptionProvider, "TranscriptionProvider")
        return provider
    elif provider_type == "deepgram":
        from ..providers.deepgram.deepgram_provider import DeepgramTranscriptionProvider

        if experiment_mode:
            from ..config import Config

            assert isinstance(params, TranscriptionParams)
            cfg = Config(
                rss="",
                transcription_provider="deepgram",
                deepgram_model=params.model_name if params.model_name else "nova-3",
                deepgram_api_key=os.getenv("DEEPGRAM_API_KEY"),
            )
            provider = DeepgramTranscriptionProvider(cfg)
        else:
            provider = DeepgramTranscriptionProvider(cfg)

        verify_protocol_compliance(provider, TranscriptionProvider, "TranscriptionProvider")
        return provider
    elif provider_type == "anthropic":
        from ..providers.anthropic.anthropic_provider import AnthropicProvider

        if experiment_mode:
            # Create a minimal Config from params for experiment mode
            from ..config import Config

            # After conversion above, params is guaranteed to be TranscriptionParams
            assert isinstance(params, TranscriptionParams)
            cfg = Config(
                rss="",  # Dummy, not used for transcription (use alias)
                transcription_provider="anthropic",
                anthropic_transcription_model=(
                    params.model_name if params.model_name else "claude-3-5-sonnet-20241022"
                ),
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),  # Load from env
            )
            provider = AnthropicProvider(cfg)
        else:
            provider = AnthropicProvider(cfg)

        # Runtime protocol verification (dev-mode only)
        verify_protocol_compliance(provider, TranscriptionProvider, "TranscriptionProvider")
        return provider
    elif provider_type == "tailnet_dgx_whisper":
        from ..providers.tailnet_dgx.whisper_provider import (
            TailnetDgxWhisperTranscriptionProvider,
        )

        if experiment_mode:
            raise ValueError("tailnet_dgx_whisper is not supported in experiment mode")
        provider = cast(
            TranscriptionProvider,
            TailnetDgxWhisperTranscriptionProvider(cfg),
        )
        verify_protocol_compliance(provider, TranscriptionProvider, "TranscriptionProvider")
        return provider
    elif provider_type == "moss":
        # MOSS is a joint transcribe+diarize model (#1177): this branch takes the transcript half,
        # the diarization factory takes the speaker half, and the DGX service caches the inference
        # so the second stage does not re-run the model.
        from ..providers.moss.moss_provider import MossTranscriptionProvider

        if experiment_mode:
            raise ValueError("moss is not supported in experiment mode")
        provider = cast(TranscriptionProvider, MossTranscriptionProvider(cfg))
        verify_protocol_compliance(provider, TranscriptionProvider, "TranscriptionProvider")
        return provider
    else:
        raise ValueError(
            f"Unsupported transcription provider: {provider_type}. "
            "Supported providers: 'whisper', 'openai', 'gemini', 'mistral', 'deepgram', "
            "'anthropic', 'tailnet_dgx_whisper', 'moss'"
        )
