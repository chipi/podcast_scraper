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


def create_transcription_provider(
    cfg_or_provider_type: Union[config.Config, str],
    params: Optional[Union[TranscriptionParams, Dict[str, Any]]] = None,
) -> TranscriptionProvider:
    """Create a transcription provider based on configuration or experiment params.

    Supports two modes:
    1. Config-based: Pass a Config object
    2. Experiment-based: Pass provider_type string and params dict/object

    Args:
        cfg_or_provider_type: Either a Config object or provider type string
            ("whisper" or "openai")
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
    else:
        # Experiment-based mode
        provider_type_str = str(cfg_or_provider_type)
        # Type narrowing: validate it's one of the allowed values
        if provider_type_str not in ("whisper", "openai"):
            raise ValueError(f"Invalid provider type: {provider_type_str}")

        provider_type_value = cast(Literal["whisper", "openai"], provider_type_str)
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
                    return cast(TranscriptionProvider, _preloaded_ml_provider)
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
            return MLProvider(cfg)
        else:
            return MLProvider(cfg)
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
            return OpenAIProvider(cfg)
        else:
            return OpenAIProvider(cfg)
    else:
        raise ValueError(
            f"Unsupported transcription provider: {provider_type}. "
            "Supported providers: 'whisper', 'openai'"
        )
