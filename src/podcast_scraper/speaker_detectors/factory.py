"""Factory for creating speaker detection providers.

This module provides a factory function to create speaker detection
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
    from podcast_scraper.providers.params import SpeakerDetectionParams
    from podcast_scraper.speaker_detectors.base import SpeakerDetector
else:
    from podcast_scraper import config
    from podcast_scraper.providers.params import SpeakerDetectionParams
    from podcast_scraper.speaker_detectors.base import SpeakerDetector


def create_speaker_detector(
    cfg_or_provider_type: Union[config.Config, str],
    params: Optional[Union[SpeakerDetectionParams, Dict[str, Any]]] = None,
) -> SpeakerDetector:
    """Create a speaker detection provider based on configuration or experiment params.

    Supports two modes:
    1. Config-based: Pass a Config object
    2. Experiment-based: Pass provider_type string and params dict/object

    Args:
        cfg_or_provider_type: Either a Config object or provider type string
            ("spacy" or "openai")
        params: Optional parameters dict or SpeakerDetectionParams object (experiment mode only)

    Returns:
        SpeakerDetector instance

    Raises:
        ValueError: If provider type is not supported or params are invalid
        TypeError: If params are provided but cfg_or_provider_type is a Config object

    Note:
        Returns MLProvider for "spacy" provider type (unified ML provider).
        Returns OpenAIProvider for "openai" provider type (unified OpenAI provider).
        Reuses preloaded MLProvider instance if available (from early preloading).

    Example (Config-based):
        >>> from podcast_scraper import Config
        >>> cfg = Config(rss_url="...", speaker_detector_provider="spacy")
        >>> detector = create_speaker_detector(cfg)

    Example (Experiment-based, new):
        >>> from podcast_scraper.providers.params import SpeakerDetectionParams
        >>> params = SpeakerDetectionParams(
        ...     model_name="en_core_web_sm"
        ... )
        >>> detector = create_speaker_detector("spacy", params)
    """
    # Determine mode: Config-based or experiment-based
    if isinstance(cfg_or_provider_type, config.Config):
        # Config-based mode
        if params is not None:
            raise TypeError(
                "Cannot provide params when using Config-based mode. "
                "Use experiment mode: create_speaker_detector(provider_type, params)"
            )
        cfg = cfg_or_provider_type
        provider_type = cfg.speaker_detector_provider
        experiment_mode = False
    else:
        # Experiment-based mode
        provider_type_str = str(cfg_or_provider_type)
        # Type narrowing: validate it's one of the allowed values
        if provider_type_str not in ("spacy", "openai"):
            raise ValueError(f"Invalid provider type: {provider_type_str}")
        experiment_mode = True
        provider_type = cast(Literal["spacy", "openai"], provider_type_str)

    # Convert params to SpeakerDetectionParams if needed
    if experiment_mode:
        if params is None:
            # Use default params if none provided
            # Use default NER model for experiment mode without params
            params = SpeakerDetectionParams(model_name=config.DEFAULT_NER_MODEL)
        elif isinstance(params, dict):
            params = SpeakerDetectionParams(**params)
        elif not isinstance(params, SpeakerDetectionParams):
            raise TypeError(f"params must be SpeakerDetectionParams or dict, got {type(params)}")

    if provider_type == "spacy":
        # Check for preloaded MLProvider instance (from early preloading)
        # Only reuse if in Config mode (experiment mode should create fresh instances)
        if not experiment_mode:
            try:
                from ..workflow import _preloaded_ml_provider

                if _preloaded_ml_provider is not None:
                    return cast(SpeakerDetector, _preloaded_ml_provider)
            except (ImportError, AttributeError):
                # workflow module not available (e.g., in tests), create new instance
                pass

        # Create new instance
        from ..providers.ml.ml_provider import MLProvider

        if experiment_mode:
            # Create a minimal Config from params for experiment mode
            from ..config import Config

            # After conversion above, params is guaranteed to be SpeakerDetectionParams
            assert isinstance(params, SpeakerDetectionParams)
            cfg = Config(
                rss="",  # Dummy, not used for speaker detection (use alias)
                speaker_detector_provider="spacy",
                ner_model=params.model_name,
            )
            return MLProvider(cfg)
        else:
            return MLProvider(cfg)
    elif provider_type == "openai":
        from ..providers.openai.openai_provider import OpenAIProvider

        if experiment_mode:
            # Create a minimal Config from params for experiment mode
            from ..config import Config

            # After conversion above, params is guaranteed to be SpeakerDetectionParams
            assert isinstance(params, SpeakerDetectionParams)
            cfg = Config(
                rss="",  # Dummy, not used for speaker detection (use alias)
                speaker_detector_provider="openai",
                openai_speaker_model=params.model_name if params.model_name else "gpt-4o-mini",
                openai_temperature=params.temperature if params.temperature is not None else 0.3,
                openai_api_key=os.getenv("OPENAI_API_KEY"),  # Load from env
            )
            return OpenAIProvider(cfg)
        else:
            return OpenAIProvider(cfg)
    else:
        raise ValueError(
            f"Unsupported speaker detector type: {provider_type}. "
            "Supported types: 'spacy', 'openai'"
        )
