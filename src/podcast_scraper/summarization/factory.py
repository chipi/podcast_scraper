"""Factory for creating summarization providers.

This module provides a factory function to create summarization
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
    from podcast_scraper.providers.params import SummarizationParams
    from podcast_scraper.summarization.base import SummarizationProvider
else:
    from podcast_scraper import config
    from podcast_scraper.providers.params import SummarizationParams
    from podcast_scraper.summarization.base import SummarizationProvider

from podcast_scraper.utils.protocol_verification import verify_protocol_compliance


def create_summarization_provider(  # noqa: C901
    cfg_or_provider_type: Union[config.Config, str],
    params: Optional[Union[SummarizationParams, Dict[str, Any]]] = None,
) -> SummarizationProvider:
    """Create a summarization provider based on configuration or experiment params.

    Supports two modes:
    1. Config-based: Pass a Config object
    2. Experiment-based: Pass provider_type string and params dict/object

    Args:
        cfg_or_provider_type: Either a Config object or provider type string
            ("transformers", "openai", "gemini", "mistral", "grok", "deepseek",
            "ollama", or "anthropic")
        params: Optional parameters dict or SummarizationParams object (experiment mode only)

    Returns:
        SummarizationProvider instance

    Raises:
        ValueError: If provider type is not supported or params are invalid
        TypeError: If params are provided but cfg_or_provider_type is a Config object

    Note:
        Returns MLProvider for "transformers" provider type (unified ML provider).
        Returns OpenAIProvider for "openai" provider type (unified OpenAI provider).
        Reuses preloaded MLProvider instance if available (from early preloading).

    Example (Config-based):
        >>> from podcast_scraper import Config
        >>> cfg = Config(rss_url="...", summary_provider="transformers")
        >>> provider = create_summarization_provider(cfg)

    Example (Experiment-based, new):
        >>> from podcast_scraper.providers.params import SummarizationParams
        >>> params = SummarizationParams(
        ...     model_name="facebook/bart-large-cnn",
        ...     max_length=150,
        ...     device="mps"
        ... )
        >>> provider = create_summarization_provider("transformers", params)
    """
    # Determine mode: Config-based or experiment-based
    if isinstance(cfg_or_provider_type, config.Config):
        # Config-based mode
        if params is not None:
            raise TypeError(
                "Cannot provide params when using Config-based mode. "
                "Use experiment mode: create_summarization_provider(provider_type, params)"
            )
        cfg = cfg_or_provider_type
        provider_type = cfg.summary_provider
        experiment_mode = False
    else:
        # Experiment-based mode
        provider_type_str = str(cfg_or_provider_type)
        # Type narrowing: validate it's one of the allowed values
        if provider_type_str not in (
            "transformers",
            "openai",
            "gemini",
            "mistral",
            "grok",
            "deepseek",
            "ollama",
            "anthropic",
        ):
            raise ValueError(f"Invalid provider type: {provider_type_str}")

        provider_type_value = cast(
            Literal[
                "transformers",
                "openai",
                "gemini",
                "mistral",
                "grok",
                "deepseek",
                "ollama",
                "anthropic",
            ],
            provider_type_str,
        )
        experiment_mode = True
        provider_type = provider_type_value

    # Convert params to SummarizationParams if needed
    if experiment_mode:
        if params is None:
            # Use default params if none provided
            # Use test default model for experiment mode without params
            params = SummarizationParams(model_name=config.TEST_DEFAULT_SUMMARY_MODEL)
        elif isinstance(params, dict):
            params = SummarizationParams(**params)
        elif not isinstance(params, SummarizationParams):
            raise TypeError(f"params must be SummarizationParams or dict, got {type(params)}")

    if provider_type == "transformers":
        # Check for preloaded MLProvider instance (from early preloading)
        # Only reuse if in Config mode (experiment mode should create fresh instances)
        if not experiment_mode:
            try:
                from ..workflow import _preloaded_ml_provider

                if _preloaded_ml_provider is not None:
                    provider = cast(SummarizationProvider, _preloaded_ml_provider)
                    # Runtime protocol verification (dev-mode only)
                    verify_protocol_compliance(
                        provider, SummarizationProvider, "SummarizationProvider"
                    )
                    return provider
            except (ImportError, AttributeError):
                # workflow module not available (e.g., in tests), create new instance
                pass

        # Create new instance
        from ..providers.ml.ml_provider import MLProvider

        if experiment_mode:
            # Create a minimal Config from params for experiment mode
            # This is a temporary bridge until providers fully support params
            from ..config import Config

            # After conversion above, params is guaranteed to be SummarizationParams
            assert isinstance(params, SummarizationParams)
            # Map old params to new map/reduce params structure
            map_max = params.max_length if params.max_length else 200
            map_min = params.min_length if params.min_length else 80
            reduce_max = params.max_length if params.max_length else 650
            reduce_min = params.min_length if params.min_length else 220

            cfg = Config(
                rss="",  # Dummy, not used for summarization (use alias)
                summary_provider="transformers",
                summary_model=params.model_name if params.model_name else None,
                summary_map_params={
                    "max_new_tokens": map_max,
                    "min_new_tokens": map_min,
                },
                summary_reduce_params={
                    "max_new_tokens": reduce_max,
                    "min_new_tokens": reduce_min,
                },
                summary_device=params.device,
                summary_reduce_model=params.reduce_model,
                summary_chunk_size=params.chunk_size,
                summary_chunk_parallelism=(
                    params.chunk_parallelism if params.chunk_parallelism else 1
                ),
                summary_word_chunk_size=params.word_chunk_size,
                summary_word_overlap=params.word_overlap,
                summary_cache_dir=params.cache_dir,
            )
            provider = MLProvider(cfg)
        else:
            provider = MLProvider(cfg)

        # Runtime protocol verification (dev-mode only)
        verify_protocol_compliance(provider, SummarizationProvider, "SummarizationProvider")
        return provider
    elif provider_type == "openai":
        from ..providers.openai.openai_provider import OpenAIProvider

        if experiment_mode:
            # Create a minimal Config from params for experiment mode
            from ..config import Config

            # After conversion above, params is guaranteed to be SummarizationParams
            assert isinstance(params, SummarizationParams)
            cfg = Config(
                rss="",  # Dummy, not used for summarization (use alias)
                summary_provider="openai",
                generate_summaries=True,  # Required for OpenAI provider initialization
                generate_metadata=True,  # Required when generate_summaries=True
                openai_summary_model=params.model_name if params.model_name else "gpt-4o-mini",
                openai_temperature=params.temperature if params.temperature is not None else 0.3,
                openai_api_key=os.getenv("OPENAI_API_KEY"),  # Load from env
                openai_max_tokens=params.max_length if params.max_length else None,
            )
            provider = OpenAIProvider(cfg)
        else:
            provider = OpenAIProvider(cfg)

        # Runtime protocol verification (dev-mode only)
        verify_protocol_compliance(provider, SummarizationProvider, "SummarizationProvider")
        return provider
    elif provider_type == "gemini":
        from ..providers.gemini.gemini_provider import GeminiProvider

        if experiment_mode:
            # Create a minimal Config from params for experiment mode
            from ..config import Config

            # After conversion above, params is guaranteed to be SummarizationParams
            assert isinstance(params, SummarizationParams)
            cfg = Config(
                rss="",  # Dummy, not used for summarization (use alias)
                summary_provider="gemini",
                generate_summaries=True,  # Required for Gemini provider initialization
                generate_metadata=True,  # Required when generate_summaries=True
                gemini_summary_model=params.model_name if params.model_name else "gemini-2.0-flash",
                gemini_temperature=params.temperature if params.temperature is not None else 0.3,
                gemini_api_key=os.getenv("GEMINI_API_KEY"),  # Load from env
                gemini_max_tokens=params.max_length if params.max_length else None,
            )
            provider = GeminiProvider(cfg)
        else:
            provider = GeminiProvider(cfg)

        # Runtime protocol verification (dev-mode only)
        verify_protocol_compliance(provider, SummarizationProvider, "SummarizationProvider")
        return provider
    elif provider_type == "mistral":
        from ..providers.mistral.mistral_provider import MistralProvider

        if experiment_mode:
            # Create a minimal Config from params for experiment mode
            from ..config import Config

            # After conversion above, params is guaranteed to be SummarizationParams
            assert isinstance(params, SummarizationParams)
            cfg = Config(
                rss="",  # Dummy, not used for summarization (use alias)
                summary_provider="mistral",
                generate_summaries=True,  # Required for Mistral provider initialization
                generate_metadata=True,  # Required when generate_summaries=True
                mistral_summary_model=(
                    params.model_name if params.model_name else "mistral-small-latest"
                ),
                mistral_temperature=params.temperature if params.temperature is not None else 0.3,
                mistral_api_key=os.getenv("MISTRAL_API_KEY"),  # Load from env
                mistral_max_tokens=params.max_length if params.max_length else None,
            )
            provider = MistralProvider(cfg)
        else:
            provider = MistralProvider(cfg)

        # Runtime protocol verification (dev-mode only)
        verify_protocol_compliance(provider, SummarizationProvider, "SummarizationProvider")
        return provider
    elif provider_type == "grok":
        from ..providers.grok.grok_provider import GrokProvider

        if experiment_mode:
            # Create a minimal Config from params for experiment mode
            from ..config import Config

            # After conversion above, params is guaranteed to be SummarizationParams
            assert isinstance(params, SummarizationParams)
            cfg = Config(
                rss="",  # Dummy, not used for summarization (use alias)
                summary_provider="grok",
                generate_summaries=True,  # Required for Grok provider initialization
                generate_metadata=True,  # Required when generate_summaries=True
                grok_summary_model=(params.model_name if params.model_name else "grok-2"),
                grok_temperature=params.temperature if params.temperature is not None else 0.3,
                grok_api_key=os.getenv("GROK_API_KEY"),  # Load from env
                grok_max_tokens=params.max_length if params.max_length else None,
            )
            provider = GrokProvider(cfg)
        else:
            provider = GrokProvider(cfg)

        # Runtime protocol verification (dev-mode only)
        verify_protocol_compliance(provider, SummarizationProvider, "SummarizationProvider")
        return provider
    elif provider_type == "deepseek":
        from ..providers.deepseek.deepseek_provider import DeepSeekProvider

        if experiment_mode:
            # Create a minimal Config from params for experiment mode
            from ..config import Config

            # After conversion above, params is guaranteed to be SummarizationParams
            assert isinstance(params, SummarizationParams)
            cfg = Config(
                rss="",  # Dummy, not used for summarization (use alias)
                summary_provider="deepseek",
                generate_summaries=True,  # Required for DeepSeek provider initialization
                generate_metadata=True,  # Required when generate_summaries=True
                deepseek_summary_model=params.model_name if params.model_name else "deepseek-chat",
                deepseek_temperature=params.temperature if params.temperature is not None else 0.3,
                deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),  # Load from env
                deepseek_max_tokens=params.max_length if params.max_length else None,
            )
            provider = DeepSeekProvider(cfg)
        else:
            provider = DeepSeekProvider(cfg)

        # Runtime protocol verification (dev-mode only)
        verify_protocol_compliance(provider, SummarizationProvider, "SummarizationProvider")
        return provider
    elif provider_type == "ollama":
        from ..providers.ollama.ollama_provider import OllamaProvider

        if experiment_mode:
            # Create a minimal Config from params for experiment mode
            from ..config import Config

            # After conversion above, params is guaranteed to be SummarizationParams
            assert isinstance(params, SummarizationParams)
            cfg = Config(
                rss="",  # Dummy, not used for summarization (use alias)
                summary_provider="ollama",
                generate_summaries=True,  # Required for Ollama provider initialization
                generate_metadata=True,  # Required when generate_summaries=True
                ollama_summary_model=params.model_name if params.model_name else "llama3.3:latest",
                ollama_temperature=params.temperature if params.temperature is not None else 0.3,
                ollama_api_base=os.getenv("OLLAMA_API_BASE", "http://localhost:11434/v1"),
                ollama_max_tokens=params.max_length if params.max_length else None,
            )
            provider = OllamaProvider(cfg)
        else:
            provider = OllamaProvider(cfg)

        # Runtime protocol verification (dev-mode only)
        verify_protocol_compliance(provider, SummarizationProvider, "SummarizationProvider")
        return provider
    elif provider_type == "anthropic":
        from ..providers.anthropic.anthropic_provider import AnthropicProvider

        if experiment_mode:
            # Create a minimal Config from params for experiment mode
            from ..config import Config

            # After conversion above, params is guaranteed to be SummarizationParams
            assert isinstance(params, SummarizationParams)
            cfg = Config(
                rss="",  # Dummy, not used for summarization (use alias)
                summary_provider="anthropic",
                generate_summaries=True,  # Required for Anthropic provider initialization
                generate_metadata=True,  # Required when generate_summaries=True
                anthropic_summary_model=(
                    params.model_name if params.model_name else "claude-3-5-sonnet-20241022"
                ),
                anthropic_temperature=(
                    params.temperature if params.temperature is not None else 0.3
                ),
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),  # Load from env
                anthropic_max_tokens=params.max_length if params.max_length else None,
            )
            provider = AnthropicProvider(cfg)
        else:
            provider = AnthropicProvider(cfg)

        # Runtime protocol verification (dev-mode only)
        verify_protocol_compliance(provider, SummarizationProvider, "SummarizationProvider")
        return provider
    else:
        raise ValueError(
            f"Unsupported summarization provider: {provider_type}. "
            "Supported providers: 'transformers', 'openai', 'gemini', 'grok', "
            "'deepseek', 'mistral', 'ollama', 'anthropic'."
        )
