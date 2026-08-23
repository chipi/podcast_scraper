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

from podcast_scraper.utils.protocol_verification import verify_protocol_compliance


def create_speaker_detector(
    cfg_or_provider_type: Union[config.Config, str],
    params: Optional[Union[SpeakerDetectionParams, Dict[str, Any]]] = None,
) -> SpeakerDetector:
    """Build the configured speaker detector, wrapped in the provider failover chain.

    Speaker detection was the one LLM stage with no failover at all. Its provider is built here
    rather than through the summarization factory, so the wrapping applied to summary / GI / KG
    never reached it — and when the OpenRouter account hit its weekly limit a production job
    died with ``ProviderRuntimeError ... no budget/credit left on this key`` instead of failing
    over to native DeepSeek, which implements ``detect_speakers`` perfectly well
    (``DeepSeekProvider`` extends the same ``OpenAICompatibleProvider`` base).

    The wrap happens HERE, at the single choke point, not at the call sites. The builder below
    has thirteen ``return provider`` statements and the pipeline resolves detectors from more
    than one place; wrapping per-site is the same allowlist mistake that left eight LLM methods
    unprotected in ``summarization/fallback.py``. One entry point, one rule.
    """
    detector = _build_speaker_detector(cfg_or_provider_type, params)
    if isinstance(cfg_or_provider_type, config.Config):
        from ..summarization.fallback import wrap_with_fallback_if_configured

        # No-op unless a ladder is configured, so experiment/test paths are unaffected. The
        # primary is THIS stage's provider, not the summary one — that is what makes "never fail
        # over to the provider that just failed" correct when the two stages differ.
        return cast(
            SpeakerDetector,
            wrap_with_fallback_if_configured(
                detector,
                cfg_or_provider_type,
                primary_name=getattr(cfg_or_provider_type, "speaker_detector_provider", None),
            ),
        )
    return detector


def _build_speaker_detector(  # noqa: C901
    cfg_or_provider_type: Union[config.Config, str],
    params: Optional[Union[SpeakerDetectionParams, Dict[str, Any]]] = None,
) -> SpeakerDetector:
    """Create a speaker detection provider based on configuration or experiment params.

    Supports two modes:
    1. Config-based: Pass a Config object
    2. Experiment-based: Pass provider_type string and params dict/object

    Args:
        cfg_or_provider_type: Either a Config object or provider type string
            ("spacy", "openai", "gemini", "mistral", "grok", or "deepseek")
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
        if provider_type_str not in (
            "spacy",
            "openai",
            "gemini",
            "mistral",
            "grok",
            "ollama",
            "deepseek",
            "anthropic",
            "vllm",
            "litellm",
            "qwen",
            "groq",
        ):
            raise ValueError(f"Invalid provider type: {provider_type_str}")
        experiment_mode = True
        provider_type = cast(
            Literal[
                "spacy",
                "openai",
                "gemini",
                "mistral",
                "grok",
                "ollama",
                "deepseek",
                "anthropic",
                "vllm",
                "groq",
            ],
            provider_type_str,
        )

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
                    provider = cast(SpeakerDetector, _preloaded_ml_provider)
                    # Runtime protocol verification (dev-mode only)
                    verify_protocol_compliance(provider, SpeakerDetector, "SpeakerDetector")
                    return provider
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
            provider = MLProvider(cfg)
        else:
            provider = MLProvider(cfg)

        # Runtime protocol verification (dev-mode only)
        verify_protocol_compliance(provider, SpeakerDetector, "SpeakerDetector")
        return provider
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
            provider = OpenAIProvider(cfg)
        else:
            provider = OpenAIProvider(cfg)

        # Runtime protocol verification (dev-mode only)
        verify_protocol_compliance(provider, SpeakerDetector, "SpeakerDetector")
        return provider
    elif provider_type == "gemini":
        from ..providers.gemini.gemini_provider import GeminiProvider

        if experiment_mode:
            # Create a minimal Config from params for experiment mode
            from ..config import Config

            # After conversion above, params is guaranteed to be SpeakerDetectionParams
            assert isinstance(params, SpeakerDetectionParams)
            cfg = Config(
                rss="",  # Dummy, not used for speaker detection (use alias)
                speaker_detector_provider="gemini",
                gemini_speaker_model=(
                    params.model_name if params.model_name else "gemini-2.5-flash-lite"
                ),
                gemini_temperature=params.temperature if params.temperature is not None else 0.3,
                gemini_api_key=os.getenv("GEMINI_API_KEY"),  # Load from env
            )
            provider = GeminiProvider(cfg)
        else:
            provider = GeminiProvider(cfg)

        # Runtime protocol verification (dev-mode only)
        verify_protocol_compliance(provider, SpeakerDetector, "SpeakerDetector")
        return provider
    elif provider_type == "mistral":
        from ..providers.mistral.mistral_provider import MistralProvider

        if experiment_mode:
            # Create a minimal Config from params for experiment mode
            from ..config import Config

            # After conversion above, params is guaranteed to be SpeakerDetectionParams
            assert isinstance(params, SpeakerDetectionParams)
            cfg = Config(
                rss="",  # Dummy, not used for speaker detection (use alias)
                speaker_detector_provider="mistral",
                mistral_speaker_model=(
                    params.model_name if params.model_name else "mistral-small-latest"
                ),
                mistral_temperature=params.temperature if params.temperature is not None else 0.3,
                mistral_api_key=os.getenv("MISTRAL_API_KEY"),  # Load from env
            )
            provider = MistralProvider(cfg)
        else:
            provider = MistralProvider(cfg)

        # Runtime protocol verification (dev-mode only)
        verify_protocol_compliance(provider, SpeakerDetector, "SpeakerDetector")
        return provider
    elif provider_type == "grok":
        from ..providers.grok.grok_provider import GrokProvider

        if experiment_mode:
            # Create a minimal Config from params for experiment mode
            from ..config import Config

            # After conversion above, params is guaranteed to be SpeakerDetectionParams
            assert isinstance(params, SpeakerDetectionParams)
            cfg = Config(
                rss="",  # Dummy, not used for speaker detection (use alias)
                speaker_detector_provider="grok",
                grok_speaker_model=(params.model_name if params.model_name else "grok-4.3"),
                grok_temperature=params.temperature if params.temperature is not None else 0.3,
                grok_api_key=os.getenv("GROK_API_KEY"),  # Load from env
            )
            provider = GrokProvider(cfg)
        else:
            provider = GrokProvider(cfg)

        # Runtime protocol verification (dev-mode only)
        verify_protocol_compliance(provider, SpeakerDetector, "SpeakerDetector")
        return provider
    elif provider_type == "deepseek":
        from ..providers.deepseek.deepseek_provider import DeepSeekProvider

        if experiment_mode:
            # Create a minimal Config from params for experiment mode
            from ..config import Config

            # After conversion above, params is guaranteed to be SpeakerDetectionParams
            assert isinstance(params, SpeakerDetectionParams)
            cfg = Config(
                rss="",  # Dummy, not used for speaker detection (use alias)
                speaker_detector_provider="deepseek",
                deepseek_speaker_model=params.model_name if params.model_name else "deepseek-chat",
                deepseek_temperature=params.temperature if params.temperature is not None else 0.3,
                deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),  # Load from env
            )
            provider = DeepSeekProvider(cfg)
        else:
            provider = DeepSeekProvider(cfg)

        # Runtime protocol verification (dev-mode only)
        verify_protocol_compliance(provider, SpeakerDetector, "SpeakerDetector")
        return provider
    elif provider_type == "groq":
        from ..providers.groq.groq_provider import GroqProvider

        if experiment_mode:
            # Create a minimal Config from params for experiment mode
            from ..config import Config

            # After conversion above, params is guaranteed to be SpeakerDetectionParams
            assert isinstance(params, SpeakerDetectionParams)
            cfg = Config(
                rss="",  # Dummy, not used for speaker detection (use alias)
                speaker_detector_provider="groq",
                groq_speaker_model=(
                    params.model_name if params.model_name else "llama-3.3-70b-versatile"
                ),
                groq_temperature=params.temperature if params.temperature is not None else 0.3,
                groq_api_key=os.getenv("GROQ_API_KEY"),  # Load from env
            )
            provider = GroqProvider(cfg)
        else:
            provider = GroqProvider(cfg)

        # Runtime protocol verification (dev-mode only)
        verify_protocol_compliance(provider, SpeakerDetector, "SpeakerDetector")
        return provider
    elif provider_type == "ollama":
        from ..providers.ollama.ollama_provider import OllamaProvider

        if experiment_mode:
            # Create a minimal Config from params for experiment mode
            from ..config import Config

            # After conversion above, params is guaranteed to be SpeakerDetectionParams
            assert isinstance(params, SpeakerDetectionParams)
            cfg = Config(
                rss="",  # Dummy, not used for speaker detection (use alias)
                speaker_detector_provider="ollama",
                ollama_speaker_model=params.model_name if params.model_name else "llama3.1:8b",
                ollama_temperature=params.temperature if params.temperature is not None else 0.3,
                ollama_api_base=os.getenv("OLLAMA_API_BASE", "http://localhost:11434/v1"),
            )
            provider = OllamaProvider(cfg)
        else:
            provider = OllamaProvider(cfg)

        # Runtime protocol verification (dev-mode only)
        verify_protocol_compliance(provider, SpeakerDetector, "SpeakerDetector")
        return provider
    elif provider_type == "vllm":
        # ADR-147: naming/NER on the DGX-local vLLM open model (sibling of openai).
        from ..providers.vllm import VLLMProvider

        if experiment_mode:
            raise NotImplementedError(
                "vLLM speaker detector is not wired for experiment-param mode; drive it with a "
                "Config using speaker_detector_provider='vllm' and the vllm_* settings."
            )
        provider = VLLMProvider(cfg)
        verify_protocol_compliance(provider, SpeakerDetector, "SpeakerDetector")
        return provider
    elif provider_type == "litellm":
        # #1356: naming/NER via the homelab LiteLLM gateway (alias-routed).
        from ..providers.litellm import LiteLLMProvider

        if experiment_mode:
            raise NotImplementedError(
                "LiteLLM speaker detector is not wired for experiment-param mode; drive it with a "
                "Config using speaker_detector_provider='litellm' and the litellm_* settings."
            )
        provider = LiteLLMProvider(cfg)
        verify_protocol_compliance(provider, SpeakerDetector, "SpeakerDetector")
        return provider
    elif provider_type == "qwen":
        # ADR-147: naming/NER on a Qwen3 endpoint (cloud host or DGX vLLM); own `qwen` telemetry.
        from ..providers.qwen import QwenProvider

        if experiment_mode:
            raise NotImplementedError(
                "Qwen speaker detector is not wired for experiment-param mode; drive it with a "
                "Config using speaker_detector_provider='qwen' and the qwen_* settings."
            )
        provider = QwenProvider(cfg)
        verify_protocol_compliance(provider, SpeakerDetector, "SpeakerDetector")
        return provider
    elif provider_type == "anthropic":
        from ..providers.anthropic.anthropic_provider import AnthropicProvider

        if experiment_mode:
            # Create a minimal Config from params for experiment mode
            from ..config import Config

            # After conversion above, params is guaranteed to be SpeakerDetectionParams
            assert isinstance(params, SpeakerDetectionParams)
            cfg = Config(
                rss="",  # Dummy, not used for speaker detection (use alias)
                speaker_detector_provider="anthropic",
                anthropic_speaker_model=(
                    params.model_name if params.model_name else "claude-3-5-sonnet-20241022"
                ),
                anthropic_temperature=(
                    params.temperature if params.temperature is not None else 0.3
                ),
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),  # Load from env
            )
            provider = AnthropicProvider(cfg)
        else:
            provider = AnthropicProvider(cfg)

        # Runtime protocol verification (dev-mode only)
        verify_protocol_compliance(provider, SpeakerDetector, "SpeakerDetector")
        return provider
    else:
        raise ValueError(
            f"Unsupported speaker detector type: {provider_type}. "
            "Supported types: 'spacy', 'openai', 'gemini', 'mistral', 'grok', "
            "'deepseek', 'ollama', 'anthropic', 'groq'"
        )
