"""The ``translation`` operation (RFC-124).

Source-language text in, English out, so every downstream intelligence stage reads one
language. Shaped like :mod:`podcast_scraper.summarization`: a Protocol, a factory, and
providers that live under ``providers/``.
"""

from .base import TranslationProvider
from .factory import (
    create_translation_provider,
    is_translation_configured,
    SUPPORTED_PROVIDERS,
    translation_provider_name,
    TranslationProviderUnavailable,
)

__all__ = [
    "SUPPORTED_PROVIDERS",
    "TranslationProvider",
    "TranslationProviderUnavailable",
    "create_translation_provider",
    "is_translation_configured",
    "translation_provider_name",
]
