"""Build the translation provider from config (RFC-124 / S2.3).

Mirrors :mod:`podcast_scraper.summarization.factory`, minus its two extra modes: there is no
experiment-params path and no provider-type-override path, because there is exactly one
translation provider today. The factory exists anyway so the call site depends on the
OPERATION rather than on ``GemmaTranslateProvider``, which is what makes a second provider
(the 27B, or an apache-2.0 fallback per ADR-156's alternatives) a config change rather than a
code change.
"""

from __future__ import annotations

import logging
from typing import Optional

from .. import config
from .base import TranslationProvider

logger = logging.getLogger(__name__)

#: Provider ids this factory can build. One today; the shape is what matters.
SUPPORTED_PROVIDERS = ("gemma_translate",)

DEFAULT_PROVIDER = "gemma_translate"


class TranslationProviderUnavailable(RuntimeError):
    """Translation was asked for and cannot be provided."""


def translation_provider_name(cfg: config.Config) -> str:
    return str(getattr(cfg, "translate_provider", None) or DEFAULT_PROVIDER)


def is_translation_configured(cfg: config.Config) -> bool:
    """Whether a translator could be built AND has an endpoint and a model to call.

    Separate from ``multilingual_ingest``: the flag says whether we WANT to translate, this says
    whether we COULD. Keeping them apart is what lets the stage record `flag_off` distinctly from
    a misconfigured endpoint — one is a decision, the other is a defect.
    """
    return bool(getattr(cfg, "translate_api_base", None) and getattr(cfg, "translate_model", None))


def create_translation_provider(
    cfg: config.Config, *, provider_type: Optional[str] = None
) -> TranslationProvider:
    """Return an initialized translation provider.

    Raises:
        TranslationProviderUnavailable: unknown provider id, or no endpoint/model configured.
    """
    name = provider_type or translation_provider_name(cfg)
    if name not in SUPPORTED_PROVIDERS:
        raise TranslationProviderUnavailable(
            f"unknown translation provider {name!r}; supported: {list(SUPPORTED_PROVIDERS)}"
        )
    if not is_translation_configured(cfg):
        raise TranslationProviderUnavailable(
            "translate_api_base / translate_model are unset, so there is no translator to build. "
            "These are registry-governed fields — set them on the profile, not ad hoc."
        )

    from ..providers.vllm.translate_provider import GemmaTranslateProvider

    provider = GemmaTranslateProvider(cfg)
    provider.initialize()
    logger.info(
        "    translation provider: %s (%s)",
        name,
        getattr(cfg, "translate_model", None),
    )
    return provider
