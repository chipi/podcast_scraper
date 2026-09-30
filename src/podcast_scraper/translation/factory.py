"""Build the translation provider from config (RFC-124 / S2.3).

Mirrors :mod:`podcast_scraper.summarization.factory`, minus its two extra modes: there is no
experiment-params path and no provider-type-override path, because there is exactly one
translation provider today. The factory exists anyway so the call site depends on the
OPERATION rather than on ``GemmaTranslateProvider``, which is what makes a second provider
(the 27B, or an apache-2.0 fallback per ADR-157's alternatives) a config change rather than a
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

    A question about DEPLOYMENT, not policy — and since 2026-09-30 it is the only such question
    the stage asks. There was a `multilingual_ingest` flag beside it meaning "do we WANT to
    translate", and it was removed: whether we ingest a language is already decided, per
    language, by `enabled` in ``config/languages.yaml``, and the flag could only ever be
    consulted for an episode that gate had ALREADY approved. Its one distinct state — an
    approved non-English episode deliberately left untranslated — is a transcript with no
    intelligence layer, which is exactly what a FAILED translation produces anyway.

    So a False here is a misconfiguration: a language was enabled with no translator deployed.
    The stage records ``translator_not_configured`` for it rather than treating it as a choice.
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
