"""OpenAI GPT API-based summarization provider implementation.

This module provides a SummarizationProvider implementation using OpenAI's GPT API
for cloud-based episode summarization.

Key Advantage: OpenAI GPT models (GPT-4, GPT-4o-mini) have much larger context windows
(128k+ tokens) compared to local transformer models (1k-16k tokens). This means we can
process full transcripts directly without chunking for most podcasts.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from openai import OpenAI

from .. import config

logger = logging.getLogger(__name__)


class OpenAISummarizationProvider:
    """OpenAI GPT API-based summarization provider.

    This provider uses OpenAI's GPT API for cloud-based summarization.
    It implements the SummarizationProvider protocol.

    Note:
        This provider leverages large context windows (128k tokens) to handle
        full transcripts without chunking for most podcasts. Uses prompt_store
        (RFC-017) for versioned, parameterized prompts.
    """

    def __init__(self, cfg: config.Config):
        """Initialize OpenAI summarization provider.

        Args:
            cfg: Configuration object with openai_api_key and summarization settings

        Raises:
            ValueError: If OpenAI API key is not provided
        """
        if not cfg.openai_api_key:
            raise ValueError(
                "OpenAI API key required for OpenAI summarization provider. "
                "Set OPENAI_API_KEY environment variable or openai_api_key in config."
            )

        self.cfg = cfg
        # Support custom base_url for E2E testing with mock servers
        client_kwargs: dict[str, Any] = {"api_key": cfg.openai_api_key}
        if cfg.openai_api_base:
            client_kwargs["base_url"] = cfg.openai_api_base
        self.client = OpenAI(**client_kwargs)
        # Default to gpt-4o-mini (cost-effective with large context window)
        self.model = getattr(cfg, "openai_summary_model", "gpt-4o-mini")
        self.temperature = getattr(cfg, "openai_temperature", 0.3)
        # GPT-4o-mini supports 128k context window - can handle full transcripts
        self.max_context_tokens = 128000  # Conservative estimate
        self._initialized = False

    def initialize(self) -> None:
        """Initialize provider (no local model loading needed for API).

        This method is called to prepare the provider for use.
        For OpenAI API, initialization is a no-op but we track it for consistency.
        """
        if self._initialized:
            return

        logger.debug("Initializing OpenAI summarization provider (model: %s)", self.model)
        self._initialized = True
        logger.debug("OpenAI summarization provider initialized successfully")

    def summarize(
        self,
        text: str,
        episode_title: Optional[str] = None,
        episode_description: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Summarize text using OpenAI GPT API.

        Can handle full transcripts directly due to large context window (128k+ tokens).
        No chunking needed for most podcast transcripts.

        Args:
            text: Transcript text to summarize
            episode_title: Optional episode title
            episode_description: Optional episode description
            params: Optional parameters dict with:
                - max_length: Maximum summary length in tokens (default from config)
                - min_length: Minimum summary length in tokens (default from config)
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
        if not self._initialized:
            raise RuntimeError(
                "OpenAISummarizationProvider not initialized. Call initialize() first."
            )

        # Extract parameters with defaults from config
        max_length = (params.get("max_length") if params else None) or self.cfg.summary_max_length
        min_length = (params.get("min_length") if params else None) or self.cfg.summary_min_length
        custom_prompt = params.get("prompt") if params else None

        logger.debug(
            "Summarizing text via OpenAI API (model: %s, max_length: %d)",
            self.model,
            max_length,
        )

        try:
            # Build prompts using prompt_store (RFC-017)
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

            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=max_length,
            )

            summary = response.choices[0].message.content
            if not summary:
                logger.warning("OpenAI API returned empty summary")
                summary = ""

            logger.debug("OpenAI summarization completed: %d characters", len(summary))

            # Get prompt metadata for tracking (RFC-017)
            from ..prompt_store import get_prompt_metadata

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

            return {
                "summary": summary,
                # OpenAI provider doesn't generate short summaries separately
                "summary_short": None,
                "metadata": {
                    "model": self.model,
                    "provider": "openai",
                    "max_length": max_length,
                    "min_length": min_length,
                    "prompts": prompt_metadata,
                },
            }

        except Exception as exc:
            logger.error("OpenAI API error in summarization: %s", exc)
            raise ValueError(f"OpenAI summarization failed: {exc}") from exc

    def _build_summarization_prompts(
        self,
        text: str,
        episode_title: Optional[str],
        episode_description: Optional[str],
        max_length: int,
        min_length: int,
        custom_prompt: Optional[str],
    ) -> tuple[str, str, Optional[str], str, int, int]:
        """Build system and user prompts for summarization using prompt_store (RFC-017).

        Args:
            text: Transcript text to summarize
            episode_title: Optional episode title
            episode_description: Optional episode description
            max_length: Maximum summary length in tokens
            min_length: Minimum summary length in tokens
            custom_prompt: Optional custom prompt (overrides default)

        Returns:
            Tuple of (system_prompt, user_prompt, system_prompt_name,
            user_prompt_name, paragraphs_min, paragraphs_max)
        """
        from ..prompt_store import render_prompt

        # Use prompt_store to load versioned prompt templates (RFC-017)
        system_prompt_name = self.cfg.openai_summary_system_prompt or "summarization/system_v1"
        user_prompt_name = self.cfg.openai_summary_user_prompt

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

    def cleanup(self) -> None:
        """Cleanup provider resources (no-op for API provider)."""
        # No resources to clean up for API provider
        pass
