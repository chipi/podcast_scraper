"""OpenAI API-based speaker detection provider implementation.

This module provides a SpeakerDetector implementation using OpenAI's GPT API
for automatic speaker name detection from episode metadata.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Set, Tuple

from openai import OpenAI

from .. import config, models

logger = logging.getLogger(__name__)

# Default speaker names when detection fails
DEFAULT_SPEAKER_NAMES = ["Host", "Guest"]


class OpenAISpeakerDetector:
    """OpenAI API-based speaker detection provider.

    This provider uses OpenAI's GPT API for automatic speaker name detection.
    It implements the SpeakerDetector protocol.

    Note:
        This provider uses prompt_store (RFC-017) for versioned, parameterized prompts.
        Prompts are loaded from prompts/ner/ directory.
    """

    def __init__(self, cfg: config.Config):
        """Initialize OpenAI speaker detection provider.

        Args:
            cfg: Configuration object with openai_api_key and speaker detection settings

        Raises:
            ValueError: If OpenAI API key is not provided
        """
        if not cfg.openai_api_key:
            raise ValueError(
                "OpenAI API key required for OpenAI speaker detector. "
                "Set OPENAI_API_KEY environment variable or openai_api_key in config."
            )

        self.cfg = cfg
        self.client = OpenAI(api_key=cfg.openai_api_key)
        # Default to gpt-4o-mini (cost-effective with good quality)
        self.model = getattr(cfg, "openai_speaker_model", "gpt-4o-mini")
        self.temperature = getattr(cfg, "openai_temperature", 0.3)
        self._initialized = False

    def initialize(self) -> None:
        """Initialize provider (no local model loading needed for API).

        This method is called to prepare the provider for use.
        For OpenAI API, initialization is a no-op but we track it for consistency.
        """
        if self._initialized:
            return

        logger.debug("Initializing OpenAI speaker detection provider (model: %s)", self.model)
        self._initialized = True
        logger.debug("OpenAI speaker detection provider initialized successfully")

    def detect_speakers(
        self,
        episode_title: str,
        episode_description: str | None,
        known_hosts: Set[str],
    ) -> Tuple[list[str], Set[str], bool]:
        """Detect speaker names from episode metadata using OpenAI API.

        Args:
            episode_title: Episode title
            episode_description: Optional episode description
            known_hosts: Set of known host names (for context)

        Returns:
            Tuple of:
            - List of detected speaker names (hosts + guests)
            - Set of detected host names (subset of known_hosts)
            - Success flag (True if detection succeeded)

        Raises:
            ValueError: If detection fails or API key is invalid
            RuntimeError: If provider is not initialized
        """
        if not self._initialized:
            raise RuntimeError("OpenAISpeakerDetector not initialized. Call initialize() first.")

        if not self.cfg.auto_speakers:
            logger.debug("Auto-speakers disabled, detection failed")
            return DEFAULT_SPEAKER_NAMES.copy(), set(), False

        logger.debug("Detecting speakers via OpenAI API for episode: %s", episode_title[:50])

        try:
            # Build prompt using prompt_store (RFC-017)
            user_prompt = self._build_speaker_detection_prompt(
                episode_title, episode_description, known_hosts
            )

            # Get system prompt from prompt_store (RFC-017)
            from ..prompt_store import render_prompt

            system_prompt_name = self.cfg.openai_speaker_system_prompt or "ner/system_ner_v1"
            system_prompt = render_prompt(system_prompt_name)

            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=300,
                response_format={"type": "json_object"},  # Request JSON response
            )

            response_text = response.choices[0].message.content
            if not response_text:
                logger.warning("OpenAI API returned empty response")
                return DEFAULT_SPEAKER_NAMES.copy(), set(), False

            # Parse JSON response
            speakers, detected_hosts, success = self._parse_speakers_from_response(
                response_text, known_hosts
            )

            logger.debug(
                "OpenAI speaker detection completed: %d speakers, %d hosts, success=%s",
                len(speakers),
                len(detected_hosts),
                success,
            )

            # Note: Prompt metadata tracking could be added here if needed for results
            # For now, we focus on summarization metadata tracking

            return speakers, detected_hosts, success

        except json.JSONDecodeError as exc:
            logger.error("Failed to parse OpenAI API JSON response: %s", exc)
            logger.debug(
                "Response text: %s", response_text if "response_text" in locals() else "N/A"
            )
            return DEFAULT_SPEAKER_NAMES.copy(), set(), False
        except Exception as exc:
            logger.error("OpenAI API error in speaker detection: %s", exc)
            raise ValueError(f"OpenAI speaker detection failed: {exc}") from exc

    def analyze_patterns(
        self,
        episodes: list[models.Episode],
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

    def _build_speaker_detection_prompt(
        self,
        episode_title: str,
        episode_description: str | None,
        known_hosts: Set[str],
    ) -> str:
        """Build prompt for speaker detection using prompt_store (RFC-017).

        Args:
            episode_title: Episode title
            episode_description: Optional episode description
            known_hosts: Set of known host names

        Returns:
            Rendered prompt string
        """
        from ..prompt_store import render_prompt

        # Use prompt_store to load versioned prompt template (RFC-017)
        prompt_name = self.cfg.openai_speaker_user_prompt

        # Merge config params with template params
        template_params = {
            "episode_title": episode_title,
            "episode_description": episode_description or "",
            "known_hosts": ", ".join(sorted(known_hosts)) if known_hosts else "",
        }
        template_params.update(self.cfg.ner_prompt_params)

        return render_prompt(prompt_name, **template_params)

    def _parse_speakers_from_response(
        self, response_text: str, known_hosts: Set[str]
    ) -> Tuple[list[str], Set[str], bool]:
        """Parse speaker names from OpenAI API JSON response.

        Args:
            response_text: JSON response from OpenAI API
            known_hosts: Set of known host names (for filtering)

        Returns:
            Tuple of (speaker_names_list, detected_hosts_set, detection_succeeded)
        """
        try:
            data = json.loads(response_text)

            # Extract speakers, hosts, and guests from JSON
            all_speakers = data.get("speakers", [])
            detected_hosts_list = data.get("hosts", [])
            guests_list = data.get("guests", [])

            # Normalize names (strip whitespace, filter empty)
            all_speakers = [name.strip() for name in all_speakers if name.strip()]
            detected_hosts_list = [name.strip() for name in detected_hosts_list if name.strip()]
            guests_list = [name.strip() for name in guests_list if name.strip()]

            # Filter hosts to only include those in known_hosts
            detected_hosts = {host for host in detected_hosts_list if host in known_hosts}

            # Build speaker names list: hosts first, then guests
            speaker_names = list(detected_hosts) + guests_list

            # Ensure we have at least MIN_SPEAKERS_REQUIRED speakers
            min_speakers = getattr(self.cfg, "screenplay_num_speakers", 2)
            if len(speaker_names) < min_speakers:
                # Add default speakers if needed
                defaults_needed = min_speakers - len(speaker_names)
                speaker_names.extend(DEFAULT_SPEAKER_NAMES[:defaults_needed])

            # Detection succeeded if we have real names (not just defaults)
            detection_succeeded = bool(detected_hosts or guests_list or (len(all_speakers) > 0))

            return speaker_names[:min_speakers], detected_hosts, detection_succeeded

        except (json.JSONDecodeError, KeyError, AttributeError) as exc:
            logger.warning("Failed to parse OpenAI response as JSON: %s", exc)
            logger.debug("Response text: %s", response_text[:200])
            # Fallback: try to extract names from text response
            return self._parse_speakers_from_text(response_text, known_hosts)

    def _parse_speakers_from_text(
        self, response_text: str, known_hosts: Set[str]
    ) -> Tuple[list[str], Set[str], bool]:
        """Fallback: Parse speaker names from text response (not JSON).

        Args:
            response_text: Text response from OpenAI API
            known_hosts: Set of known host names

        Returns:
            Tuple of (speaker_names_list, detected_hosts_set, detection_succeeded)
        """
        # Try to extract names using simple patterns
        # Look for common patterns like "Speakers: Name1, Name2" or "Hosts: Name1"
        names = set()

        # Pattern: "speakers": ["Name1", "Name2"]
        json_pattern = r'"speakers"\s*:\s*\[(.*?)\]'
        match = re.search(json_pattern, response_text, re.IGNORECASE)
        if match:
            names_str = match.group(1)
            # Extract quoted strings
            quoted_names = re.findall(r'"([^"]+)"', names_str)
            names.update(quoted_names)

        # Pattern: "Hosts: Name1, Name2"
        hosts_pattern = r"hosts?\s*:\s*([^\n]+)"
        match = re.search(hosts_pattern, response_text, re.IGNORECASE)
        if match:
            hosts_str = match.group(1)
            host_names = [n.strip() for n in re.split(r"[,;]", hosts_str)]
            names.update(host_names)

        # Pattern: "Guests: Name1, Name2"
        guests_pattern = r"guests?\s*:\s*([^\n]+)"
        match = re.search(guests_pattern, response_text, re.IGNORECASE)
        if match:
            guests_str = match.group(1)
            guest_names = [n.strip() for n in re.split(r"[,;]", guests_str)]
            names.update(guest_names)

        # Filter out generic names
        names = {n for n in names if n.lower() not in ("host", "guest", "speaker")}

        # Separate hosts and guests
        detected_hosts = {name for name in names if name in known_hosts}
        guests = [name for name in names if name not in known_hosts]

        # Build speaker names list
        speaker_names = list(detected_hosts) + guests
        min_speakers = getattr(self.cfg, "screenplay_num_speakers", 2)
        if len(speaker_names) < min_speakers:
            speaker_names.extend(DEFAULT_SPEAKER_NAMES[: min_speakers - len(speaker_names)])

        detection_succeeded = bool(detected_hosts or guests)

        return speaker_names[:min_speakers], detected_hosts, detection_succeeded

    def cleanup(self) -> None:
        """Cleanup provider resources (no-op for API provider)."""
        # No resources to clean up for API provider
        pass
