# RFC-034: DeepSeek Provider Implementation

- **Status**: Draft
- **Authors**:
- **Stakeholders**: Maintainers, users wanting DeepSeek API integration, cost-conscious users
- **Related PRDs**:
  - `docs/prd/PRD-011-deepseek-provider-integration.md`
- **Related RFCs**:
  - `docs/rfc/RFC-013-openai-provider-implementation.md` (reference implementation)
  - `docs/rfc/RFC-032-anthropic-provider-implementation.md` (similar pattern - no transcription)
  - `docs/rfc/RFC-021-modularization-refactoring-plan.md` (architecture foundation)

## Abstract

Design and implement DeepSeek AI providers for speaker detection and summarization capabilities. DeepSeek offers an OpenAI-compatible API at significantly lower cost (90-95% cheaper), making it ideal for cost-conscious users. Like Anthropic, DeepSeek does NOT support audio transcription. This RFC follows the established provider patterns and leverages the existing OpenAI SDK for implementation.

**Architecture Alignment:** This RFC follows the protocol-based provider system established in RFC-021. DeepSeek providers implement the same protocols (`SpeakerDetector`, `SummarizationProvider`) and integrate via the existing factory pattern. The key implementation detail is using the OpenAI SDK with a custom `base_url` pointing to DeepSeek's API.

## Problem Statement

Users want the option to use DeepSeek AI as an extremely cost-effective alternative for:

1. **Speaker Detection**: Entity extraction using DeepSeek chat models
2. **Summarization**: High-quality summaries using DeepSeek chat models

**Note:** Transcription is NOT supported by DeepSeek (no audio API).

Key advantages of DeepSeek:

- **95% cheaper** than OpenAI for text processing
- **OpenAI-compatible API** - no new SDK required
- **Strong reasoning** with DeepSeek-R1 model

Requirements:

- No changes to end-user experience when using defaults
- Secure API key management
- Per-capability provider selection
- Use OpenAI SDK with custom base_url (no new dependency)
- Handle capability gaps gracefully (transcription not supported)

## Constraints & Assumptions

**Constraints:**

- **Prerequisite**: OpenAI provider implementation (RFC-013) ✅ Completed
- **Backward Compatibility**: Default providers must remain unchanged
- **API Key Security**: API keys never in source code
- **Capability Gap**: DeepSeek does not support audio transcription
- **SDK Reuse**: Use existing OpenAI SDK, no new dependency

**Assumptions:**

- DeepSeek API maintains OpenAI compatibility
- DeepSeek API is stable and available
- Prompts may need optimization for DeepSeek models

## Design & Implementation

### 0. DeepSeek API Overview

DeepSeek provides an OpenAI-compatible API:

| Feature | OpenAI | DeepSeek |
| ------- | ------ | -------- |
| **Base URL** | `https://api.openai.com/v1` | `https://api.deepseek.com` |
| **Chat Endpoint** | `/v1/chat/completions` | `/v1/chat/completions` |
| **Audio API** | ✅ Whisper | ❌ Not available |
| **Context Window** | 128k tokens | 64k tokens |
| **SDK** | `openai` | `openai` (with custom base_url) |

### 1. Architecture Overview

```text
podcast_scraper/
├── deepseek/                       # NEW: Shared DeepSeek utilities
│   ├── __init__.py
│   └── deepseek_provider.py        # Shared client using OpenAI SDK
├── speaker_detectors/
│   ├── base.py                     # SpeakerDetector protocol (existing)
│   ├── factory.py                  # Updated to include DeepSeek
│   └── deepseek_detector.py        # NEW: DeepSeek implementation
├── summarization/
│   ├── base.py                     # SummarizationProvider protocol (existing)
│   ├── factory.py                  # Updated to include DeepSeek
│   └── deepseek_provider.py        # NEW: DeepSeek implementation
├── prompts/
│   └── deepseek/                   # NEW: DeepSeek-specific prompts
│       ├── summarization/
│       │   ├── system_v1.j2
│       │   └── long_v1.j2
│       └── ner/
│           ├── system_ner_v1.j2
│           └── guest_host_v1.j2
└── config.py                       # Updated with DeepSeek fields
```

### 2. Configuration

Add to `config.py`:

```python
from typing import Literal, Optional

# Provider Selection (updated to include deepseek)

speaker_detector_provider: Literal["spacy", "openai", "anthropic", "mistral", "deepseek"] = Field(
    default="spacy",
    description="Speaker detection provider"
)

summary_provider: Literal["transformers", "openai", "anthropic", "mistral", "deepseek"] = Field(
    default="transformers",
    description="Summarization provider"
)

# DeepSeek API Configuration

deepseek_api_key: Optional[str] = Field(
    default=None,
    description="DeepSeek API key (prefer DEEPSEEK_API_KEY env var or .env file)"
)

deepseek_api_base: str = Field(
    default="https://api.deepseek.com",
    description="DeepSeek API base URL"
)

# DeepSeek Model Selection

deepseek_speaker_model: str = Field(
    default="deepseek-chat",
    description="DeepSeek model for speaker detection"
)

deepseek_summary_model: str = Field(
    default="deepseek-chat",
    description="DeepSeek model for summarization"
)

deepseek_temperature: float = Field(
    default=0.3,
    ge=0.0,
    le=2.0,
    description="Temperature for DeepSeek generation"
)

deepseek_max_tokens: Optional[int] = Field(
    default=4096,
    description="Max tokens for DeepSeek generation"
)

# DeepSeek Prompt Configuration

deepseek_summary_system_prompt: str = Field(
    default="deepseek/summarization/system_v1",
    description="DeepSeek system prompt for summarization"
)

deepseek_summary_user_prompt: str = Field(
    default="deepseek/summarization/long_v1",
    description="DeepSeek user prompt for summarization"
)

deepseek_ner_system_prompt: str = Field(
    default="deepseek/ner/system_ner_v1",
    description="DeepSeek system prompt for speaker detection"
)

deepseek_ner_user_prompt: str = Field(
    default="deepseek/ner/guest_host_v1",
    description="DeepSeek user prompt for speaker detection"
)
```

## 3. API Key Management and Validation

```python

# config.py - API Key Loading and Validation

from dotenv import load_dotenv

load_dotenv(override=False)

@field_validator('deepseek_api_key', mode='before')
@classmethod
def load_deepseek_api_key_from_env(cls, v: Any) -> Optional[str]:
    """Load API key from environment variable if not provided."""
    if v is not None:
        return v
    return os.getenv('DEEPSEEK_API_KEY')

@model_validator(mode='after')
def validate_deepseek_config(self) -> 'Config':
    """Validate DeepSeek provider configuration."""
    needs_key = (
        self.speaker_detector_provider == "deepseek" or
        self.summary_provider == "deepseek"
    )
    if needs_key and not self.deepseek_api_key:
        raise ValueError(
            "DeepSeek API key required when using DeepSeek providers. "
            "Set DEEPSEEK_API_KEY environment variable, add it to .env file, "
            "or set deepseek_api_key in config file."
        )
    return self
```

## 4. Provider Capability Validation

```python

# config.py - Capability Validation

@model_validator(mode='after')
def validate_provider_capabilities(self) -> 'Config':
    """Validate provider supports requested capability."""
    # Anthropic doesn't support transcription
    if self.transcription_provider == "anthropic":
        raise ValueError(
            "Anthropic provider does not support transcription. "
            "Use 'whisper' (local), 'openai', or 'mistral' instead."
        )
    # DeepSeek doesn't support transcription
    if self.transcription_provider == "deepseek":
        raise ValueError(
            "DeepSeek provider does not support transcription. "
            "Use 'whisper' (local), 'openai', or 'mistral' instead."
        )
    return self
```

## 5. DeepSeek Provider Implementations

### 5.1 Shared DeepSeek Utilities

**File**: `podcast_scraper/deepseek/deepseek_provider.py`

```python
"""Shared DeepSeek provider utilities.

This module provides shared utilities for DeepSeek API providers.
Key insight: DeepSeek uses an OpenAI-compatible API, so we reuse the OpenAI SDK.
"""

from __future__ import annotations

import logging
from typing import Any

from openai import OpenAI

from .. import config

logger = logging.getLogger(__name__)

def create_deepseek_client(cfg: config.Config) -> OpenAI:

```text

    """Create DeepSeek client using OpenAI SDK with custom base_url.

```
```text

    Args:
        cfg: Configuration object with deepseek_api_key and deepseek_api_base

```
```text

    Returns:
        OpenAI client configured for DeepSeek API

```
```text

    Raises:
        ValueError: If API key is not provided
    """
    if not cfg.deepseek_api_key:
        raise ValueError(
            "DeepSeek API key required. "
            "Set DEEPSEEK_API_KEY environment variable or deepseek_api_key in config."
        )

```
```text

    # Use OpenAI SDK with DeepSeek's base URL
    return OpenAI(
        api_key=cfg.deepseek_api_key,
        base_url=cfg.deepseek_api_base,
    )

```

#### 5.2 Speaker Detection Provider

**File**: `podcast_scraper/speaker_detectors/deepseek_detector.py`

```python

"""DeepSeek AI-based speaker detection provider.

This module provides a SpeakerDetector implementation using DeepSeek's chat API
for cloud-based speaker/guest detection from episode metadata.

Note: Uses OpenAI SDK with DeepSeek's base_url for API compatibility.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from .. import config, models
from ..deepseek.deepseek_provider import create_deepseek_client

logger = logging.getLogger(__name__)

class DeepSeekSpeakerDetector:

```text
    """DeepSeek AI-based speaker detection provider.
```

    This provider uses DeepSeek's chat API for speaker detection.
    It implements the SpeakerDetector protocol.
    """

```python
    def __init__(self, cfg: config.Config):
        """Initialize DeepSeek speaker detector.
```
```text
        Args:
            cfg: Configuration object with deepseek_api_key and speaker settings
```
```text
        Raises:
            ValueError: If DeepSeek API key is not provided
        """
        self.cfg = cfg
        self.client = create_deepseek_client(cfg)
        self.model = cfg.deepseek_speaker_model
        self.temperature = cfg.deepseek_temperature
        self._initialized = False
```
```python
    def initialize(self) -> None:
        """Initialize provider (no local model loading needed for API)."""
        if self._initialized:
            return
```
```text
        logger.debug("Initializing DeepSeek speaker detector (model: %s)", self.model)
        self._initialized = True
        logger.debug("DeepSeek speaker detector initialized successfully")
```
```python
    def detect_hosts(
        self,
        feed_title: str,
        feed_description: Optional[str],
        feed_authors: Optional[List[str]],
    ) -> Set[str]:
        """Detect hosts from feed metadata using DeepSeek API.
```
```text
        Args:
            feed_title: Title of the podcast feed
            feed_description: Optional description of the feed
            feed_authors: Optional list of feed authors
```
```text
        Returns:
            Set of detected host names
        """
        if not self._initialized:
            self.initialize()
```

        system_prompt, user_prompt = self._build_host_detection_prompts(
            feed_title, feed_description, feed_authors
        )

```text
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.cfg.deepseek_max_tokens or 500,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
```

            content = response.choices[0].message.content
            hosts = self._parse_hosts_from_response(content)

```text
            logger.debug("DeepSeek detected hosts: %s", hosts)
            return hosts
```
```python
        except Exception as e:
            logger.error("DeepSeek API error in host detection: %s", e)
            raise ValueError(f"DeepSeek host detection failed: {e}") from e
```
```python
    def detect_speakers(
        self,
        episode_title: str,
        episode_description: Optional[str],
        known_hosts: Set[str],
    ) -> Tuple[List[str], Set[str], bool]:
        """Detect speakers for an episode using DeepSeek API.
```
```text
        Args:
            episode_title: Title of the episode
            episode_description: Optional description of the episode
            known_hosts: Set of known host names
```
```text
        Returns:
            Tuple of (speaker_names, detected_hosts, success)
        """
        if not self._initialized:
            self.initialize()
```

        system_prompt, user_prompt = self._build_speaker_detection_prompts(
            episode_title, episode_description, known_hosts
        )

```text
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.cfg.deepseek_max_tokens or 500,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
```

            content = response.choices[0].message.content
            speakers, detected_hosts, success = self._parse_speakers_from_response(
                content, known_hosts
            )

```text
            logger.debug("DeepSeek detected speakers: %s", speakers)
            return speakers, detected_hosts, success
```
```python
        except Exception as e:
            logger.error("DeepSeek API error in speaker detection: %s", e)
            raise ValueError(f"DeepSeek speaker detection failed: {e}") from e
```
```python
    def analyze_patterns(
        self,
        episodes: List[models.Episode],
        known_hosts: Set[str],
    ) -> Optional[Dict[str, Any]]:
        """Analyze episode patterns (optional, can use local logic)."""
        return None
```
```python
    def cleanup(self) -> None:
        """Cleanup provider resources (no-op for API provider)."""
        pass
```
```python
    def _build_host_detection_prompts(
        self,
        feed_title: str,
        feed_description: Optional[str],
        feed_authors: Optional[List[str]],
    ) -> Tuple[str, str]:
        """Build prompts for host detection using prompt_store."""
        from ..prompt_store import render_prompt
```
```text
        system_prompt = render_prompt(self.cfg.deepseek_ner_system_prompt)
        user_prompt = render_prompt(
            self.cfg.deepseek_ner_user_prompt,
            feed_title=feed_title,
            feed_description=feed_description or "",
            feed_authors=", ".join(feed_authors) if feed_authors else "",
            task="host_detection",
        )
        return system_prompt, user_prompt
```
```python
    def _build_speaker_detection_prompts(
        self,
        episode_title: str,
        episode_description: Optional[str],
        known_hosts: Set[str],
    ) -> Tuple[str, str]:
        """Build prompts for speaker detection using prompt_store."""
        from ..prompt_store import render_prompt
```
```text
        system_prompt = render_prompt(self.cfg.deepseek_ner_system_prompt)
        user_prompt = render_prompt(
            self.cfg.deepseek_ner_user_prompt,
            episode_title=episode_title,
            episode_description=episode_description or "",
            known_hosts=", ".join(known_hosts) if known_hosts else "",
            task="speaker_detection",
        )
        return system_prompt, user_prompt
```
```python
    def _parse_hosts_from_response(self, response_text: str) -> Set[str]:
        """Parse host names from API response."""
        try:
            data = json.loads(response_text)
            if isinstance(data, dict) and "hosts" in data:
                return set(data["hosts"])
            if isinstance(data, list):
                return set(data)
        except json.JSONDecodeError:
            pass
```
```text
        hosts = set()
        for line in response_text.strip().split("\n"):
            for name in line.split(","):
                name = name.strip().strip("-").strip("*").strip()
                if name and len(name) > 1:
                    hosts.add(name)
        return hosts
```
```python
    def _parse_speakers_from_response(
        self, response_text: str, known_hosts: Set[str]
    ) -> Tuple[List[str], Set[str], bool]:
        """Parse speakers from API response."""
        try:
            data = json.loads(response_text)
            if isinstance(data, dict):
                speakers = data.get("speakers", [])
                hosts = set(data.get("hosts", []))
                guests = data.get("guests", [])
                all_speakers = list(hosts) + guests if not speakers else speakers
                return all_speakers, hosts, True
        except json.JSONDecodeError:
            pass
```

        speakers = []
        for line in response_text.strip().split("\n"):

```text
            for name in line.split(","):
                name = name.strip().strip("-").strip("*").strip()
                if name and len(name) > 1:
                    speakers.append(name)
```
```text
        detected_hosts = set(s for s in speakers if s in known_hosts)
        return speakers, detected_hosts, len(speakers) > 0
```

#### 5.3 Summarization Provider

**File**: `podcast_scraper/summarization/deepseek_provider.py`

```python
"""DeepSeek AI-based summarization provider.

This module provides a SummarizationProvider implementation using DeepSeek's chat API
for cloud-based episode summarization.

Note: Uses OpenAI SDK with DeepSeek's base_url for API compatibility.
Key advantage: DeepSeek is 90-95% cheaper than OpenAI.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .. import config
from ..deepseek.deepseek_provider import create_deepseek_client

logger = logging.getLogger(__name__)

class DeepSeekSummarizationProvider:

```text

    """DeepSeek AI-based summarization provider.

```
    This provider uses DeepSeek's chat API for cloud-based summarization.
    It implements the SummarizationProvider protocol.
    """

```python

    def __init__(self, cfg: config.Config):
        """Initialize DeepSeek summarization provider.

```
```text

        Args:
            cfg: Configuration object with deepseek_api_key and summarization settings

```
```text

        Raises:
            ValueError: If DeepSeek API key is not provided
        """
        self.cfg = cfg
        self.client = create_deepseek_client(cfg)
        self.model = cfg.deepseek_summary_model
        self.temperature = cfg.deepseek_temperature
        # DeepSeek supports 64k context window
        self.max_context_tokens = 64000
        self._initialized = False
        # API providers are thread-safe
        self._requires_separate_instances = False

```
```python

    def initialize(self) -> None:
        """Initialize provider (no local model loading needed for API)."""
        if self._initialized:
            return

```
```text

        logger.debug("Initializing DeepSeek summarization provider (model: %s)", self.model)
        self._initialized = True
        logger.debug("DeepSeek summarization provider initialized successfully")

```
```python

    def summarize(
        self,
        text: str,
        episode_title: Optional[str] = None,
        episode_description: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Summarize text using DeepSeek chat API.

```
```text

        Args:
            text: Transcript text to summarize
            episode_title: Optional episode title
            episode_description: Optional episode description
            params: Optional parameters dict

```
```text

        Returns:
            Dictionary with summary results

```
```text

        Raises:
            ValueError: If summarization fails
            RuntimeError: If provider is not initialized
        """
        if not self._initialized:
            raise RuntimeError(
                "DeepSeekSummarizationProvider not initialized. Call initialize() first."
            )

```
```text

        max_length = (params.get("max_length") if params else None) or self.cfg.summary_max_length
        min_length = (params.get("min_length") if params else None) or self.cfg.summary_min_length

```
        logger.debug(
            "Summarizing text via DeepSeek API (model: %s, max_length: %d)",
            self.model,
            max_length,
        )

```text

        try:
            (
                system_prompt,
                user_prompt,
                system_prompt_name,
                user_prompt_name,
                paragraphs_min,
                paragraphs_max,
            ) = self._build_summarization_prompts(
                text, episode_title, episode_description, max_length, min_length
            )

```json
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.cfg.deepseek_max_tokens or max_length,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

            summary = response.choices[0].message.content
            if not summary:

```text

                logger.warning("DeepSeek API returned empty summary")
                summary = ""

```
```text

            logger.debug("DeepSeek summarization completed: %d characters", len(summary))

```
```python

            from ..prompt_store import get_prompt_metadata

```
            prompt_metadata = {}
            if system_prompt_name:

```text

                prompt_metadata["system"] = get_prompt_metadata(system_prompt_name)
            user_params = {
                "transcript": text[:100] + "..." if len(text) > 100 else text,
                "title": episode_title or "",
                "paragraphs_min": paragraphs_min,
                "paragraphs_max": paragraphs_max,
            }
            prompt_metadata["user"] = get_prompt_metadata(user_prompt_name, params=user_params)

```
```text

            return {
                "summary": summary,
                "summary_short": None,
                "metadata": {
                    "model": self.model,
                    "provider": "deepseek",
                    "max_length": max_length,
                    "min_length": min_length,
                    "prompts": prompt_metadata,
                },
            }

```
```python

        except Exception as exc:
            logger.error("DeepSeek API error in summarization: %s", exc)
            raise ValueError(f"DeepSeek summarization failed: {exc}") from exc

```
```python

    def _build_summarization_prompts(
        self,
        text: str,
        episode_title: Optional[str],
        episode_description: Optional[str],
        max_length: int,
        min_length: int,
    ) -> tuple[str, str, str, str, int, int]:
        """Build system and user prompts using prompt_store."""
        from ..prompt_store import render_prompt

```
        system_prompt_name = self.cfg.deepseek_summary_system_prompt
        user_prompt_name = self.cfg.deepseek_summary_user_prompt

```text

        system_prompt = render_prompt(system_prompt_name)

```
```text

        paragraphs_min = max(1, min_length // 100)
        paragraphs_max = max(paragraphs_min, max_length // 100)

```
        template_params = {
            "transcript": text,
            "title": episode_title or "",
            "paragraphs_min": paragraphs_min,
            "paragraphs_max": paragraphs_max,
        }
        template_params.update(self.cfg.summary_prompt_params)

```text

        user_prompt = render_prompt(user_prompt_name, **template_params)

```
```text

        return (
            system_prompt,
            user_prompt,
            system_prompt_name,
            user_prompt_name,
            paragraphs_min,
            paragraphs_max,
        )

```
```python

    def cleanup(self) -> None:
        """Cleanup provider resources (no-op for API provider)."""
        pass

```

### 6. Factory Updates

#### 6.1 Speaker Detector Factory

**File**: `podcast_scraper/speaker_detectors/factory.py` (update)

```python

def create_speaker_detector(cfg: config.Config) -> Optional[SpeakerDetector]:
    """Create a speaker detector based on configuration."""
    if not cfg.auto_speakers:
        return None

    provider_type = cfg.speaker_detector_provider

    if provider_type in ("spacy", "ner"):
        from .ner_detector import NERSpeakerDetector
        return NERSpeakerDetector(cfg)
    elif provider_type == "openai":
        from .openai_detector import OpenAISpeakerDetector
        return OpenAISpeakerDetector(cfg)
    elif provider_type == "anthropic":
        from .anthropic_detector import AnthropicSpeakerDetector
        return AnthropicSpeakerDetector(cfg)
    elif provider_type == "mistral":
        from .mistral_detector import MistralSpeakerDetector
        return MistralSpeakerDetector(cfg)
    elif provider_type == "deepseek":

```python
        from .deepseek_detector import DeepSeekSpeakerDetector
        return DeepSeekSpeakerDetector(cfg)
    else:
        raise ValueError(
            f"Unsupported speaker detector provider: {provider_type}. "
            "Supported providers: 'spacy', 'openai', 'anthropic', 'mistral', 'deepseek'."
        )
```

#### 6.2 Summarization Factory

**File**: `podcast_scraper/summarization/factory.py` (update)

```python
def create_summarization_provider(cfg: config.Config) -> Optional[SummarizationProvider]:
    """Create a summarization provider based on configuration."""
    if not cfg.generate_summaries:
        return None

    provider_type = cfg.summary_provider

    if provider_type in ("transformers", "local"):
        from .local_provider import LocalSummarizationProvider
        return LocalSummarizationProvider(cfg)
    elif provider_type == "openai":
        from .openai_provider import OpenAISummarizationProvider
        return OpenAISummarizationProvider(cfg)
    elif provider_type == "anthropic":
        from .anthropic_provider import AnthropicSummarizationProvider
        return AnthropicSummarizationProvider(cfg)
    elif provider_type == "mistral":
        from .mistral_provider import MistralSummarizationProvider
        return MistralSummarizationProvider(cfg)
    elif provider_type == "deepseek":

```python

        from .deepseek_provider import DeepSeekSummarizationProvider
        return DeepSeekSummarizationProvider(cfg)
    else:
        raise ValueError(
            f"Unsupported summarization provider: {provider_type}. "
            "Supported providers: 'transformers', 'openai', 'anthropic', 'mistral', 'deepseek'."
        )

```

### 7. DeepSeek-Specific Prompt Templates

#### 7.1 Summarization System Prompt

**File**: `prompts/deepseek/summarization/system_v1.j2`

```jinja2

You are an expert podcast summarizer. Your task is to create concise, informative summaries of podcast episode transcripts.

Guidelines:
- Focus on key insights, decisions, and takeaways
- Ignore sponsor reads, ads, and housekeeping announcements
- Do not use direct quotes or speaker attributions
- Do not invent information not present in the transcript
- Write in a clear, professional tone
- Structure the summary with logical flow

```
#### 7.2 Summarization User Prompt

**File**: `prompts/deepseek/summarization/long_v1.j2`

```jinja2

Please summarize the following podcast transcript.

{% if title %}Episode Title: {{ title }}{% endif %}

Target length: {{ paragraphs_min }} to {{ paragraphs_max }} paragraphs.

Transcript:
{{ transcript }}

Provide a comprehensive summary covering the main topics, key insights, and important takeaways.

```
#### 7.3 NER System Prompt

**File**: `prompts/deepseek/ner/system_ner_v1.j2`

```jinja2

You are an expert at identifying people mentioned in podcast metadata. Your task is to extract speaker names from podcast episode information.

Guidelines:
- Focus on identifying hosts, guests, and speakers
- Return names in a consistent format
- Distinguish between hosts (regular presenters) and guests (episode-specific)
- Respond in JSON format with "hosts" and "guests" arrays

```
#### 7.4 NER User Prompt

**File**: `prompts/deepseek/ner/guest_host_v1.j2`

```jinja2

{% if task == "host_detection" %}
Identify the hosts of this podcast from the following information:

Podcast Title: {{ feed_title }}
{% if feed_description %}Description: {{ feed_description }}{% endif %}
{% if feed_authors %}Listed Authors: {{ feed_authors }}{% endif %}

Return a JSON object with format: {"hosts": ["Name1", "Name2"]}
{% else %}
Identify speakers in this podcast episode:

Episode Title: {{ episode_title }}
{% if episode_description %}Description: {{ episode_description }}{% endif %}
{% if known_hosts %}Known Hosts: {{ known_hosts }}{% endif %}

Return a JSON object with format: {"speakers": [...], "hosts": [...], "guests": [...]}
{% endif %}

```
### 8. E2E Server Mock Endpoints

DeepSeek uses OpenAI-compatible API, so we can reuse OpenAI mock endpoints. Add URL helper:

```python

class E2EServerURLs:
    """URL helper class for E2E server."""

    def deepseek_api_base(self) -> str:
        """Get DeepSeek API base URL for E2E testing.

        Note: DeepSeek uses OpenAI-compatible API, so we reuse the same mock endpoint.
        """
        return f"http://{self.host}:{self.port}"

```
### 9. Dependencies

No new dependencies required - DeepSeek uses OpenAI SDK:

```toml

# pyproject.toml - No changes needed
# DeepSeek uses existing openai package with custom base_url

```yaml

## Testing Strategy

### Test Coverage

| Test Type | Description | Location |
| --------- | ----------- | -------- |
| **Unit Tests** | Mock DeepSeek API calls | `tests/unit/podcast_scraper/test_deepseek_providers.py` |
| **Integration Tests** | Test with E2E server mock | `tests/integration/test_deepseek_providers.py` |
| **E2E Tests** | Full pipeline with DeepSeek | `tests/e2e/test_deepseek_provider_integration_e2e.py` |

### Test Organization

```text

tests/
├── unit/
│   └── podcast_scraper/
│       └── test_deepseek_providers.py
├── integration/
│   └── test_deepseek_providers.py
└── e2e/
    └── test_deepseek_provider_integration_e2e.py

```
### Test Markers

```python

@pytest.mark.unit           # Unit tests
@pytest.mark.integration    # Integration tests
@pytest.mark.e2e           # End-to-end tests
@pytest.mark.llm           # Uses LLM APIs
@pytest.mark.deepseek      # Uses DeepSeek specifically

```go

## Rollout & Monitoring

### Rollout Plan

1. **Phase 1**: Core implementation
   - Create `deepseek/` package with shared utilities
   - Implement `DeepSeekSpeakerDetector`
   - Implement `DeepSeekSummarizationProvider`
   - Add configuration fields

2. **Phase 2**: Integration
   - Update factories
   - Add provider capability validation
   - Create DeepSeek-specific prompts
   - Update `.env.example`

3. **Phase 3**: Testing
   - Reuse OpenAI mock endpoints
   - Write unit tests
   - Write integration tests
   - Write E2E tests

4. **Phase 4**: Documentation
   - Update Provider Configuration Quick Reference
   - Update Provider Implementation Guide
   - Add cost comparison documentation

### Success Criteria

1. ✅ DeepSeek providers implement same interfaces as other providers
2. ✅ Users can select DeepSeek for speaker detection and summarization
3. ✅ Clear error when attempting transcription with DeepSeek
4. ✅ API keys managed securely via environment variables
5. ✅ No new SDK dependency (uses OpenAI SDK)
6. ✅ E2E tests pass with mock endpoints
7. ✅ Default behavior (local providers) unchanged

## Alternatives Considered

### 1. Dedicated DeepSeek SDK

**Description**: Use official DeepSeek Python SDK instead of OpenAI SDK

**Pros**:

- May support DeepSeek-specific features
- Official support

**Cons**:

- Additional dependency
- May not be as well-maintained as OpenAI SDK
- OpenAI compatibility is documented and stable

**Why Rejected**: OpenAI SDK works well with DeepSeek's OpenAI-compatible API, avoiding new dependency.

## Open Questions

1. **Cache Hit Pricing**: How can users maximize cache hits for lower costs?
2. **DeepSeek-R1 Integration**: Should we add support for reasoning tokens?
3. **Regional Availability**: Document any regional restrictions?

## References

- **Related PRD**: `docs/prd/PRD-011-deepseek-provider-integration.md`
- **OpenAI Provider RFC**: `docs/rfc/RFC-013-openai-provider-implementation.md`
- **Anthropic Provider RFC**: `docs/rfc/RFC-032-anthropic-provider-implementation.md`
- **DeepSeek API Documentation**: https://platform.deepseek.com/docs
- **DeepSeek Pricing**: https://platform.deepseek.com/pricing
