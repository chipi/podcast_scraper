# RFC-033: Mistral Provider Implementation

- **Status**: Draft
- **Authors**:
- **Stakeholders**: Maintainers, users wanting Mistral API integration, developers implementing providers
- **Related PRDs**:
  - `docs/prd/PRD-010-mistral-provider-integration.md`
- **Related RFCs**:
  - `docs/rfc/RFC-013-openai-provider-implementation.md` (reference implementation)
  - `docs/rfc/RFC-032-anthropic-provider-implementation.md` (similar pattern)
  - `docs/rfc/RFC-021-modularization-refactoring-plan.md` (architecture foundation)
  - `docs/rfc/RFC-017-prompt-management.md` (prompt system)

## Abstract

Design and implement Mistral AI providers for transcription, speaker detection, and summarization capabilities. Mistral is unique among cloud providers in supporting ALL three capabilities, making it a complete OpenAI alternative. This RFC builds on the existing modularization architecture (RFC-021) and follows identical patterns to OpenAI (RFC-013) and Anthropic (RFC-032) implementations.

**Architecture Alignment:** This RFC follows the protocol-based provider system established in RFC-021. Mistral providers implement the same protocols (`TranscriptionProvider`, `SpeakerDetector`, `SummarizationProvider`) and integrate via the existing factory pattern.

## Problem Statement

Users want the option to use Mistral AI as a complete alternative to OpenAI for:

1. **Transcription**: Audio-to-text using Voxtral models
2. **Speaker Detection**: Entity extraction using Mistral chat models
3. **Summarization**: High-quality summaries using Mistral chat models

Unlike Anthropic, Mistral supports ALL three capabilities, making it a true OpenAI alternative.

Requirements:

- No changes to end-user experience or workflow when using defaults
- Secure API key management (environment variables, never in source code)
- Per-capability provider selection (can mix local, OpenAI, Anthropic, and Mistral)
- Build on existing modularization and provider architecture
- Use Mistral-specific prompts (prompts are provider-specific)
- Handle Voxtral API differences from OpenAI Whisper API

## Constraints & Assumptions

**Constraints:**

- **Prerequisite**: Modularization refactoring (RFC-021) ✅ Completed
- **Prerequisite**: OpenAI provider implementation (RFC-013) ✅ Completed
- **Prerequisite**: Anthropic provider implementation (RFC-032) ✅ Completed
- **Backward Compatibility**: Default providers (local) must remain unchanged
- **API Key Security**: API keys must never be in source code or committed files
- **Rate Limits**: Must respect Mistral API rate limits and implement retry logic

**Assumptions:**

- Mistral API is stable and well-documented
- Mistral Python SDK follows similar patterns to OpenAI/Anthropic SDKs
- Voxtral transcription API follows similar patterns to OpenAI Whisper API
- Prompts need to be optimized for Mistral (may differ from GPT/Claude)

## Design & Implementation

### 0. Mistral API Overview

Mistral's API is similar to OpenAI but with some differences:

| Feature | OpenAI | Mistral |
| ------- | ------ | ------- |
| **Chat Endpoint** | `/v1/chat/completions` | `/v1/chat/completions` |
| **Transcription Endpoint** | `/v1/audio/transcriptions` | `/v1/audio/transcriptions` |
| **Audio Models** | whisper-1 | voxtral-mini-latest |
| **Context Window** | 128k tokens | 256k tokens (large) |
| **Temperature Range** | 0.0 - 2.0 | 0.0 - 1.0 |
| **Python SDK** | `openai` | `mistralai` |

### 1. Architecture Overview

```text
podcast_scraper/
├── mistral/                        # NEW: Shared Mistral utilities
│   ├── __init__.py
│   └── mistral_provider.py         # Shared client, rate limiting
├── transcription/
│   ├── base.py                     # TranscriptionProvider protocol (existing)
│   ├── factory.py                  # Updated to include Mistral
│   ├── whisper_provider.py         # Local Whisper (existing)
│   ├── openai_provider.py          # OpenAI (existing)
│   └── mistral_provider.py         # NEW: Mistral/Voxtral implementation
├── speaker_detectors/
│   ├── base.py                     # SpeakerDetector protocol (existing)
│   ├── factory.py                  # Updated to include Mistral
│   ├── ner_detector.py             # Local NER (existing)
│   ├── openai_detector.py          # OpenAI (existing)
│   ├── anthropic_detector.py       # Anthropic (existing)
│   └── mistral_detector.py         # NEW: Mistral implementation
├── summarization/
│   ├── base.py                     # SummarizationProvider protocol (existing)
│   ├── factory.py                  # Updated to include Mistral
│   ├── local_provider.py           # Local transformers (existing)
│   ├── openai_provider.py          # OpenAI (existing)
│   ├── anthropic_provider.py       # Anthropic (existing)
│   └── mistral_provider.py         # NEW: Mistral implementation
├── prompts/
│   ├── summarization/              # OpenAI prompts (existing)
│   ├── ner/                        # OpenAI prompts (existing)
│   ├── anthropic/                  # Anthropic prompts (existing)
│   └── mistral/                    # NEW: Mistral-specific prompts
│       ├── summarization/
│       │   ├── system_v1.j2
│       │   └── long_v1.j2
│       └── ner/
│           ├── system_ner_v1.j2
│           └── guest_host_v1.j2
└── config.py                       # Updated with Mistral fields
```

### 2. Configuration

Add to `config.py`:

```python
from typing import Literal, Optional

# Provider Selection (updated to include mistral)

transcription_provider: Literal["whisper", "openai", "mistral"] = Field(
    default="whisper",
    description="Transcription provider: 'whisper' (local), 'openai', or 'mistral'"
)

speaker_detector_provider: Literal["spacy", "openai", "anthropic", "mistral"] = Field(
    default="spacy",
    description="Speaker detection provider: 'spacy' (local), 'openai', 'anthropic', or 'mistral'"
)

summary_provider: Literal["transformers", "openai", "anthropic", "mistral"] = Field(
    default="transformers",
    description="Summarization provider: 'transformers' (local), 'openai', 'anthropic', or 'mistral'"
)

# Mistral API Configuration

mistral_api_key: Optional[str] = Field(
    default=None,
    description="Mistral API key (prefer MISTRAL_API_KEY env var or .env file)"
)

mistral_api_base: Optional[str] = Field(
    default=None,
    description="Custom Mistral API base URL (for E2E testing)"
)

# Mistral Model Selection

mistral_transcription_model: str = Field(
    default="voxtral-mini-latest",
    description="Mistral Voxtral model for transcription"
)

mistral_speaker_model: str = Field(
    default="mistral-small-latest",
    description="Mistral model for speaker detection"
)

mistral_summary_model: str = Field(
    default="mistral-small-latest",
    description="Mistral model for summarization"
)

mistral_temperature: float = Field(
    default=0.3,
    ge=0.0,
    le=1.0,
    description="Temperature for Mistral generation (0.0-1.0)"
)

mistral_max_tokens: Optional[int] = Field(
    default=4096,
    description="Max tokens for Mistral generation"
)

# Mistral Prompt Configuration

mistral_summary_system_prompt: str = Field(
    default="mistral/summarization/system_v1",
    description="Mistral system prompt for summarization"
)

mistral_summary_user_prompt: str = Field(
    default="mistral/summarization/long_v1",
    description="Mistral user prompt for summarization"
)

mistral_ner_system_prompt: str = Field(
    default="mistral/ner/system_ner_v1",
    description="Mistral system prompt for speaker detection"
)

mistral_ner_user_prompt: str = Field(
    default="mistral/ner/guest_host_v1",
    description="Mistral user prompt for speaker detection"
)
```

## 3. API Key Management

Follow identical pattern to OpenAI/Anthropic:

```python

# config.py - API Key Loading

from dotenv import load_dotenv

# Load .env file automatically

load_dotenv(override=False)

@field_validator('mistral_api_key', mode='before')
@classmethod
def load_mistral_api_key_from_env(cls, v: Any) -> Optional[str]:
    """Load API key from environment variable if not provided."""
    if v is not None:
        return v
    return os.getenv('MISTRAL_API_KEY')

@model_validator(mode='after')
def validate_mistral_config(self) -> 'Config':

```text

    """Validate Mistral provider configuration."""
    needs_key = (
        self.transcription_provider == "mistral" or
        self.speaker_detector_provider == "mistral" or
        self.summary_provider == "mistral"
    )
    if needs_key and not self.mistral_api_key:
        raise ValueError(
            "Mistral API key required when using Mistral providers. "
            "Set MISTRAL_API_KEY environment variable, add it to .env file, "
            "or set mistral_api_key in config file."
        )
    return self

```
## 4. Mistral Provider Implementations

### 4.1 Shared Mistral Utilities

**File**: `podcast_scraper/mistral/mistral_provider.py`

```python

"""Shared Mistral provider utilities.

This module provides shared utilities for Mistral API providers,
including client initialization and rate limiting.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from mistralai import Mistral

from .. import config

logger = logging.getLogger(__name__)

def create_mistral_client(cfg: config.Config) -> Mistral:

```text
    """Create Mistral client with configuration.
```

    Args:
        cfg: Configuration object with mistral_api_key and optional mistral_api_base

```
    Raises:
        ValueError: If API key is not provided
    """
    if not cfg.mistral_api_key:
        raise ValueError(
            "Mistral API key required. "
            "Set MISTRAL_API_KEY environment variable or mistral_api_key in config."
        )

```

    # Support custom server_url for E2E testing with mock servers
    if cfg.mistral_api_base:
        client_kwargs["server_url"] = cfg.mistral_api_base

```
#### 4.2 Transcription Provider

**File**: `podcast_scraper/transcription/mistral_provider.py`

```python

"""Mistral Voxtral-based transcription provider.

This module provides a TranscriptionProvider implementation using Mistral's
Voxtral models for cloud-based audio transcription.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .. import config, models
from ..mistral.mistral_provider import create_mistral_client

logger = logging.getLogger(__name__)

class MistralTranscriptionProvider:

```text

    """Mistral Voxtral-based transcription provider.

```python

    def __init__(self, cfg: config.Config):
        """Initialize Mistral transcription provider.

```
            cfg: Configuration object with mistral_api_key and transcription settings

```

        """
        self.cfg = cfg
        self.client = create_mistral_client(cfg)
        self.model = cfg.mistral_transcription_model
        self._initialized = False

```python

    def initialize(self) -> None:
        """Initialize provider (no local model loading needed for API)."""
        if self._initialized:
            return

```

        logger.debug("Mistral transcription provider initialized successfully")

```python

    def transcribe(self, audio_path: Path, language: Optional[str] = None) -> str:
        """Transcribe audio file using Mistral Voxtral API.

```

            language: Optional language code (ISO 639-1)

```
            Transcribed text

```

            RuntimeError: If provider is not initialized
        """
        if not self._initialized:
            raise RuntimeError(
                "MistralTranscriptionProvider not initialized. Call initialize() first."
            )

```
            with open(audio_path, "rb") as audio_file:
                # Mistral Voxtral API call
                transcription_kwargs: Dict[str, Any] = {
                    "model": self.model,
                    "file": {
                        "file_name": audio_path.name,
                        "content": audio_file.read(),
                    },
                }

```

                    transcription_kwargs["language"] = language

```
```

            logger.debug("Mistral transcription completed: %d characters", len(text))
            return text

```python

        except Exception as exc:
            logger.error("Mistral API error in transcription: %s", exc)
            raise ValueError(f"Mistral transcription failed: {exc}") from exc

```python

    def transcribe_with_segments(
        self, audio_path: Path, language: Optional[str] = None
    ) -> tuple[str, List[Dict[str, Any]]]:
        """Transcribe audio file with timestamp segments.

```
            language: Optional language code

```

            Tuple of (full_text, list of segment dictionaries)
        """
        if not self._initialized:
            raise RuntimeError(
                "MistralTranscriptionProvider not initialized. Call initialize() first."
            )

```
```

            with open(audio_path, "rb") as audio_file:
                transcription_kwargs: Dict[str, Any] = {
                    "model": self.model,
                    "file": {
                        "file_name": audio_path.name,
                        "content": audio_file.read(),
                    },
                    "timestamp_granularities": ["segment"],
                }

```

```

```
            segments = []
            if hasattr(response, 'segments') and response.segments:
                for seg in response.segments:
                    segments.append({
                        "start": getattr(seg, 'start', 0),
                        "end": getattr(seg, 'end', 0),
                        "text": getattr(seg, 'text', ''),
                    })

```

                len(segments),
            )
            return text, segments

```python

        except Exception as exc:
            logger.error("Mistral API error in transcription with segments: %s", exc)
            raise ValueError(f"Mistral transcription failed: {exc}") from exc

```python

    def cleanup(self) -> None:
        """Cleanup provider resources (no-op for API provider)."""
        pass

```
#### 4.3 Speaker Detection Provider

**File**: `podcast_scraper/speaker_detectors/mistral_detector.py`

```python

"""Mistral AI-based speaker detection provider.

This module provides a SpeakerDetector implementation using Mistral's chat API
for cloud-based speaker/guest detection from episode metadata.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from .. import config, models
from ..mistral.mistral_provider import create_mistral_client

logger = logging.getLogger(__name__)

class MistralSpeakerDetector:

```text

    """Mistral AI-based speaker detection provider.

```

    """

```python

    def __init__(self, cfg: config.Config):
        """Initialize Mistral speaker detector.

```

```
        """
        self.cfg = cfg
        self.client = create_mistral_client(cfg)
        self.model = cfg.mistral_speaker_model
        self.temperature = cfg.mistral_temperature
        self._initialized = False

```python

    def initialize(self) -> None:
        """Initialize provider (no local model loading needed for API)."""
        if self._initialized:
            return

```
        logger.debug("Mistral speaker detector initialized successfully")

```python

    def detect_hosts(
        self,
        feed_title: str,
        feed_description: Optional[str],
        feed_authors: Optional[List[str]],
    ) -> Set[str]:
        """Detect hosts from feed metadata using Mistral API.

```
            feed_description: Optional description of the feed
            feed_authors: Optional list of feed authors

```

        """
        if not self._initialized:
            self.initialize()

```
        )

```text

        try:
            response = self.client.chat.complete(
                model=self.model,
                max_tokens=self.cfg.mistral_max_tokens or 500,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

```

            logger.debug("Mistral detected hosts: %s", hosts)
            return hosts

```python

        except Exception as e:
            logger.error("Mistral API error in host detection: %s", e)
            raise ValueError(f"Mistral host detection failed: {e}") from e

```python

    def detect_speakers(
        self,
        episode_title: str,
        episode_description: Optional[str],
        known_hosts: Set[str],
    ) -> Tuple[List[str], Set[str], bool]:
        """Detect speakers for an episode using Mistral API.

```

            episode_description: Optional description of the episode
            known_hosts: Set of known host names

```
        """
        if not self._initialized:
            self.initialize()

```

        )

```text

        try:
            response = self.client.chat.complete(
                model=self.model,
                max_tokens=self.cfg.mistral_max_tokens or 500,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

```

                content, known_hosts
            )

```text

            logger.debug("Mistral detected speakers: %s", speakers)
            return speakers, detected_hosts, success

```python

        except Exception as e:
            logger.error("Mistral API error in speaker detection: %s", e)
            raise ValueError(f"Mistral speaker detection failed: {e}") from e

```python

    def analyze_patterns(
        self,
        episodes: List[models.Episode],
        known_hosts: Set[str],
    ) -> Optional[Dict[str, Any]]:
        """Analyze episode patterns (optional, can use local logic)."""
        return None

```python

    def cleanup(self) -> None:
        """Cleanup provider resources (no-op for API provider)."""
        pass

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

            self.cfg.mistral_ner_user_prompt,
            feed_title=feed_title,
            feed_description=feed_description or "",
            feed_authors=", ".join(feed_authors) if feed_authors else "",
            task="host_detection",
        )
        return system_prompt, user_prompt

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

            self.cfg.mistral_ner_user_prompt,
            episode_title=episode_title,
            episode_description=episode_description or "",
            known_hosts=", ".join(known_hosts) if known_hosts else "",
            task="speaker_detection",
        )
        return system_prompt, user_prompt

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

            for name in line.split(","):
                name = name.strip().strip("-").strip("*").strip()
                if name and len(name) > 1:
                    hosts.add(name)
        return hosts

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

            for name in line.split(","):
                name = name.strip().strip("-").strip("*").strip()
                if name and len(name) > 1:
                    speakers.append(name)

```
#### 4.4 Summarization Provider

**File**: `podcast_scraper/summarization/mistral_provider.py`

```python

"""Mistral AI-based summarization provider.

This module provides a SummarizationProvider implementation using Mistral's chat API
for cloud-based episode summarization.

Key Advantage: Mistral Large has 256k token context window, enabling full transcript
processing without chunking for most podcasts.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .. import config
from ..mistral.mistral_provider import create_mistral_client

logger = logging.getLogger(__name__)

class MistralSummarizationProvider:

```text

    """Mistral AI-based summarization provider.

```python

    def __init__(self, cfg: config.Config):
        """Initialize Mistral summarization provider.

```

```

            ValueError: If Mistral API key is not provided
        """
        self.cfg = cfg
        self.client = create_mistral_client(cfg)
        self.model = cfg.mistral_summary_model
        self.temperature = cfg.mistral_temperature
        # Mistral Large supports 256k context window
        self.max_context_tokens = 256000
        self._initialized = False
        # API providers are thread-safe
        self._requires_separate_instances = False

```python

    def initialize(self) -> None:
        """Initialize provider (no local model loading needed for API)."""
        if self._initialized:
            return

```

        self._initialized = True
        logger.debug("Mistral summarization provider initialized successfully")

```python

    def summarize(
        self,
        text: str,
        episode_title: Optional[str] = None,
        episode_description: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Summarize text using Mistral chat API.

```

            text: Transcript text to summarize
            episode_title: Optional episode title
            episode_description: Optional episode description
            params: Optional parameters dict

```

```

            ValueError: If summarization fails
            RuntimeError: If provider is not initialized
        """
        if not self._initialized:
            raise RuntimeError(
                "MistralSummarizationProvider not initialized. Call initialize() first."
            )

```

```

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

            response = self.client.chat.complete(
                model=self.model,
                max_tokens=self.cfg.mistral_max_tokens or max_length,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

            summary = response.choices[0].message.content
            if not summary:

```text

                logger.warning("Mistral API returned empty summary")
                summary = ""

```
```python

            from ..prompt_store import get_prompt_metadata

```
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

                "summary": summary,
                "summary_short": None,
                "metadata": {
                    "model": self.model,
                    "provider": "mistral",
                    "max_length": max_length,
                    "min_length": min_length,
                    "prompts": prompt_metadata,
                },
            }

```python

        except Exception as exc:
            logger.error("Mistral API error in summarization: %s", exc)
            raise ValueError(f"Mistral summarization failed: {exc}") from exc

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
        system_prompt = render_prompt(system_prompt_name)

```

        paragraphs_max = max(paragraphs_min, max_length // 100)

```
            "paragraphs_min": paragraphs_min,
            "paragraphs_max": paragraphs_max,
        }
        template_params.update(self.cfg.summary_prompt_params)

```text

        user_prompt = render_prompt(user_prompt_name, **template_params)

```
            user_prompt,
            system_prompt_name,
            user_prompt_name,
            paragraphs_min,
            paragraphs_max,
        )

```python

    def cleanup(self) -> None:
        """Cleanup provider resources (no-op for API provider)."""
        pass

```
### 5. Factory Updates

#### 5.1 Transcription Factory

**File**: `podcast_scraper/transcription/factory.py` (update)

```python

def create_transcription_provider(cfg: config.Config) -> Optional[TranscriptionProvider]:
    """Create a transcription provider based on configuration."""
    provider_type = cfg.transcription_provider

    if provider_type in ("whisper", "local"):
        from .whisper_provider import WhisperTranscriptionProvider
        return WhisperTranscriptionProvider(cfg)
    elif provider_type == "openai":
        from .openai_provider import OpenAITranscriptionProvider
        return OpenAITranscriptionProvider(cfg)
    elif provider_type == "mistral":
        from .mistral_provider import MistralTranscriptionProvider
        return MistralTranscriptionProvider(cfg)
    else:
        raise ValueError(
            f"Unsupported transcription provider: {provider_type}. "
            "Supported providers: 'whisper', 'openai', 'mistral'."
        )

```
#### 5.2 Speaker Detector Factory

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
    else:
        raise ValueError(
            f"Unsupported speaker detector provider: {provider_type}. "
            "Supported providers: 'spacy', 'openai', 'anthropic', 'mistral'."
        )

```
#### 5.3 Summarization Factory

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
    else:
        raise ValueError(
            f"Unsupported summarization provider: {provider_type}. "
            "Supported providers: 'transformers', 'openai', 'anthropic', 'mistral'."
        )

```
### 6. Mistral-Specific Prompt Templates

#### 6.1 Summarization System Prompt

**File**: `prompts/mistral/summarization/system_v1.j2`

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
#### 6.2 Summarization User Prompt

**File**: `prompts/mistral/summarization/long_v1.j2`

```jinja2

Please summarize the following podcast transcript.

{% if title %}Episode Title: {{ title }}{% endif %}

Target length: {{ paragraphs_min }} to {{ paragraphs_max }} paragraphs.

Transcript:
{{ transcript }}

Provide a comprehensive summary covering the main topics, key insights, and important takeaways.

```
#### 6.3 NER System Prompt

**File**: `prompts/mistral/ner/system_ner_v1.j2`

```jinja2

You are an expert at identifying people mentioned in podcast metadata. Your task is to extract speaker names from podcast episode information.

Guidelines:
- Focus on identifying hosts, guests, and speakers
- Return names in a consistent format
- Distinguish between hosts (regular presenters) and guests (episode-specific)
- Respond in JSON format with "hosts" and "guests" arrays

```
#### 6.4 NER User Prompt

**File**: `prompts/mistral/ner/guest_host_v1.j2`

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
### 7. E2E Server Mock Endpoints

Add Mistral mock endpoints to `tests/e2e/fixtures/e2e_http_server.py`:

```python

def do_POST(self):
    """Handle POST requests."""
    path = self.path.split("?")[0]

    # Existing endpoints...

    # Mistral chat completions endpoint
    if path == "/v1/chat/completions":
        self._handle_mistral_chat_completions()
        return

    # Mistral audio transcriptions endpoint
    if path == "/v1/audio/transcriptions":
        self._handle_mistral_transcriptions()
        return

    self.send_error(404, "Endpoint not found")

def _handle_mistral_chat_completions(self):

```text

    """Handle Mistral chat completions API requests."""
    try:
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        request_data = json.loads(body.decode("utf-8"))

```

        # Get system and user messages
        system_content = next(
            (m.get("content", "") for m in messages if m.get("role") == "system"),
            ""
        )
        user_content = next(
            (m.get("content", "") for m in messages if m.get("role") == "user"),
            ""
        )

```
                "speakers": ["Host", "Guest"],
                "hosts": ["Host"],
                "guests": ["Guest"],
            })
        else:
            response_content = (
                "This is a test summary of the podcast episode. "
                "The episode covers various topics discussed by the hosts and guests."
            )

```json

        # Build Mistral response format
        response_data = {
            "id": "chat-test-12345",
            "object": "chat.completion",
            "created": 1234567890,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_content,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        }

```
        self.send_header("Content-Length", str(len(response_json)))
        self.end_headers()
        self.wfile.write(response_json.encode("utf-8"))

```

        self.send_error(500, f"Error handling Mistral chat: {e}")

```python

def _handle_mistral_transcriptions(self):

```python

    """Handle Mistral Voxtral transcription API requests."""
    try:
        # Parse multipart form data for audio file
        # For mock purposes, return a test transcription
        response_data = {
            "text": "This is a test transcription from the Mistral Voxtral API.",
            "segments": [
                {"start": 0.0, "end": 5.0, "text": "This is a test transcription"},
                {"start": 5.0, "end": 10.0, "text": "from the Mistral Voxtral API."},
            ],
        }

```
        self.send_header("Content-Length", str(len(response_json)))
        self.end_headers()
        self.wfile.write(response_json.encode("utf-8"))

```

        self.send_error(500, f"Error handling Mistral transcription: {e}")

```python

class E2EServerURLs:
    """URL helper class for E2E server."""

    def mistral_api_base(self) -> str:
        """Get Mistral API base URL for E2E testing."""
        return f"http://{self.host}:{self.port}"

```

### 8. Dependencies

Add to `pyproject.toml`:

```toml

[project.optional-dependencies]
mistral = [
    "mistralai>=1.0.0,<2.0.0",
]

# All AI providers

ai = [
    "openai>=1.0.0,<2.0.0",
    "anthropic>=0.30.0,<1.0.0",
    "mistralai>=1.0.0,<2.0.0",
    "tenacity>=8.2.0,<9.0.0",
]

```

# For Mistral support only

pip install -e ".[mistral]"

# For all AI providers

pip install -e ".[ai]"

```yaml

## Testing Strategy

### Test Coverage

| Test Type | Description | Location |
| --------- | ----------- | -------- |
| **Unit Tests** | Mock Mistral API calls | `tests/unit/podcast_scraper/test_mistral_providers.py` |
| **Integration Tests** | Test with E2E server mock | `tests/integration/test_mistral_providers.py` |
| **E2E Tests** | Full pipeline with Mistral | `tests/e2e/test_mistral_provider_integration_e2e.py` |
| **E2E Server Tests** | Verify mock endpoints | `tests/e2e/test_e2e_server.py` |

### Test Organization

```text

tests/
├── unit/
│   └── podcast_scraper/
│       └── test_mistral_providers.py
├── integration/
│   └── test_mistral_providers.py
└── e2e/
    └── test_mistral_provider_integration_e2e.py

```
### Test Markers

```python

@pytest.mark.unit           # Unit tests
@pytest.mark.integration    # Integration tests
@pytest.mark.e2e           # End-to-end tests
@pytest.mark.llm           # Uses LLM APIs
@pytest.mark.mistral       # Uses Mistral specifically
@pytest.mark.transcription # Tests transcription

```go

## Rollout & Monitoring

### Rollout Plan

1. **Phase 1**: Core implementation
   - Create `mistral/` package with shared utilities
   - Implement `MistralTranscriptionProvider`
   - Implement `MistralSpeakerDetector`
   - Implement `MistralSummarizationProvider`
   - Add configuration fields

2. **Phase 2**: Integration
   - Update all factories
   - Create Mistral-specific prompts
   - Update `.env.example`

3. **Phase 3**: Testing
   - Add E2E server mock endpoints
   - Write unit tests
   - Write integration tests
   - Write E2E tests

4. **Phase 4**: Documentation
   - Update Provider Configuration Quick Reference
   - Update Provider Implementation Guide
   - Add capability matrix updates

### Success Criteria

1. ✅ Mistral providers implement same interfaces as other providers
2. ✅ Users can select Mistral for ALL three capabilities
3. ✅ Mistral is a complete OpenAI alternative
4. ✅ API keys managed securely via environment variables
5. ✅ E2E tests pass with Mistral mock endpoints
6. ✅ Default behavior (local providers) unchanged
7. ✅ Documentation complete and clear

## Alternatives Considered

### 1. OpenAI-Compatible Mode

**Description**: Use OpenAI SDK with Mistral's OpenAI-compatible endpoint

**Pros**:

- Less code, reuse OpenAI provider
- Simpler maintenance

**Cons**:

- May not support all Mistral features
- Voxtral API differences may not work
- Less control over Mistral-specific optimizations

**Why Rejected**: Native Mistral SDK provides better feature support and Voxtral integration.

### 2. Shared LLM Provider Base Class

**Description**: Create abstract base class for all LLM providers

**Pros**:

- Reduces code duplication
- Easier to add new providers

**Cons**:

- API differences between providers make abstraction complex
- Prompt optimization varies by model

**Why Rejected**: Current per-provider approach is simpler and allows provider-specific optimizations.

## Open Questions

1. **Voxtral Pricing**: Need to confirm exact pricing for Voxtral transcription
2. **Rate Limits**: Need to document Mistral's rate limits for each endpoint
3. **Audio Formats**: Need to verify which audio formats Voxtral supports

## References

- **Related PRD**: `docs/prd/PRD-010-mistral-provider-integration.md`
- **OpenAI Provider RFC**: `docs/rfc/RFC-013-openai-provider-implementation.md`
- **Anthropic Provider RFC**: `docs/rfc/RFC-032-anthropic-provider-implementation.md`
- **Mistral API Documentation**: https://docs.mistral.ai/
- **Mistral Python SDK**: https://github.com/mistralai/mistral-python
- **Voxtral Documentation**: https://docs.mistral.ai/capabilities/audio_transcription
