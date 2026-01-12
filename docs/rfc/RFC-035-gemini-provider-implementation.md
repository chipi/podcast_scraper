# RFC-035: Google Gemini Provider Implementation

- **Status**: Draft
- **Authors**:
- **Stakeholders**: Maintainers, users wanting Google Gemini integration
- **Related PRDs**:
  - `docs/prd/PRD-012-gemini-provider-integration.md`
- **Related RFCs**:
  - `docs/rfc/RFC-013-openai-provider-implementation.md` (reference)
  - `docs/rfc/RFC-033-mistral-provider-implementation.md` (similar - full capabilities)

## Abstract

Design and implement Google Gemini providers for transcription, speaker detection, and summarization capabilities. Gemini is unique in offering native multimodal audio understanding (no separate ASR step) and an industry-leading 2 million token context window. This RFC follows established provider patterns.

**Architecture Alignment:** Gemini providers implement the same protocols (`TranscriptionProvider`, `SpeakerDetector`, `SummarizationProvider`) and integrate via the existing factory pattern.

## Problem Statement

Users want Google Gemini as a provider option for:

1. **Transcription**: Native audio understanding via multimodal input
2. **Speaker Detection**: Entity extraction using Gemini chat models
3. **Summarization**: High-quality summaries with massive context window

Key advantages of Gemini:

- **Native audio understanding** - no separate transcription step
- **2M token context** - process entire seasons
- **Generous free tier** - excellent for development
- **Google ecosystem** - familiar for Google Cloud users

## Constraints & Assumptions

**Constraints:**

- Must use Google AI Python SDK (`google-genai`)
- Audio files must be uploaded or passed as inline data
- Rate limits apply (especially free tier)

**Assumptions:**

- Gemini API is stable
- Audio transcription quality is comparable to Whisper
- Prompts may need Gemini-specific optimization

## Design & Implementation

### 0. Gemini API Overview

| Feature | OpenAI | Gemini |
| ------- | ------ | ------ |
| **SDK** | `openai` | `google-genai` |
| **Audio** | Whisper (separate API) | Native multimodal |
| **Context Window** | 128k tokens | 2M tokens |
| **Audio Input** | File upload | File or inline data |

### 1. Architecture Overview

```text
podcast_scraper/
├── gemini/                         # NEW: Shared Gemini utilities
│   ├── __init__.py
│   └── gemini_provider.py          # Shared client
├── transcription/
│   └── gemini_provider.py          # NEW: Gemini transcription
├── speaker_detectors/
│   └── gemini_detector.py          # NEW: Gemini speaker detection
├── summarization/
│   └── gemini_provider.py          # NEW: Gemini summarization
├── prompts/
│   └── gemini/                     # NEW: Gemini-specific prompts
│       ├── summarization/
│       │   ├── system_v1.j2
│       │   └── long_v1.j2
│       └── ner/
│           ├── system_ner_v1.j2
│           └── guest_host_v1.j2
└── config.py                       # Updated with Gemini fields
```

### 2. Configuration

Add to `config.py`:

```python
from typing import Literal, Optional

# Provider Selection (updated)

transcription_provider: Literal["whisper", "openai", "mistral", "gemini"] = Field(
    default="whisper",
    description="Transcription provider"
)

speaker_detector_provider: Literal["spacy", "openai", "anthropic", "mistral", "deepseek", "gemini"] = Field(
    default="spacy",
    description="Speaker detection provider"
)

summary_provider: Literal["transformers", "openai", "anthropic", "mistral", "deepseek", "gemini"] = Field(
    default="transformers",
    description="Summarization provider"
)

# Gemini API Configuration

gemini_api_key: Optional[str] = Field(
    default=None,
    description="Google AI API key (prefer GEMINI_API_KEY env var)"
)

# Gemini Model Selection

gemini_transcription_model: str = Field(
    default="gemini-2.0-flash",
    description="Gemini model for transcription"
)

gemini_speaker_model: str = Field(
    default="gemini-2.0-flash",
    description="Gemini model for speaker detection"
)

gemini_summary_model: str = Field(
    default="gemini-2.0-flash",
    description="Gemini model for summarization"
)

gemini_temperature: float = Field(
    default=0.3,
    ge=0.0,
    le=2.0,
    description="Temperature for Gemini generation"
)

gemini_max_tokens: Optional[int] = Field(
    default=8192,
    description="Max tokens for Gemini generation"
)

# Gemini Prompt Configuration

gemini_summary_system_prompt: str = Field(
    default="gemini/summarization/system_v1",
    description="Gemini system prompt for summarization"
)

gemini_summary_user_prompt: str = Field(
    default="gemini/summarization/long_v1",
    description="Gemini user prompt for summarization"
)

gemini_ner_system_prompt: str = Field(
    default="gemini/ner/system_ner_v1",
    description="Gemini system prompt for speaker detection"
)

gemini_ner_user_prompt: str = Field(
    default="gemini/ner/guest_host_v1",
    description="Gemini user prompt for speaker detection"
)
```

## 3. API Key Management

```python

# config.py

@field_validator('gemini_api_key', mode='before')
@classmethod
def load_gemini_api_key_from_env(cls, v: Any) -> Optional[str]:
    """Load API key from environment variable if not provided."""
    if v is not None:
        return v
    return os.getenv('GEMINI_API_KEY')

@model_validator(mode='after')
def validate_gemini_config(self) -> 'Config':
    """Validate Gemini provider configuration."""
    needs_key = (
        self.transcription_provider == "gemini" or
        self.speaker_detector_provider == "gemini" or
        self.summary_provider == "gemini"
    )
    if needs_key and not self.gemini_api_key:
        raise ValueError(
            "Gemini API key required when using Gemini providers. "
            "Set GEMINI_API_KEY environment variable or gemini_api_key in config."
        )
    return self
```

## 4. Provider Implementations

### 4.1 Shared Gemini Utilities

**File**: `podcast_scraper/gemini/gemini_provider.py`

```python
"""Shared Gemini provider utilities."""

from __future__ import annotations

import logging
from typing import Any

from google import genai
from google.genai import types

from .. import config

logger = logging.getLogger(__name__)

def create_gemini_client(cfg: config.Config) -> genai.Client:
    """Create Gemini client with configuration.

    Args:
        cfg: Configuration object with gemini_api_key

```text

    Returns:
        Gemini client instance

```
        ValueError: If API key is not provided
    """
    if not cfg.gemini_api_key:
        raise ValueError(
            "Gemini API key required. "
            "Set GEMINI_API_KEY environment variable or gemini_api_key in config."
        )

```

    return genai.Client(api_key=cfg.gemini_api_key)

```
#### 4.2 Transcription Provider

**File**: `podcast_scraper/transcription/gemini_provider.py`

```python

"""Gemini-based transcription provider.

This module uses Gemini's native multimodal audio understanding
for transcription - no separate ASR step required.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from google.genai import types

from .. import config
from ..gemini.gemini_provider import create_gemini_client

logger = logging.getLogger(__name__)

class GeminiTranscriptionProvider:

```text
    """Gemini-based transcription provider using native audio understanding."""
```python

    def __init__(self, cfg: config.Config):
        """Initialize Gemini transcription provider."""
        self.cfg = cfg
        self.client = create_gemini_client(cfg)
        self.model = cfg.gemini_transcription_model
        self._initialized = False

```python
    def initialize(self) -> None:
        """Initialize provider."""
        if self._initialized:
            return
```

        logger.debug("Initializing Gemini transcription provider (model: %s)", self.model)
        self._initialized = True

```python
    def transcribe(self, audio_path: Path, language: Optional[str] = None) -> str:
        """Transcribe audio file using Gemini's native audio understanding.
```

        Args:
            audio_path: Path to audio file
            language: Optional language code

```
        """
        if not self._initialized:
            raise RuntimeError("Provider not initialized")
```

        logger.debug("Transcribing %s via Gemini", audio_path)

```
            with open(audio_path, "rb") as f:
                audio_data = f.read()
```

            # Determine MIME type
            suffix = audio_path.suffix.lower()
            mime_types = {
                ".mp3": "audio/mpeg",
                ".wav": "audio/wav",
                ".m4a": "audio/mp4",
                ".ogg": "audio/ogg",
                ".flac": "audio/flac",
            }
            mime_type = mime_types.get(suffix, "audio/mpeg")

```
```

            prompt = "Transcribe this audio file. Return only the transcription text, no additional commentary."
            if language:
                prompt += f" The audio is in {language}."

            response = self.client.models.generate_content(
                model=self.model,
                contents=[audio_part, prompt],
            )

            text = response.text
            logger.debug("Gemini transcription completed: %d characters", len(text))
            return text

```python
        except Exception as exc:
            logger.error("Gemini API error in transcription: %s", exc)
            raise ValueError(f"Gemini transcription failed: {exc}") from exc
```python

    def transcribe_with_segments(
        self, audio_path: Path, language: Optional[str] = None
    ) -> tuple[str, List[Dict[str, Any]]]:
        """Transcribe with segments (Gemini provides text only, no segments)."""
        text = self.transcribe(audio_path, language)
        # Gemini doesn't provide native segments, return empty list
        return text, []

```python
    def cleanup(self) -> None:
        """Cleanup resources."""
        pass
```

#### 4.3 Speaker Detection Provider

**File**: `podcast_scraper/speaker_detectors/gemini_detector.py`

```python
"""Gemini-based speaker detection provider."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from .. import config, models
from ..gemini.gemini_provider import create_gemini_client

logger = logging.getLogger(__name__)

class GeminiSpeakerDetector:
    """Gemini-based speaker detection provider."""

    def __init__(self, cfg: config.Config):
        """Initialize Gemini speaker detector."""
        self.cfg = cfg
        self.client = create_gemini_client(cfg)
        self.model = cfg.gemini_speaker_model
        self.temperature = cfg.gemini_temperature
        self._initialized = False

```python

    def initialize(self) -> None:
        """Initialize provider."""
        if self._initialized:
            return
        logger.debug("Initializing Gemini speaker detector (model: %s)", self.model)
        self._initialized = True

```python

    def detect_hosts(
        self,
        feed_title: str,
        feed_description: Optional[str],
        feed_authors: Optional[List[str]],
    ) -> Set[str]:
        """Detect hosts from feed metadata."""
        if not self._initialized:
            self.initialize()

```

            feed_title, feed_description, feed_authors
        )

```text

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=[system_prompt + "\n\n" + user_prompt],
                config={"temperature": self.temperature},
            )

```

            hosts = self._parse_hosts_from_response(content)
            logger.debug("Gemini detected hosts: %s", hosts)
            return hosts

```python

        except Exception as e:
            logger.error("Gemini API error in host detection: %s", e)
            raise ValueError(f"Gemini host detection failed: {e}") from e

```python

    def detect_speakers(
        self,
        episode_title: str,
        episode_description: Optional[str],
        known_hosts: Set[str],
    ) -> Tuple[List[str], Set[str], bool]:
        """Detect speakers for an episode."""
        if not self._initialized:
            self.initialize()

```

```text

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=[system_prompt + "\n\n" + user_prompt],
                config={"temperature": self.temperature},
            )

```
            )
            logger.debug("Gemini detected speakers: %s", speakers)
            return speakers, detected_hosts, success

```python

        except Exception as e:
            logger.error("Gemini API error in speaker detection: %s", e)
            raise ValueError(f"Gemini speaker detection failed: {e}") from e

```python

    def analyze_patterns(
        self,
        episodes: List[models.Episode],
        known_hosts: Set[str],
    ) -> Optional[Dict[str, Any]]:
        """Analyze patterns (use local logic)."""
        return None

```python

    def cleanup(self) -> None:
        """Cleanup resources."""
        pass

```python

    def _build_host_detection_prompts(self, feed_title, feed_description, feed_authors):
        from ..prompt_store import render_prompt
        system_prompt = render_prompt(self.cfg.gemini_ner_system_prompt)
        user_prompt = render_prompt(
            self.cfg.gemini_ner_user_prompt,
            feed_title=feed_title,
            feed_description=feed_description or "",
            feed_authors=", ".join(feed_authors) if feed_authors else "",
            task="host_detection",
        )
        return system_prompt, user_prompt

```python

    def _build_speaker_detection_prompts(self, episode_title, episode_description, known_hosts):
        from ..prompt_store import render_prompt
        system_prompt = render_prompt(self.cfg.gemini_ner_system_prompt)
        user_prompt = render_prompt(
            self.cfg.gemini_ner_user_prompt,
            episode_title=episode_title,
            episode_description=episode_description or "",
            known_hosts=", ".join(known_hosts) if known_hosts else "",
            task="speaker_detection",
        )
        return system_prompt, user_prompt

```python

    def _parse_hosts_from_response(self, response_text: str) -> Set[str]:
        try:
            data = json.loads(response_text)
            if isinstance(data, dict) and "hosts" in data:
                return set(data["hosts"])
        except json.JSONDecodeError:
            pass
        hosts = set()
        for line in response_text.strip().split("\n"):
            for name in line.split(","):
                name = name.strip().strip("-").strip("*").strip()
                if name and len(name) > 1:
                    hosts.add(name)
        return hosts

```python

    def _parse_speakers_from_response(self, response_text: str, known_hosts: Set[str]):
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
        speakers = []
        for line in response_text.strip().split("\n"):
            for name in line.split(","):
                name = name.strip().strip("-").strip("*").strip()
                if name and len(name) > 1:
                    speakers.append(name)
        detected_hosts = set(s for s in speakers if s in known_hosts)
        return speakers, detected_hosts, len(speakers) > 0

```
#### 4.4 Summarization Provider

**File**: `podcast_scraper/summarization/gemini_provider.py`

```python

"""Gemini-based summarization provider.

Key advantage: Gemini 1.5 Pro supports 2M token context window,
enabling full transcript processing without chunking.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .. import config
from ..gemini.gemini_provider import create_gemini_client

logger = logging.getLogger(__name__)

class GeminiSummarizationProvider:
    """Gemini-based summarization provider."""

```python
    def __init__(self, cfg: config.Config):
        """Initialize Gemini summarization provider."""
        self.cfg = cfg
        self.client = create_gemini_client(cfg)
        self.model = cfg.gemini_summary_model
        self.temperature = cfg.gemini_temperature
        # Gemini 1.5 Pro supports 2M context
        self.max_context_tokens = 2000000
        self._initialized = False
        self._requires_separate_instances = False
```python

    def initialize(self) -> None:
        """Initialize provider."""
        if self._initialized:
            return
        logger.debug("Initializing Gemini summarization provider (model: %s)", self.model)
        self._initialized = True

```python
    def summarize(
        self,
        text: str,
        episode_title: Optional[str] = None,
        episode_description: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Summarize text using Gemini."""
        if not self._initialized:
            raise RuntimeError("Provider not initialized")
```

        max_length = (params.get("max_length") if params else None) or self.cfg.summary_max_length
        min_length = (params.get("min_length") if params else None) or self.cfg.summary_min_length

```

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

```

                contents=[system_prompt + "\n\n" + user_prompt],
                config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.cfg.gemini_max_tokens or max_length,
                },
            )

            summary = response.text
            if not summary:

```text

                logger.warning("Gemini returned empty summary")
                summary = ""

```python

            from ..prompt_store import get_prompt_metadata
            prompt_metadata = {}
            if system_prompt_name:
                prompt_metadata["system"] = get_prompt_metadata(system_prompt_name)
            prompt_metadata["user"] = get_prompt_metadata(user_prompt_name)

```
                "summary_short": None,
                "metadata": {
                    "model": self.model,
                    "provider": "gemini",
                    "max_length": max_length,
                    "min_length": min_length,
                    "prompts": prompt_metadata,
                },
            }
```python

        except Exception as exc:
            logger.error("Gemini API error: %s", exc)
            raise ValueError(f"Gemini summarization failed: {exc}") from exc

```python
    def _build_summarization_prompts(self, text, episode_title, episode_description, max_length, min_length):
        from ..prompt_store import render_prompt
        system_prompt_name = self.cfg.gemini_summary_system_prompt
        user_prompt_name = self.cfg.gemini_summary_user_prompt
        system_prompt = render_prompt(system_prompt_name)
        paragraphs_min = max(1, min_length // 100)
        paragraphs_max = max(paragraphs_min, max_length // 100)
        template_params = {
            "transcript": text,
            "title": episode_title or "",
            "paragraphs_min": paragraphs_min,
            "paragraphs_max": paragraphs_max,
        }
        template_params.update(self.cfg.summary_prompt_params)
        user_prompt = render_prompt(user_prompt_name, **template_params)
        return system_prompt, user_prompt, system_prompt_name, user_prompt_name, paragraphs_min, paragraphs_max
```python

    def cleanup(self) -> None:
        """Cleanup resources."""
        pass

```
### 5. Factory Updates

Update all factories to include `"gemini"` option following the same pattern as other providers.

### 6. Dependencies

Add to `pyproject.toml`:

```toml

[project.optional-dependencies]
gemini = [
    "google-genai>=0.5.0,<1.0.0",
]

```
## Testing Strategy

Same pattern as other providers: unit tests, integration tests, E2E tests with mock endpoints.

## Success Criteria

1. ✅ Gemini supports all three capabilities
2. ✅ Native audio transcription works
3. ✅ 2M context window is available
4. ✅ Free tier works for development
5. ✅ E2E tests pass

## References

- **Related PRD**: `docs/prd/PRD-012-gemini-provider-integration.md`
- **Google AI Documentation**: https://ai.google.dev/docs
- **Gemini API Reference**: https://ai.google.dev/api/rest
