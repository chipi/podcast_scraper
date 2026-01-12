# RFC-036: Groq Provider Implementation

- **Status**: Draft
- **Authors**:
- **Stakeholders**: Maintainers, users wanting ultra-fast inference
- **Related PRDs**:
  - `docs/prd/PRD-013-groq-provider-integration.md`
- **Related RFCs**:
  - `docs/rfc/RFC-034-deepseek-provider-implementation.md` (same pattern - OpenAI SDK)

## Abstract

Design and implement Groq providers for speaker detection and summarization capabilities. Groq offers **ultra-fast inference** (10x faster than other providers) by running models on custom LPU hardware. This RFC follows the DeepSeek pattern of using the OpenAI SDK with a custom base_url.

**Key Advantage:** Groq processes at 500+ tokens/second vs ~50-100 for other providers.

## Problem Statement

Users want Groq for:

1. **Speaker Detection**: Entity extraction using Llama/Mixtral models
2. **Summarization**: High-quality summaries at 10x speed

**Note:** Transcription is NOT supported (Groq hosts LLMs only, no audio models).

Key advantages:

- **10x faster** inference than any other provider
- **Open source models** (Llama 3.3, Mixtral, Gemma)
- **OpenAI-compatible API** - no new SDK required
- **Generous free tier** - 14,400 tokens/minute

## Constraints & Assumptions

**Constraints:**

- No audio transcription support
- Rate limits on free tier (30 RPM, 14,400 TPM)
- Model selection limited to Groq-hosted models

**Assumptions:**

- Groq API maintains OpenAI compatibility
- Llama 3.3 70B quality is comparable to GPT-4o-mini

## Design & Implementation

### 0. Groq API Overview

| Feature | OpenAI | Groq |
| ------- | ------ | ---- |
| **Base URL** | `https://api.openai.com/v1` | `https://api.groq.com/openai/v1` |
| **SDK** | `openai` | `openai` (with custom base_url) |
| **Audio API** | ✅ Whisper | ❌ Not available |
| **Speed** | ~100 tokens/sec | **500+ tokens/sec** |
| **Models** | Proprietary | Open source (Llama, Mixtral, Gemma) |

### 1. Architecture Overview

```text
podcast_scraper/
├── groq/                           # NEW: Shared Groq utilities
│   ├── __init__.py
│   └── groq_provider.py            # Shared client using OpenAI SDK
├── speaker_detectors/
│   └── groq_detector.py            # NEW: Groq speaker detection
├── summarization/
│   └── groq_provider.py            # NEW: Groq summarization
├── prompts/
│   └── groq/                       # NEW: Groq/Llama-specific prompts
│       ├── summarization/
│       │   ├── system_v1.j2
│       │   └── long_v1.j2
│       └── ner/
│           ├── system_ner_v1.j2
│           └── guest_host_v1.j2
└── config.py                       # Updated with Groq fields
```

### 2. Configuration

```python
from typing import Literal, Optional

# Provider Selection (updated)

speaker_detector_provider: Literal[
    "spacy", "openai", "anthropic", "mistral", "deepseek", "gemini", "groq"
] = Field(default="spacy")

summary_provider: Literal[
    "transformers", "openai", "anthropic", "mistral", "deepseek", "gemini", "groq"
] = Field(default="transformers")

# Groq API Configuration

groq_api_key: Optional[str] = Field(
    default=None,
    description="Groq API key (prefer GROQ_API_KEY env var)"
)

groq_api_base: str = Field(
    default="https://api.groq.com/openai/v1",
    description="Groq API base URL"
)

# Groq Model Selection

groq_speaker_model: str = Field(
    default="llama-3.3-70b-versatile",
    description="Groq model for speaker detection"
)

groq_summary_model: str = Field(
    default="llama-3.3-70b-versatile",
    description="Groq model for summarization"
)

groq_temperature: float = Field(
    default=0.3,
    ge=0.0,
    le=2.0,
    description="Temperature for Groq generation"
)

groq_max_tokens: Optional[int] = Field(
    default=4096,
    description="Max tokens for Groq generation"
)

# Groq Prompt Configuration

groq_summary_system_prompt: str = Field(
    default="groq/summarization/system_v1",
    description="Groq system prompt for summarization"
)

groq_summary_user_prompt: str = Field(
    default="groq/summarization/long_v1",
    description="Groq user prompt for summarization"
)

groq_ner_system_prompt: str = Field(
    default="groq/ner/system_ner_v1",
    description="Groq system prompt for speaker detection"
)

groq_ner_user_prompt: str = Field(
    default="groq/ner/guest_host_v1",
    description="Groq user prompt for speaker detection"
)
```

## 3. API Key Management and Validation

```python

# config.py

@field_validator('groq_api_key', mode='before')
@classmethod
def load_groq_api_key_from_env(cls, v: Any) -> Optional[str]:
    if v is not None:
        return v
    return os.getenv('GROQ_API_KEY')

@model_validator(mode='after')
def validate_groq_config(self) -> 'Config':
    needs_key = (
        self.speaker_detector_provider == "groq" or
        self.summary_provider == "groq"
    )
    if needs_key and not self.groq_api_key:
        raise ValueError(
            "Groq API key required when using Groq providers. "
            "Set GROQ_API_KEY environment variable or groq_api_key in config."
        )
    return self
```

## 4. Provider Capability Validation

```python

# config.py - Update existing validation

@model_validator(mode='after')
def validate_provider_capabilities(self) -> 'Config':
    # Providers that don't support transcription
    no_transcription = {"anthropic", "deepseek", "groq"}
    if self.transcription_provider in no_transcription:
        raise ValueError(
            f"{self.transcription_provider.title()} provider does not support transcription. "
            "Use 'whisper' (local), 'openai', 'mistral', or 'gemini' instead."
        )
    return self
```

## 5. Provider Implementations

### 5.1 Shared Groq Utilities

**File**: `podcast_scraper/groq/groq_provider.py`

```python
"""Shared Groq provider utilities.

Uses OpenAI SDK with custom base_url (same pattern as DeepSeek).
"""

from __future__ import annotations

import logging
from openai import OpenAI
from .. import config

logger = logging.getLogger(__name__)

def create_groq_client(cfg: config.Config) -> OpenAI:
    """Create Groq client using OpenAI SDK with custom base_url."""
    if not cfg.groq_api_key:
        raise ValueError(
            "Groq API key required. "
            "Set GROQ_API_KEY environment variable or groq_api_key in config."
        )

```text

    return OpenAI(
        api_key=cfg.groq_api_key,
        base_url=cfg.groq_api_base,
    )

```
#### 5.2 Speaker Detection Provider

**File**: `podcast_scraper/speaker_detectors/groq_detector.py`

```python

"""Groq-based speaker detection provider.

Uses Llama/Mixtral models hosted on Groq's ultra-fast LPU infrastructure.
10x faster than other cloud providers.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from .. import config, models
from ..groq.groq_provider import create_groq_client

logger = logging.getLogger(__name__)

class GroqSpeakerDetector:

```text
    """Groq-based speaker detection provider."""
```python

    def __init__(self, cfg: config.Config):
        self.cfg = cfg
        self.client = create_groq_client(cfg)
        self.model = cfg.groq_speaker_model
        self.temperature = cfg.groq_temperature
        self._initialized = False

```python
    def initialize(self) -> None:
        if self._initialized:
            return
        logger.debug("Initializing Groq speaker detector (model: %s)", self.model)
        self._initialized = True
```python

    def detect_hosts(
        self,
        feed_title: str,
        feed_description: Optional[str],
        feed_authors: Optional[List[str]],
    ) -> Set[str]:
        if not self._initialized:
            self.initialize()

```
```text

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.cfg.groq_max_tokens or 500,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

```

```python

        except Exception as e:
            logger.error("Groq API error: %s", e)
            raise ValueError(f"Groq host detection failed: {e}") from e

```python
    def detect_speakers(
        self,
        episode_title: str,
        episode_description: Optional[str],
        known_hosts: Set[str],
    ) -> Tuple[List[str], Set[str], bool]:
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
                max_tokens=self.cfg.groq_max_tokens or 500,
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
            logger.debug("Groq detected speakers: %s", speakers)
            return speakers, detected_hosts, success

```python
        except Exception as e:
            logger.error("Groq API error: %s", e)
            raise ValueError(f"Groq speaker detection failed: {e}") from e
```python

    def analyze_patterns(self, episodes, known_hosts):
        return None

```python
    def cleanup(self) -> None:
        pass
```python

    def _build_host_detection_prompts(self, feed_title, feed_description, feed_authors):
        from ..prompt_store import render_prompt
        system_prompt = render_prompt(self.cfg.groq_ner_system_prompt)
        user_prompt = render_prompt(
            self.cfg.groq_ner_user_prompt,
            feed_title=feed_title,
            feed_description=feed_description or "",
            feed_authors=", ".join(feed_authors) if feed_authors else "",
            task="host_detection",
        )
        return system_prompt, user_prompt

```python
    def _build_speaker_detection_prompts(self, episode_title, episode_description, known_hosts):
        from ..prompt_store import render_prompt
        system_prompt = render_prompt(self.cfg.groq_ner_system_prompt)
        user_prompt = render_prompt(
            self.cfg.groq_ner_user_prompt,
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

#### 5.3 Summarization Provider

**File**: `podcast_scraper/summarization/groq_provider.py`

```python
"""Groq-based summarization provider.

Key advantage: Ultra-fast inference (500+ tokens/second).
Processes in ~1/10th the time of other providers.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .. import config
from ..groq.groq_provider import create_groq_client

logger = logging.getLogger(__name__)

class GroqSummarizationProvider:
    """Groq-based summarization provider."""

```python

    def __init__(self, cfg: config.Config):
        self.cfg = cfg
        self.client = create_groq_client(cfg)
        self.model = cfg.groq_summary_model
        self.temperature = cfg.groq_temperature
        # Llama 3.3 70B supports 128k context
        self.max_context_tokens = 128000
        self._initialized = False
        self._requires_separate_instances = False

```python

    def initialize(self) -> None:
        if self._initialized:
            return
        logger.debug("Initializing Groq summarization provider (model: %s)", self.model)
        self._initialized = True

```python

    def summarize(
        self,
        text: str,
        episode_title: Optional[str] = None,
        episode_description: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self._initialized:
            raise RuntimeError("Provider not initialized")

```
```

        logger.debug("Summarizing via Groq (model: %s)", self.model)

```
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
                max_tokens=self.cfg.groq_max_tokens or max_length,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

            summary = response.choices[0].message.content
            if not summary:

```text

                logger.warning("Groq returned empty summary")
                summary = ""

```

            logger.debug("Groq summarization completed: %d characters", len(summary))

```python

            from ..prompt_store import get_prompt_metadata
            prompt_metadata = {}
            if system_prompt_name:
                prompt_metadata["system"] = get_prompt_metadata(system_prompt_name)
            prompt_metadata["user"] = get_prompt_metadata(user_prompt_name)

```

            return {
                "summary": summary,
                "summary_short": None,
                "metadata": {
                    "model": self.model,
                    "provider": "groq",
                    "max_length": max_length,
                    "min_length": min_length,
                    "prompts": prompt_metadata,
                },
            }

```python

        except Exception as exc:
            logger.error("Groq API error: %s", exc)
            raise ValueError(f"Groq summarization failed: {exc}") from exc

```python

    def _build_summarization_prompts(self, text, episode_title, episode_description, max_length, min_length):
        from ..prompt_store import render_prompt
        system_prompt_name = self.cfg.groq_summary_system_prompt
        user_prompt_name = self.cfg.groq_summary_user_prompt
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
        pass

```

### 6. Dependencies

No new dependencies - uses existing `openai` package:

```toml

# pyproject.toml - No changes needed
# Groq uses OpenAI SDK with custom base_url

```

## Testing Strategy

Same pattern as DeepSeek: reuse OpenAI mock endpoints with Groq base_url.

## Success Criteria

1. ✅ Groq providers are 10x faster than other cloud providers
2. ✅ Clear error when attempting transcription
3. ✅ Free tier works for development
4. ✅ No new SDK dependency
5. ✅ E2E tests pass

## References

- **Related PRD**: `docs/prd/PRD-013-groq-provider-integration.md`
- **DeepSeek RFC**: `docs/rfc/RFC-034-deepseek-provider-implementation.md`
- **Groq Documentation**: https://console.groq.com/docs
- **Groq Models**: https://console.groq.com/docs/models
