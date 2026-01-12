# RFC-037: Ollama Provider Implementation

- **Status**: Draft
- **Authors**:
- **Stakeholders**: Maintainers, privacy-conscious users, offline users
- **Related PRDs**:
  - `docs/prd/PRD-014-ollama-provider-integration.md`
- **Related RFCs**:
  - `docs/rfc/RFC-034-deepseek-provider-implementation.md` (OpenAI SDK pattern)
  - `docs/rfc/RFC-036-groq-provider-implementation.md` (similar pattern)

## Abstract

Design and implement Ollama providers for speaker detection and summarization capabilities. Ollama is unique in being a **fully local/offline solution** that runs open-source LLMs on your own hardware with ZERO API costs. This RFC follows the established pattern of using the OpenAI SDK with a custom base_url.

**Key Advantages:**

- No internet required
- No API costs
- No rate limits
- Complete data privacy

## Problem Statement

Users want Ollama for:

1. **Speaker Detection**: Entity extraction using local Llama/Mistral models
2. **Summarization**: High-quality summaries using local models

**Note:** Transcription is NOT supported (Ollama hosts LLMs only).

Key advantages:

- **Fully offline** - no internet required
- **Zero cost** - no per-token pricing
- **Complete privacy** - data never leaves local machine
- **OpenAI-compatible API** - uses same SDK

## Constraints & Assumptions

**Constraints:**

- Ollama must be installed and running locally
- Models must be pulled before use (`ollama pull llama3.3`)
- Performance depends on local hardware
- No audio transcription support

**Assumptions:**

- User has sufficient hardware for model of choice
- Ollama server is running on default port (11434)
- User has already pulled desired models

## Design & Implementation

### 0. Ollama API Overview

| Feature | OpenAI | Ollama |
| ------- | ------ | ------ |
| **Base URL** | `https://api.openai.com/v1` | `http://localhost:11434/v1` |
| **API Key** | Required | Not required |
| **SDK** | `openai` | `openai` (with custom base_url) |
| **Audio API** | ✅ Whisper | ❌ Not available |
| **Pricing** | Per token | **Free** |
| **Rate Limits** | Yes | **No** |

### 1. Architecture Overview

```text
podcast_scraper/
├── ollama/                         # NEW: Shared Ollama utilities
│   ├── __init__.py
│   └── ollama_provider.py          # Shared client, model validation
├── speaker_detectors/
│   └── ollama_detector.py          # NEW: Ollama speaker detection
├── summarization/
│   └── ollama_provider.py          # NEW: Ollama summarization
├── prompts/
│   └── ollama/                     # NEW: Ollama/Llama-specific prompts
│       ├── summarization/
│       │   ├── system_v1.j2
│       │   └── long_v1.j2
│       └── ner/
│           ├── system_ner_v1.j2
│           └── guest_host_v1.j2
└── config.py                       # Updated with Ollama fields
```

### 2. Configuration

```python
from typing import Literal, Optional

# Provider Selection (updated)

speaker_detector_provider: Literal[
    "spacy", "openai", "anthropic", "mistral", "deepseek", "gemini", "groq", "ollama"
] = Field(default="spacy")

summary_provider: Literal[
    "transformers", "openai", "anthropic", "mistral", "deepseek", "gemini", "groq", "ollama"
] = Field(default="transformers")

# Ollama API Configuration (NO API KEY NEEDED)

ollama_api_base: str = Field(
    default="http://localhost:11434/v1",
    description="Ollama API base URL"
)

# Ollama Model Selection

ollama_speaker_model: str = Field(
    default="llama3.3:latest",
    description="Ollama model for speaker detection"
)

ollama_summary_model: str = Field(
    default="llama3.3:latest",
    description="Ollama model for summarization"
)

ollama_temperature: float = Field(
    default=0.3,
    ge=0.0,
    le=2.0,
    description="Temperature for Ollama generation"
)

ollama_max_tokens: Optional[int] = Field(
    default=4096,
    description="Max tokens for Ollama generation"
)

# Ollama Prompt Configuration

ollama_summary_system_prompt: str = Field(
    default="ollama/summarization/system_v1",
    description="Ollama system prompt for summarization"
)

ollama_summary_user_prompt: str = Field(
    default="ollama/summarization/long_v1",
    description="Ollama user prompt for summarization"
)

ollama_ner_system_prompt: str = Field(
    default="ollama/ner/system_ner_v1",
    description="Ollama system prompt for speaker detection"
)

ollama_ner_user_prompt: str = Field(
    default="ollama/ner/guest_host_v1",
    description="Ollama user prompt for speaker detection"
)

# Ollama Connection Settings

ollama_timeout: int = Field(
    default=120,
    description="Timeout in seconds for Ollama API calls (local inference can be slow)"
)
```

## 3. Provider Capability Validation

```python

# config.py - Update existing validation

@model_validator(mode='after')
def validate_provider_capabilities(self) -> 'Config':
    # Providers that don't support transcription
    no_transcription = {"anthropic", "deepseek", "groq", "ollama"}
    if self.transcription_provider in no_transcription:
        raise ValueError(
            f"{self.transcription_provider.title()} provider does not support transcription. "
            "Use 'whisper' (local), 'openai', 'mistral', or 'gemini' instead."
        )
    return self
```

## 4. Provider Implementations

### 4.1 Shared Ollama Utilities

**File**: `podcast_scraper/ollama/ollama_provider.py`

```python
"""Shared Ollama provider utilities.

Ollama provides a local LLM server with an OpenAI-compatible API.
No API key required - runs entirely on local hardware.
"""

from __future__ import annotations

import logging
import httpx
from openai import OpenAI
from .. import config

logger = logging.getLogger(__name__)

OLLAMA_NOT_RUNNING_ERROR = """
Ollama server is not running. Please start it with:

    ollama serve

Or install Ollama from: https://ollama.ai
"""

MODEL_NOT_FOUND_ERROR = """
Model '{model}' is not available in Ollama. Install it with:

    ollama pull {model}

Available models can be listed with:

    ollama list
"""

def create_ollama_client(cfg: config.Config) -> OpenAI:

```text

    """Create Ollama client using OpenAI SDK.

```python

    Note: Ollama doesn't require an API key.
    """
    # Test if Ollama is running
    try:
        base_url = cfg.ollama_api_base.rstrip('/v1')  # Remove /v1 for health check
        response = httpx.get(f"{base_url}/api/version", timeout=5.0)
        response.raise_for_status()
    except (httpx.ConnectError, httpx.TimeoutException) as e:
        raise ConnectionError(OLLAMA_NOT_RUNNING_ERROR) from e

```

    # OpenAI SDK with Ollama - use dummy key (Ollama ignores it)
    return OpenAI(
        api_key="ollama",  # Ollama ignores API key, but SDK requires one
        base_url=cfg.ollama_api_base,
        timeout=cfg.ollama_timeout,
    )

```python
def validate_model_available(cfg: config.Config, model: str) -> None:

```text

    """Check if model is available in Ollama."""
    try:
        base_url = cfg.ollama_api_base.rstrip('/v1')
        response = httpx.get(f"{base_url}/api/tags", timeout=10.0)
        response.raise_for_status()
        data = response.json()
        available_models = [m.get("name", "") for m in data.get("models", [])]

```
            raise ValueError(MODEL_NOT_FOUND_ERROR.format(model=model))

```

    except httpx.HTTPError as e:
        logger.warning("Could not validate model availability: %s", e)

```
#### 4.2 Speaker Detection Provider

**File**: `podcast_scraper/speaker_detectors/ollama_detector.py`

```python

"""Ollama-based speaker detection provider.

Runs entirely locally on user's hardware.
No API costs, no rate limits, complete privacy.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from .. import config, models
from ..ollama.ollama_provider import create_ollama_client, validate_model_available

logger = logging.getLogger(__name__)

class OllamaSpeakerDetector:

```text
    """Ollama-based speaker detection provider."""
```python

    def __init__(self, cfg: config.Config):
        self.cfg = cfg
        self.model = cfg.ollama_speaker_model
        self.temperature = cfg.ollama_temperature
        self._client = None
        self._initialized = False

```python
    def initialize(self) -> None:
        if self._initialized:
            return
```

        logger.debug("Initializing Ollama speaker detector (model: %s)", self.model)

```
        # Create client
        self._client = create_ollama_client(self.cfg)
        self._initialized = True
        logger.debug("Ollama speaker detector initialized")

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
            feed_title, feed_description, feed_authors
        )

```text

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                max_tokens=self.cfg.ollama_max_tokens or 500,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

```
            hosts = self._parse_hosts_from_response(content)
            logger.debug("Ollama detected hosts: %s", hosts)
            return hosts

```python

        except Exception as e:
            logger.error("Ollama error: %s", e)
            raise ValueError(f"Ollama host detection failed: {e}") from e

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
```text

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                max_tokens=self.cfg.ollama_max_tokens or 500,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

```

            )
            logger.debug("Ollama detected speakers: %s", speakers)
            return speakers, detected_hosts, success

```python

        except Exception as e:
            logger.error("Ollama error: %s", e)
            raise ValueError(f"Ollama speaker detection failed: {e}") from e

```python

    def analyze_patterns(self, episodes, known_hosts):
        return None

```python

    def cleanup(self) -> None:
        pass

```python

    def _build_host_detection_prompts(self, feed_title, feed_description, feed_authors):
        from ..prompt_store import render_prompt
        system_prompt = render_prompt(self.cfg.ollama_ner_system_prompt)
        user_prompt = render_prompt(
            self.cfg.ollama_ner_user_prompt,
            feed_title=feed_title,
            feed_description=feed_description or "",
            feed_authors=", ".join(feed_authors) if feed_authors else "",
            task="host_detection",
        )
        return system_prompt, user_prompt

```python

    def _build_speaker_detection_prompts(self, episode_title, episode_description, known_hosts):
        from ..prompt_store import render_prompt
        system_prompt = render_prompt(self.cfg.ollama_ner_system_prompt)
        user_prompt = render_prompt(
            self.cfg.ollama_ner_user_prompt,
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

#### 4.3 Summarization Provider

**File**: `podcast_scraper/summarization/ollama_provider.py`

```python

"""Ollama-based summarization provider.

Runs entirely locally - no API costs, complete privacy.
Performance depends on local hardware.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .. import config
from ..ollama.ollama_provider import create_ollama_client, validate_model_available

logger = logging.getLogger(__name__)

class OllamaSummarizationProvider:
    """Ollama-based summarization provider."""

```python

    def __init__(self, cfg: config.Config):
        self.cfg = cfg
        self.model = cfg.ollama_summary_model
        self.temperature = cfg.ollama_temperature
        self._client = None
        self._initialized = False
        self._requires_separate_instances = False
        # Context depends on model - assume 128k for modern models
        self.max_context_tokens = 128000

```python

    def initialize(self) -> None:
        if self._initialized:
            return

```

        # Validate model is available
        validate_model_available(self.cfg, self.model)

```
        self._initialized = True
        logger.debug("Ollama summarization provider initialized")

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

```
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

            response = self._client.chat.completions.create(
                model=self.model,
                max_tokens=self.cfg.ollama_max_tokens or max_length,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

            summary = response.choices[0].message.content
            if not summary:

```text

                logger.warning("Ollama returned empty summary")
                summary = ""

```

```python

            from ..prompt_store import get_prompt_metadata
            prompt_metadata = {}
            if system_prompt_name:
                prompt_metadata["system"] = get_prompt_metadata(system_prompt_name)
            prompt_metadata["user"] = get_prompt_metadata(user_prompt_name)

```

                "summary": summary,
                "summary_short": None,
                "metadata": {
                    "model": self.model,
                    "provider": "ollama",
                    "max_length": max_length,
                    "min_length": min_length,
                    "prompts": prompt_metadata,
                },
            }

```python

        except Exception as exc:
            logger.error("Ollama error: %s", exc)
            raise ValueError(f"Ollama summarization failed: {exc}") from exc

```python

    def _build_summarization_prompts(self, text, episode_title, episode_description, max_length, min_length):
        from ..prompt_store import render_prompt
        system_prompt_name = self.cfg.ollama_summary_system_prompt
        user_prompt_name = self.cfg.ollama_summary_user_prompt
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

### 5. Dependencies

Add `httpx` for health checks (likely already installed):

```toml

[project.optional-dependencies]
ollama = [
    "httpx>=0.24.0,<1.0.0",  # For Ollama health checks
]

```

## Testing Strategy

### Unit Tests

Mock the OpenAI client and httpx calls.

### Integration Tests

Two approaches:

1. **Mock Ollama**: Use E2E server with OpenAI-compatible endpoints
2. **Real Ollama**: Skip if Ollama not running (`pytest.mark.skipif`)

### E2E Tests

Mark tests as `@pytest.mark.ollama` and skip if Ollama not available.

## Success Criteria

1. ✅ Works completely offline
2. ✅ Zero API costs
3. ✅ Clear error when Ollama not running
4. ✅ Clear error when model not installed
5. ✅ E2E tests pass

## References

- **Related PRD**: `docs/prd/PRD-014-ollama-provider-integration.md`
- **Ollama Documentation**: https://ollama.ai
- **Ollama API Reference**: https://github.com/ollama/ollama/blob/main/docs/api.md
- **OpenAI Compatibility**: https://ollama.ai/blog/openai-compatibility
