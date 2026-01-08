# RFC-032: Anthropic Provider Implementation

- **Status**: Draft
- **Authors**:
- **Stakeholders**: Maintainers, users wanting Anthropic API integration, developers implementing providers
- **Related PRDs**:
  - `docs/prd/PRD-009-anthropic-provider-integration.md`
- **Related RFCs**:
  - `docs/rfc/RFC-013-openai-provider-implementation.md` (reference implementation)
  - `docs/rfc/RFC-021-modularization-refactoring-plan.md` (architecture foundation)
  - `docs/rfc/RFC-017-prompt-management.md` (prompt system)

## Abstract

Design and implement Anthropic Claude API providers for speaker detection and summarization capabilities. This RFC builds on the existing modularization architecture (RFC-021) and follows identical patterns to the OpenAI provider implementation (RFC-013) to add Anthropic as pluggable providers while maintaining backward compatibility.

**Architecture Alignment:** This RFC follows the protocol-based provider system established in RFC-021 and uses the same patterns as RFC-013 (OpenAI). Anthropic providers implement the same protocols (`SpeakerDetector`, `SummarizationProvider`) and integrate via the existing factory pattern.

## Problem Statement

Users want the option to use Anthropic Claude API as an alternative to OpenAI for:

1. **Speaker Detection**: Entity extraction using Claude models
2. **Summarization**: High-quality summaries using Claude models

**Note:** Transcription is NOT supported by Anthropic (no audio API).

Requirements:

- No changes to end-user experience or workflow when using defaults
- Secure API key management (environment variables, never in source code)
- Per-capability provider selection (can mix local, OpenAI, and Anthropic)
- Build on existing modularization and provider architecture
- Handle capability gaps gracefully (transcription not supported)
- Use Anthropic-specific prompts (prompts are provider-specific)

## Constraints & Assumptions

**Constraints:**

- **Prerequisite**: Modularization refactoring (RFC-021) ✅ Completed
- **Prerequisite**: OpenAI provider implementation (RFC-013) ✅ Completed
- **Backward Compatibility**: Default providers (local) must remain unchanged
- **API Key Security**: API keys must never be in source code or committed files
- **Capability Gap**: Anthropic does not support audio transcription
- **Rate Limits**: Must respect Anthropic API rate limits and implement retry logic

**Assumptions:**

- Anthropic API is stable and well-documented
- Anthropic Python SDK follows similar patterns to OpenAI SDK
- Prompts need to be optimized for Claude (different from GPT prompts)
- Users understand the capability matrix (what each provider supports)

## Design & Implementation

### 0. Anthropic API Overview

Anthropic's API differs from OpenAI in several ways:

| Feature | OpenAI | Anthropic |
| ------- | ------ | --------- |
| **Chat Endpoint** | `/v1/chat/completions` | `/v1/messages` |
| **Audio API** | ✅ Whisper API | ❌ Not available |
| **Context Window** | 128k tokens | 200k tokens |
| **Temperature Range** | 0.0 - 2.0 | 0.0 - 1.0 |
| **System Message** | In messages array | Separate `system` parameter |

### 1. Architecture Overview

Follow the same structure as OpenAI provider:

```text
podcast_scraper/
├── anthropic/                      # NEW: Shared Anthropic utilities
│   ├── __init__.py
│   └── anthropic_provider.py       # Shared client, rate limiting
├── speaker_detectors/
│   ├── base.py                     # SpeakerDetector protocol (existing)
│   ├── factory.py                  # Updated to include Anthropic
│   ├── ner_detector.py             # Local NER (existing)
│   ├── openai_detector.py          # OpenAI (existing)
│   └── anthropic_detector.py       # NEW: Anthropic implementation
├── summarization/
│   ├── base.py                     # SummarizationProvider protocol (existing)
│   ├── factory.py                  # Updated to include Anthropic
│   ├── local_provider.py           # Local transformers (existing)
│   ├── openai_provider.py          # OpenAI (existing)
│   └── anthropic_provider.py       # NEW: Anthropic implementation
├── prompts/
│   ├── summarization/              # OpenAI prompts (existing)
│   ├── ner/                        # OpenAI prompts (existing)
│   └── anthropic/                  # NEW: Anthropic-specific prompts
│       ├── summarization/
│       │   ├── system_v1.j2
│       │   └── long_v1.j2
│       └── ner/
│           ├── system_ner_v1.j2
│           └── guest_host_v1.j2
└── config.py                       # Updated with Anthropic fields
```

### 2. Configuration

Add to `config.py`:

```python
from typing import Literal, Optional

# Provider Selection (updated)

speaker_detector_provider: Literal["spacy", "openai", "anthropic"] = Field(
    default="spacy",
    description="Speaker detection provider: 'spacy' (local), 'openai', or 'anthropic'"
)

summary_provider: Literal["transformers", "openai", "anthropic"] = Field(
    default="transformers",
    description="Summarization provider: 'transformers' (local), 'openai', or 'anthropic'"
)

# Anthropic API Configuration

anthropic_api_key: Optional[str] = Field(
    default=None,
    description="Anthropic API key (prefer ANTHROPIC_API_KEY env var or .env file)"
)

anthropic_api_base: Optional[str] = Field(
    default=None,
    description="Custom Anthropic API base URL (for E2E testing)"
)

# Anthropic Model Selection

anthropic_speaker_model: str = Field(
    default="claude-3-5-haiku-latest",
    description="Anthropic model for speaker detection"
)

anthropic_summary_model: str = Field(
    default="claude-3-5-haiku-latest",
    description="Anthropic model for summarization"
)

anthropic_temperature: float = Field(
    default=0.3,
    ge=0.0,
    le=1.0,
    description="Temperature for Anthropic generation (0.0-1.0)"
)

anthropic_max_tokens: Optional[int] = Field(
    default=4096,
    description="Max tokens for Anthropic generation"
)

# Anthropic Prompt Configuration

anthropic_summary_system_prompt: str = Field(
    default="anthropic/summarization/system_v1",
    description="Anthropic system prompt for summarization"
)

anthropic_summary_user_prompt: str = Field(
    default="anthropic/summarization/long_v1",
    description="Anthropic user prompt for summarization"
)

anthropic_ner_system_prompt: str = Field(
    default="anthropic/ner/system_ner_v1",
    description="Anthropic system prompt for speaker detection"
)

anthropic_ner_user_prompt: str = Field(
    default="anthropic/ner/guest_host_v1",
    description="Anthropic user prompt for speaker detection"
)
```

## 3. API Key Management

Follow identical pattern to OpenAI (RFC-013 Section 2.2):

```python

# config.py - API Key Loading

from dotenv import load_dotenv

# Load .env file automatically

load_dotenv(override=False)

@field_validator('anthropic_api_key', mode='before')
@classmethod
def load_anthropic_api_key_from_env(cls, v: Any) -> Optional[str]:
    """Load API key from environment variable if not provided."""
    if v is not None:
        return v
    return os.getenv('ANTHROPIC_API_KEY')

@model_validator(mode='after')
def validate_anthropic_config(self) -> 'Config':

```text

    """Validate Anthropic provider configuration."""
    needs_key = (
        self.speaker_detector_provider == "anthropic" or
        self.summary_provider == "anthropic"
    )
    if needs_key and not self.anthropic_api_key:
        raise ValueError(
            "Anthropic API key required when using Anthropic providers. "
            "Set ANTHROPIC_API_KEY environment variable, add it to .env file, "
            "or set anthropic_api_key in config file."
        )
    return self

```
## 4. Provider Capability Validation

Add validation to prevent using Anthropic for transcription:

```python

# config.py - Capability Validation

@model_validator(mode='after')
def validate_provider_capabilities(self) -> 'Config':
    """Validate provider supports requested capability."""
    if self.transcription_provider == "anthropic":
        raise ValueError(
            "Anthropic provider does not support transcription. "
            "Use 'whisper' (local) or 'openai' instead. "
            "See provider capability matrix in documentation."
        )
    return self

```
## 5. Anthropic Provider Implementations

### 5.1 Shared Anthropic Utilities

**File**: `podcast_scraper/anthropic/anthropic_provider.py`

```python

"""Shared Anthropic provider utilities.

This module provides shared utilities for Anthropic API providers,
including client initialization and rate limiting.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from anthropic import Anthropic

from .. import config

logger = logging.getLogger(__name__)

def create_anthropic_client(cfg: config.Config) -> Anthropic:

```text

    """Create Anthropic client with configuration.

```

    Args:
        cfg: Configuration object with anthropic_api_key and optional anthropic_api_base

```
#### 5.2 Speaker Detection Provider

**File**: `podcast_scraper/speaker_detectors/anthropic_detector.py`

```python

"""Anthropic Claude API-based speaker detection provider.

This module provides a SpeakerDetector implementation using Anthropic's Claude API
for cloud-based speaker/guest detection from episode metadata.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from .. import config, models
from ..anthropic.anthropic_provider import create_anthropic_client

logger = logging.getLogger(__name__)

class AnthropicSpeakerDetector:

```text

    """Anthropic Claude API-based speaker detection provider.

```python

    def __init__(self, cfg: config.Config):
        """Initialize Anthropic speaker detector.

```python

    def initialize(self) -> None:
        """Initialize provider (no local model loading needed for API)."""
        if self._initialized:
            return

```python

    def detect_hosts(
        self,
        feed_title: str,
        feed_description: Optional[str],
        feed_authors: Optional[List[str]],
    ) -> Set[str]:
        """Detect hosts from feed metadata using Anthropic API.

```

```python

        except Exception as e:
            logger.error("Anthropic API error in host detection: %s", e)
            raise ValueError(f"Anthropic host detection failed: {e}") from e

```python

    def detect_speakers(
        self,
        episode_title: str,
        episode_description: Optional[str],
        known_hosts: Set[str],
    ) -> Tuple[List[str], Set[str], bool]:
        """Detect speakers for an episode using Anthropic API.

```

                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )

```python

        except Exception as e:
            logger.error("Anthropic API error in speaker detection: %s", e)
            raise ValueError(f"Anthropic speaker detection failed: {e}") from e

```python

    def analyze_patterns(
        self,
        episodes: List[models.Episode],
        known_hosts: Set[str],
    ) -> Optional[Dict[str, Any]]:
        """Analyze episode patterns (optional, can use local logic)."""
        # Return None to use local pattern analysis
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
        )
        return system_prompt, user_prompt

```python

    def _parse_hosts_from_response(self, response_text: str) -> Set[str]:
        """Parse host names from API response."""
        try:
            # Try to parse as JSON first
            data = json.loads(response_text)
            if isinstance(data, dict) and "hosts" in data:
                return set(data["hosts"])
            if isinstance(data, list):
                return set(data)
        except json.JSONDecodeError:
            pass

```
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
                # Combine hosts and guests as speakers
                all_speakers = list(hosts) + guests if not speakers else speakers
                return all_speakers, hosts, True
        except json.JSONDecodeError:
            pass

```

```

#### 5.3 Summarization Provider

**File**: `podcast_scraper/summarization/anthropic_provider.py`

```python

"""Anthropic Claude API-based summarization provider.

This module provides a SummarizationProvider implementation using Anthropic's Claude API
for cloud-based episode summarization.

Key Advantage: Claude models have large context windows (200k tokens), enabling
full transcript processing without chunking for most podcasts.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .. import config
from ..anthropic.anthropic_provider import create_anthropic_client

logger = logging.getLogger(__name__)

class AnthropicSummarizationProvider:

```text

    """Anthropic Claude API-based summarization provider.

```python

    def __init__(self, cfg: config.Config):
        """Initialize Anthropic summarization provider.

```
```python

    def initialize(self) -> None:
        """Initialize provider (no local model loading needed for API)."""
        if self._initialized:
            return

```python

    def summarize(
        self,
        text: str,
        episode_title: Optional[str] = None,
        episode_description: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Summarize text using Anthropic Claude API.

```
                "AnthropicSummarizationProvider not initialized. Call initialize() first."
            )

```

                system_prompt_name,
                user_prompt_name,
                paragraphs_min,
                paragraphs_max,
            ) = self._build_summarization_prompts(
                text, episode_title, episode_description, max_length, min_length
            )

```
                messages=[{"role": "user", "content": user_prompt}],
            )

```python

            # Get prompt metadata for tracking (RFC-017)
            from ..prompt_store import get_prompt_metadata

```
            }
            prompt_metadata["user"] = get_prompt_metadata(user_prompt_name, params=user_params)

```python

        except Exception as exc:
            logger.error("Anthropic API error in summarization: %s", exc)
            raise ValueError(f"Anthropic summarization failed: {exc}") from exc

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

        template_params.update(self.cfg.summary_prompt_params)

```text

        user_prompt = render_prompt(user_prompt_name, **template_params)

```

            paragraphs_min,
            paragraphs_max,
        )

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

    if provider_type in ("spacy", "ner"):  # "ner" for backward compatibility
        from .ner_detector import NERSpeakerDetector
        return NERSpeakerDetector(cfg)
    elif provider_type == "openai":
        from .openai_detector import OpenAISpeakerDetector
        return OpenAISpeakerDetector(cfg)
    elif provider_type == "anthropic":
        from .anthropic_detector import AnthropicSpeakerDetector
        return AnthropicSpeakerDetector(cfg)
    else:
        raise ValueError(
            f"Unsupported speaker detector provider: {provider_type}. "
            "Supported providers: 'spacy', 'openai', 'anthropic'."
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

    if provider_type in ("transformers", "local"):  # "local" for backward compatibility
        from .local_provider import LocalSummarizationProvider
        return LocalSummarizationProvider(cfg)
    elif provider_type == "openai":
        from .openai_provider import OpenAISummarizationProvider
        return OpenAISummarizationProvider(cfg)
    elif provider_type == "anthropic":
        from .anthropic_provider import AnthropicSummarizationProvider
        return AnthropicSummarizationProvider(cfg)
    else:
        raise ValueError(
            f"Unsupported summarization provider: {provider_type}. "
            "Supported providers: 'transformers', 'openai', 'anthropic'."
        )

```

### 7. Anthropic-Specific Prompt Templates

#### 7.1 Summarization System Prompt

**File**: `prompts/anthropic/summarization/system_v1.j2`

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

**File**: `prompts/anthropic/summarization/long_v1.j2`

```jinja2

Please summarize the following podcast transcript.

{% if title %}Episode Title: {{ title }}{% endif %}

Target length: {{ paragraphs_min }} to {{ paragraphs_max }} paragraphs.

Transcript:
{{ transcript }}

Provide a comprehensive summary covering the main topics, key insights, and important takeaways.

```

#### 7.3 NER System Prompt

**File**: `prompts/anthropic/ner/system_ner_v1.j2`

```jinja2

You are an expert at identifying people mentioned in podcast metadata. Your task is to extract speaker names from podcast episode information.

Guidelines:
- Focus on identifying hosts, guests, and speakers
- Return names in a consistent format
- Distinguish between hosts (regular presenters) and guests (episode-specific)
- Respond in JSON format with "hosts" and "guests" arrays

```

#### 7.4 NER User Prompt

**File**: `prompts/anthropic/ner/guest_host_v1.j2`

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

Add Anthropic mock endpoints to `tests/e2e/fixtures/e2e_http_server.py`:

```python

def do_POST(self):
    """Handle POST requests."""
    path = self.path.split("?")[0]

    # Existing OpenAI endpoints...

    # Anthropic Messages API endpoint
    if path == "/v1/messages":
        self._handle_anthropic_messages()
        return

    self.send_error(404, "Endpoint not found")

def _handle_anthropic_messages(self):
    """Handle Anthropic messages API requests."""
    try:
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        request_data = json.loads(body.decode("utf-8"))

```text

        messages = request_data.get("messages", [])
        system = request_data.get("system", "")
        model = request_data.get("model", "claude-3-5-haiku-latest")

```
            response_content = summary_text

```json

        # Build Anthropic response format
        response_data = {
            "id": "msg_test_12345",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [
                {
                    "type": "text",
                    "text": response_content,
                }
            ],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
            },
        }

```python

class E2EServerURLs:
    """URL helper class for E2E server."""

    def anthropic_api_base(self) -> str:
        """Get Anthropic API base URL for E2E testing."""
        return f"http://{self.host}:{self.port}"

```

### 9. Dependencies

Add to `pyproject.toml`:

```toml

[project.optional-dependencies]
anthropic = [
    "anthropic>=0.30.0,<1.0.0",
]

# All AI providers

ai = [
    "openai>=1.0.0,<2.0.0",
    "anthropic>=0.30.0,<1.0.0",
    "tenacity>=8.2.0,<9.0.0",
]

```

# For Anthropic support

pip install -e ".[anthropic]"

# For all AI providers

pip install -e ".[ai]"

```yaml

## Testing Strategy

### Test Coverage

| Test Type | Description | Location |
| --------- | ----------- | -------- |
| **Unit Tests** | Mock Anthropic API calls | `tests/unit/podcast_scraper/test_anthropic_providers.py` |
| **Integration Tests** | Test with E2E server mock | `tests/integration/test_anthropic_providers.py` |
| **E2E Tests** | Full pipeline with Anthropic | `tests/e2e/test_anthropic_provider_integration_e2e.py` |
| **E2E Server Tests** | Verify mock endpoints | `tests/e2e/test_e2e_server.py` |

### Test Organization

Tests follow existing patterns from OpenAI provider tests:

```text

tests/
├── unit/
│   └── podcast_scraper/
│       └── test_anthropic_providers.py
├── integration/
│   └── test_anthropic_providers.py
└── e2e/
    └── test_anthropic_provider_integration_e2e.py

```
### Test Markers

```python

@pytest.mark.unit           # Unit tests
@pytest.mark.integration    # Integration tests
@pytest.mark.e2e           # End-to-end tests
@pytest.mark.llm           # Uses LLM APIs
@pytest.mark.anthropic     # Uses Anthropic specifically

```go

## Rollout & Monitoring

### Rollout Plan

1. **Phase 1**: Core implementation
   - Create `anthropic/` package with shared utilities
   - Implement `AnthropicSpeakerDetector`
   - Implement `AnthropicSummarizationProvider`
   - Add configuration fields

2. **Phase 2**: Integration
   - Update factories
   - Add provider capability validation
   - Create Anthropic-specific prompts
   - Update `.env.example`

3. **Phase 3**: Testing
   - Add E2E server mock endpoints
   - Write unit tests
   - Write integration tests
   - Write E2E tests

4. **Phase 4**: Documentation
   - Update Provider Configuration Quick Reference
   - Update Provider Implementation Guide
   - Add examples to documentation

### Success Criteria

1. ✅ Anthropic providers implement same interfaces as OpenAI providers
2. ✅ Users can select Anthropic providers via configuration
3. ✅ Clear error when attempting transcription with Anthropic
4. ✅ API keys managed securely via environment variables
5. ✅ E2E tests pass with Anthropic mock endpoints
6. ✅ Default behavior (local providers) unchanged
7. ✅ Documentation complete and clear

## Alternatives Considered

### 1. Unified LLM Provider

**Description**: Single provider that abstracts all LLM APIs (OpenAI, Anthropic, etc.)

**Pros**:
- Less code duplication
- Easier to add new providers

**Cons**:
- API differences make abstraction complex
- Prompt optimization varies by model
- Would require significant refactoring

**Why Rejected**: Current per-provider approach is simpler and allows provider-specific optimizations.

### 2. LangChain Integration

**Description**: Use LangChain as abstraction layer for all LLM providers

**Pros**:
- Industry-standard abstraction
- Built-in retry logic and rate limiting

**Cons**:
- Heavy dependency
- Overhead for simple use cases
- Less control over prompts

**Why Rejected**: Too heavyweight for current needs; may consider for future if complexity grows.

## Open Questions

1. **Rate Limiting**: Should we share rate limiting configuration across providers or keep separate?
2. **Prompt Versioning**: Should Anthropic prompts follow same versioning as OpenAI or independent?
3. **Cost Tracking**: Should we add cost estimation/tracking features?

## References

- **Related PRD**: `docs/prd/PRD-009-anthropic-provider-integration.md`
- **OpenAI Provider RFC**: `docs/rfc/RFC-013-openai-provider-implementation.md`
- **Modularization Plan**: `docs/rfc/RFC-021-modularization-refactoring-plan.md`
- **Anthropic API Documentation**: https://docs.anthropic.com/en/api
- **Anthropic Python SDK**: https://github.com/anthropics/anthropic-sdk-python
