# RFC-013: OpenAI Provider Implementation

- **Status**: Completed
- **Authors**:
- **Stakeholders**: Maintainers, users wanting OpenAI API integration, developers implementing providers
- **Related PRDs**: `docs/prd/PRD-006-openai-provider-integration.md`
- **Related RFCs**: `docs/rfc/RFC-017-prompt-management.md`
- **Related RFCs**: `docs/rfc/RFC-021-modularization-refactoring-plan.md` (historical reference - modularization plan)
- **Related Issues**: (to be created)

## Abstract

Design and implement OpenAI API providers for speaker detection, transcription, and summarization capabilities. This RFC builds on the modularization refactoring plan to add OpenAI as pluggable providers while maintaining backward compatibility and zero changes to end-user experience with default (local) providers.

## Problem Statement

**Note:** For detailed model selection guidance and cost analysis, see `docs/prd/PRD-006-openai-provider-integration.md` section "OpenAI Model Selection and Cost Analysis".

Users want the option to use OpenAI API for:

1. **Speaker Detection**: Higher accuracy entity extraction using GPT models
2. **Transcription**: Faster or higher-quality transcription using Whisper API
3. **Summarization**: Higher-quality summaries using GPT models

Requirements:

- No changes to end-user experience or workflow when using defaults
- Secure API key management (environment variables, never in source code)
- Per-capability provider selection (can mix local and OpenAI)
- Maintain parallelism and performance characteristics
- Build on existing modularization refactoring

## Constraints & Assumptions

- **Prerequisite**: Modularization refactoring (RFC-021) must be completed first (✅ **Completed**)
- **Backward Compatibility**: Default providers (transformers/local) must remain unchanged
- **API Key Security**: API keys must never be in source code or committed files
- **Environment Support**: Must work in both development and production environments
- **Rate Limits**: Must respect OpenAI API rate limits and implement retry logic
- **Cost Awareness**: API usage incurs costs; users should be aware
- **Network Dependency**: Requires internet connectivity when using OpenAI providers
- **Error Handling**: API failures must be handled gracefully with clear error messages

## Design & Implementation

### 0. Preprocessing Strategy (Provider-Agnostic)

**Key Principle**: All preprocessing steps should be provider-agnostic and applied BEFORE passing data to any provider (local or OpenAI).

**Current Preprocessing Steps:**

1. **Transcript Cleaning** (for summarization):
   - Remove timestamps `[00:12:34]` (language-agnostic)
   - Remove generic speaker tags (`Speaker 1:`, `Host:`, etc.) while preserving actual names
   - Collapse excessive blank lines
   - Optionally remove filler words (disabled by default for multi-language support)

2. **Sponsor Block Removal** (for summarization):
   - Remove sponsor/advertisement blocks
   - Remove outro blocks (subscription prompts, etc.)

3. **Name Sanitization** (for speaker detection):
   - Remove parentheses, punctuation
   - Normalize whitespace
   - Preserve actual speaker names

**Implementation Location:**

- Preprocessing happens in `metadata.py` BEFORE calling providers
- Functions like `clean_transcript()` and `remove_sponsor_blocks()` are in `summarizer.py` but are provider-agnostic utilities
- After modularization, these should remain in a shared preprocessing module or utility functions

**Benefits:**

- ✅ Consistent preprocessing regardless of provider
- ✅ More efficient (do once, not per provider)
- ✅ Easier to maintain (single implementation)
- ✅ Providers receive clean, standardized input

**After Modularization:**

- Preprocessing functions should be moved to a shared module (e.g., `podcast_scraper/preprocessing.py`)
- Called in `metadata.py` or `workflow.py` BEFORE provider selection
- All providers (local and OpenAI) receive preprocessed text
- No provider-specific preprocessing needed

### 1. Architecture Overview

Build on the provider abstraction from modularization refactoring:

````text
podcast_scraper/
├── speaker_detectors/
│   ├── base.py              # SpeakerDetector protocol
│   ├── factory.py           # Factory (selects provider)
│   ├── ner_detector.py      # Local NER provider (existing)
│   └── openai_detector.py   # NEW: OpenAI provider
├── transcription/
│   ├── base.py              # TranscriptionProvider protocol
│   ├── factory.py           # Factory (selects provider)
│   ├── whisper_provider.py  # Local Whisper provider (existing)
│   └── openai_provider.py   # NEW: OpenAI Whisper API provider
├── summarization/
│   ├── base.py              # SummarizationProvider protocol
│   ├── factory.py           # Factory (selects provider)
│   ├── local_provider.py    # Local transformers provider (existing)
│   └── openai_provider.py   # NEW: OpenAI GPT provider
└── config.py                # Provider type fields + API key config
```text

Add to `config.py` (already planned in refactoring):

```python

# Speaker Detection Provider

speaker_detector_type: Literal["ner", "openai"] = Field(
    default="ner",
    description="Speaker detection provider: 'ner' (local spaCy) or 'openai' (GPT API)"
)

# Transcription Provider

transcription_provider: Literal["whisper", "openai"] = Field(
    default="whisper",
    description="Transcription provider: 'whisper' (local) or 'openai' (Whisper API)"
)

# Summarization Provider

summary_provider: Literal["transformers", "openai"] = Field(
    default="transformers",
    description="Summarization provider: 'transformers' (HuggingFace) or 'openai' (GPT API)"
)

# Keep existing fields for backward compatibility

ner_model: Optional[str] = Field(default=None, alias="ner_model")
whisper_model: str = Field(default="base", alias="whisper_model")
summary_model: Optional[str] = Field(default=None, alias="summary_model")
```text

We use `python-dotenv` to manage environment variables via `.env` files, providing a convenient way to configure API keys per environment (development, staging, production) without hardcoding them.

**Implementation:**

```python

# config.py (at module level, before Config class)

from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root (if it exists)

# This happens automatically when config module is imported

_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path, override=False)  # Don't override existing env vars
else:

    # Also check current working directory (for flexibility)

    load_dotenv(override=False)

# Config class

openai_api_key: Optional[str] = Field(
    default=None,
    description="OpenAI API key (prefer OPENAI_API_KEY environment variable or .env file)"
)

@field_validator('openai_api_key', mode='before')
@classmethod
def load_api_key_from_env(cls, v: Any) -> Optional[str]:

```python

    """Load API key from environment variable if not provided."""
    if v is not None:
        return v
    return os.getenv('OPENAI_API_KEY')

```python
@model_validator(mode='after')
def validate_openai_config(self) -> 'Config':

```text

    """Validate OpenAI provider configuration."""

```
# .env (add to .gitignore!)

# OpenAI API Configuration

OPENAI_API_KEY=sk-your-actual-api-key-here

# Optional: OpenAI Organization ID (if you're in multiple orgs)

OPENAI_ORGANIZATION=org-your-org-id

# Optional: Custom API base URL (for proxies)

# OPENAI_API_BASE=https://api.openai.com/v1

# Other environment variables

LOG_LEVEL=INFO

```text

2. **Create `examples/.env.example`** template (commit this to git):

```bash

# examples/.env.example

# Copy this file to .env and fill in your actual values

# DO NOT commit .env to git!

# OpenAI API Configuration

OPENAI_API_KEY=sk-your-api-key-here

# OPENAI_ORGANIZATION=org-your-org-id

# OPENAI_API_BASE=https://api.openai.com/v1

# Logging

LOG_LEVEL=INFO

```text

3. **Add to `.gitignore`**:

```gitignore

# Environment variables

.env
.env.local
.env.*.local

```text
```python

# cli.py or service.py

from podcast_scraper import config  # This loads .env automatically

# ... rest of code

```text
```bash

# Development

.env.development

# Staging

.env.staging

# Production

.env.production

```python

from dotenv import load_dotenv
from pathlib import Path

env_file = Path(".env.production")  # or from ENV environment variable
load_dotenv(env_file, override=False)

```text

- ✅ **Load at startup** - `.env` loaded automatically when config module imports
- ✅ **Don't override existing vars** - `override=False` respects system environment variables
- ✅ **Never log API keys** - Sanitize logs, never print full keys
- ✅ **Validate key format** - Check that key starts with `sk-` if provided
- ✅ **Separate keys per environment** - Use different keys for dev/staging/prod
- ✅ **Rotate keys periodically** - Update keys regularly for security

**Dependencies:**

Add to `pyproject.toml` dependencies:

```toml

"python-dotenv>=1.0.0,<2.0.0",  # For .env file support

```text

1. Copy `examples/.env.example` to `.env`:

   ```bash

   cp examples/.env.example .env

`   ```

1. Edit `.env` and add your OpenAI API key:

   ```bash

   OPENAI_API_KEY=sk-your-actual-key-here

   ```

2. Verify setup:

   ```bash

   python -c "from podcast_scraper import config; print('Config loaded successfully')"

   ```

**Note:** The `.env` file is automatically loaded when the `podcast_scraper` package is imported.

**Fallback Priority:**

API key resolution order (highest to lowest priority):

1. **Config file** (`openai_api_key` field in YAML/JSON config)
2. **System environment variable** (`OPENAI_API_KEY` from shell/system)
3. **`.env` file** (`OPENAI_API_KEY` from `.env` file)
4. **None** (raises error if OpenAI provider is selected)

## 3. OpenAI Provider Implementations

### 3.1 Speaker Detection Provider

**File**: `podcast_scraper/speaker_detectors/openai_detector.py`

````python

from typing import List, Set, Optional, Dict, Any, Tuple
from openai import OpenAI
from .. import config, models
from .base import SpeakerDetector

class OpenAISpeakerDetector:
    """OpenAI API-based speaker detection provider."""

    def __init__(self, cfg: config.Config):
        if not cfg.openai_api_key:
            raise ValueError("OpenAI API key required for OpenAI speaker detector")
        self.client = OpenAI(api_key=cfg.openai_api_key)
        self.cfg = cfg
        self.model = getattr(cfg, 'openai_speaker_model', 'gpt-4o-mini')  # Configurable

    def detect_hosts(
        self,
        feed_title: str,
        feed_description: Optional[str],
        feed_authors: Optional[List[str]],
    ) -> Set[str]:

```python

        """Detect hosts from feed metadata using OpenAI API."""
        prompt = self._build_host_detection_prompt(feed_title, feed_description, feed_authors)

```python

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at identifying podcast hosts from metadata."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200,
            )

```python

    def detect_speakers(
        self,
        episode_title: str,
        episode_description: Optional[str],
        known_hosts: Set[str],
    ) -> Tuple[List[str], Set[str], bool]:
        """Detect speakers for an episode using OpenAI API."""
        prompt = self._build_speaker_detection_prompt(
            episode_title, episode_description, known_hosts
        )

```json

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at identifying speakers and guests in podcast episodes."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300,
            )

```
```python

    def analyze_patterns(
        self,
        episodes: List[models.Episode],
        known_hosts: Set[str],
    ) -> Optional[Dict[str, Any]]:
        """Analyze episode patterns using OpenAI API (optional, can use local logic)."""

```python

    def _build_host_detection_prompt(
        self,
        feed_title: str,
        feed_description: Optional[str],
        feed_authors: Optional[List[str]],
    ) -> str:
        """Build prompt for host detection using prompt_store (RFC-017).

```python

        Prompts are loaded from versioned files via prompt_store, enabling
        prompt engineering without code changes.
        """
        from ..prompt_store import render_prompt

```text

        return render_prompt(
            self.cfg.ner_user_prompt or "ner/guest_host_v1",
            feed_title=feed_title,
            feed_description=feed_description or "",
            feed_authors=", ".join(feed_authors) if feed_authors else "",
            **self.cfg.ner_prompt_params,
        )

```python

    def _parse_hosts_from_response(self, response_text: str) -> Set[str]:
        """Parse host names from API response."""

```python

- Structured prompts for consistent results
- Parse JSON or structured text from API responses
- Handle API errors gracefully with retries
- Maintain same return types as NER provider

**Prompt Implementation:**

All prompts are loaded via `prompt_store` (see RFC-017) for versioning and parameterization:

```python

from ..prompt_store import render_prompt

# In detect_speakers():

user_prompt = render_prompt(
    self.cfg.ner_user_prompt or "ner/guest_host_v1",
    episode_title=episode_title,
    episode_description=episode_description or "",
    known_hosts=", ".join(known_hosts) if known_hosts else "",
    **self.cfg.ner_prompt_params,
)

system_prompt = render_prompt(
    self.cfg.ner_system_prompt or "ner/system_ner_v1",
    **self.cfg.ner_prompt_params,
) if self.cfg.ner_system_prompt else None

```python

from typing import Dict, Optional, Tuple, Any
from openai import OpenAI
from .. import config
from .base import TranscriptionProvider

class OpenAITranscriptionProvider:
    """OpenAI Whisper API-based transcription provider."""

    def __init__(self, cfg: config.Config):
        if not cfg.openai_api_key:
            raise ValueError("OpenAI API key required for OpenAI transcription provider")
        self.client = OpenAI(api_key=cfg.openai_api_key)
        self.cfg = cfg
        self.model = getattr(cfg, 'openai_whisper_model', 'whisper-1')  # Default Whisper API model

    def initialize(self, cfg: config.Config) -> Optional[Any]:
        """Initialize provider (no local model loading needed for API)."""
        return self  # Return self as resource

```python

    def transcribe(
        self,
        media_path: str,
        cfg: config.Config,
        resource: Any,
    ) -> Tuple[Dict[str, Any], float]:
        """Transcribe media file using OpenAI Whisper API."""
        import time

```
                    response_format="verbose_json",  # Get segments
                )

```json

            result = {
                'text': transcript.text,
                'segments': [
                    {
                        'id': i,
                        'start': seg.start,
                        'end': seg.end,
                        'text': seg.text,
                    }
                    for i, seg in enumerate(transcript.segments)
                ] if hasattr(transcript, 'segments') else [],
                'language': transcript.language if hasattr(transcript, 'language') else None,
            }

```text

            return result, elapsed

```python

    def cleanup(self, resource: Any) -> None:
        """Cleanup provider resources (no-op for API provider)."""
        pass

```
## 3.3 Summarization Provider

**File**: `podcast_scraper/summarization/openai_provider.py`

**Key Advantage**: OpenAI GPT models (GPT-4, GPT-4o-mini) have much larger context windows (128k+ tokens) compared to local transformer models (1k-16k tokens). This means we can process full transcripts directly without chunking, simplifying the implementation significantly.

```python

from typing import List, Optional, Dict, Any
from openai import OpenAI
from .. import config
from .base import SummarizationProvider

class OpenAISummarizationProvider:
    """OpenAI GPT API-based summarization provider."""

    def __init__(self, cfg: config.Config):
        if not cfg.openai_api_key:
            raise ValueError("OpenAI API key required for OpenAI summarization provider")
        self.client = OpenAI(api_key=cfg.openai_api_key)
        self.cfg = cfg
        self.model = getattr(cfg, 'openai_summary_model', 'gpt-4o-mini')  # Cost-effective default

        # GPT-4o-mini supports 128k context window - can handle full transcripts

        self.max_context_tokens = 128000  # Conservative estimate

```python

    def initialize(self, cfg: config.Config) -> Optional[Any]:
        """Initialize provider (no local model loading needed for API)."""
        return self  # Return self as resource

```python

    def summarize(
        self,
        text: str,
        cfg: config.Config,
        resource: Any,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Summarize text using OpenAI GPT API.

```json

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at creating concise, informative summaries with key takeaways."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=max_length,
            )

```python

    def summarize_chunks(
        self,
        chunks: List[str],
        cfg: config.Config,
        resource: Any,
    ) -> List[str]:
        """Summarize multiple text chunks (MAP phase).

```python

    def combine_summaries(
        self,
        summaries: List[str],
        cfg: config.Config,
        resource: Any,
    ) -> str:
        """Combine multiple summaries into final summary (REDUCE phase).

```json

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at combining multiple summaries into a coherent final summary."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=getattr(cfg, 'summary_max_length', 150),
            )

```python

    def cleanup(self, resource: Any) -> None:
        """Cleanup provider resources (no-op for API provider)."""
        pass

```

- **Use GPT-4o-mini**: Cost-effective default with large context window
- **Prompt Management**: Use `prompt_store` (RFC-017) for versioned, parameterized prompts

**Prompt Implementation:**

All prompts are loaded via `prompt_store` (see RFC-017) for versioning and parameterization:

```python

from ..prompt_store import render_prompt

# In summarize():

system_prompt = render_prompt(
    self.cfg.summary_system_prompt or "summarization/system_v1",
    **self.cfg.summary_prompt_params,
) if self.cfg.summary_system_prompt else None

user_prompt = render_prompt(
    self.cfg.summary_user_prompt or "summarization/long_v1",
    transcript=text,
    title=episode_title or "",
    paragraphs_min=(min_length or cfg.summary_min_length) // 100,
    paragraphs_max=(max_length or cfg.summary_max_length) // 100,
    **self.cfg.summary_prompt_params,
)

```python

# podcast_scraper/speaker_detectors/factory.py

@staticmethod
def create(cfg: config.Config) -> Optional[SpeakerDetector]:
    if not cfg.auto_speakers:
        return None

    detector_type = cfg.speaker_detector_type
    if detector_type == 'ner':
        from .ner_detector import NERSpeakerDetector
        return NERSpeakerDetector(cfg)
    elif detector_type == 'openai':
        from .openai_detector import OpenAISpeakerDetector
        return OpenAISpeakerDetector(cfg)
    return None

# Similar updates for transcription and summarization factories

```yaml

- **Provider-Specific**: Each provider loads prompts internally (not part of protocol)
- **Config-Driven**: Prompt selection via config fields, not code

**Example Integration:**

```python

# In OpenAISummarizationProvider.__init__():

from ..prompt_store import render_prompt, get_prompt_metadata

# Prompts are loaded on-demand when needed, cached automatically

# No initialization required - prompt_store handles caching

# In summarize() method:

system_prompt = None
if cfg.summary_system_prompt:
    system_prompt = render_prompt(
        cfg.summary_system_prompt,
        **cfg.summary_prompt_params,
    )

user_prompt = render_prompt(
    cfg.summary_user_prompt or "summarization/long_v1",
    transcript=text,
    title=episode_title or "",
    paragraphs_min=min_length // 100,
    paragraphs_max=max_length // 100,
    **cfg.summary_prompt_params,
)

```yaml

- ✅ **Reproducibility**: SHA256 hashes track exact prompt versions
- ✅ **Provider Autonomy**: Each provider handles prompts internally
- ✅ **Protocol Compliance**: Prompts don't affect protocol interfaces

See RFC-017 for complete prompt management design.

## 5. Parallelism Considerations

### 5.1 Current Parallelism

**Local Providers:**

- Transcription: Sequential (one model instance)
- Summarization: Parallel chunk processing (multiple worker threads with model instances)

**OpenAI Providers:**

- Transcription: Can parallelize API calls (no shared state)
- Summarization: Can parallelize chunk API calls (no shared state)
- Speaker Detection: Can parallelize episode processing (no shared state)

#### 5.2 Implementation Strategy

**For Transcription:**

- Current: Sequential processing
- With OpenAI: Can use ThreadPoolExecutor for parallel API calls
- Rate Limiting: Implement semaphore or rate limiter to respect API limits

**For Summarization:**

- Current: Parallel chunk processing with worker threads
- With OpenAI: Same pattern, but API calls instead of local model inference
- Rate Limiting: Use rate limiter to respect API rate limits

**Rate Limiting Implementation:**

```python

from threading import Semaphore
import time

class RateLimiter:
    """Rate limiter for OpenAI API calls."""

    def __init__(self, max_calls_per_minute: int = 60):
        self.semaphore = Semaphore(max_calls_per_minute)
        self.call_times = []
        self.lock = threading.Lock()

    def acquire(self):
        """Acquire permission to make API call."""
        with self.lock:
            now = time.time()

            # Remove calls older than 1 minute

            self.call_times = [t for t in self.call_times if now - t < 60]

```text

            if len(self.call_times) >= self.max_calls_per_minute:

```

                    self.call_times = [t for t in self.call_times if now - t < 60]

```python

    def release(self):
        """Release after API call completes."""
        self.semaphore.release()

```python

# Global rate limiter (shared across provider instances)

_openai_rate_limiter = RateLimiter(max_calls_per_minute=60)

class OpenAISummarizationProvider:
    def summarize_chunks(self, chunks: List[str], cfg: config.Config, resource: Any) -> List[str]:
        summaries = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for chunk in chunks:
                _openai_rate_limiter.acquire()
                future = executor.submit(self._summarize_chunk, chunk, cfg, resource)
                futures.append(future)

            for future in futures:
                try:
                    summary = future.result()
                    summaries.append(summary)
                finally:

```text

                    _openai_rate_limiter.release()
        return summaries

```python

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import RateLimitError, APIError

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((RateLimitError, APIError))
)
def _call_openai_api(self, ...):
    """Make OpenAI API call with retry logic."""

    # API call implementation

    pass

```text

- Clear error messages indicating which provider failed
- Actionable error messages (e.g., "Check OPENAI_API_KEY environment variable")
- Log API errors with context (but not API keys)

## 7. Dependencies

**Add to `pyproject.toml`:**

```toml

[project.optional-dependencies]

# ... existing dependencies ...

openai = [
    "openai>=1.0.0,<2.0.0",
    "tenacity>=8.0.0,<9.0.0",  # For retry logic
]

````

```bash

# For OpenAI support

pip install -e ".[openai]"

# Or with all ML dependencies

```
## 8.2 Integration Tests

- Optional tests with real API (requires API key)
- Test end-to-end workflow with OpenAI providers
- Test parallelism with API providers
- Test error scenarios (invalid key, rate limits)

### 8.3 Backward Compatibility Tests

- Verify default behavior unchanged (local providers)
- Verify existing tests still pass
- Verify no breaking changes to workflow

### 9. Documentation

#### 9.1 User Documentation

- How to set up OpenAI API key
- How to configure providers
- Cost considerations
- Performance characteristics
- Example configurations

#### 9.2 Developer Documentation

- Provider interface documentation
- How to add new providers
- Testing strategies
- Error handling patterns

## Migration Path

1. **Complete Modularization Refactoring** (prerequisite)
   - Implement provider abstractions
   - Refactor local providers
   - Update workflow to use factories

2. **Add Configuration Fields**
   - Add provider type fields to config
   - Add API key management
   - Add validation logic

3. **Implement OpenAI Providers**
   - Speaker detection provider
   - Transcription provider
   - Summarization provider

4. **Update Factories**
   - Add OpenAI provider support to factories
   - Test provider selection

5. **Add Parallelism Support**
   - Implement rate limiting
   - Update parallelism for API providers
   - Test concurrent API calls

6. **Testing & Documentation**
   - Write tests
   - Update documentation
   - Add examples

## Extensibility & Public API Design

### Extension Points (Public API)

The provider system is designed to be extensible by external contributors. The following are **public APIs** that contributors can use:

#### 1. Protocol Interfaces (Public API)

**Location**: `podcast_scraper/speaker_detectors/base.py`, `podcast_scraper/transcription/base.py`, `podcast_scraper/summarization/base.py`

```python

# Public API - Protocol definitions

from typing import Protocol

class SpeakerDetector(Protocol):
    """Public protocol for speaker detection providers."""
    def detect_hosts(...) -> Set[str]: ...
    def detect_speakers(...) -> Tuple[List[str], Set[str], bool]: ...
    def analyze_patterns(...) -> Optional[Dict[str, Any]]: ...

class TranscriptionProvider(Protocol):
    """Public protocol for transcription providers."""
    def initialize(...) -> Optional[Any]: ...
    def transcribe(...) -> Tuple[Dict[str, Any], float]: ...
    def cleanup(...) -> None: ...

class SummarizationProvider(Protocol):
    """Public protocol for summarization providers."""
    def initialize(...) -> Optional[Any]: ...
    def summarize(...) -> Dict[str, Any]: ...
    def summarize_chunks(...) -> List[str]: ...
    def combine_summaries(...) -> str: ...
    def cleanup(...) -> None: ...

```python

# External contributor can implement protocol

from podcast_scraper.speaker_detectors.base import SpeakerDetector

class CustomSpeakerDetector:
    """Custom implementation by contributor."""
    def detect_hosts(self, ...) -> Set[str]:

        # Custom implementation

        pass

    def detect_speakers(self, ...) -> Tuple[List[str], Set[str], bool]:

        # Custom implementation

        pass

```python

    def analyze_patterns(self, ...) -> Optional[Dict[str, Any]]:

```
# Type checker will verify protocol compliance

detector: SpeakerDetector = CustomSpeakerDetector()  # ✅ Type-safe

```text
```python

# Public API - Factory extension points

class SpeakerDetectorFactory:
    """Factory for creating speaker detectors."""

    @staticmethod
    def create(cfg: config.Config) -> Optional[SpeakerDetector]:
        """Create detector based on config.

        Contributors can extend this to support custom providers.
        """
        if not cfg.auto_speakers:
            return None

        detector_type = cfg.speaker_detector_type
        if detector_type == 'ner':
            from .ner_detector import NERSpeakerDetector
            return NERSpeakerDetector(cfg)
        elif detector_type == 'openai':

```python

            from .openai_detector import OpenAISpeakerDetector
            return OpenAISpeakerDetector(cfg)

```python

        elif detector_type == 'custom':
            from external_package import CustomSpeakerDetector
            return CustomSpeakerDetector(cfg)
        return None

```python

# Public API - Config fields for provider selection

class Config(BaseModel):
    """Configuration model - public API for provider selection."""

    # Public fields for provider selection

    speaker_detector_provider: Literal["ner", "openai", "custom"] = Field(default="ner")
    transcription_provider: Literal["whisper", "openai", "custom"] = Field(default="whisper")
    summary_provider: Literal["transformers", "openai", "custom"] = Field(default="transformers")

    # Contributors can extend with custom config fields

    custom_provider_config: Optional[Dict[str, Any]] = Field(default=None)

```text

- Located in `podcast_scraper/speaker_detectors/ner_detector.py` (internal)
- Located in `podcast_scraper/transcription/whisper_provider.py` (internal)
- Located in `podcast_scraper/summarization/local_provider.py` (internal)
- Located in `podcast_scraper/speaker_detectors/openai_detector.py` (internal)
- Located in `podcast_scraper/transcription/openai_provider.py` (internal)
- Located in `podcast_scraper/summarization/openai_provider.py` (internal)

These serve as:

- **Reference implementations** showing how to implement protocols
- **Default providers** for users who don't need custom implementations
- **Examples** for contributors

## Contributor Implementations

We expect and encourage contributors to create their own provider implementations.

**Example: Custom Transcription Provider**

```python

# external_package/deepgram_provider.py

from typing import Dict, Optional, Tuple, Any
from podcast_scraper.transcription.base import TranscriptionProvider
from podcast_scraper import config

class DeepgramTranscriptionProvider:
    """Custom Deepgram transcription provider by contributor."""

    def __init__(self, cfg: config.Config):
        import deepgram
        self.client = deepgram.DeepgramClient(cfg.deepgram_api_key)
        self.cfg = cfg

    def initialize(self, cfg: config.Config) -> Optional[Any]:
        """Initialize Deepgram client."""
        return self

```python

    def transcribe(
        self,
        media_path: str,
        cfg: config.Config,
        resource: Any,
    ) -> Tuple[Dict[str, Any], float]:
        """Transcribe using Deepgram API."""
        import time
        start_time = time.time()

        with open(media_path, 'rb') as audio_file:
            response = self.client.transcription.sync_prerecorded(
                {"buffer": audio_file},
                {"punctuate": True, "model": "nova"}
            )

        # Return same format as protocol requires
        result = {
            "text": response["results"]["channels"][0]["alternatives"][0]["transcript"],
            "segments": [],  # Convert Deepgram format to standard format
        }

```text

        elapsed = time.time() - start_time
        return result, elapsed

```python

    def cleanup(self, resource: Any) -> None:
        """Cleanup resources."""
        pass

```python

# In factory or plugin system

from external_package.deepgram_provider import DeepgramTranscriptionProvider

# Add to factory

if cfg.transcription_provider == 'deepgram':
    return DeepgramTranscriptionProvider(cfg)

```text
```python

# tests/test_provider_protocols.py

def test_speaker_detector_protocol():
    """Test that any SpeakerDetector implementation follows protocol."""
    from podcast_scraper.speaker_detectors.base import SpeakerDetector

    # Mock implementation

    class MockDetector:
        def detect_hosts(self, ...) -> Set[str]:
            return {"Host"}
        def detect_speakers(self, ...) -> Tuple[List[str], Set[str], bool]:
            return (["Host", "Guest"], {"Host"}, True)
        def analyze_patterns(self, ...) -> Optional[Dict[str, Any]]:
            return None

    # Type checker verifies protocol compliance

```text

    detector: SpeakerDetector = MockDetector()  # Must pass type check

```python

# tests/test_factories.py

def test_factory_provider_selection():
    """Test factory correctly selects providers."""
    cfg = Config(speaker_detector_provider="ner")
    detector = SpeakerDetectorFactory.create(cfg)
    assert isinstance(detector, NERSpeakerDetector)

    cfg = Config(speaker_detector_provider="openai", openai_api_key="test")
    detector = SpeakerDetectorFactory.create(cfg)
    assert isinstance(detector, OpenAISpeakerDetector)

```python

# tests/test_workflow_with_providers.py

def test_workflow_with_mock_provider():
    """Test workflow works with any provider implementation."""
    mock_detector = MockSpeakerDetector()

    # Test that workflow uses detector correctly

    # Verify no provider-specific code in workflow

```python

# tests/speaker_detectors/test_ner_detector.py

class TestNERSpeakerDetector:
    """Tests for NER speaker detector implementation."""

    def test_detect_hosts(self):
        """Test host detection."""
        detector = NERSpeakerDetector(cfg)
        hosts = detector.detect_hosts("Test Podcast", None, ["John Doe"])
        assert isinstance(hosts, set)
        assert len(hosts) > 0

    def test_detect_speakers(self):
        """Test speaker detection."""
        detector = NERSpeakerDetector(cfg)
        speakers, detected_hosts, success = detector.detect_speakers(
            "Episode with Guest", None, {"Host"}
        )
        assert isinstance(speakers, list)
        assert isinstance(detected_hosts, set)
        assert isinstance(success, bool)

```python

    def test_protocol_compliance(self):
        """Verify protocol interface compliance."""
        detector = NERSpeakerDetector(cfg)

```

- ✅ External implementations should follow same testing standards
- ✅ Mock providers for testing workflow without real providers
- ✅ Integration tests with real providers (optional, requires API keys)

## Documentation & Examples

**New Extensibility Documentation** (`docs/EXTENSIBILITY.md`):

### Architecture Overview

- How provider system works (protocol-based design)
- Factory pattern usage
- Provider lifecycle (initialization, usage, cleanup)
- Configuration-driven provider selection

#### Creating Custom Providers

**Step-by-step guides:**

1. **Creating a Custom Speaker Detector**:
   - Implement `SpeakerDetector` protocol
   - Register in factory
   - Add config field
   - Write tests
   - Document usage

2. **Creating a Custom Transcription Provider**:
   - Implement `TranscriptionProvider` protocol
   - Handle file uploads
   - Return standard format
   - Register in factory
   - Write tests

3. **Creating a Custom Summarization Provider**:
   - Implement `SummarizationProvider` protocol
   - Handle long texts (if needed)
   - Return standard format
   - Register in factory
   - Write tests

#### Example: Minimal Provider

```python

# Minimal speaker detector implementation

from typing import Set, List, Tuple, Optional, Dict, Any
from podcast_scraper.speaker_detectors.base import SpeakerDetector
from podcast_scraper import config

class MinimalSpeakerDetector:
    """Minimal example of speaker detector implementation."""

    def __init__(self, cfg: config.Config):
        self.cfg = cfg

    def detect_hosts(
        self,
        feed_title: str,
        feed_description: Optional[str],
        feed_authors: Optional[List[str]],
    ) -> Set[str]:

```text

        """Detect hosts - minimal implementation."""
        if feed_authors:
            return set(feed_authors)
        return set()

```python

    def detect_speakers(
        self,
        episode_title: str,
        episode_description: Optional[str],
        known_hosts: Set[str],
    ) -> Tuple[List[str], Set[str], bool]:
        """Detect speakers - minimal implementation."""
        speakers = list(known_hosts)
        return speakers, known_hosts, True

```python

    def analyze_patterns(
        self,
        episodes: List[models.Episode],
        known_hosts: Set[str],
    ) -> Optional[Dict[str, Any]]:
        """Analyze patterns - optional."""
        return None

```python

# Full-featured provider with error handling, logging, etc.

class FullFeaturedSpeakerDetector:
    """Full-featured example with error handling."""

    def __init__(self, cfg: config.Config):
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)

        # Initialize resources

    def detect_hosts(self, ...) -> Set[str]:
        """Detect hosts with error handling."""
        try:

            # Implementation

            pass
        except Exception as e:

```text

            self.logger.error(f"Error detecting hosts: {e}")
            raise

```python

# tests/test_custom_provider.py

def test_custom_provider_protocol():
    """Test custom provider follows protocol."""
    from podcast_scraper.speaker_detectors.base import SpeakerDetector

    custom_detector = CustomSpeakerDetector(cfg)

    # Type check

    detector: SpeakerDetector = custom_detector

    # Runtime checks

    assert hasattr(custom_detector, 'detect_hosts')
    assert hasattr(custom_detector, 'detect_speakers')
    assert hasattr(custom_detector, 'analyze_patterns')

```text

    # Functional tests

```go

1. **Code Organization**:
   - Follow existing provider structure
   - Place in appropriate package (`speaker_detectors/`, `transcription/`, `summarization/`)
   - Or create external package

2. **Naming Conventions**:
   - Provider classes: `{Service}Provider` or `{Service}Detector`
   - Files: `{service}_provider.py` or `{service}_detector.py`

3. **Documentation**:
   - Docstrings for all public methods
   - Usage examples
   - Configuration requirements
   - Error handling documentation

4. **Testing**:
   - Unit tests for all methods
   - Protocol compliance tests
   - Error scenario tests
   - Integration tests (if applicable)

5. **Pull Request Process**:
   - Add provider to factory
   - Add config field (if needed)
   - Add tests
   - Update documentation
   - Add examples

## Open Questions

1. **Rate Limiting**: ~~What are OpenAI API rate limits? Should we make rate limiter configurable?~~ ✅ **RESOLVED** - See Appendix B
2. **Cost Tracking**: Should we add cost tracking/monitoring?
3. **Fallback**: Should we support fallback (try OpenAI, fallback to local)?
4. **Model Selection**: ~~Should model selection be more granular (different models for different tasks)?~~ ✅ **RESOLVED** - See PRD-006 and Appendix A
5. **Plugin System**: Should we support external packages registering providers via entry points?

## Alternatives Considered

1. **Single Provider Selection**: One config for all capabilities
   - **Rejected**: Users may want to mix providers (e.g., local transcription, OpenAI summarization)

2. **API Key in Config File**: Store API key in YAML config
   - **Rejected**: Security risk, prefer environment variables

3. **Synchronous Only**: No parallelism for API calls
   - **Rejected**: Would slow down batch processing significantly

## Success Criteria

- ✅ OpenAI providers implement same interfaces as local providers
- ✅ Users can select OpenAI providers via configuration
- ✅ API keys managed securely via environment variables
- ✅ Parallelism works correctly with API providers
- ✅ Error handling is clear and actionable
- ✅ Default behavior (local providers) unchanged
- ✅ Documentation complete and clear

---

## Future Documentation (Before Stage 6)

The following documentation should be created during Stages 1-5 to support Stage 6 (OpenAI implementation):

### 1. Testing Strategy Document (`docs/wip/TESTING_STRATEGY_MODULARIZATION.md`)

**Purpose:** Provide concrete examples and patterns for testing providers

**Contents:**

- Protocol compliance test examples
- Mock provider patterns
- Integration test requirements
- Performance benchmark baselines
- Test data fixtures
- Example test cases for each provider type

**Example:**

```python

def test_summarization_provider_protocol_compliance():
    """Verify provider implements SummarizationProvider protocol."""
    provider = LocalSummarizationProvider(cfg)

    # Protocol interface check

    assert hasattr(provider, 'initialize')
    assert hasattr(provider, 'summarize')
    assert hasattr(provider, 'cleanup')

    # Signature validation

    import inspect
    sig = inspect.signature(provider.summarize)
    assert 'text' in sig.parameters
    assert 'cfg' in sig.parameters

    # Return type validation

```text

    result = provider.summarize("test text", cfg)
    assert isinstance(result, dict)
    assert 'summary' in result

```

### 2. Provider Implementation Guide (`docs/guides/PROVIDER_IMPLEMENTATION_GUIDE.md`)

**Purpose:** Enable external contributors to create custom providers

**Contents:**

- Step-by-step provider creation guide
- Protocol interface documentation
- Factory registration pattern
- Testing requirements
- Documentation requirements
- Pull request process
- Three example implementations:
  1. **Minimal Example** (Hello World provider)
  2. **Full-Featured Example** (with error handling, retries, logging)
  3. **Custom Config Example** (provider with custom configuration)

**Example Structure:**

```markdown

# Custom Provider Guide

## Quick Start

1. Create provider class implementing protocol
2. Add to factory
3. Add tests
4. Submit PR

## Minimal Example: Custom Summarization Provider

\`\`\`python
class MyCustomSummarizationProvider:
    """Custom summarization provider example."""

    def __init__(self, cfg: config.Config):
        self.cfg = cfg

```python

    def initialize(self) -> None:
        """Initialize provider resources."""
        pass

```python

    def summarize(self, text: str, cfg: config.Config) -> Dict[str, Any]:
        """Summarize text using custom logic."""

```python

    def cleanup(self) -> None:
        """Clean up provider resources."""
        pass

```python

## Registering Your Provider

\`\`\`python

# In podcast_scraper/summarization/factory.py

def create(cfg: config.Config):

```python

    if cfg.summary_provider == "my-custom":
        from .my_custom_provider import MyCustomSummarizationProvider
        return MyCustomSummarizationProvider(cfg)

```
## 3. Environment Variable Documentation (`docs/api/CONFIGURATION.md`)

**Purpose:** Comprehensive reference for all environment variables

**Contents:**

- Complete list of supported environment variables
- Usage examples
- Security best practices
- Troubleshooting guide
- Platform-specific instructions (macOS, Linux, Windows)
- Docker/container environment setup

**Example Structure:**

```markdown

# Environment Variables

## OpenAI Configuration

### OPENAI_API_KEY (Required for OpenAI providers)

**Description:** OpenAI API authentication key

**Format:** `sk-...` (starts with `sk-`)

**Usage:**
\`\`\`bash

# macOS/Linux

export OPENAI_API_KEY="sk-..."

# Windows (PowerShell)

$env:OPENAI_API_KEY="sk-..."

# Docker

docker run -e OPENAI_API_KEY="sk-..." ...

# Docker Compose

environment:

  - OPENAI_API_KEY=${OPENAI_API_KEY}
\`\`\`

**Security:**
- Never commit API keys to source control
- Use `.env` files (add to `.gitignore`)
- Rotate keys periodically
- Use separate keys for dev/prod

**Troubleshooting:**
- Check key starts with `sk-`
- Verify key hasn't been revoked: https://platform.openai.com/api-keys
- Test with: `python -c "import os; print(os.getenv('OPENAI_API_KEY')[:10])"`

## OPENAI_ORGANIZATION (Optional)

**Description:** OpenAI organization ID (for users in multiple orgs)

**Format:** `org-...`

**Usage:**
\`\`\`bash
export OPENAI_ORGANIZATION="org-..."
\`\`\`

### OPENAI_API_BASE (Optional)

**Description:** Custom API base URL (for proxies or alternative endpoints)

**Default:** `https://api.openai.com/v1`

**Usage:**
\`\`\`bash
export OPENAI_API_BASE="https://your-proxy.example.com/v1"
\`\`\`

## Other Environment Variables

### LOG_LEVEL (Optional)

**Description:** Set logging verbosity

**Values:** `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`

**Default:** `INFO`

**Usage:**
\`\`\`bash
export LOG_LEVEL="DEBUG"
\`\`\`

## Complete Example (.env file)

\`\`\`bash

# .env - Add to .gitignore!

# This file is automatically loaded by python-dotenv when config.py is imported

# OpenAI API Configuration

OPENAI_API_KEY=sk-your-actual-api-key-here
OPENAI_ORGANIZATION=org-your-org-id-here  # Optional

# Logging

LOG_LEVEL=INFO
\`\`\`

**Note:** The `.env` file is automatically loaded via `python-dotenv` when the `podcast_scraper.config` module is imported. See section 2.2 for implementation details.

**Timeline:** Create before Stage 6 (OpenAI implementation)

---

## Appendix A: Provider Naming Convention

**Decision Date:** December 10, 2025
**Status:** Approved

### Current State (Inconsistent)

```python

# Naming in planning documents before standardization:

speaker_detector_type: Literal["ner", "openai"]          # ✅ Technology-based
transcription_provider: Literal["whisper", "openai"]     # ✅ Technology-based
summary_provider: Literal["local", "openai"]             # ❌ Location-based (ambiguous)

```text

**Standardized Naming:**

```python

speaker_detector_type: Literal["ner", "openai"] = "ner"
transcription_provider: Literal["whisper", "openai"] = "whisper"
summary_provider: Literal["transformers", "openai"] = "transformers"  # CHANGED

```text

2. **Consistency:** All values now refer to technology (ner, whisper, transformers, openai)
3. **Extensibility:** Easy to add more technologies (e.g., "aws-comprehend")
4. **User Understanding:** Users immediately know what technology they're selecting

**Future Extensibility:**

```python

# Examples of future additions:

speaker_detector_type: Literal["ner", "openai", "aws-comprehend", "google-nlp"]
transcription_provider: Literal["whisper", "openai", "deepgram", "assemblyai"]
summary_provider: Literal["transformers", "openai", "cohere"]

```text
```python

@field_validator('summary_provider', mode='before')
def migrate_local_to_transformers(cls, v):
    if v == "local":
        warnings.warn(
            "summary_provider='local' is deprecated, use 'transformers'. "
            "Support for 'local' will be removed in v3.0.0",
            DeprecationWarning
        )
        return "transformers"
    return v

```text

## OpenAI Rate Limits (as of December 2025)

OpenAI uses a tiered rate limiting system based on usage:

| Tier | Requirement | RPM (gpt-4o-mini) | TPM (Tokens) | RPD (Requests) |
| ---- | ----------- | ----------------- | ------------ | --------------- |
| **Free** | New account | 500 | 200,000 | 10,000 |
| **Tier 1** | $5+ spent | 500 | 2,000,000 | 10,000 |
| **Tier 2** | $50+ spent + 7 days | 5,000 | 10,000,000 | 100,000 |
| **Tier 3** | $100+ spent + 7 days | 5,000 | 10,000,000 | 200,000 |
| **Tier 4** | $250+ spent + 14 days | 10,000 | 30,000,000 | 300,000 |
| **Tier 5** | $1,000+ spent + 30 days | 10,000 | 80,000,000 | 500,000 |

**Legend:** RPM = Requests Per Minute, TPM = Tokens Per Minute, RPD = Requests Per Day

**Note:** Check [OpenAI Rate Limits](https://platform.openai.com/docs/guides/rate-limits) for current limits.

### Configuration Fields

Add to `config.py` when implementing OpenAI providers:

```python

# OpenAI Rate Limiting

openai_max_concurrent_requests: int = Field(
    default=5,
    ge=1,
    le=100,
    description="Maximum concurrent OpenAI API requests (controls parallelism)"
)

openai_requests_per_minute: int = Field(
    default=50,
    ge=1,
    le=10000,
    description="Rate limit: requests per minute (adjust based on OpenAI tier)"
)

openai_tokens_per_minute: int = Field(
    default=100000,
    ge=1000,
    le=80000000,
    description="Rate limit: tokens per minute (adjust based on OpenAI tier)"
)

openai_retry_max_attempts: int = Field(
    default=3,
    ge=1,
    le=10,
    description="Maximum retry attempts for failed OpenAI API requests"
)

openai_retry_backoff_factor: float = Field(
    default=2.0,
    ge=1.0,
    le=10.0,
    description="Exponential backoff factor for retries (delay = backoff^attempt)"
)

openai_retry_max_delay: int = Field(
    default=60,
    ge=1,
    le=300,
    description="Maximum retry delay in seconds (caps exponential backoff)"
)

openai_timeout: int = Field(
    default=60,
    ge=5,
    le=600,
    description="Request timeout in seconds for OpenAI API calls"
)

```text

openai_max_concurrent_requests: 5
openai_requests_per_minute: 50
openai_tokens_per_minute: 100000

```text

openai_max_concurrent_requests: 10
openai_requests_per_minute: 500
openai_tokens_per_minute: 500000

```text

openai_max_concurrent_requests: 20
openai_requests_per_minute: 1000
openai_tokens_per_minute: 1000000

```text

   - Semaphore for concurrency control
   - Sliding window for rate tracking

2. **Retry Logic** (using `tenacity` library):
   - Automatic retry for rate limit, timeout, and connection errors
   - Exponential backoff (2x multiplier, capped at 60s)
   - 3 attempts by default

3. **Error Handling**:
   - **Retry-able:** RateLimitError, APITimeoutError, APIConnectionError, InternalServerError
   - **Non-retry-able:** AuthenticationError, PermissionDeniedError, BadRequestError, NotFoundError

## Error Messages

**Rate Limit Error (after retries):**

```text

OpenAI rate limit exceeded. Suggestions:

1. Reduce parallelism: --openai-max-concurrent-requests 3
2. Reduce rate: --openai-requests-per-minute 30
3. Wait and retry: The limit resets every minute
4. Check tier limits: https://platform.openai.com/account/limits

```text
```text

OpenAI API authentication failed.
Please check:

1. OPENAI_API_KEY environment variable is set
2. API key is valid (starts with 'sk-')
3. API key has not been revoked
Get your API key: https://platform.openai.com/api-keys

```text

# Add to pyproject.toml dependencies

"tenacity>=8.2.0,<9.0.0",  # For retry logic with exponential backoff

```text
```text

- Mock rate limiter for unit tests (no delays)
- Unit tests for token bucket logic
- Integration tests for API error handling
- Protocol compliance tests

## References

- [OpenAI Rate Limits Documentation](https://platform.openai.com/docs/guides/rate-limits)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Tenacity Library](https://tenacity.readthedocs.io/)

````
