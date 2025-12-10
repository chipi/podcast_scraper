# RFC-013: OpenAI Provider Implementation

- **Status**: Draft
- **Authors**:
- **Stakeholders**: Maintainers, users wanting OpenAI API integration, developers implementing providers
- **Related PRDs**: `docs/prd/PRD-006-openai-provider-integration.md`
- **Related Documents**: `docs/wip/MODULARIZATION_REFACTORING_PLAN.md`
- **Related Issues**: (to be created)

## Abstract

Design and implement OpenAI API providers for speaker detection, transcription, and summarization capabilities. This RFC builds on the modularization refactoring plan to add OpenAI as pluggable providers while maintaining backward compatibility and zero changes to end-user experience with default (local) providers.

## Problem Statement

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

- **Prerequisite**: Modularization refactoring (MODULARIZATION_REFACTORING_PLAN.md) must be completed first
- **Backward Compatibility**: Default providers (local) must remain unchanged
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

```text
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
```

### 2. Configuration

#### 2.1 Provider Selection Fields

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
summary_provider: Literal["local", "openai"] = Field(
    default="local",
    description="Summarization provider: 'local' (transformers) or 'openai' (GPT API)"
)

# Keep existing fields for backward compatibility
ner_model: Optional[str] = Field(default=None, alias="ner_model")
whisper_model: str = Field(default="base", alias="whisper_model")
summary_model: Optional[str] = Field(default=None, alias="summary_model")
```

#### 2.2 API Key Management

**Environment Variable Approach:**

```python
# config.py
openai_api_key: Optional[str] = Field(
    default=None,
    description="OpenAI API key (prefer OPENAI_API_KEY environment variable)"
)

@field_validator('openai_api_key', mode='before')
@classmethod
def load_api_key_from_env(cls, v: Any) -> Optional[str]:
    """Load API key from environment variable if not provided."""
    if v is not None:
        return v
    return os.getenv('OPENAI_API_KEY')

@model_validator(mode='after')
def validate_openai_config(self) -> 'Config':
    """Validate OpenAI provider configuration."""
    # Check if OpenAI provider is selected but API key is missing
    needs_key = (
        self.speaker_detector_type == "openai" or
        self.transcription_provider == "openai" or
        self.summary_provider == "openai"
    )
    
    if needs_key and not self.openai_api_key:
        raise ValueError(
            "OpenAI API key required when using OpenAI providers. "
            "Set OPENAI_API_KEY environment variable or openai_api_key config field."
        )
    return self
```

**Security Best Practices:**

- Never log API keys
- Never commit API keys to source control
- Use environment variables as primary method
- Support config file for convenience (but document security implications)
- Validate API key format (starts with `sk-`) if possible

### 3. OpenAI Provider Implementations

#### 3.1 Speaker Detection Provider

**File**: `podcast_scraper/speaker_detectors/openai_detector.py`

```python
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
        """Detect hosts from feed metadata using OpenAI API."""
        prompt = self._build_host_detection_prompt(feed_title, feed_description, feed_authors)
        
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
            
            hosts_text = response.choices[0].message.content
            hosts = self._parse_hosts_from_response(hosts_text)
            return hosts
            
        except Exception as e:
            logger.error(f"OpenAI API error in host detection: {e}")
            raise
    
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
            
            speakers_text = response.choices[0].message.content
            speakers, detected_hosts, success = self._parse_speakers_from_response(
                speakers_text, known_hosts
            )
            return speakers, detected_hosts, success
            
        except Exception as e:
            logger.error(f"OpenAI API error in speaker detection: {e}")
            raise
    
    def analyze_patterns(
        self,
        episodes: List[models.Episode],
        known_hosts: Set[str],
    ) -> Optional[Dict[str, Any]]:
        """Analyze episode patterns using OpenAI API (optional, can use local logic)."""
        # Can use local pattern analysis or OpenAI API
        # For now, return None to use local logic
        return None
    
    def _build_host_detection_prompt(
        self,
        feed_title: str,
        feed_description: Optional[str],
        feed_authors: Optional[List[str]],
    ) -> str:
        """Build prompt for host detection."""
        # Implementation details...
        pass
    
    def _parse_hosts_from_response(self, response_text: str) -> Set[str]:
        """Parse host names from API response."""
        # Implementation details...
        pass
```

**Key Design Decisions:**

- Use GPT-4o-mini or GPT-3.5-turbo for cost efficiency (configurable)
- Structured prompts for consistent results
- Parse JSON or structured text from API responses
- Handle API errors gracefully with retries
- Maintain same return types as NER provider

#### 3.2 Transcription Provider

**File**: `podcast_scraper/transcription/openai_provider.py`

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
    
    def transcribe(
        self,
        media_path: str,
        cfg: config.Config,
        resource: Any,
    ) -> Tuple[Dict[str, Any], float]:
        """Transcribe media file using OpenAI Whisper API."""
        import time
        
        start_time = time.time()
        
        try:
            with open(media_path, 'rb') as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model=self.model,
                    file=audio_file,
                    language=getattr(cfg, 'whisper_language', None),  # Optional language hint
                    response_format="verbose_json",  # Get segments
                )
            
            elapsed = time.time() - start_time
            
            # Convert to same format as local Whisper provider
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
            
            return result, elapsed
            
        except Exception as e:
            logger.error(f"OpenAI Whisper API error: {e}")
            raise
    
    def cleanup(self, resource: Any) -> None:
        """Cleanup provider resources (no-op for API provider)."""
        pass
```

**Key Design Decisions:**

- Use Whisper API (`whisper-1` model)
- Support language hints (optional)
- Return same format as local provider (text + segments)
- Handle file uploads correctly
- Support parallelism (multiple API calls can run concurrently)

#### 3.3 Summarization Provider

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
    
    def initialize(self, cfg: config.Config) -> Optional[Any]:
        """Initialize provider (no local model loading needed for API)."""
        return self  # Return self as resource
    
    def summarize(
        self,
        text: str,
        cfg: config.Config,
        resource: Any,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Summarize text using OpenAI GPT API.
        
        Can handle full transcripts directly due to large context window (128k+ tokens).
        No chunking needed for most podcast transcripts.
        """
        max_length = max_length or getattr(cfg, 'summary_max_length', 150)
        min_length = min_length or getattr(cfg, 'summary_min_length', 30)
        
        prompt = self._build_summarization_prompt(text, max_length, min_length)
        
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
            
            summary = response.choices[0].message.content
            
            return {
                'summary': summary,
                'metadata': {
                    'model': self.model,
                    'provider': 'openai',
                }
            }
            
        except Exception as e:
            logger.error(f"OpenAI API error in summarization: {e}")
            raise
    
    def summarize_chunks(
        self,
        chunks: List[str],
        cfg: config.Config,
        resource: Any,
    ) -> List[str]:
        """Summarize multiple text chunks (MAP phase).
        
        NOTE: For OpenAI, this is typically not needed due to large context window.
        However, we maintain this interface for compatibility. If chunks can fit in
        one context window, we combine them and summarize once. Otherwise, we
        summarize each chunk separately.
        """
        # Check if we can combine chunks into single API call
        combined_text = "\n\n".join(chunks)
        estimated_tokens = len(combined_text.split()) * 1.3  # Rough token estimate
        
        if estimated_tokens < self.max_context_tokens * 0.8:  # 80% safety margin
            # Can fit all chunks in one call - more efficient
            logger.debug("Combining chunks for single OpenAI API call (fits in context window)")
            result = self.summarize(combined_text, cfg, resource)
            # Return as single summary (workflow will handle this correctly)
            return [result['summary']]
        else:
            # Too long, summarize chunks separately (rare case)
            logger.debug("Chunks too long, summarizing separately")
            summaries = []
            for chunk in chunks:
                result = self.summarize(chunk, cfg, resource)
                summaries.append(result['summary'])
            return summaries
    
    def combine_summaries(
        self,
        summaries: List[str],
        cfg: config.Config,
        resource: Any,
    ) -> str:
        """Combine multiple summaries into final summary (REDUCE phase).
        
        NOTE: For OpenAI, if summarize_chunks() combined chunks into one call,
        this method may receive a single summary. In that case, we can return it
        directly or refine it. For multiple summaries, we combine them.
        """
        if len(summaries) == 1:
            # Already combined in summarize_chunks() - can return directly or refine
            return summaries[0]
        
        # Multiple summaries to combine
        combined_text = "\n\n".join(summaries)
        prompt = self._build_combination_prompt(combined_text)
        
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
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI API error in summary combination: {e}")
            raise
    
    def cleanup(self, resource: Any) -> None:
        """Cleanup provider resources (no-op for API provider)."""
        pass
```

**Key Design Decisions:**

- **Leverage Large Context Window**: GPT-4o-mini supports 128k tokens - can handle full transcripts without chunking
- **Simplified Processing**: Most transcripts fit in one API call, eliminating MAP/REDUCE complexity
- **Maintain Interface Compatibility**: Still implement MAP/REDUCE methods for protocol compliance, but optimize internally
- **Smart Chunk Handling**: If chunks are provided, check if they fit in context window and combine them
- **Cost Efficiency**: Single API call for full transcript is more cost-effective than multiple chunk calls
- **Use GPT-4o-mini**: Cost-effective default with large context window

### 4. Factory Updates

Update factories to support OpenAI providers:

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
```

### 5. Parallelism Considerations

#### 5.1 Current Parallelism

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
            
            if len(self.call_times) >= self.max_calls_per_minute:
                # Wait until we can make a call
                sleep_time = 60 - (now - self.call_times[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    self.call_times = [t for t in self.call_times if now - t < 60]
        
        self.semaphore.acquire()
        with self.lock:
            self.call_times.append(time.time())
    
    def release(self):
        """Release after API call completes."""
        self.semaphore.release()
```

**Usage in Providers:**

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
                    _openai_rate_limiter.release()
        return summaries
```

### 6. Error Handling & Retries

**Retry Strategy:**

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
```

**Error Messages:**

- Clear error messages indicating which provider failed
- Actionable error messages (e.g., "Check OPENAI_API_KEY environment variable")
- Log API errors with context (but not API keys)

### 7. Dependencies

**Add to `pyproject.toml`:**

```toml
[project.optional-dependencies]
# ... existing dependencies ...
openai = [
    "openai>=1.0.0,<2.0.0",
    "tenacity>=8.0.0,<9.0.0",  # For retry logic
]
```

**Installation:**

```bash
# For OpenAI support
pip install -e ".[openai]"

# Or with all ML dependencies
pip install -e ".[ml,openai]"
```

### 8. Testing Strategy

#### 8.1 Unit Tests

- Mock OpenAI API responses
- Test provider interfaces match protocols
- Test error handling and retries
- Test rate limiting

#### 8.2 Integration Tests

- Optional tests with real API (requires API key)
- Test end-to-end workflow with OpenAI providers
- Test parallelism with API providers
- Test error scenarios (invalid key, rate limits)

#### 8.3 Backward Compatibility Tests

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

## Open Questions

1. **Rate Limiting**: What are OpenAI API rate limits? Should we make rate limiter configurable?

2. **Cost Tracking**: Should we add cost tracking/monitoring?
3. **Fallback**: Should we support fallback (try OpenAI, fallback to local)?
4. **Model Selection**: Should model selection be more granular (different models for different tasks)?

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
