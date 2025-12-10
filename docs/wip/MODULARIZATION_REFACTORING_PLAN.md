# Modularization Refactoring Plan

## Overview

This document outlines the refactoring plan to modularize the podcast scraper architecture, enabling easy integration of OpenAI API as a replacement for on-device AI/ML components.

**North Star Goal:** Easily plug in OpenAI API as replacement for on-device AI/ML (speaker detection, transcription, summarization) without major refactoring.

**Scope:** This refactoring focuses on three key areas:

1. **Speaker Detection** (NER → OpenAI API)
2. **Transcription** (Whisper local → OpenAI Whisper API)
3. **Summarization** (Local transformers → OpenAI API)

**Out of Scope:** RSS feed source abstraction (can be addressed separately later)

---

## Current Architecture Assessment

### Modularity Scores

| Area | Score | Status | Priority |
| ---- | ----- | ------ | -------- |
| Speaker Detection | 5/10 | 🟡 Moderate | **HIGH** |
| Transcription | 6/10 | 🟡 Moderate | **HIGH** |
| Summarization | 4/10 | 🔴 Low | **HIGH** |
| RSS Feed Source | 2/10 | 🔴 Low | **IGNORED** |

---

## 1. Speaker Detection Abstraction

### Current State

**Moderate Coupling Points:**

1. **`workflow.py`** directly imports speaker detection:
   ```python
   from . import speaker_detection
   
   nlp = speaker_detection.get_ner_model(cfg)  # NER-specific
   hosts = speaker_detection.detect_hosts_from_feed(...)  # NER-specific
   ```

2. **`config.py`** has NER-specific fields:
   ```python
   ner_model: Optional[str] = Field(default=None, alias="ner_model")
   auto_speakers: bool = Field(default=True, alias="auto_speakers")
   ```

3. **`speaker_detection.py`** is tightly coupled to spaCy:
   - Direct `spacy.load()` calls
   - NER-specific entity extraction
   - No abstraction layer
   - Large functions (`detect_speaker_names()` ~250 lines, `extract_person_entities()` ~170 lines)

**Current Modularity Score: 5/10**

- ✅ Speaker detection is isolated in its own module
- ✅ Functions are reasonably abstract (`detect_speaker_names()`)
- ❌ Hardcoded to NER/spaCy implementation
- ❌ Cannot easily swap to OpenAI API or other services
- ❌ Config tied to NER model names
- ❌ Large functions with multiple responsibilities

### Target Architecture

**Protocol-Based Provider System:**

```python
# podcast_scraper/speaker_detectors/base.py
from typing import Protocol, List, Set, Optional, Dict, Any, Tuple
from .. import config, models

class SpeakerDetector(Protocol):
    """Protocol for speaker detection providers."""
    
    def detect_hosts(
        self,
        feed_title: str,
        feed_description: Optional[str],
        feed_authors: Optional[List[str]],
    ) -> Set[str]:
        """Detect hosts from feed metadata."""
        ...
    
    def detect_speakers(
        self,
        episode_title: str,
        episode_description: Optional[str],
        known_hosts: Set[str],
    ) -> Tuple[List[str], Set[str], bool]:
        """Detect speakers for an episode.
        
        Returns:
            (speaker_names, detected_hosts, success)
        """
        ...
    
    def analyze_patterns(
        self,
        episodes: List[models.Episode],
        known_hosts: Set[str],
    ) -> Optional[Dict[str, Any]]:
        """Analyze episode patterns for heuristics."""
        ...
```

**Factory Pattern:**

```python
# podcast_scraper/speaker_detectors/factory.py
from typing import Optional
from .. import config
from .base import SpeakerDetector

class SpeakerDetectorFactory:
    """Factory for creating speaker detectors."""
    
    @staticmethod
    def create(cfg: config.Config) -> Optional[SpeakerDetector]:
        if not cfg.auto_speakers:
            return None
        
        detector_type = cfg.speaker_detector_type  # 'ner', 'openai', etc.
        if detector_type == 'ner':
            from .ner_detector import NERSpeakerDetector
            return NERSpeakerDetector(cfg)
        elif detector_type == 'openai':
            from .openai_detector import OpenAISpeakerDetector
            return OpenAISpeakerDetector(cfg)
        return None
```

### Implementation Steps

#### Phase 1: Quick Wins (No Breaking Changes)

1. **Add provider type field to `config.py`:**
   ```python
   speaker_detector_type: Literal["ner", "openai"] = Field(default="ner")
   # Keep ner_model for backward compatibility
   ner_model: Optional[str] = Field(default=None, alias="ner_model")
   ```

2. **Create protocol definitions:**
   - Create `podcast_scraper/speaker_detectors/` package
   - Define `SpeakerDetector` protocol in `base.py`
   - No implementation changes yet

3. **Create factory function:**
   - Create `factory.py` with `SpeakerDetectorFactory.create()`
   - Returns current NER implementation wrapped in protocol

#### Phase 2: Refactor Current Implementation - Speaker Detection

1. **Refactor `speaker_detection.py` → `speaker_detectors/ner_detector.py`:**
   - Extract helper functions from large functions:
     - `_calculate_heuristic_score()` - Extract from `detect_speaker_names()`
     - `_build_guest_candidates()` - Process title/description guests
     - `_select_best_guest()` - Select guest with highest score
     - `_extract_entities_from_text()` - Core NER extraction
     - `_extract_entities_from_segments()` - Segment-based fallback
     - `_pattern_based_fallback()` - Pattern matching fallback
   - Implement `SpeakerDetector` protocol
   - Wrap existing functions as methods

2. **Update `workflow.py`:**
   ```python
   from .speaker_detectors import SpeakerDetectorFactory
   
   detector = SpeakerDetectorFactory.create(cfg)
   if detector:
       hosts = detector.detect_hosts(feed.title, feed.description, feed.authors)
   ```

#### Phase 3: Add OpenAI Provider - Speaker Detection (Future)

1. **Create `speaker_detectors/openai_detector.py`:**
   - Implement `SpeakerDetector` protocol
   - Use OpenAI API for entity extraction
   - Map OpenAI responses to expected format

2. **Update factory:**
   - Add OpenAI detector to factory
   - Update config with OpenAI options

**Benefits:**

- ✅ Easy to add OpenAI API for speaker detection
- ✅ Can use multiple detectors (fallback chain)
- ✅ Testable with mock detectors
- ✅ Backward compatible (NER remains default)
- ✅ Better code organization (smaller functions)

**Effort:** Medium (2-3 days)

---

## 2. Transcription Abstraction

### Current State (Transcription)

**Moderate Coupling Points:**

1. **`workflow.py`** directly imports Whisper:
   ```python
   from . import whisper_integration as whisper
   
   whisper_model = whisper.load_whisper_model(cfg)  # Whisper-specific
   result, elapsed = whisper.transcribe_with_whisper(...)  # Whisper-specific
   ```

2. **`episode_processor.py`** has Whisper-specific code:
   ```python
   from . import whisper_integration as whisper
   
   result, tc_elapsed = whisper.transcribe_with_whisper(whisper_model, temp_media, cfg)
   ```

3. **`config.py`** has Whisper-specific fields:
   ```python
   whisper_model: str = Field(default="base", alias="whisper_model")
   transcribe_missing: bool = Field(default=False, alias="transcribe_missing")
   ```

4. **`_TranscriptionResources`** has Whisper model hardcoded:
   ```python
   class _TranscriptionResources(NamedTuple):
       whisper_model: Any  # Whisper-specific type
   ```

**Current Modularity Score: 6/10**

- ✅ Transcription logic is isolated in `whisper_integration.py`
- ✅ Functions are reasonably abstract (`transcribe_with_whisper()`)
- ❌ Hardcoded to Whisper library
- ❌ Cannot easily swap to OpenAI Whisper API or other services
- ❌ Config tied to Whisper model names
- ❌ Resource management assumes local model loading

### Target Architecture (Transcription)

**Protocol-Based Provider System:**

```python
# podcast_scraper/transcription/base.py
from typing import Protocol, Dict, Optional, Tuple, Any
from .. import config

class TranscriptionProvider(Protocol):
    """Protocol for transcription providers."""
    
    def initialize(self, cfg: config.Config) -> Optional[Any]:
        """Initialize provider (load model, setup API client, etc.).
        
        Returns:
            Provider-specific resource object or None if initialization fails
        """
        ...
    
    def transcribe(
        self,
        media_path: str,
        cfg: config.Config,
        resource: Any,  # Provider-specific resource
    ) -> Tuple[Dict[str, Any], float]:
        """Transcribe media file.
        
        Args:
            media_path: Path to media file
            cfg: Configuration object
            resource: Provider-specific resource (model, client, etc.)
        
        Returns:
            Tuple of (result_dict, elapsed_seconds)
            result_dict should have 'text' and optionally 'segments'
        """
        ...
    
    def cleanup(self, resource: Any) -> None:
        """Cleanup provider resources."""
        ...
```

**Factory Pattern:**

```python
# podcast_scraper/transcription/factory.py
from typing import Optional
from .. import config
from .base import TranscriptionProvider

class TranscriptionProviderFactory:
    """Factory for creating transcription providers."""
    
    @staticmethod
    def create(cfg: config.Config) -> Optional[TranscriptionProvider]:
        if not cfg.transcribe_missing:
            return None
        
        provider_type = cfg.transcription_provider  # 'whisper', 'openai', etc.
        if provider_type == 'whisper':
            from .whisper_provider import WhisperTranscriptionProvider
            return WhisperTranscriptionProvider()
        elif provider_type == 'openai':
            from .openai_provider import OpenAITranscriptionProvider
            return OpenAITranscriptionProvider(cfg)
        return None
```

### Implementation Steps (Transcription)

#### Phase 1: Quick Wins - Transcription (No Breaking Changes)

1. **Add provider type field to `config.py`:**
   ```python
   transcription_provider: Literal["whisper", "openai"] = Field(default="whisper")
   # Keep whisper_model for backward compatibility
   whisper_model: str = Field(default="base", alias="whisper_model")
   ```

2. **Create protocol definitions:**
   - Create `podcast_scraper/transcription/` package
   - Define `TranscriptionProvider` protocol in `base.py`
   - No implementation changes yet

3. **Create factory function:**
   - Create `factory.py` with `TranscriptionProviderFactory.create()`
   - Returns current Whisper implementation wrapped in protocol

#### Phase 2: Refactor Current Implementation - Transcription

1. **Refactor `whisper_integration.py` → `transcription/whisper_provider.py`:**
   - Keep all current logic
   - Implement `TranscriptionProvider` protocol
   - Wrap existing functions as methods
   - Extract helper functions from `transcribe_media_to_text()`:
     - `_format_transcript_if_needed()` - Screenplay formatting logic
     - `_save_transcript_file()` - File writing logic
     - `_cleanup_temp_media()` - Cleanup logic

2. **Update `workflow.py`:**
   ```python
   from .transcription import TranscriptionProviderFactory
   
   provider = TranscriptionProviderFactory.create(cfg)
   if provider:
       resource = provider.initialize(cfg)
       result, elapsed = provider.transcribe(media_path, cfg, resource)
   ```

3. **Update `_TranscriptionResources`:**
   ```python
   class _TranscriptionResources(NamedTuple):
       provider: Optional[TranscriptionProvider]
       resource: Any  # Provider-specific resource
       temp_dir: Optional[str]
       transcription_jobs: List[models.TranscriptionJob]
       # ... rest
   ```

#### Phase 3: Add OpenAI Provider - Transcription (Future)

1. **Create `transcription/openai_provider.py`:**
   - Implement `TranscriptionProvider` protocol
   - Use OpenAI Whisper API
   - Handle API authentication and rate limiting
   - Map OpenAI responses to expected format

2. **Update factory:**
   - Add OpenAI provider to factory
   - Update config with OpenAI options

**Benefits:**

- ✅ Easy to add OpenAI Whisper API
- ✅ Can support both local (Whisper) and cloud (API) providers
- ✅ Testable with mock providers
- ✅ Backward compatible (Whisper remains default)
- ✅ Better resource management (provider-specific cleanup)

**Effort:** Medium-High (3-4 days)

---

## 3. Summarization Abstraction

### Current State (Summarization)

**Tight Coupling Points:**

1. **`workflow.py`** directly imports summarizer:
   ```python
   from . import summarizer
   
   model_name = summarizer.select_summary_model(cfg)  # Local model-specific
   summary_model = summarizer.SummaryModel(...)  # Local model-specific
   ```

2. **`metadata.py`** has summarization logic:
   ```python
   from . import summarizer
   
   summary_metadata = _generate_episode_summary(...)  # Uses local models
   ```

3. **`config.py`** has local model-specific fields:
   ```python
   summary_model: Optional[str] = Field(default=None, alias="summary_model")
   summary_provider: Literal["local"] = Field(default="local")
   generate_summaries: bool = Field(default=False, alias="generate_summaries")
   ```

4. **`summarizer.py`** is tightly coupled to HuggingFace transformers:
   - Direct `AutoModelForSeq2SeqLM.from_pretrained()` calls
   - Local model loading and caching
   - No abstraction layer

**Current Modularity Score: 4/10**

- ✅ Summarization logic is isolated in `summarizer.py`
- ❌ Hardcoded to local HuggingFace models
- ❌ Cannot easily swap to OpenAI API
- ❌ Config tied to HuggingFace model names
- ❌ Resource management assumes local model loading
- ❌ Large `metadata.py` functions (`generate_episode_metadata()` ~200 lines)

### Target Architecture (Summarization)

**Protocol-Based Provider System:**

```python
# podcast_scraper/summarization/base.py
from typing import Protocol, Optional, Dict, Any
from .. import config

class SummarizationProvider(Protocol):
    """Protocol for summarization providers."""
    
    def initialize(self, cfg: config.Config) -> Optional[Any]:
        """Initialize provider (load model, setup API client, etc.).
        
        Returns:
            Provider-specific resource object or None if initialization fails
        """
        ...
    
    def summarize(
        self,
        text: str,
        cfg: config.Config,
        resource: Any,  # Provider-specific resource
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Summarize text.
        
        Args:
            text: Text to summarize
            cfg: Configuration object
            resource: Provider-specific resource (model, client, etc.)
            max_length: Maximum summary length
            min_length: Minimum summary length
        
        Returns:
            Dictionary with 'summary' and optionally 'chunks', 'metadata'
        """
        ...
    
    def summarize_chunks(
        self,
        chunks: List[str],
        cfg: config.Config,
        resource: Any,
    ) -> List[str]:
        """Summarize multiple text chunks (MAP phase).
        
        Returns:
            List of chunk summaries
        """
        ...
    
    def combine_summaries(
        self,
        summaries: List[str],
        cfg: config.Config,
        resource: Any,
    ) -> str:
        """Combine multiple summaries into final summary (REDUCE phase).
        
        Returns:
            Final combined summary
        """
        ...
    
    def cleanup(self, resource: Any) -> None:
        """Cleanup provider resources."""
        ...
```

**Factory Pattern:**

```python
# podcast_scraper/summarization/factory.py
from typing import Optional
from .. import config
from .base import SummarizationProvider

class SummarizationProviderFactory:
    """Factory for creating summarization providers."""
    
    @staticmethod
    def create(cfg: config.Config) -> Optional[SummarizationProvider]:
        if not cfg.generate_summaries:
            return None
        
        provider_type = cfg.summary_provider  # 'transformers', 'openai', etc.
        if provider_type == 'transformers':
            from .transformers_provider import TransformersSummarizationProvider
            return TransformersSummarizationProvider(cfg)
        elif provider_type == 'openai':
            from .openai_provider import OpenAISummarizationProvider
            return OpenAISummarizationProvider(cfg)
        return None
```

### Implementation Steps (Summarization)

#### Phase 1: Quick Wins - Summarization (No Breaking Changes)

1. **Add provider type field to `config.py`:**
   ```python
   summary_provider: Literal["transformers", "openai"] = Field(default="transformers")
   # Keep summary_model for backward compatibility
   summary_model: Optional[str] = Field(default=None, alias="summary_model")
   ```

2. **Create protocol definitions:**
   - Create `podcast_scraper/summarization/` package
   - Define `SummarizationProvider` protocol in `base.py`
   - No implementation changes yet

3. **Create factory function:**
   - Create `factory.py` with `SummarizationProviderFactory.create()`
   - Returns current local implementation wrapped in protocol

#### Phase 2: Refactor Current Implementation - Summarization

1. **Extract Preprocessing to Shared Module:**
   - Create `podcast_scraper/preprocessing.py` module
   - Move `clean_transcript()`, `remove_sponsor_blocks()`, `clean_for_summarization()` from `summarizer.py`
   - These functions are provider-agnostic and should be called BEFORE provider selection
   - Update `metadata.py` to use shared preprocessing module

2. **Refactor `summarizer.py` → `summarization/transformers_provider.py`:**
   - Keep all current logic
   - Implement `SummarizationProvider` protocol
   - Wrap existing `SummaryModel` class as provider
   - Remove preprocessing functions (moved to shared module)
   - Extract helper functions from `metadata.py`:
     - `_build_feed_metadata()` - Construct FeedMetadata
     - `_build_episode_metadata()` - Construct EpisodeMetadata
     - `_build_content_metadata()` - Construct ContentMetadata
     - `_build_processing_metadata()` - Construct ProcessingMetadata

2. **Update `workflow.py`:**
   ```python
   from .summarization import SummarizationProviderFactory
   
   provider = SummarizationProviderFactory.create(cfg)
   if provider:
       resource = provider.initialize(cfg)
   ```

3. **Update `metadata.py`:**
   - Refactor `generate_episode_metadata()` to use provider
   - Extract helper functions for model building
   - Use provider for summarization instead of direct model calls

#### Phase 3: Add OpenAI Provider - Summarization (Future)

1. **Create `summarization/openai_provider.py`:**
   - Implement `SummarizationProvider` protocol
   - Use OpenAI API for summarization
   - Handle chunking for long texts (MAP phase)
   - Combine summaries (REDUCE phase)
   - Handle API authentication and rate limiting

2. **Update factory:**
   - Add OpenAI provider to factory
   - Update config with OpenAI options

**Benefits:**

- ✅ Easy to add OpenAI API for summarization
- ✅ Can support both local (transformers) and cloud (API) providers
- ✅ Testable with mock providers
- ✅ Backward compatible (local remains default)
- ✅ Better resource management
- ✅ Cleaner `metadata.py` (smaller functions)

**Effort:** Medium-High (3-4 days)

---

## Implementation Priority & Timeline

### Recommended Order

1. **Phase 1: Transcription Abstraction** (Highest Impact)
   - Already has good isolation
   - Most likely to need alternatives (OpenAI Whisper API)
   - Medium complexity
   - **Effort:** 3-4 days

2. **Phase 2: Speaker Detection Abstraction** (Medium Impact)
   - Good isolation already
   - May want OpenAI API for better accuracy
   - Medium complexity
   - **Effort:** 2-3 days

3. **Phase 3: Summarization Abstraction** (High Impact)
   - Most tightly coupled currently
   - OpenAI API would be most valuable here
   - Medium-High complexity
   - **Effort:** 3-4 days

**Total Estimated Effort:** 8-11 days for all three abstractions

### Quick Wins (Can Do Now - No Breaking Changes)

1. **Add Provider Type Fields to Config:**
   ```python
   # config.py
   speaker_detector_type: Literal["ner", "openai"] = Field(default="ner")
   transcription_provider: Literal["whisper", "openai"] = Field(default="whisper")
   summary_provider: Literal["transformers", "openai"] = Field(default="transformers")
   ```

2. **Create Protocol/ABC Definitions:**
   - Define interfaces now
   - Implement later when needed
   - Makes intent clear
   - Enables type checking

3. **Extract Provider Factories:**
   - Create factory functions that return current implementations
   - Use factories in workflow
   - Makes swapping easier later

**Effort:** 1 day

---

## File Structure (Proposed)

```text
podcast_scraper/
├── preprocessing.py         # NEW: Provider-agnostic preprocessing utilities
│                           # - clean_transcript() (timestamp removal, speaker normalization)
│                           # - remove_sponsor_blocks() (ad removal)
│                           # - clean_for_summarization() (combined cleaning)
│                           # Called BEFORE provider selection in metadata.py/workflow.py
├── speaker_detectors/
│   ├── __init__.py
│   ├── base.py              # SpeakerDetector protocol
│   ├── factory.py           # SpeakerDetectorFactory
│   ├── ner_detector.py      # Current NER implementation (refactored)
│   └── openai_detector.py   # Future OpenAI implementation
├── transcription/
│   ├── __init__.py
│   ├── base.py              # TranscriptionProvider protocol
│   ├── factory.py           # TranscriptionProviderFactory
│   ├── whisper_provider.py # Current Whisper implementation (refactored)
│   └── openai_provider.py  # Future OpenAI Whisper API implementation
├── summarization/
│   ├── __init__.py
│   ├── base.py                    # SummarizationProvider protocol
│   ├── factory.py                 # SummarizationProviderFactory
│   ├── transformers_provider.py   # Current HuggingFace transformers implementation (refactored)
│   └── openai_provider.py         # Future OpenAI API implementation
├── workflow.py              # Uses factories, calls preprocessing
├── metadata.py              # Refactored (smaller functions), calls preprocessing BEFORE providers
├── config.py                # Has provider type fields
└── ...
```

---

## Key Principles

1. **Backward Compatibility First**
   - Default to current implementations
   - Add new fields as optional
   - Don't break existing code
   - Keep existing config fields for compatibility

2. **Protocols Over Inheritance**
   - Use `Protocol` for flexibility
   - Easier to mock and test
   - No forced inheritance hierarchy
   - Enables duck typing

3. **Factory Pattern**
   - Centralized provider creation
   - Easy to swap implementations
   - Configuration-driven
   - Single point of change

4. **Gradual Migration**
   - Can implement incrementally
   - Test at each step
   - No big-bang refactoring
   - Each phase delivers value

5. **Testability**
   - Mock providers for unit tests
   - Integration tests with real providers
   - Backward compatibility tests
   - Provider-specific tests

6. **North Star: OpenAI API Ready**
   - All abstractions designed with OpenAI API in mind
   - Easy to add OpenAI providers after refactoring
   - No additional refactoring needed for OpenAI integration

---

## Migration Strategy

### Backward Compatibility

- Keep all existing config fields
- Default to current implementations
- Add new fields as optional
- Deprecate old fields gradually (if needed)

### Testing Strategy

- Create mock providers for testing
- Test workflow with different providers
- Ensure backward compatibility tests pass
- Test provider switching

### Rollout Plan

1. **Week 1:** Quick wins + Transcription abstraction
2. **Week 2:** Speaker detection abstraction
3. **Week 3:** Summarization abstraction
4. **Week 4:** Testing, documentation, cleanup

---

## Success Criteria

✅ Can add OpenAI API providers without modifying core workflow  
✅ All existing functionality preserved  
✅ Tests pass with both old and new providers  
✅ Config remains backward compatible  
✅ Code is more maintainable (smaller functions, clearer structure)  
✅ Ready for OpenAI API integration as next step  

---

## Next Steps After Refactoring

Once this refactoring is complete, adding OpenAI API providers will be straightforward:

1. **OpenAI Speaker Detection:**
   - Create `speaker_detectors/openai_detector.py`
   - Implement `SpeakerDetector` protocol
   - Use OpenAI API for entity extraction
   - Add to factory

2. **OpenAI Whisper Transcription:**
   - Create `transcription/openai_provider.py`
   - Implement `TranscriptionProvider` protocol
   - Use OpenAI Whisper API
   - Add to factory

3. **OpenAI Summarization:**
   - Create `summarization/openai_provider.py`
   - Implement `SummarizationProvider` protocol
   - Use OpenAI API for summarization
   - Handle MAP/REDUCE phases
   - Add to factory

**Estimated Effort for OpenAI Integration:** 3-5 days (after refactoring)

---

## Notes

- All refactoring should maintain existing functionality
- Add tests for extracted functions and new providers
- Update docstrings to reflect new structure
- Consider backward compatibility for public APIs
- Document provider interfaces clearly
- Provide examples for each provider type
