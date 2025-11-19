# Architecture Modularity Assessment

## Overview

This document assesses the current codebase architecture for modularity and extensibility in three key areas:

1. **Podcast Source** (currently RSS-only)
2. **Speaker/Host Detection** (currently NER-based)
3. **Transcription** (currently Whisper-based)

The goal is to identify coupling points and provide recommendations for making the architecture more modular and easier to extend without major refactoring.

---

## 1. Podcast Source Abstraction

### Current State (Podcast Source)

**Tight Coupling Points:**

1. **`workflow.py`** directly imports and calls RSS-specific functions:

   ```python
   from .rss_parser import (
       fetch_and_parse_rss,  # RSS-specific
       create_episode_from_item,  # RSS-specific
       extract_episode_metadata,  # RSS-specific
   )
   
   feed = fetch_and_parse_rss(cfg)  # Hardcoded RSS fetching
   ```

2. **`config.py`** has RSS-specific fields:

   ```python
   rss_url: Optional[str] = Field(default=None, alias="rss")
   ```

3. **`models.py`** has RSS-specific data structures:

   ```python
   class RssFeed:  # RSS-specific model
   ```

4. **Workflow assumes RSS structure** throughout:
   - `_fetch_and_parse_feed()` returns `models.RssFeed`
   - `_prepare_episodes_from_feed()` expects RSS items
   - Episode creation assumes RSS item structure

### Impact Assessment (RSS Feed)

**Current Modularity Score: 2/10** (Low - Hard to extend)

- ❌ Cannot add alternative sources (YouTube, Spotify, direct API) without modifying core workflow
- ❌ RSS-specific logic scattered across workflow
- ❌ Config tied to RSS URL format
- ✅ RSS parsing is isolated in `rss_parser.py` module

### Recommendations

#### Option A: Abstract Feed Provider Interface (Recommended)

Create a pluggable feed provider system:

```python
# podcast_scraper/feed_providers/__init__.py
from abc import ABC, abstractmethod
from typing import List, Protocol

class FeedProvider(Protocol):
    """Protocol for feed providers."""
    
    def fetch_feed(self, source: str, cfg: config.Config) -> Feed:
        """Fetch feed from source."""
        ...
    
    def parse_feed(self, raw_data: Any, cfg: config.Config) -> Feed:
        """Parse raw feed data into Feed object."""
        ...

class Feed(Protocol):
    """Abstract feed representation."""
    title: str
    items: List[FeedItem]
    base_url: str
    authors: Optional[List[str]]
```

**Implementation Steps:**

1. Create `feed_providers/` package with:
   - `base.py` - Abstract base classes/protocols
   - `rss_provider.py` - Current RSS implementation
   - `registry.py` - Provider registry/factory

2. Refactor `workflow.py`:

   ```python
   from .feed_providers import get_feed_provider
   
   provider = get_feed_provider(cfg.feed_source_type)  # 'rss', 'youtube', etc.
   feed = provider.fetch_feed(cfg.feed_source, cfg)
   ```

3. Update `config.py`:

   ```python
   feed_source_type: Literal["rss", "youtube", "spotify"] = Field(default="rss")
   feed_source: Optional[str] = Field(default=None, alias="rss")  # Generic source
   ```

**Benefits:**

- ✅ Easy to add new sources (YouTube, Spotify, API)
- ✅ Minimal changes to workflow
- ✅ Clear separation of concerns
- ✅ Testable with mock providers

**Effort:** Medium (2-3 days)

---

## 2. Speaker Detection Abstraction

### Current State (Speaker Detection)

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

### Impact Assessment (Speaker Detection)

**Current Modularity Score: 5/10** (Moderate - Somewhat extensible)

- ✅ Speaker detection is isolated in its own module
- ✅ Functions are reasonably abstract (`detect_speaker_names()`)
- ❌ Hardcoded to NER/spaCy implementation
- ❌ Cannot easily swap to alternative services (OpenAI API, AWS Comprehend, etc.)
- ❌ Config tied to NER model names

### Recommendations (Speaker Detection)

#### Option A: Strategy Pattern with Provider Interface (Recommended)

Create a pluggable speaker detection system:

```python
# podcast_scraper/speaker_detectors/__init__.py
from abc import ABC, abstractmethod
from typing import List, Set, Optional, Dict, Any

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
    ) -> tuple[List[str], Set[str], bool]:
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

class SpeakerDetectorFactory:
    """Factory for creating speaker detectors."""
    
    @staticmethod
    def create(cfg: config.Config) -> Optional[SpeakerDetector]:
        if not cfg.auto_speakers:
            return None
        
        detector_type = cfg.speaker_detector_type  # 'ner', 'openai', 'aws', etc.
        if detector_type == 'ner':
            return NERSpeakerDetector(cfg)
        elif detector_type == 'openai':
            return OpenAISpeakerDetector(cfg)
        # ... other providers
        return None
```

**Implementation Steps:**

1. Create `speaker_detectors/` package:
   - `base.py` - Protocol/ABC definitions
   - `ner_detector.py` - Current NER implementation (refactored)
   - `openai_detector.py` - Future OpenAI API implementation
   - `factory.py` - Detector factory

2. Refactor `speaker_detection.py` → `speaker_detectors/ner_detector.py`:
   - Keep all current logic
   - Implement `SpeakerDetector` protocol
   - Wrap existing functions as methods

3. Update `workflow.py`:

   ```python
   from .speaker_detectors import SpeakerDetectorFactory
   
   detector = SpeakerDetectorFactory.create(cfg)
   if detector:
       hosts = detector.detect_hosts(feed.title, None, feed.authors)
   ```

4. Update `config.py`:

   ```python
   speaker_detector_type: Literal["ner", "openai", "aws"] = Field(default="ner")
   # Keep ner_model for backward compatibility
   ner_model: Optional[str] = Field(default=None, alias="ner_model")
   ```

**Benefits:**

- ✅ Easy to add new detection methods (OpenAI, AWS, custom)
- ✅ Can use multiple detectors (fallback chain)
- ✅ Testable with mock detectors
- ✅ Backward compatible (NER remains default)

**Effort:** Medium (2-3 days)

---

## 3. Transcription Abstraction

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

### Impact Assessment (Transcription)

**Current Modularity Score: 6/10** (Moderate - Somewhat extensible)

- ✅ Transcription logic is isolated in `whisper_integration.py`
- ✅ Functions are reasonably abstract (`transcribe_with_whisper()`)
- ❌ Hardcoded to Whisper library
- ❌ Cannot easily swap to alternative services (Deepgram, AssemblyAI, etc.)
- ❌ Config tied to Whisper model names
- ❌ Resource management assumes local model loading

### Recommendations (Transcription)

#### Option A: Transcription Provider Interface (Recommended)

Create a pluggable transcription system:

```python
# podcast_scraper/transcription/__init__.py
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Any

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

class TranscriptionProviderFactory:
    """Factory for creating transcription providers."""
    
    @staticmethod
    def create(cfg: config.Config) -> Optional[TranscriptionProvider]:
        if not cfg.transcribe_missing:
            return None
        
        provider_type = cfg.transcription_provider  # 'whisper', 'deepgram', 'assemblyai', etc.
        if provider_type == 'whisper':
            return WhisperTranscriptionProvider()
        elif provider_type == 'deepgram':
            return DeepgramTranscriptionProvider(cfg)
        elif provider_type == 'assemblyai':
            return AssemblyAITranscriptionProvider(cfg)
        # ... other providers
        return None
```

**Implementation Steps:**

1. Create `transcription/` package:
   - `base.py` - Protocol/ABC definitions
   - `whisper_provider.py` - Current Whisper implementation (refactored)
   - `deepgram_provider.py` - Future Deepgram implementation
   - `factory.py` - Provider factory

2. Refactor `whisper_integration.py` → `transcription/whisper_provider.py`:
   - Keep all current logic
   - Implement `TranscriptionProvider` protocol
   - Wrap existing functions as methods

3. Update `workflow.py`:

   ```python
   from .transcription import TranscriptionProviderFactory
   
   provider = TranscriptionProviderFactory.create(cfg)
   if provider:
       resource = provider.initialize(cfg)
       result, elapsed = provider.transcribe(media_path, cfg, resource)
   ```

4. Update `_TranscriptionResources`:

   ```python
   class _TranscriptionResources(NamedTuple):
       provider: Optional[TranscriptionProvider]
       resource: Any  # Provider-specific resource
       temp_dir: Optional[str]
       transcription_jobs: List[models.TranscriptionJob]
       # ... rest
   ```

5. Update `config.py`:

   ```python
   transcription_provider: Literal["whisper", "deepgram", "assemblyai"] = Field(default="whisper")
   # Keep whisper_model for backward compatibility
   whisper_model: str = Field(default="base", alias="whisper_model")
   ```

**Benefits:**

- ✅ Easy to add new transcription services (Deepgram, AssemblyAI, etc.)
- ✅ Can support both local (Whisper) and cloud (API) providers
- ✅ Testable with mock providers
- ✅ Backward compatible (Whisper remains default)
- ✅ Better resource management (provider-specific cleanup)

**Effort:** Medium-High (3-4 days)

---

## Summary & Priority Recommendations

### Current Architecture Scores

| Area | Score | Status |
| --- | --- | --- |
| Podcast Source | 2/10 | 🔴 Needs Improvement |
| Speaker Detection | 5/10 | 🟡 Moderate |
| Transcription | 6/10 | 🟡 Moderate |

### Recommended Implementation Order

1. **Phase 1: Transcription Abstraction** (Highest Impact)
   - Already has good isolation
   - Most likely to need alternatives (cloud APIs)
   - Medium complexity

2. **Phase 2: Speaker Detection Abstraction** (Medium Impact)
   - Good isolation already
   - May want cloud services (OpenAI, AWS)
   - Medium complexity

3. **Phase 3: Feed Source Abstraction** (Highest Complexity)
   - Most tightly coupled
   - Requires more refactoring
   - But enables biggest flexibility

### Quick Wins (Can Do Now)

1. **Add Provider Type Fields to Config** (No breaking changes):

   ```python
   feed_source_type: Literal["rss"] = Field(default="rss")  # Future: "youtube", "spotify"
   speaker_detector_type: Literal["ner"] = Field(default="ner")  # Future: "openai", "aws"
   transcription_provider: Literal["whisper"] = Field(default="whisper")  # Future: "deepgram"
   ```

2. **Create Protocol/ABC Definitions** (No implementation changes):
   - Define interfaces now
   - Implement later when needed
   - Makes intent clear

3. **Extract Provider Factories** (Minimal changes):
   - Create factory functions that return current implementations
   - Use factories in workflow
   - Makes swapping easier later

### Migration Strategy

**Backward Compatibility:**

- Keep all existing config fields
- Default to current implementations
- Add new fields as optional
- Deprecate old fields gradually

**Testing Strategy:**

- Create mock providers for testing
- Test workflow with different providers
- Ensure backward compatibility tests pass

---

## Conclusion

The current architecture has **moderate modularity** with room for improvement. The recommended approach uses **Strategy Pattern** and **Protocol-based interfaces** to enable pluggable providers while maintaining backward compatibility.

**Key Principles:**

1. ✅ **Protocols over Inheritance** - Use `Protocol` for flexibility
2. ✅ **Factory Pattern** - Centralized provider creation
3. ✅ **Backward Compatibility** - Don't break existing code
4. ✅ **Gradual Migration** - Can implement incrementally
5. ✅ **Testability** - Easy to mock and test

**Estimated Total Effort:** 7-10 days for all three abstractions

**Risk Level:** Low - Changes are additive and backward compatible
