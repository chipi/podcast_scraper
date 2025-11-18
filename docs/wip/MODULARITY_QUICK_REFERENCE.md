# Modularity Quick Reference

## Current Coupling Points

### 🔴 High Coupling (Hard to Change)

### 1. RSS Feed Source

- `workflow.py` → `rss_parser.fetch_and_parse_rss()` (direct call)
- `config.rss_url` (RSS-specific)
- `models.RssFeed` (RSS-specific structure)
- **Impact:** Cannot add YouTube/Spotify/etc. without major refactoring

### 2. Speaker Detection

- `workflow.py` → `speaker_detection.get_ner_model()` (NER-specific)
- `config.ner_model` (spaCy-specific)
- `speaker_detection.py` → `spacy.load()` (direct dependency)
- **Impact:** Cannot swap to OpenAI/AWS/etc. without code changes

### 3. Transcription

- `workflow.py` → `whisper.load_whisper_model()` (Whisper-specific)
- `episode_processor.py` → `whisper.transcribe_with_whisper()` (Whisper-specific)
- `config.whisper_model` (Whisper-specific)
- **Impact:** Cannot swap to Deepgram/AssemblyAI/etc. without code changes

---

## Quick Wins (Can Implement Now)

### 1. Add Provider Type Fields (No Breaking Changes)

**File:** `config.py`

```python
# Add these fields (with defaults to current behavior)
feed_source_type: Literal["rss"] = Field(default="rss")
speaker_detector_type: Literal["ner"] = Field(default="ner")
transcription_provider: Literal["whisper"] = Field(default="whisper")
```

**Benefit:** Makes future extensibility explicit, no breaking changes

### 2. Create Protocol Definitions (No Implementation Changes)

**New File:** `podcast_scraper/providers/__init__.py`

```python
from typing import Protocol, List, Set, Optional, Dict, Any, Tuple
from . import config, models

class FeedProvider(Protocol):
    def fetch_feed(self, source: str, cfg: config.Config) -> models.RssFeed:
        """Fetch feed from source."""
        ...

class SpeakerDetector(Protocol):
    def detect_hosts(self, feed_title: str, feed_authors: Optional[List[str]]) -> Set[str]:
        """Detect hosts from feed metadata."""
        ...
    
    def detect_speakers(self, episode_title: str, known_hosts: Set[str]) -> Tuple[List[str], Set[str], bool]:
        """Detect speakers for an episode."""
        ...

class TranscriptionProvider(Protocol):
    def initialize(self, cfg: config.Config) -> Optional[Any]:
        """Initialize provider."""
        ...
    
    def transcribe(self, media_path: str, cfg: config.Config, resource: Any) -> Tuple[Dict[str, Any], float]:
        """Transcribe media file."""
        ...
```

**Benefit:** Documents expected interfaces, enables type checking

### 3. Extract Factory Functions (Minimal Changes)

**New File:** `podcast_scraper/providers/factory.py`

```python
from typing import Optional
from . import config
from .feed_providers import FeedProvider
from .speaker_detectors import SpeakerDetector
from .transcription import TranscriptionProvider

def get_feed_provider(cfg: config.Config) -> FeedProvider:
    """Get feed provider based on config."""
    # For now, always return RSS provider
    from .feed_providers.rss_provider import RSSFeedProvider
    return RSSFeedProvider()

def get_speaker_detector(cfg: config.Config) -> Optional[SpeakerDetector]:
    """Get speaker detector based on config."""
    if not cfg.auto_speakers:
        return None
    # For now, always return NER detector
    from .speaker_detectors.ner_detector import NERSpeakerDetector
    return NERSpeakerDetector(cfg)

def get_transcription_provider(cfg: config.Config) -> Optional[TranscriptionProvider]:
    """Get transcription provider based on config."""
    if not cfg.transcribe_missing:
        return None
    # For now, always return Whisper provider
    from .transcription.whisper_provider import WhisperTranscriptionProvider
    return WhisperTranscriptionProvider()
```

**Benefit:** Centralizes provider creation, makes swapping easier later

---

## Migration Path (When Ready)

### Step 1: Create Provider Interfaces

- Define protocols/ABCs
- No implementation changes yet

### Step 2: Refactor Current Implementations

- Move RSS logic to `feed_providers/rss_provider.py`
- Move NER logic to `speaker_detectors/ner_detector.py`
- Move Whisper logic to `transcription/whisper_provider.py`

### Step 3: Update Workflow to Use Factories

- Replace direct calls with factory functions
- Test thoroughly

### Step 4: Add New Providers (As Needed)

- Implement new providers following protocols
- Add to factory
- Update config with new options

---

## Testing Strategy

### Mock Providers for Testing

```python
# tests/mocks/mock_providers.py
class MockFeedProvider:
    def fetch_feed(self, source: str, cfg: config.Config) -> models.RssFeed:
        return models.RssFeed(title="Test Feed", items=[], base_url="https://test.com")

class MockSpeakerDetector:
    def detect_hosts(self, feed_title: str, feed_authors: Optional[List[str]]) -> Set[str]:
        return {"Test Host"}
    
    def detect_speakers(self, episode_title: str, known_hosts: Set[str]) -> Tuple[List[str], Set[str], bool]:
        return (["Host", "Guest"], {"Host"}, True)

class MockTranscriptionProvider:
    def initialize(self, cfg: config.Config) -> Any:
        return "mock_resource"
    
    def transcribe(self, media_path: str, cfg: config.Config, resource: Any) -> Tuple[Dict[str, Any], float]:
        return ({"text": "Mock transcript", "segments": []}, 1.0)
```

---

## File Structure (Proposed)

```text
podcast_scraper/
├── providers/
│   ├── __init__.py          # Protocol definitions
│   ├── factory.py            # Provider factories
│   ├── feed_providers/
│   │   ├── __init__.py
│   │   ├── base.py          # FeedProvider protocol
│   │   └── rss_provider.py  # Current RSS implementation
│   ├── speaker_detectors/
│   │   ├── __init__.py
│   │   ├── base.py          # SpeakerDetector protocol
│   │   └── ner_detector.py  # Current NER implementation
│   └── transcription/
│       ├── __init__.py
│       ├── base.py          # TranscriptionProvider protocol
│       └── whisper_provider.py  # Current Whisper implementation
├── workflow.py              # Uses factories
├── config.py                # Has provider type fields
└── ...
```

---

## Key Principles

1. **Backward Compatibility First**
   - Default to current implementations
   - Add new fields as optional
   - Don't break existing code

2. **Protocols Over Inheritance**
   - Use `Protocol` for flexibility
   - Easier to mock and test
   - No forced inheritance hierarchy

3. **Factory Pattern**
   - Centralized provider creation
   - Easy to swap implementations
   - Configuration-driven

4. **Gradual Migration**
   - Can implement incrementally
   - Test at each step
   - No big-bang refactoring

5. **Testability**
   - Mock providers for unit tests
   - Integration tests with real providers
   - Backward compatibility tests
