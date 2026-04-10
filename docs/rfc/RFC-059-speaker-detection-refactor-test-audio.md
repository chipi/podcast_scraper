# RFC-059: Speaker Detection Refactor & Test Audio Improvements

- **Status**: Draft
- **Authors**: Architecture Review
- **Stakeholders**: Core Pipeline, ML Providers, Test Infrastructure
- **Related PRDs**:
  - `docs/prd/PRD-020-audio-speaker-diarization.md`
  - `docs/prd/PRD-008-speaker-name-detection.md`
  - `docs/prd/PRD-002-whisper-fallback.md`
- **Related RFCs**:
  - `docs/rfc/RFC-010-speaker-name-detection.md` (current NER design)
  - `docs/rfc/RFC-058-audio-speaker-diarization.md` (companion: diarization capability)
  - `docs/rfc/RFC-060-diarization-aware-commercial-cleaning.md` (downstream: diarization-aware sponsor detection)
  - `docs/rfc/RFC-040-audio-preprocessing-pipeline.md` (audio preprocessing)
  - `docs/rfc/RFC-018-test-structure-reorganization.md` (test organization)
  - `docs/rfc/RFC-054-e2e-mock-response-strategy.md` (E2E test strategy)
- **Related Issues**:
  - Issue #269: Refactor `speaker_detection.py` into `speaker_detectors/` submodules
  - Issue #111: Use unique voices for each speaker in generated mock podcast audio
  - Issue #109: Add commercial segments to mock podcast transcripts
  - Issue #414: Audio Pipeline Separation
- **Related Documents**:
  - `tests/fixtures/FIXTURES_SPEC.md`
  - `tests/fixtures/scripts/transcripts_to_mp3.py`

## Abstract

This RFC addresses two related infrastructure improvements that collectively prepare the audio and speaker detection systems for the diarization capability defined in RFC-058:

1. **Speaker detection modularization** — Refactor `speaker_detection.py` (1,376 lines) into focused submodules within `speaker_detectors/` (Issue #269)
2. **Test audio improvements** — Assign unique TTS voices per speaker and add commercial segments to mock transcripts (Issues #111, #109)

These are prerequisite and companion work for RFC-058 (diarization): modular speaker detection is needed to cleanly integrate diarization name-mapping, and distinct test voices are required for diarization to actually distinguish speakers in test audio.

**Note:** Audio chunking for API providers (Issue #286) was originally scoped in this RFC but has been moved out as a standalone item — it is independent of diarization and speaker detection.

**Architecture Alignment:** Follows RFC-016 modularization principles and matches existing patterns in `summarization/` (chunking, map_reduce, prompts submodules).

## Problem Statement

Two distinct but related problems need solving:

### Problem 1: Monolithic Speaker Detection (Issue #269)

`speaker_detection.py` is 1,376 lines containing 23+ functions spanning seven concerns: spaCy model loading, entity extraction, name validation, pattern analysis, title position analysis, speaker detection orchestration, and guest detection scoring. This monolith:

- Makes it hard to add new detection strategies (e.g., diarization-based mapping from RFC-058)
- Complicates testing (tests need the whole module to test one concern)
- Violates the modularization pattern used elsewhere (e.g., `summarization/` with `chunking.py`, `map_reduce.py`, `prompts.py`)

### Problem 2: Indistinguishable Test Audio (Issue #111)

The TTS audio generation script (`transcripts_to_mp3.py`) uses only **two voices** for all speakers:

- **Host voice**: `Samantha` (en_US) for all hosts (Maya, Ethan, Rina, Leo, Nora)
- **Guest voice**: `Daniel` (en_GB) for all guests

This means all hosts sound identical and all guests sound identical. When RFC-058 adds diarization, pyannote cannot distinguish speakers in test audio because they share the same voice. Additionally, there are no commercial segments (Issue #109), making it impossible to test sponsor-block cleaning against realistic fixtures.

**Use Cases:**

1. **Diarization integration**: Clean speaker detection module with clear extension points for diarization name-mapping
2. **Diarization testing**: Test audio where pyannote can actually distinguish speakers by voice
3. **Cleaning validation**: Test transcripts with commercial segments for `remove_sponsor_blocks()` testing

## Goals

1. **Modularize speaker detection** into 6-7 focused submodules with clear responsibilities
2. **Maintain backward compatibility**: `speaker_detection.py` remains as public API (thin re-export wrapper)
3. **Unique TTS voices**: Each fixture speaker gets a distinct macOS `say` voice with accent variety
4. **Commercial segments**: Add realistic sponsor blocks to all 15 mock transcripts
5. **Enable diarization testing**: Test audio fixtures usable for pyannote speaker separation validation

## Constraints & Assumptions

**Constraints:**

- Speaker detection refactor must not break any existing tests (zero-diff behavior)
- Test audio generation requires macOS (uses `/usr/bin/say`) — CI runs on Linux, so audio fixtures are pre-generated and checked in (Git LFS)
- Commercial segments must use patterns detectable by existing `remove_sponsor_blocks()` and the evolving eval tooling

**Assumptions:**

- macOS `say` has sufficient voice variety (~20+ English voices with different accents)
- pyannote can distinguish macOS TTS voices that use different voice models
- Fixture regeneration is an infrequent, developer-initiated operation

## Design & Implementation

### 1. Speaker Detection Modularization (Issue #269)

**Current structure:**

```text
src/podcast_scraper/
├── providers/ml/
│   └── speaker_detection.py        # 1,376 lines — ALL logic
└── speaker_detectors/
    ├── __init__.py
    ├── base.py                     # SpeakerDetector protocol (61 lines)
    └── factory.py                  # Factory function (67 lines)
```

**Target structure:**

```text
src/podcast_scraper/
├── providers/ml/
│   └── speaker_detection.py        # ~80 lines — thin re-export wrapper
└── speaker_detectors/
    ├── __init__.py                  # Re-exports public API
    ├── base.py                      # SpeakerDetector protocol (existing)
    ├── factory.py                   # Factory function (existing)
    ├── constants.py                 # All threshold/configuration constants
    ├── ner.py                       # spaCy model loading & management
    ├── entities.py                  # Entity extraction from text (NER)
    ├── normalization.py             # Name validation & sanitization
    ├── patterns.py                  # Pattern analysis for titles/descriptions
    ├── guests.py                    # Guest detection scoring & selection
    ├── hosts.py                     # Host detection from feed/transcript
    └── detection.py                 # Main orchestration (detect_speaker_names)
```

**Module responsibilities:**

| Module | Lines (est.) | Responsibility | Key functions |
| --- | --- | --- | --- |
| `constants.py` | ~60 | All threshold and scoring constants | `MIN_NAME_LENGTH`, `POSITION_SCORE_BONUS`, etc. |
| `ner.py` | ~100 | spaCy model loading, validation, download | `_load_spacy_model`, `get_ner_model`, `_validate_model_name` |
| `entities.py` | ~160 | Extract PERSON entities from text via spaCy | `extract_person_entities`, `_extract_entities_from_doc`, `_extract_entities_from_segments` |
| `normalization.py` | ~80 | Name sanitization and validation | `_sanitize_person_name`, `_validate_person_entity`, `_extract_confidence_score` |
| `patterns.py` | ~160 | Title position analysis, prefix/suffix, pattern fallback | `analyze_episode_patterns`, `_pattern_based_fallback`, `_find_common_patterns` |
| `guests.py` | ~180 | Guest candidate scoring, selection, context-aware filtering | `_build_guest_candidates`, `_select_best_guest`, `_is_likely_actual_guest` |
| `hosts.py` | ~120 | Host detection from feed metadata and transcript intro | `detect_hosts_from_feed`, `detect_hosts_from_transcript_intro` |
| `detection.py` | ~200 | Main orchestration, speaker list building | `detect_speaker_names`, `_build_speaker_names_list` |

**Migration approach (preserving backward compatibility):**

```python
# speaker_detection.py — becomes thin re-export wrapper
"""Backward-compatible re-export of speaker detection public API.

All logic has moved to speaker_detectors/ submodules.
This module re-exports the public API for existing imports.
"""
from ..speaker_detectors.constants import DEFAULT_SPEAKER_NAMES
from ..speaker_detectors.detection import detect_speaker_names
from ..speaker_detectors.entities import extract_person_entities
from ..speaker_detectors.guests import _has_guest_intent_cue
from ..speaker_detectors.hosts import detect_hosts_from_feed
from ..speaker_detectors.ner import get_ner_model
from ..speaker_detectors.normalization import (
    filter_default_speaker_names,
    is_default_speaker_name,
)

__all__ = [
    "detect_speaker_names",
    "detect_hosts_from_feed",
    "get_ner_model",
    "extract_person_entities",
    "DEFAULT_SPEAKER_NAMES",
    "is_default_speaker_name",
    "filter_default_speaker_names",
    "_has_guest_intent_cue",
]
```

**Phased extraction order:**

1. `constants.py` — no dependencies, pure values
2. `normalization.py` — depends only on constants
3. `ner.py` — depends on config, constants
4. `entities.py` — depends on normalization, constants
5. `patterns.py` — depends on entities, constants
6. `guests.py` — depends on entities, patterns, constants
7. `hosts.py` — depends on entities, constants
8. `detection.py` — depends on guests, hosts, entities, normalization, constants

After each phase: run `make ci-fast` to verify zero regressions.

### 2. Test Audio Voice Improvements (Issue #111)

**Speaker-to-voice mapping:**

```python
SPEAKER_VOICE_MAP: dict[str, str] = {
    # Hosts (varied accents)
    "Maya": "Samantha",       # en_US female
    "Ethan": "Alex",          # en_US male
    "Rina": "Karen",          # en_AU female
    "Leo": "Daniel",          # en_GB male
    "Nora": "Moira",          # en_IE female
    "Alex": "Evan",           # en_US male (p07-p09)

    # Guests (varied accents for speaker distinction)
    "Liam": "Fred",           # en_US male
    "Sophie": "Flo",          # en_GB female
    "Noah": "Tom",            # en_US male
    "Priya": "Isha",          # en_IN female
    "Jonas": "Eddy",          # en_US male (distinct character)
    "Camila": "Paulina",      # es_MX female
    "Marco": "Luca",          # it_IT male
    "Hanna": "Anna",          # de_DE female
    "Owen": "Reed",           # en_US male
    "Ava": "Kathy",           # en_US female
    "Tariq": "Rishi",         # en_IN male
    "Elise": "Amelie",        # fr_CA female
    "Daniel": "Oliver",       # en_GB male
    "Isabel": "Monica",       # es_ES female
    "Kasper": "Ralph",        # en_US male
}
```

**Fallback strategy:**

```python
# Hash-based fallback for unmapped speakers
FALLBACK_VOICES = ["Albert", "Bruce", "Junior", "Nicky", "Ralph", "Shelley", "Trinoids"]

def get_voice_for_speaker(speaker_name: str, is_host: bool) -> str:
    if speaker_name in SPEAKER_VOICE_MAP:
        return SPEAKER_VOICE_MAP[speaker_name]
    # Deterministic hash-based selection
    idx = hash(speaker_name) % len(FALLBACK_VOICES)
    return FALLBACK_VOICES[idx]
```

**Changes to `transcripts_to_mp3.py`:**

1. Replace binary host/guest voice logic with per-speaker voice lookup
2. Parse `Speaker: text` lines to extract actual speaker name (not just host vs. non-host)
3. Each unique speaker name maps to a unique voice via `SPEAKER_VOICE_MAP`
4. Keep `--host-voice` and `--guest-voice` CLI args as fallback defaults
5. Add `--list-voices` to show available macOS voices

### 3. Commercial Segments in Transcripts (Issue #109)

**Placement per episode:**

1. **Opening ad** — After intro greeting, before main content (~30-60s)
2. **Mid-roll ad** — At ~50% through content, natural conversation break (~30-60s)
3. **Closing ad** — Before outro/signoff (~30-60s)

**Template for host-read ads:**

```text
Maya: This episode is brought to you by Stripe. Stripe makes it easy to accept
payments online and in person. Whether you're building a marketplace or a
subscription service, Stripe handles the complexity. Get started at
stripe.com/podcast. That's stripe.com/podcast.
```

**Template for mid-roll break:**

```text
Maya: We'll be right back after a quick word from our sponsors.

Ad: Today's episode is sponsored by Linear. Linear is the issue tracker built
for speed. With keyboard shortcuts, powerful search, and beautiful design,
Linear makes project management effortless. Try Linear free at
linear.app/podcast.

Maya: Welcome back. We were just discussing...
```

**Brand distribution across episodes:**

- Use well-known podcast sponsor brands: Figma, Stripe, Linear, Notion, Vanta, Miro, Zapier, HubSpot, Squarespace, Shopify
- Each episode gets 3 different brands (opening, mid-roll, closing)
- No brand repeats within an episode; brands can repeat across episodes
- Commercial content uses phrases matching existing `SPONSOR_PATTERNS` in `preprocessing.py`

**Speaker label for ads:**

- Host-read ads use the host's speaker label (e.g., `Maya:`)
- Pre-recorded ads use `Ad:` label — this exercises multi-speaker parsing
- `transcripts_to_mp3.py` needs an `Ad` voice entry (e.g., `"Ad": "Zarvox"` for distinctly synthetic)

## Key Decisions

1. **Speaker detection: modularize, don't rewrite**
   - **Decision**: Extract functions into submodules with identical behavior; `speaker_detection.py` becomes a re-export wrapper
   - **Rationale**: Zero behavior change reduces risk; existing 245+ tests validate correctness through the refactor

2. **Voice mapping: explicit over hash-based**
   - **Decision**: Curated `SPEAKER_VOICE_MAP` with hash-based fallback for unknown speakers
   - **Rationale**: Explicit mapping ensures voice quality and deterministic, reviewable assignments; fallback handles future fixture additions

3. **Commercial segments: host-read style**
   - **Decision**: Most ads are host-read (using host speaker label); one mid-roll uses `Ad:` label
   - **Rationale**: Matches real podcast patterns; exercises existing cleaning detection; `Ad:` label tests multi-speaker parsing

## Alternatives Considered

1. **Rewrite speaker detection from scratch**
   - **Why Rejected**: Too risky; existing logic is well-tested and functional. Modularization preserves behavior while improving structure.

2. **Use pyttsx3 instead of macOS `say` for test audio**
   - **Why Rejected**: pyttsx3 voice variety is limited and platform-dependent; macOS `say` has 20+ distinct English voices with accent variety. Test audio generation is already macOS-only.

3. **Generate fixture audio in CI (Linux)**
   - **Why Rejected**: Linux TTS options (espeak) have lower quality; fixture audio should be pre-generated and stable. Regeneration is an infrequent developer operation.

## Testing Strategy

**Speaker detection refactor:**

- All existing tests in `tests/unit/podcast_scraper/test_speaker_detection.py` must pass unchanged
- All integration tests in `tests/integration/providers/ollama/test_*_speaker.py` must pass
- After refactor, add per-module unit tests in `tests/unit/podcast_scraper/speaker_detectors/`
- Run `make ci-fast` after each extraction phase

**Test audio improvements:**

- Regenerate all fixture audio with unique voices: `cd tests/fixtures/scripts && bash generate_audio.sh`
- Manual verification: listen to samples to confirm voice distinction
- If RFC-058 diarization is implemented, run pyannote on fixture audio to verify it can distinguish speakers

**Commercial segments:**

- After adding commercials, verify `remove_sponsor_blocks()` correctly removes commercial segments
- Validate with the evolving eval tooling that cleaning quality improves
- Check that summaries generated from cleaned transcripts exclude sponsor content

**Test Organization:**

```text
tests/
├── unit/podcast_scraper/
│   └── speaker_detectors/       # NEW — per-module unit tests
│       ├── test_constants.py
│       ├── test_ner.py
│       ├── test_entities.py
│       ├── test_normalization.py
│       ├── test_patterns.py
│       ├── test_guests.py
│       ├── test_hosts.py
│       └── test_detection.py
```

## Rollout & Monitoring

**Rollout Plan:**

- **Phase 1** (~6h): Speaker detection modularization (Issue #269)
  - Extract modules one at a time; `make ci-fast` after each
  - PR with zero-diff behavior guarantee
- **Phase 2** (~4h): Test audio voice improvements (Issue #111)
  - Update `transcripts_to_mp3.py` with voice map
  - Regenerate fixture audio
  - Verify voice distinction
- **Phase 3** (~4h): Commercial segments (Issue #109)
  - Add sponsor blocks to all 15 transcripts
  - Regenerate audio
  - Validate cleaning with evolving eval tooling

**Success Criteria:**

1. All existing tests pass after speaker detection refactor (zero behavior change)
2. Each fixture speaker has a distinct TTS voice (verified by listening)
3. Commercial segments are detectable by the cleaning pipeline and validated via eval tooling

## Relationship to Other RFCs

This RFC (RFC-059) is the companion to RFC-058 (Audio-Based Speaker Diarization):

1. **RFC-058**: Adds the diarization *capability* (pyannote integration)
2. **RFC-059** (this): Improves the *infrastructure* that diarization depends on

**Dependencies:**

- RFC-058's diarization name-mapping integrates into the modularized `speaker_detectors/detection.py`
- RFC-058's integration tests depend on unique-voice test audio from this RFC

## Benefits

1. **Maintainable speaker detection**: 7 focused modules instead of 1 monolith
2. **Testable in isolation**: Each concern can be unit-tested independently
3. **Diarization-ready**: Clean extension points for RFC-058 name-mapping
4. **Realistic test fixtures**: Unique voices enable meaningful diarization testing
5. **Cleaning validation**: Commercial segments enable `remove_sponsor_blocks()` testing

## Migration Path

1. **Speaker detection**: Existing imports from `speaker_detection.py` continue to work (re-export wrapper). No external API changes.
2. **Test audio**: Regenerate fixtures; Git LFS handles the audio file updates. No test code changes (same filenames).
3. **Commercial segments**: Transcript files updated in-place; tests that check transcript content may need minor updates.

## Open Questions

1. Should the refactored speaker detection modules live in `providers/ml/speaker_detection/` (closer to current location) or `speaker_detectors/` (current plan, already has `base.py` and `factory.py`)?
2. Should voice assignments in `SPEAKER_VOICE_MAP` try to match speaker name demographics (e.g., Priya → Indian accent)?
3. Should commercial segments use exact timestamps or relative positions?

## References

- **Issue #269**: [Refactor `speaker_detection.py` into `speaker_detectors/` submodules](https://github.com/chipi/podcast_scraper/issues/269)
- **Issue #111**: [Use unique voices for each speaker in generated mock podcast audio](https://github.com/chipi/podcast_scraper/issues/111)
- **Issue #109**: [Add commercial segments to mock podcast transcripts](https://github.com/chipi/podcast_scraper/issues/109)
- **Issue #414**: [Audio Pipeline Separation](https://github.com/chipi/podcast_scraper/issues/414)
- **Source Code**: `src/podcast_scraper/providers/ml/speaker_detection.py`
- **Source Code**: `tests/fixtures/scripts/transcripts_to_mp3.py`
- **Fixture Spec**: `tests/fixtures/FIXTURES_SPEC.md`
