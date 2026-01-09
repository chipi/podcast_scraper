# Offline Podcast Fixtures (RSS + Transcripts + Audio)

This directory contains **synthetic podcast fixtures** used for **offline,
deterministic end-to-end testing**.

These fixtures are intentionally designed to:

- avoid hitting real RSS feeds or audio hosting services
- mimic realistic podcast workloads (long audio, transcripts, metadata)
- be safe to commit to git
- be repeatable and regenerable if needed

These fixtures are **test infrastructure**, not editorial content.

---

## Folder layout

tests/fixtures/
├─ README.md              # this file (overview & navigation)
├─ FIXTURES_SPEC.md       # authoritative spec + metadata (source of truth)
├─ VERIFICATION.md        # how to verify fixtures after generation
├─ rss/                   # RSS2 + iTunes XML feeds
├─ transcripts/           # synthetic transcripts (input to audio generation)
├─ scripts/               # audio generation scripts
└─ audio/                 # generated MP3 files

---

## Podcasts and RSS feeds

There are **9 synthetic podcasts** with varying episode counts:

### Standard Podcasts (p01-p05)

**5 podcasts**, each with **3 episodes** (~10-35 minutes each):

- **p01 — Mountain biking**
  RSS: [rss/p01_mtb.xml](./rss/p01_mtb.xml)

- **p02 — Software engineering**
  RSS: [rss/p02_software.xml](./rss/p02_software.xml)

- **p03 — Scuba diving**
  RSS: [rss/p03_scuba.xml](./rss/p03_scuba.xml)

- **p04 — Photography**
  RSS: [rss/p04_photo.xml](./rss/p04_photo.xml)

- **p05 — Investing**
  RSS: [rss/p05_investing.xml](./rss/p05_investing.xml)

### Long-Form Podcasts (p06-p09)

**4 podcasts** designed for **long-context testing** and **threshold boundary scenarios**:

- **p06 — Edge Cases**
  RSS: [rss/p06_edge_cases.xml](./rss/p06_edge_cases.xml)

  6 episodes for testing edge cases

- **p07 — The Long View: Sustainability**
  RSS: [rss/p07_sustainability.xml](./rss/p07_sustainability.xml)

  1 extra-long episode (~1h 45m, 14,471 words)

- **p08 — The Long View: Solar Energy**
  RSS: [rss/p08_solar.xml](./rss/p08_solar.xml)

  1 extra-long episode (~2h 21m, 19,251 words)

- **p09 — The Long View: Biohacking**
  RSS: [rss/p09_biohacking.xml](./rss/p09_biohacking.xml)

  3 long episodes (~20 minutes each, ~3,000 words each)

---

## Naming and mapping conventions (critical)

All assets use a **shared deterministic ID**:
pXX_eYY

Where:

- `pXX` = podcast ID (`p01` .. `p09`)
- `eYY` = episode ID (`e01` .. `e06`)

This ID is used consistently across:

- RSS `<guid>`
- transcript filename
- audio filename

Example:
p02_e03
├─ RSS guid: p02_e03
├─ Transcript: transcripts/p02_e03.txt
└─ Audio: audio/p02_e03.mp3

---

## Episodes overview

| Podcast | RSS file | Episodes (GUID / transcript / audio basename) |
| ------ | ------- | ---------------------------------------------- |
| p01 | p01_mtb.xml | p01_e01, p01_e02, p01_e03 |
| p02 | p02_software.xml | p02_e01, p02_e02, p02_e03 |
| p03 | p03_scuba.xml | p03_e01, p03_e02, p03_e03 |
| p04 | p04_photo.xml | p04_e01, p04_e02, p04_e03 |
| p05 | p05_investing.xml | p05_e01, p05_e02, p05_e03 |
| p06 | p06_edge_cases.xml | p06_e01, p06_e02, p06_e03, p06_e04, p06_e05, p06_e06 |
| p07 | p07_sustainability.xml | p07_e01 |
| p08 | p08_solar.xml | p08_e01 |
| p09 | p09_biohacking.xml | p09_e01, p09_e02, p09_e03 |

---

## Transcripts

Location:
tests/fixtures/transcripts/

Properties:

- plain text (`.txt`)
- fully **synthetic**
- conversational format
- includes speaker labels (e.g. `Maya:`, `Ethan:`)
- may include timestamps like `[00:00]`
- may include stage directions like `*Short break.*`

Target spoken duration:

- `e01`, `e02`: ~10–12 minutes (medium)
- `e03`: ~30–35 minutes (long)
- Long-form podcasts (p07-p09): ~20 minutes to 2+ hours

Actual duration depends on TTS voice and rate.

---

## Audio files

Location:
audio/

Files:
audio/pXX_eYY.mp3

Properties:

- generated locally from transcripts
- mono MP3
- realistic duration and file size
- suitable for streaming, partial reads, and timeout testing

### Fast Test Fixtures

For fast test execution, minimal fixtures are available:

- **Fast RSS Feed**: `rss/p01_fast.xml` - Single episode with 1-minute duration
- **Fast Transcript**: `transcripts/p01_e01_fast.txt` - First ~1 minute of p01_e01 transcript
- **Fast Audio**: `audio/p01_e01_fast.mp3` - 60-second audio file (469 KB vs 5.8 MB for full episode)

**Purpose**: Reduce E2E-fast test execution time from ~180-240 seconds to ~30-45 seconds by
using shorter audio files.

**Usage**: Automatically used by E2E-fast tests when `_allowed_podcasts` is set (fast mode
detection). Integration-fast tests use `/feed-no-transcript.xml` which serves the fast audio file.

**Generation**: Fast audio was created by extracting the first 60 seconds of `p01_e01.mp3` using ffmpeg.

### Multi-Episode Test Fixtures

For testing multi-episode processing logic, multi-episode test fixtures are available:

- **Multi-Episode RSS Feed**: `rss/p01_multi.xml` - 5 episodes with 10-15 second duration each
- **Multi-Episode Transcripts**: `transcripts/p01_multi_e01.txt` through `p01_multi_e05.txt` -
  Short transcripts (2-3 sentences each)

- **Multi-Episode Audio**: `audio/p01_multi_e01.mp3` through `p01_multi_e05.mp3` - Very short
  audio files (~50-100 KB each)

**Purpose**: Test episode iteration/looping logic, concurrent processing, job queues, and error
handling across multiple episodes without the overhead of long audio files. Total processing time:
~2-3 minutes for 5 episodes.

**Usage**: Used by E2E tests for multi-episode processing. These tests can process multiple
episodes (5 episodes) to validate multi-episode processing logic.

**Test Strategy**:

- **Fast Feed** (`p01_fast.xml`): 1 episode, 1 minute - for regular E2E tests (code quality)
- **Multi-Episode Feed** (`p01_multi.xml`): 5 episodes, 10-15 seconds each - for multi-episode processing tests
- **Original Mock Data** (`p01_mtb.xml`, etc.): Full episodes - for nightly data quality tests
- **Long-Form Fixtures** (`p07`, `p08`, `p09`): Extra-long episodes (1-2+ hours) - for threshold boundary testing and long-context model evaluation (related to issue #283)

### Long-Form Test Fixtures (Threshold Boundary Testing)

For testing summarization threshold boundaries and long-context model behavior:

- **p07 - Sustainability**: 1 episode, **14,471 words**, ~1h 45m
  - Purpose: Test 3-4k combined summary token threshold (issue #283)
  - Expected: ~22 chunks → ~3,300 combined summary tokens

- **p08 - Solar Energy**: 1 episode, **19,251 words**, ~2h 21m
  - Purpose: Test >4k combined summary token threshold
  - Expected: ~30 chunks → ~4,500 combined summary tokens

- **p09 - Biohacking**: 3 episodes, **~3,000 words each**, ~20 min each
  - Purpose: Test medium-to-long episode processing
  - Expected: ~4-5 chunks per episode → ~600-750 combined tokens each

**Use cases:**
- Validate LED model vs BART/PEGASUS threshold behavior
- Test hierarchical reduce vs extractive fallback decision logic
- Verify compression ratio warnings are model-specific
- Ensure consistent quality across varying episode lengths

---

## Audio generation

Audio is generated **offline** using macOS built-in TTS.

### Engine

- Platform: macOS
- TTS engine: `say`
- No online services used

### Voices

- **Host voice:** Samantha
- **Guest voice:** Daniel

### Parameters

- Speech rate: `145`
- MP3 bitrate: `64k`
- Channels: mono

### Voice assignment logic

- Host name is inferred from filename prefix:
  - `p01_*` → Maya
  - `p02_*` → Ethan
  - `p03_*` → Rina
  - `p04_*` → Leo
  - `p05_*` → Nora
  - `p06_*` → TBD (edge cases)
  - `p07_*` → Alex (The Long View - Sustainability)
  - `p08_*` → Alex (The Long View - Solar Energy)
  - `p09_*` → Alex (The Long View - Biohacking)
- When a transcript line speaker matches the host name → host voice
- All other speakers → guest voice
- Timestamps, headings, and stage directions are removed from speech

---

## Generation scripts

Scripts:

- `fixtures/scripts/transcripts_to_mp3.py` — main generator
- `fixtures/scripts/generate_audio.sh` — wrapper with defaults

Generate all audio (from repo root):

```bash
chmod +x tests/fixtures/scripts/generate_audio.sh
./tests/fixtures/scripts/generate_audio.sh
```