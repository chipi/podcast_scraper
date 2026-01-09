# Fixture Specification (Authoritative)

This document describes how the podcast fixtures were generated and
serves as the single source of truth for regeneration.

If fixtures need to be regenerated, updated, or extended,
this file should be used as the input specification.

---

## Podcasts

There are **9 synthetic podcasts** with varying episode counts.

Naming conventions:

- Podcast IDs: `p01` .. `p09`
- Episode IDs: `e01` .. `e06` (varies by podcast)
- Combined ID / GUID: `pXX_eYY`
- Filenames must match IDs exactly

---

# p01 — Singletrack Sessions

- Topic: Mountain biking
- Host: Maya

Episodes:

- e01 — *Building Trails That Last* (guest: Liam) — medium
- e02 — *Enduro Skills Without the Hype* (guest: Sophie) — medium
- e03 — *The Mechanics of a Quiet, Fast Bike* (guest: Noah) — long

---

## p02 — Practical Systems

- Topic: Software engineering
- Host: Ethan

Episodes:

- e01 — *On-Call That Doesn't Break People* (guest: Priya) — medium
- e02 — *Staff Engineer Communication Patterns* (guest: Jonas) — medium
- e03 — *Security as Design, Not a Checklist* (guest: Camila) — long

---

### p03 — Below the Surface

- Topic: Scuba diving
- Host: Rina

Episodes:

- e01 — *Wreck Diving Fundamentals* (guest: Marco) — medium
- e02 — *Marine Biology for Divers* (guest: Hanna) — medium
- e03 — *Calm Under Pressure* (guest: Owen) — long

---

### p04 — Frame & Light

- Topic: Photography
- Host: Leo

Episodes:

- e01 — *Underwater Images That Feel Alive* (guest: Ava) — medium
- e02 — *Documentary Workflow in the Field* (guest: Tariq) — medium
- e03 — *Lighting Decisions That Save a Shoot* (guest: Elise) — long

---

### p05 — Long Horizon Notes

- Topic: Investing
- Host: Nora

Episodes:

- e01 — *Index Investing Without the Myths* (guest: Daniel) — medium
- e02 — *Real Estate: Numbers Before Narratives* (guest: Isabel) — medium
- e03 — *Risk Management for People Who Hate Spreadsheets* (guest: Kasper) — long

---

### p06 — Edge Cases

- Topic: Various edge case scenarios for testing
- Host: TBD

Episodes:

- e01 through e06 — Various edge case episodes

---

## Transcript Design

- All transcripts are **synthetic**
- Format: plain text
- Speaker labels are required (e.g. `Maya:`, `Noah:`)
- Optional timestamps like `[00:00]` may appear
- Stage directions may appear as `*Short break.*`

Target spoken duration (approx):

- medium episodes: ~11 minutes
- long episodes: ~33 minutes
- extra-long episodes (p07-p09): 20 minutes to 2+ hours

Actual duration depends on TTS voice and rate.

---

## Audio Generation

Audio is generated locally from transcripts.

### Engine

- Platform: macOS
- TTS engine: `say`

### Voices

- Host voice: `Samantha`
- Guest voice: `Daniel`

### TTS parameters

- Speech rate: `145`
- MP3 bitrate: `64k`
- Channels: mono

### Voice assignment logic

- Host voice is used when speaker name matches the podcast host
- Guest voice is used for all other speakers
- Host detection is inferred from filename prefix:
  - `p01_*` → Maya
  - `p02_*` → Ethan
  - `p03_*` → Rina
  - `p04_*` → Leo
  - `p05_*` → Nora
  - `p06_*` → TBD (edge cases)
  - `p07_*` → Alex (The Long View - Sustainability)
  - `p08_*` → Alex (The Long View - Solar Energy)
  - `p09_*` → Alex (The Long View - Biohacking)

---

## Audio Generation Scripts

- `fixtures/scripts/transcripts_to_mp3.py` — main generator (supports individual files or directories)
- `fixtures/scripts/generate_audio.sh` — wrapper for default execution

Audio output:

- `audio/pXX_eYY.mp3`

**Generate all audio:**

```bash
./tests/fixtures/scripts/generate_audio.sh
```

**Generate specific files:**

```bash
python3 tests/fixtures/scripts/transcripts_to_mp3.py \
    tests/fixtures/transcripts/p07_e01.txt \
    tests/fixtures/transcripts/p08_e01.txt \
    --overwrite
```yaml

---

Fast Test Fixtures

For fast test execution, minimal fixtures are available to reduce test runtime:

Fast Episode (p01_e01_fast)
	•	RSS Feed: rss/p01_fast.xml - Single episode with 1-minute duration
	•	Transcript: transcripts/p01_e01_fast.txt - First ~1 minute of p01_e01 transcript
	•	Audio: audio/p01_e01_fast.mp3 - 60-second audio file (469 KB)

Purpose: Reduce E2E-fast test execution time by ~75-85% (from ~3-4 minutes to ~30-45 seconds).

Generation: Fast audio was created by extracting the first 60 seconds of p01_e01.mp3 using ffmpeg:

ffmpeg -i audio/p01_e01.mp3 -t 60 -c copy audio/p01_e01_fast.mp3

⸻

### p07 — The Long View: Sustainability

- Topic: Sustainability, systems thinking, long-term societal challenges
- Host: Alex Morgan
- **Purpose:** Long-context testing, threshold boundary scenarios (issue #283)

Episodes:

- e01 — *What Sustainability Really Means (And Why Everyone Is Talking About It)* (guest: Dr. Elena Fischer) — **extra-long**

**Actual specs:**

- **Words:** 14,471 words
- **Duration:** ~1 hour 45 minutes
- **File size:** ~48 MB
- **Expected chunks:** ~22 chunks (at 650 words/chunk)
- **Expected combined summary tokens:** ~3,300 tokens
- **Test scenario:** 3-4k token threshold boundary (hierarchical reduce vs extractive)

Transcript requirements:

- Single episode only
- Plain text transcript
- Speaker labels required (Alex Morgan:, Dr. Elena Fischer:)
- No markdown formatting
- Natural long-form podcast conversation
- Reflective, explanatory tone

Output files:

- Transcript: `transcripts/p07_e01.txt`
- RSS feed: `rss/p07_sustainability.xml`
- Audio: `audio/p07_e01.mp3`

---

### p08 — The Long View: Solar Energy

- Topic: Solar energy and its role in the near future
- Host: Alex Morgan
- **Purpose:** Very long-context testing, >4k token threshold validation

Episodes:

- e01 — *The Role of Solar Energy in the Near Future* (guest: Dr. Rafael Mendes) — **extra-long**

**Actual specs:**

- **Words:** 19,251 words
- **Duration:** ~2 hours 21 minutes
- **File size:** ~65 MB
- **Expected chunks:** ~30 chunks (at 650 words/chunk)
- **Expected combined summary tokens:** ~4,500 tokens
- **Test scenario:** >4k token threshold (extractive fallback validation)

Transcript requirements:

- Single episode only
- Plain text transcript
- Speaker labels required (Alex Morgan:, Dr. Rafael Mendes:)
- No markdown formatting
- Natural long-form podcast conversation
- Analytical and forward-looking tone

Output files:

- Transcript: `transcripts/p08_e01.txt`
- RSS feed: `rss/p08_solar.xml`
- Audio: `audio/p08_e01.mp3`

---

### p09 — The Long View: Biohacking

- Topic: Biohacking and the state of the field in 2025
- Host: Alex Morgan
- Format: Solo host (no guests)
- **Purpose:** Medium-to-long episode testing, multi-episode processing

Episodes:

- e01 — *Biohacking in 2025: From Fringe to Framework* — long
- e02 — *Sleep, Metabolism, and Measurement: What Actually Works* — long
- e03 — *Ethics, Limits, and the Next Decade of Human Optimization* — long

**Actual specs:**

- **Words per episode:** ~3,000 words each (2,964 / 2,937 / 2,954)
- **Duration per episode:** ~20 minutes each
- **File size per episode:** ~9 MB each
- **Expected chunks per episode:** ~4-5 chunks
- **Expected combined summary tokens per episode:** ~600-750 tokens
- **Test scenario:** Medium-length episodes with consistent processing

Transcript requirements:

- Plain text transcripts
- Speaker label only: Alex Morgan:
- No markdown formatting
- Long-form, reflective monologue style

Output files:

- Transcripts:
  - `transcripts/p09_e01.txt`
  - `transcripts/p09_e02.txt`
  - `transcripts/p09_e03.txt`
- RSS feed: `rss/p09_biohacking.xml`
- Audio:
  - `audio/p09_e01.mp3`
  - `audio/p09_e02.mp3`
  - `audio/p09_e03.mp3`

---

LD Fixture Generation Notes
	•	LD fixtures intentionally exceed normal episode lengths
	•	Audio generation MAY:
	•	Be chunked internally for TTS
	•	Use the same voices and parameters as defined above
	•	RSS <itunes:duration> SHOULD reflect full episode length
	•	These fixtures are expected to be slow to generate and large on disk

Purpose:
	•	Evaluate long-context model behavior
	•	Test transcript chunking strategies
	•	Test RSS + audio handling for very large episodes
	•	Benchmark summarization, NER, and topic coherence over long inputs

⸻

## Git & Storage

- Audio files may be large (up to 65 MB for extra-long episodes)
- Git LFS is recommended for `audio/*.mp3`
- Transcripts and RSS files are normal git-tracked text
