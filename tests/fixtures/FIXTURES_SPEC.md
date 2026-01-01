# Fixture Specification (Authoritative)

This document describes **how the podcast fixtures were generated** and
serves as the **single source of truth** for regeneration.

If fixtures need to be regenerated, updated, or extended,
this file should be used as the input specification.

---

## Podcasts

There are **5 synthetic podcasts**, each with **3 episodes**.

Naming conventions:

- Podcast IDs: `p01` .. `p05`
- Episode IDs: `e01` .. `e03`
- Combined ID / GUID: `pXX_eYY`
- Filenames must match IDs exactly

---

### p01 — Singletrack Sessions

- Topic: Mountain biking
- Host: Maya

Episodes:

- e01 — *Building Trails That Last* (guest: Liam) — medium
- e02 — *Enduro Skills Without the Hype* (guest: Sophie) — medium
- e03 — *The Mechanics of a Quiet, Fast Bike* (guest: Noah) — long

---

### p02 — Practical Systems

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

## Transcript Design

- All transcripts are **synthetic**
- Format: plain text
- Speaker labels are required (e.g. `Maya:`, `Noah:`)
- Optional timestamps like `[00:00]` may appear
- Stage directions may appear as `*Short break.*`

Target spoken duration (approx):

- medium episodes: ~11 minutes
- long episodes: ~33 minutes

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

---

## Audio Generation Scripts

- `fixtures/scripts/transcripts_to_mp3.py` — main generator
- `fixtures/scripts/generate_audio.sh` — wrapper for default execution

Audio output:

- `audio/pXX_eYY.mp3`

---

## Fast Test Fixtures

For fast test execution, minimal fixtures are available to reduce test runtime:

### Fast Episode (p01_e01_fast)

- **RSS Feed**: `rss/p01_fast.xml` - Single episode with 1-minute duration
- **Transcript**: `transcripts/p01_e01_fast.txt` - First ~1 minute of p01_e01 transcript
- **Audio**: `audio/p01_e01_fast.mp3` - 60-second audio file (469 KB)

**Purpose**: Reduce E2E-fast test execution time by ~75-85% (from ~3-4 minutes to ~30-45 seconds).

**Generation**: Fast audio was created by extracting the first 60 seconds of `p01_e01.mp3` using ffmpeg:

```bash
ffmpeg -i audio/p01_e01.mp3 -t 60 -c copy audio/p01_e01_fast.mp3
```yaml

---

## Git & Storage

- Audio files may be large
- Git LFS is recommended for `audio/*.mp3`
- Transcripts and RSS files are normal git-tracked text
