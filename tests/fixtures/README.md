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

There are **5 synthetic podcasts**, each with **3 episodes**.

RSS feeds (relative links):

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

---

## Naming and mapping conventions (critical)

All assets use a **shared deterministic ID**:
pXX_eYY

Where:

- `pXX` = podcast ID (`p01` .. `p05`)
- `eYY` = episode ID (`e01` .. `e03`)

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
