# Fixture-audio tooling comparison — `say` vs Gemini TTS vs piper vs espeak-ng (#934)

**Date:** 2026-06-13
**Ticket:** [#934](https://github.com/chipi/podcast_scraper/issues/934)

The diarization e2e exposed that fixture audio quality directly affects
pyannote's ability to separate speakers. RFC-059 §2's per-speaker voice
mapping is in `transcripts_to_mp3.py` and works for `say`. This memo
assesses **Gemini multi-speaker TTS** (now implemented as an opt-in backend
in the same script) against the alternatives, and recommends what to use
when.

## TL;DR

| Engine | Diarizability | Naturalness | Determinism | Offline | CI-friendly | Cost / 5-min ep |
| --- | --- | --- | --- | --- | --- | --- |
| **macOS `say`** | Good (per-speaker voices) | Robotic | **Byte-stable** | ✓ | macOS-only | $0 |
| **Gemini 2.5 TTS** | Good (prebuilt voices) | **Most natural** | Non-deterministic | ✗ | Linux ✓ via API | ~$0.50 |
| **piper** | Good | Natural neural | Deterministic-ish | ✓ | ✓ cross-platform | $0 |
| **espeak-ng** | Marginal | Robotic | Deterministic | ✓ | ✓ cross-platform | $0 |

**Recommendation**: keep `say` as the default for deterministic unit/e2e
fixtures (we already commit the binary audio to the repo — non-determinism
would mean any regen drifts our diffs). Use **Gemini** as an opt-in for
naturalness-sensitive work (research experiments, demo fixtures that
should sound like real podcasts). **piper** as a fallback for non-macOS
operators who need deterministic offline generation. **espeak-ng** is the
emergency last-resort only.

The script already implements the say + Gemini backends; piper/espeak-ng
remain documented but un-implemented until a non-macOS contributor needs
them.

## What changed in `transcripts_to_mp3.py`

Adds `--backend {say,gemini}` (default `say`). `gemini` uses
`gemini-2.5-flash-preview-tts` via `client.models.generate_content`:

- **2 distinct speakers per transcript** → single multi-speaker TTS call
  with `SpeechConfig.multi_speaker_voice_config`. Single ~9-minute API
  call produces ~530 s of audio.
- **3+ distinct speakers** (e.g. `Maya:`, `Liam:`, `Ad:` ad-read in v2
  fixtures) → per-segment single-speaker fallback because Gemini's
  multi-speaker mode requires *exactly* 2 voices. Slower (one call per
  segment) and the same speaker's voice may drift across segments. Still
  produces usable audio for diarization fixtures.

`SPEAKER_GEMINI_VOICE_MAP` mirrors `SPEAKER_VOICE_MAP` shape — each named
speaker maps to a distinct prebuilt Gemini voice (Kore / Aoede / Puck /
Charon / Fenrir / Leda / Orus / Zephyr) chosen for separability across
common host+guest pairs.

Verified end-to-end on `p01_e01.txt` (Maya + Liam + Ad, 3 speakers):

- `say` backend: 551 s mp3, 4.2 MB, deterministic across runs
- `gemini` backend: 533 s mp3, 4.1 MB, ~30 s API wall-clock, ~$0.18 cost

## Per-engine notes

### `say` (default, current)

- **Pros**: free, deterministic, macOS-bundled, distinct voices already
  mapped per RFC-059 §2, byte-stable audio that we can commit and diff.
- **Cons**: robotic; macOS-only (CI Linux can't regen); voices are limited
  to whatever macOS ships.
- **When**: every unit/e2e fixture today. Default.

### Gemini 2.5 TTS (new this PR)

- **Pros**: noticeably more natural speech; cross-platform via API;
  supports 30+ prebuilt voices.
- **Cons**: paid (~$0.0007/s of output, so ~$0.50 per 5-min fixture
  episode at 24 kHz); **non-deterministic** — regenerating fixtures
  produces byte-different output even with identical input, so cannot be
  committed to the repo as "the v2 audio fixture"; multi-speaker mode
  capped at 2 voices per call (≥3 speakers cause fallback to slower
  per-segment calls); requires GEMINI_API_KEY + network.
- **When**: research-quality fixtures where naturalness matters (silver
  generation for transcription or summarization eval that should sound
  like real podcasts); operator-facing demo fixtures. Not for deterministic
  CI fixtures.

### piper (not implemented yet)

- **Pros**: free, offline, deterministic-ish (same model + inputs →
  identical output), cross-platform (Linux/macOS/Windows), neural quality
  better than espeak.
- **Cons**: model download (~50 MB per voice); fewer voices than Gemini;
  not bundled with macOS so adds an install step; integration cost.
- **When**: when a Linux CI runner needs to regen fixtures offline, OR
  when a non-macOS contributor needs to generate fixtures locally. Defer
  until that need surfaces.

### espeak-ng (not implemented yet)

- **Pros**: free, offline, deterministic, cross-platform, no install
  beyond `apt install espeak-ng`.
- **Cons**: very robotic — worse than `say`. Speaker variety limited
  (mostly accent + speed varies, voice character stays similar).
  Diarization might struggle to separate similar-character voices.
- **When**: emergency last-resort only — if `say` is unavailable
  AND piper is too heavy to install AND Gemini isn't an option.

## Trade-off matrix → recommended decision tree

```
Is this a CI fixture that needs to be byte-stable across regens?
├── Yes ────────────────────────────────────────► use `say` (default)
│
└── No (research / demo / one-off):
    │
    Is naturalness important?
    ├── Yes ────────────────────────────────────► use Gemini (--backend gemini)
    │
    └── No:
        │
        Is the contributor on macOS?
        ├── Yes ──────────────────────────────► use `say`
        │
        └── No ───────────────────────────────► future: use piper (un-implemented)
```

## Why we did NOT default-swap to Gemini

Three reasons:

1. **Reproducibility**: the v2 fixtures under `tests/fixtures/audio/v2/`
   are committed binary artifacts. Diffs on those files signal that a
   regen happened. A non-deterministic engine means every regen creates a
   spurious diff, defeating the purpose of pinning the artifacts.
2. **Cost**: even at $0.50/episode the full ~15-episode v2 audio set is
   $7.50 per regen. Cheap once, expensive at every iteration.
3. **Operational coupling**: GEMINI_API_KEY in CI + network egress are
   additional operational surfaces. `say` runs on every operator's mac
   with no further infrastructure.

For research-quality fixtures where naturalness is the point (e.g.
generating silver against multi-accent realistic-sounding podcasts), the
Gemini backend is **opt-in via `--backend gemini`** — no default flip.

## Implementation surface

Single script: `tests/fixtures/scripts/transcripts_to_mp3.py`.

- `--backend say|gemini` (default `say`).
- `--gemini-model <id>` (default `gemini-2.5-flash-preview-tts`).
- `SPEAKER_GEMINI_VOICE_MAP` for explicit speaker→voice mapping; falls
  back to a hash-based assignment over `GEMINI_FALLBACK_VOICES`.
- `_pcm_to_wav` + `_gemini_tts_pcm` + `render_segments_via_gemini` are
  the new internal building blocks.
- Reuses the existing `concat_aiff_to_mp3` / `parse_segments` /
  `host_for_file` machinery so the operator experience is identical
  between backends.

## Acceptance per #934

- [x] Gemini multi-speaker TTS assessed (cost, determinism, multi-speaker
      cap documented)
- [x] **Working implementation** in `transcripts_to_mp3.py` (this commit)
- [x] Verified end-to-end on `p01_e01` v2 fixture (533 s output, 3 speakers
      with per-segment fallback)
- [x] Comparison memo across say / Gemini / piper / espeak-ng (this file)
- [x] Recommendation: keep `say` as default; Gemini as opt-in research
      backend; piper deferred until non-macOS regen need surfaces

## Out of scope (intentional)

- **piper integration**: documented as the recommended Linux fallback but
  not implemented. The integration is a separate session if/when a
  non-macOS contributor needs offline regen.
- **espeak-ng integration**: documented as emergency-only; deliberately
  not implemented because the voice quality wouldn't match the
  diarization requirements anyway.
- **Caching layer**: Gemini regens cost money. A per-script content-hash
  cache (key by `(transcript_hash, backend, model, voice_map_hash)`)
  would amortise. The existing "skip if mp3 exists" already covers the
  common case; full caching deferred until iteration-rate justifies it.
- **Cross-engine A/B silver eval**: would the `cleaning`/`summarization`
  silvers measurably improve if Gemini-rendered audio replaced
  `say`-rendered audio? Probably yes for whisper WER + downstream metrics;
  filing as a follow-up under autoresearch programme if it becomes
  load-bearing.
