# Transcription Autoresearch Plan

Same methodology as v2 summarization eval applied to the transcription stage.
Whisper API is ~90% of pipeline cost (#577). Never evaluated: are we using
the right provider, model, and settings? Is the quality/cost tradeoff optimal?

**Related:** #577 (Whisper cost optimization), #591 (full pipeline validation)

---

## The question

We use OpenAI Whisper API (`whisper-1`) as default. Never compared against:
- Local Whisper (free, various model sizes)
- Other cloud ASR providers (Gemini, Deepgram, AssemblyAI)
- Audio preprocessing (compression, silence removal, downsampling)

Is Whisper API the right choice? Can we get equivalent quality cheaper/faster?

---

## Phase 1: Research — what's available (half day)

### Providers to investigate (researched 2026-04-18)

| Provider | Type | Cost/min | Diarization | Timestamps | Fit |
|----------|------|:--------:|:-----------:|:----------:|:---:|
| **OpenAI Whisper API** (current) | Cloud | $0.006 | No (separate) | Segment | Baseline |
| **Local Whisper** (transformers) | Local | $0 | No | Segment | Already supported; tiny→large-v3 |
| **Deepgram Nova-3** | Cloud | $0.0043 | **Built-in, free** | **Word-level** | **#1 candidate** — 28% cheaper, better diarization |
| **AssemblyAI Nano** | Cloud | $0.0065 | Built-in, free | Word-level | #2 — comparable price, excellent SDK |
| **Gladia** | Cloud | $0.0061 | Built-in | Word-level | #3 — near Whisper price, newer entrant |
| Gemini | Cloud | ~$0.01-0.04 | No API | No timestamps | **Skip** — no structured ASR output |
| Google Chirp 2 | Cloud | $0.016 | Built-in | Word-level | Overpriced for podcast-only |
| Rev.ai | Cloud | $0.02 | Built-in | Word-level | 3× current cost, skip |

**Key insight:** Deepgram Nova-3 is cheaper than Whisper AND has built-in
diarization + word-level timestamps — exactly what GI quote grounding needs.
Would require a new provider implementation (`deepgram_provider.py`).

### What to research per provider

- Supported audio formats + max duration
- Language support (en primary, multilingual nice-to-have)
- Speaker diarization capability (built-in vs separate)
- Word-level timestamps (needed for GI quote grounding)
- Segment-level output format (compatibility with our pipeline)
- Pricing model (per-minute, per-token, free tier)

---

## Phase 2: Silver transcripts + baseline (half day)

### Generate silver references

Use OpenAI Whisper API (`whisper-1`) as silver — it's our current baseline
and generally considered high quality for English podcasts.

- Run on 5 held-out episodes (already have audio?)
- Store as `data/eval/references/silver/silver_whisper1_benchmark_v2/`
- Each reference: full transcript text + segments JSON

**Key question:** do we have the raw audio files for held-out episodes?
If not, need to source them. The materialized transcripts exist but we
need audio to re-transcribe with different providers.

### Baseline measurement

Score current Whisper API output on:
- **WER** (Word Error Rate) against a manual gold transcript (if available)
  or against itself (as reference quality baseline)
- **Downstream quality:** run summary + GI + KG on the transcript, score
  against existing silvers. This measures: does transcript quality affect
  downstream quality?

---

## Phase 3: Provider matrix (1-2 days)

### Local Whisper model sweep

| Model | Params | Speed (est.) | Quality (est.) |
|-------|--------|-------------|----------------|
| tiny | 39M | ~10x real-time | Lowest |
| base | 74M | ~7x real-time | Low |
| small | 244M | ~4x real-time | Medium |
| medium | 769M | ~2x real-time | Good |
| large-v3 | 1.5B | ~0.5x real-time | Best local |

Run each on same 5 episodes. Measure:
- WER against Whisper API silver
- Wall-clock time per episode
- Downstream summary/GI quality (chain test)

### Cloud provider comparison (if new providers worth adding)

- Same 5 episodes through each provider
- Score WER + downstream quality
- Add cost per episode

### Audio preprocessing experiments

| Knob | Options | Expected impact |
|------|---------|----------------|
| Sample rate | 44.1kHz → 16kHz | Whisper works at 16kHz internally; saves upload bandwidth |
| Bitrate | 128kbps → 64kbps | Smaller files, faster upload |
| Silence removal | VAD-based trimming | Less audio = less cost (per-minute pricing) |
| Format | mp3 → opus/ogg | Better compression at same quality |
| Chunking | Full episode vs segments | May help with long episodes |

---

## Phase 4: Downstream chain test

**The critical experiment:** take the best cheap transcript option and measure
whether downstream quality actually drops.

```
Whisper API transcript → summary → GI → KG → bridge
     vs
Local Whisper-medium transcript → summary → GI → KG → bridge
     vs
Compressed audio → Whisper API → summary → GI → KG → bridge
```

If summary/GI/KG quality is unchanged (±2%), the cheaper option wins.
Podcast audio is clean speech — most ASR errors are on names/jargon, which
may not affect summary quality at all.

---

## Phase 5: Optimize + update defaults

Based on findings:
- Update `config_constants.py` transcription defaults
- Document cost/quality tradeoffs in AI Provider Guide
- Add transcription tier to the 4-tier strategy (like summarization has)
- Update #577 with concrete recommendations

---

## Estimated budget

- Local Whisper runs: $0 (GPU time only, ~2-4 hrs for all models × 5 eps)
- Cloud alternatives: ~$5-10 if testing new providers
- Downstream scoring: reuse existing scorers, ~$3-5 in judge API costs
- **Total: under $20**

---

## Connection to other work

- **Must happen AFTER #591** (pipeline validation) — no point optimizing
  transcription if the downstream pipeline has integration bugs
- **Informs #577** (Whisper cost optimization) with data-backed recommendations
- **May unblock local-only deployment** — if local Whisper-medium is "good enough,"
  the pipeline can run fully offline (Whisper + Ollama + local ML)
