# Audio Pipeline Guide

Operator and developer reference for **transcription**, **speaker attribution**, **API audio
chunking**, **commercial/sponsor cleaning**, and related provider paths. Covers **Audio Wave 1**
(merged PR #850) and **Wave 2** (diarization + commercial Phase 2).

For strategic architecture see [Architecture](../architecture/ARCHITECTURE.md). For every config
field see [CONFIGURATION.md](../api/CONFIGURATION.md). For CLI flags see [CLI.md](../api/CLI.md).

---

## Wave summary

| Wave | Issues | What shipped |
| ---- | ------ | ------------ |
| **Wave 1** (#850) | #269, #286, #486, #19, #597 | `speaker_detectors/` refactor; `AudioChunker` for oversized API files; `CommercialDetector` Phase 1; single Whisper progress bar; Deepgram Nova-3 provider |
| **Wave 2** | #482, #488 | pyannote neural diarization (default on for local Whisper); commercial Phase 2 diarization signals; DGX `tailnet_dgx_whisper` diarize support |

Design specs: [RFC-058](../rfc/RFC-058-audio-speaker-diarization.md) (diarization),
[RFC-060](../rfc/RFC-060-diarization-aware-commercial-cleaning.md) (commercial cleaning),
[RFC-059](../rfc/RFC-059-speaker-detection-refactor-test-audio.md) (speaker detection refactor).

---

## Transcription providers

| Provider | Config value | When to use |
| -------- | ------------- | ----------- |
| Local Whisper | `whisper` | Default local path; supports screenplay + neural diarization |
| DGX Whisper (tailnet) | `tailnet_dgx_whisper` | Prod primary on DGX with cloud fallback ([ADR-096](../adr/ADR-096-dgx-spark-prod-primary-with-fallback.md)); same diarize/screenplay rules as local Whisper |
| OpenAI | `openai` | Cloud STT (`whisper-1`); plain text output; no local diarization |
| Gemini | `gemini` | Cloud STT |
| Mistral | `mistral` | Cloud STT |
| Deepgram | `deepgram` | Nova-3 with built-in utterance diarization in API response (#597); requires `DEEPGRAM_API_KEY` / `[llm]` extra |

**Screenplay + neural diarization** apply only to **`whisper`** and **`tailnet_dgx_whisper`**. Other
providers coerce `screenplay: false` and `diarize: false` at config validation.

---

## Speaker detection (#269)

NER and heuristic speaker name detection live under `src/podcast_scraper/speaker_detectors/`
(`constants`, `normalization`, `ner`, `entities`, `patterns`, `guests`, `hosts`, `detection`).
`providers/ml/speaker_detection.py` remains a thin re-export facade for backward compatibility.

Detected host/guest names are mapped onto screenplay speaker labels (gap-based or diarized).

---

## Screenplay formatting and diarization (#482)

### Gap-based (fallback)

When `diarize` is off or pyannote fails, `format_screenplay_from_segments()` rotates speakers on
silence gaps (`screenplay_gap_s`). This is **not** voice identity — same speaker can be split across
labels in rapid exchanges.

### Neural diarization (default for local Whisper)

After Whisper transcription, an optional **pyannote.audio** second pass assigns speaker IDs from
voice embeddings, aligned to Whisper segments by maximum overlap.

| Setting | Default | Notes |
| ------- | ------- | ----- |
| `diarize` | `true` | Coerced to `false` for API transcription providers |
| `screenplay` | auto-enabled when `diarize=true` on Whisper paths | Explicit `screenplay: false` still respected |
| `hf_token` | from `HF_TOKEN` env or `~/.huggingface/token` | Required for gated pyannote models |
| `diarization_num_speakers` | auto | Set when speaker count is known |
| `diarization_min_speakers` | `2` | Auto-detect floor |
| `diarization_max_speakers` | `20` | Auto-detect ceiling |
| `diarization_device` | `auto` | `cpu`, `cuda`, or `mps` |
| `diarization_model` | `pyannote/speaker-diarization-3.1` | HuggingFace pipeline id |

**Install:** `pip install -e ".[ml]"` (pyannote + torchaudio bundled in `[ml]`; pinned in `[dev]` for
CI). Lazy-imported — package loads without pyannote when diarization is off.

**CLI:** `--diarize`, `--no-diarize`, `--hf-token`, `--diarization-num-speakers`, etc.

**Module:** `src/podcast_scraper/providers/ml/diarization/`

**Cache:** Results are stored under `<output_dir>/.cache/diarization/` keyed by audio hash and
diarization config fingerprint. Re-runs skip pyannote when a valid cache entry exists.

On failure (missing token, import error, runtime error), the pipeline logs a warning and falls back
to gap-based screenplay.

---

## API audio chunking (#286)

Cloud transcription providers enforce upload size limits. **`AudioChunker`**
(`preprocessing/audio/chunker.py`) splits oversized files with ffmpeg stream-copy, transcribes each
chunk, and merges text with overlap deduplication.

Triggered from `workflow/episode_processor.py` when post-preprocess audio still exceeds the provider
cap (OpenAI, Gemini, Mistral, Deepgram paths). Local Whisper is not chunked by this mechanism.

---

## Commercial / sponsor cleaning

### Phase 1 (#486, Wave 1)

`CommercialDetector` (`cleaning/commercial/`) replaces the old four-phrase `remove_sponsor_blocks`
heuristic with confidence-scored **text patterns** + **positional heuristics** (intro / mid-roll /
outro clusters). `preprocessing/core.py` delegates sponsor removal; duplicate logic removed from
`summarizer.py`.

Works on **transcript text** — benefits all transcription providers.

### Phase 2 (#488, Wave 2)

When diarization segments are available, optional signals (`diarization_signals.py`) adjust
confidence: host monologue boost, guest disqualify, duration/topic hints. Wired through
`CommercialDetector` when callers pass diarization metadata. Summarization cleaning loads sibling
`.segments.json` and infers a host speaker id when pyannote labels are present.

---

## Deepgram (#597)

```bash
export DEEPGRAM_API_KEY=your-key
python -m podcast_scraper.cli https://example.com/feed.xml \
  --transcription-provider deepgram \
  --deepgram-model nova-3
```

Config: `transcription_provider: deepgram`, `deepgram_api_key`, `deepgram_model` (default `nova-3`).
SDK ships in **`[llm]`** extra (`deepgram-sdk`).

Deepgram returns utterance-level speaker labels in the API response — separate from local pyannote
diarization.

---

## DGX prod profile

[`cloud_with_dgx_whisper_primary.yaml`](https://github.com/chipi/podcast_scraper/blob/main/config/profiles/cloud_with_dgx_whisper_primary.yaml)
sets `transcription.primary: tailnet_dgx_whisper` with OpenAI fallback, plus `screenplay: true` and
`diarize: true`. Diarization runs on the machine that holds the audio file after transcription
(local pipeline host), not on the remote DGX Whisper HTTP service unless you colocate processing.

See [DGX Runbook](DGX_RUNBOOK.md) for tailnet and validation steps.

---

## Deployment profiles (diarize / screenplay)

Local and DGX deployment profiles under `config/profiles/` set **`screenplay: true`** and
**`diarize: true`** by default. Cloud API transcription profiles (`cloud_balanced`, `cloud_quality`,
`cloud_thin`) rely on config coercion — `diarize` and `screenplay` are turned off because those
providers emit plain text.

Details: [Profiles README](https://github.com/chipi/podcast_scraper/blob/main/config/profiles/README.md) in the repository.

---

## Progress indicator (#19)

Per-episode nested Whisper `"Transcribing"` progress was removed. The batch-level transcription bar
in the processing stage is the sole progress indicator during local Whisper runs.

---

## Troubleshooting quick reference

| Symptom | Likely cause | Action |
| ------- | ------------- | ------ |
| Diarization skipped with warning | No `HF_TOKEN` / model terms not accepted | Export `HF_TOKEN`; accept terms at HuggingFace for `pyannote/speaker-diarization-3.1` |
| `ProviderDependencyError` for pyannote | `[ml]` not installed | `pip install -e ".[ml]"` |
| `diarize` false despite YAML `true` | API transcription provider | Expected — only local Whisper paths diarize |
| Deepgram validation error | Missing API key | Set `DEEPGRAM_API_KEY` |
| Oversize API upload still fails | File exceeds chunk merge limits | Check ffmpeg; see chunker logs; lower preprocessing bitrate |

Full guide: [Troubleshooting](TROUBLESHOOTING.md).

---

## Related documents

- [DEPENDENCIES_GUIDE.md](DEPENDENCIES_GUIDE.md) — `[ml]` / `[dev]` pyannote pins
- [PREPROCESSING_PROFILES_GUIDE.md](PREPROCESSING_PROFILES_GUIDE.md) — text cleaning profiles (includes sponsor step)
- [AI Provider Comparison Guide](AI_PROVIDER_COMPARISON_GUIDE.md) — provider matrix
- [ADR-058](../adr/ADR-058-additive-pyannote-diarization-with-separate-extra.md) — diarization decision + amendment
- [ADR-059](../adr/ADR-059-confidence-scored-multi-signal-commercial-detection.md) — commercial detection decision
