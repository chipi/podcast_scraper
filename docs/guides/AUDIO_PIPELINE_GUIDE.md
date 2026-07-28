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
| **Wave 3** | #414, #547 | `pipeline_stage` (`full` / `audio_only` / `enrich_only`); transcript cache fingerprint by STT provider/model; persist episode MP3 under corpus **`media/`**; `GET /api/corpus/media`; viewer transcript dialog `<audio>` with seek to `timestamp_start_ms` (local serve only) |

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
| `diarization_model` | `pyannote/speaker-diarization-community-1` | HuggingFace pipeline id (v4, non-gated; 3.1 fallback) |
| `diarization_clustering_threshold` | `None` | pyannote clustering-threshold override; higher merges → fewer speakers (curbs over-segmentation) |
| `diarization_min_cluster_size` | `None` | Clusters smaller than this (≈12) reassigned to nearest speaker — drops short over-seg fragments |
| `diarization_min_segment_ms` | `None` | Squelch: drop any speaker whose longest segment < this (ms); kills phantom micro-clusters, keeps real cameos. Per-feed overridable |

**Install:** `pip install -e ".[ml]"` (pyannote + torchaudio bundled in `[ml]`; pinned in `[dev]` for
CI). Lazy-imported — package loads without pyannote when diarization is off.

**CLI:** `--diarize`, `--no-diarize`, `--hf-token`, `--diarization-num-speakers`, etc.

**Module:** `src/podcast_scraper/providers/ml/diarization/`

**Cache:** Results are stored under `<output_dir>/.cache/diarization/` keyed by audio hash and
diarization config fingerprint. Re-runs skip pyannote when a valid cache entry exists.

On failure (missing token, import error, runtime error), the pipeline logs a warning and falls back
to gap-based screenplay.

---

## Host/guest resolution flow (end-to-end)

The single most important thing to hold onto: **WHO comes from metadata; WHICH-VOICE comes from the
conversation; the roster is the one place the two fuse** (ADR-110). Transcription runs *before*
diarization runs *before* the roster. Metadata detection (feed + title/description) happens first
and is carried in as constraints — it never reads the audio.

```text
 RSS FETCH  →  per episode: audio URL + METADATA { title, description, feed author }
     │
 ── LEVEL 1: METADATA DETECTION — "WHO" (pre-transcription) ─────────────────────
     ├─ detect_hosts_from_feed (speaker_detectors/hosts.py)
     │     feed states the hosts → known_hosts   (host COUNT here = the roster cap)
     └─ detect_speaker_names   (speaker_detectors/detection.py, NER over TITLE+DESC)
           → detected_guests (corroborated)  +  metadata_named (everyone stated)
     │
 ── LEVEL 2: TRANSCRIPTION  (audio → words) ─────────────────────────────────────
     └─ transcribe_media_to_text (workflow/episode_processor.py)
           → transcript TEXT + time-coded SEGMENTS   (no speaker labels yet)
     │
 ── LEVEL 3: DIARIZATION  (words → WHICH cluster) ───────────────────────────────
     └─ apply_diarization_to_result (providers/ml/diarization/pipeline.py)
           → SPEAKER_00, SPEAKER_01 … + per-voice segments   (anonymous: no name/role)
     │
 ── LEVEL 4: ROSTER RESOLUTION  ★ THE FUSION: metadata × conversation ★ ──────────
     └─ resolve_speaker_roster (providers/ml/diarization/roster.py)
        (a) SELF-INTRO extraction   _self_intro_voice_names()  → voice_intro
              └─ CANONICAL-NAME FUSION: snap ASR spelling to the stated name
                 ("Kevin Russo"→"Kevin Roose"; "Professor Pape"→"Robert Pape")
        (b) ON-AIR INTRO reader     _intro_reader_voice_names() ("my guest is X" → next voice)
        (c) CONVERSATIONAL ROLES    roles_from_conversation()   → conv_hosts / conv_guests
              ("welcome to the show" = host act; "thanks for having me" = guest act)
        (d) HOST-VOICE SELECTION  (cap = len(host_pool) — metadata says how MANY)
              1. self-intro name ∈ known_hosts        ← strongest (cross-reference)
              2. performs a host act, capped at the count
              3. the opener (does the intro)
              4. fill any COUNTED-but-empty seat from intro voices
              GUARD stated_non_host_voices: a voice that SAYS a name NOT in the host
                pool is blocked from steps 2/3/4 — a guest may not fill an absent
                co-host's seat (No Priors/Andy Fang, Unhedged/Joshua Franklin)
        (e) LLM VOICE RESOLUTION    _resolve_voices_via_llm()  (naming only, CLOSED list)
        (f) GUEST NAMING            _name_guest_voices()  (forced 1-name-1-voice; never positional)
     │
     ▼  SpeakerRole per voice = { name, role(host|guest|unknown), named, source }
        → .speakers.diagnostics.json   (per-voice + summary/exposed metric)
     │
 ── LEVEL 5: DURABLE OUTPUT ─────────────────────────────────────────────────────
     └─ _build_speakers_from_diarized_segments (workflow/metadata_generation.py)
           → content.speakers = [{name, role}]  →  .metadata.json / .manifest.json
```

**The name/role sources, and where each lands:**

| Source | Level | Function | Produces |
| ------ | ----- | -------- | -------- |
| Title + description | 1 | `detect_speaker_names` (NER) | `detected_guests`, `metadata_named` |
| Feed metadata | 1 | `detect_hosts_from_feed` | `known_hosts` (host count → cap) |
| Intro text (spoken) | 4a/4b | `_self_intro_voice_names`, `_intro_reader_voice_names` | `voice_intro` (per-voice) |
| Conversational roles | 4c | `roles_from_conversation` | `conv_hosts` / `conv_guests` |
| Canonical-name fusion | 4a | inside `_self_intro_voice_names` | ASR spelling → stated name |
| LLM (naming only) | 4e | `_resolve_voices_via_llm` | fills unresolved voices, closed list |

Every voice — **named or unnamed** — carries a `role` of `host`, `guest`, or `unknown` in the
diagnostics sidecar; `role=unknown` is the explicit "cannot tell host vs guest" bucket, and the
summary rolls these up as `exposed.voices_unknown` / `voices_unidentified`.

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

[`cloud_with_dgx_primary.yaml`](https://github.com/chipi/podcast_scraper/blob/main/config/profiles/cloud_with_dgx_primary.yaml)
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
| Diarization skipped with warning | Only when using the gated `3.1` fallback with no `HF_TOKEN` (the default `community-1` is non-gated — no token needed) | Keep the `community-1` default, or export `HF_TOKEN` + accept terms for `pyannote/speaker-diarization-3.1` |
| `ProviderDependencyError` for pyannote | `[ml]` not installed | `pip install -e ".[ml]"` |
| `diarize` false despite YAML `true` | API transcription provider | Expected — only local Whisper paths diarize |
| Deepgram validation error | Missing API key | Set `DEEPGRAM_API_KEY` |
| Oversize API upload still fails | File exceeds chunk merge limits | Check ffmpeg; see chunker logs; lower preprocessing bitrate |

Full guide: [Troubleshooting](TROUBLESHOOTING.md).

---

## Overlap with the LLM stage (#1180)

Audio work (Whisper, diarization, screenplay formatting) does NOT block LLM
work (metadata, summary, GI, KG) at the pipeline level. The orchestration
starts a dedicated **ProcessingProcessor** thread that consumes a
`processing_jobs` queue populated by the transcription thread the moment a
transcript is saved. So `transcript(ep1) → LLM(ep1)` runs concurrently with
`transcript(ep2)`.

- **Knob:** `transcription_parallelism` (default 1) drives how many episodes
  are transcribed simultaneously. **Whisper local is safest at 1** (GPU/CPU
  contention). **API providers** (OpenAI, Deepgram, DGX) can safely go higher.
- **Knob:** `processing_parallelism` (default 2) drives how many episodes are
  in LLM work simultaneously. Rate-limit-bounded for API summarizers.

**MPS exclusive mode** is the one place we deliberately serialize. When both
Whisper AND local summarization run on Apple MPS (single GPU), the pipeline
holds LLM work until all transcription is done (`stages/processing.py`
`should_serialize_mps`). On dedicated hardware (Whisper on GPU + LLM via
API/DGX) the two overlap in full.

**How to check whether overlap is actually happening** — every run's summary
JSON reports `processing_overlap_ratio`, `processing_thread_busy_ratio`,
`processing_thread_queue_idle_seconds`, `inline_processed_episodes_count`,
`safety_net_processed_episodes_count`, and
`handoff_latency_seconds_per_episode`. Cross-referenced in
[Pipeline and workflow](PIPELINE_AND_WORKFLOW.md#parallelism-observability-1180)
and [Performance](PERFORMANCE.md#tuning-parallelism-1180).

## Related documents

- [DEPENDENCIES_GUIDE.md](DEPENDENCIES_GUIDE.md) — `[ml]` / `[dev]` pyannote pins
- [PREPROCESSING_PROFILES_GUIDE.md](PREPROCESSING_PROFILES_GUIDE.md) — text cleaning profiles (includes sponsor step)
- [AI Provider Comparison Guide](AI_PROVIDER_COMPARISON_GUIDE.md) — provider matrix
- [ADR-058](../adr/ADR-058-additive-pyannote-diarization-with-separate-extra.md) — diarization decision + amendment
- [ADR-059](../adr/ADR-059-confidence-scored-multi-signal-commercial-detection.md) — commercial detection decision
