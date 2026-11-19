# Audio Pipeline Guide

Operator and developer reference for **transcription**, **speaker attribution**, **API audio
chunking**, **commercial/sponsor cleaning**, and related provider paths.

For strategic architecture see [Architecture](../architecture/ARCHITECTURE.md). For every config
field see [CONFIGURATION.md](../api/CONFIGURATION.md). For CLI flags see [CLI.md](../api/CLI.md).

---

## What this repository ships, and what it does not

This repository registers **no built-in pipeline stages**
(`src/podcast_scraper/builtin_stage_hooks.py`). Speaker detection and neural diarization, ad
cleaning (sponsor/commercial cleaning and the ad-free transcript) and audio sync are provided by an
installed extension through `Extension.pipeline_hooks` (ADR-176,
[Extension seams](../adr/ADR-176-extension-seams-for-the-split.md)). Without one, each hook in
`podcast_scraper.stage_hooks` answers its plain default.

| Part | Ships here | Needs an installed extension |
| ---- | ---------- | ---------------------------- |
| Transcription providers (`whisper`, `tailnet_dgx_whisper`, `openai`, `gemini`, `mistral`, `groq`, `deepgram`, `moss`) and the fallback chain | yes | |
| Audio preprocessing (`preprocessing/audio`, ffmpeg) and API audio chunking (`AudioChunker`) | yes | |
| Provider-native speaker labels (Deepgram, Moss) and gap-based screenplay (`screenplay: true`) | yes | |
| Pattern-based transcript cleaning before summarization (`cleaning/pattern_based.py`) | yes | |
| Neural diarization (pyannote), speaker, host and guest naming | | yes |
| Sponsor/commercial removal and the ad-free transcript (`.adfree.*`) | | yes |

A public-only install therefore transcribes episodes and can render a gap-based or
provider-labelled screenplay. It does not run pyannote, detect or name speakers, or remove ads:
`remove_commercials` and `excise_ads` return the text unchanged and `adfree_base` writes nothing.
The `diarize`, `diarization_*` and `hf_token` settings below are accepted and validated by the
public config, but the pass they control runs in the extension. The pyannote stack
(`pyannote.audio`, `torchaudio`, `torchcodec`) is in neither the `[ml]` nor the `[dev]` extra
(`pyproject.toml`); the package that installs the diarization stage depends on it.

The sections on diarization, speaker resolution and commercial cleaning describe the behaviour of
that extension, so you know what to expect when one is installed.

---

## Transcription providers

| Provider | Config value | When to use |
| -------- | ------------- | ----------- |
| Local Whisper | `whisper` | Default local path; supports gap-based screenplay, and neural diarization when the extension is installed |
| Self-hosted GPU Whisper (private network) | `tailnet_dgx_whisper` | Self-hosted GPU primary with cloud fallback (ADR-096, not in this repository); same diarize/screenplay rules as local Whisper |
| OpenAI | `openai` | Cloud STT (`whisper-1`) |
| Gemini | `gemini` | Cloud STT; plain text output |
| Mistral | `mistral` | Cloud STT; plain text output |
| Groq | `groq` | Cloud STT (Whisper models) |
| Deepgram | `deepgram` | Nova-3 with built-in utterance diarization in API response; requires `DEEPGRAM_API_KEY` / `[llm]` extra |
| Moss | `moss` | Cloud STT that returns the transcript and speaker labels in one pass |

`transcription_fallback_provider(s)` adds a fallback chain behind the primary provider.

**Diarization eligibility** (config validation, `config.py`): `whisper`, `tailnet_dgx_whisper`,
`openai`, `groq`, `deepgram` and `moss`. `diarize` defaults on for `whisper` and
`tailnet_dgx_whisper` and off for the other eligible providers unless set. `gemini` and `mistral`
cannot diarize: an explicit `diarize: true` with them raises a config error
(`PODCAST_SCRAPER_DIARIZE_LAX=1` restores the deprecated silent coercion). The diarizer is chosen
with `diarization_provider`: `local` (pyannote, the default), `tailnet_dgx`, `gemini`, `deepgram` or
`moss`. The pass itself is run by the extension (see above).

**Native screenplay:** `deepgram` and `moss` return speaker labels in the transcription call, so
their screenplay is built without a separate diarization pass, even with `diarize: false`.

---

## Persisted episode audio

With `persist_episode_media` (default on) the episode audio is kept under the corpus `media/`
directory, and `GET /api/corpus/media` serves it from a local `serve` process
(`server/routes/corpus_media.py`).

---

## Speaker detection and neural diarization (extension)

Speaker name detection (NER and heuristics over feed, title and description) and the neural
diarization pass are provided by an installed extension, not by this repository. The settings
below are the ones the public config defines for it.

Detected host/guest names are mapped onto screenplay speaker labels (gap-based or diarized).

### Gap-based screenplay (ships here)

With `screenplay: true`, and when no diarized labels exist, the transcription provider's
`format_screenplay_from_segments()` rotates speakers on silence gaps (`screenplay_gap`, default
1.25 s). This is **not** voice identity: the same speaker can be split across labels in rapid
exchanges.

### Neural diarization (extension)

After Whisper transcription, a **pyannote.audio** second pass assigns speaker IDs from voice
embeddings, aligned to Whisper segments by maximum overlap.

| Setting | Default | Notes |
| ------- | ------- | ----- |
| `diarize` | `true` for `whisper` and `tailnet_dgx_whisper`; `false` for `openai`, `groq`, `deepgram`, `moss` unless set | An explicit `diarize: true` with `gemini` or `mistral` raises a config error (unless `PODCAST_SCRAPER_DIARIZE_LAX=1`, deprecated) |
| `screenplay` | auto-enabled when `diarize=true` on Whisper paths | Explicit `screenplay: false` still respected |
| `hf_token` | from `HF_TOKEN` env or `~/.huggingface/token` | Required for gated pyannote models |
| `diarization_num_speakers` | auto | Set when speaker count is known |
| `diarization_min_speakers` | `2` | Auto-detect floor |
| `diarization_max_speakers` | `20` | Auto-detect ceiling |
| `diarization_device` | `auto` | `cpu`, `cuda`, or `mps` |
| `diarization_model` | `pyannote/speaker-diarization-community-1` | HuggingFace pipeline id (v4, gated: needs `HF_TOKEN`; 3.1 fallback) |
| `diarization_min_segment_ms` | `None` | Squelch: drop any speaker whose longest segment < this (ms); kills phantom micro-clusters, keeps real cameos. Per-feed overridable |

**Install:** the pyannote stack is not in this repository's extras. It comes with the package that
installs the diarization stage.

**CLI:** `--diarize`, `--no-diarize`, `--hf-token`, `--diarization-num-speakers`, etc. (the flags
are defined in `cli.py`; the pass they enable is the extension's).

On failure (missing token, import error, runtime error), the extension's pass is expected to fall
back to the gap-based screenplay.

---

## Host/guest resolution (extension)

The model, as implemented by the extension: **WHO comes from metadata; WHICH-VOICE comes from the
conversation; the roster is the one place the two fuse**. Transcription runs *before* diarization,
and diarization runs *before* the roster. Metadata detection (feed, title, description) happens
first and is carried in as constraints; it never reads the audio.

1. **Metadata (who).** Feed metadata states the hosts (the host count caps the roster); NER over
   title and description yields guests and everyone the episode names.
2. **Transcription (words).** Audio becomes a transcript and time-coded segments, raw, with ads
   still in and no labels. This step is in this repository.
3. **Diarization (which voice).** Anonymous `SPEAKER_00`, `SPEAKER_01`, ... clusters with
   per-voice segments. Ad regions are detected here and used to type which voices are ads.
4. **Roster resolution (the fusion).** Spoken self-introductions and on-air introductions
   ("my guest is X") name voices, snapping ASR spelling to the stated name; conversational acts
   ("welcome to the show" is a host act, "thanks for having me" a guest act) assign roles; an LLM
   names only the voices still unresolved, from a closed list.
5. **Durable output.** The ad-free transcript is written after labelling, the cleaned transcript
   used for summarization after that, and the speakers land in `.metadata.json` and
   `.manifest.json`.

Every voice, named or unnamed, carries a `role` of `host`, `guest`, or `unknown` in the
diagnostics sidecar; `role=unknown` is the explicit "cannot tell host vs guest" bucket.

---

## API audio chunking

Cloud transcription providers enforce upload size limits. **`AudioChunker`**
(`preprocessing/audio/chunker.py`) splits oversized files with ffmpeg stream-copy, transcribes each
chunk, and merges text with overlap deduplication.

Triggered from `workflow/episode_processor.py` when post-preprocess audio exceeds the provider cap
(`openai`, `groq`, `gemini`, `mistral`, `deepgram` and `moss`; `_API_CHUNKING_PROVIDERS`). The
duration governor is `transcription_max_chunk_duration_seconds`. Local Whisper is not chunked by
this mechanism.

---

## Commercial / sponsor cleaning (extension)

Sponsor removal runs through the `remove_commercials` stage hook. `preprocessing/core.py`
(`remove_sponsor_blocks`) delegates to it; without an owner the text comes back unchanged. The
extension's cleaner scores **text patterns** and **positional heuristics** (intro / mid-roll /
outro clusters) on transcript text, so it benefits all transcription providers. When diarization
segments are available, they adjust confidence (host monologue boost, guest disqualify,
duration/topic hints).

What ships here is the pattern-based cleaner used before summarization
(`cleaning/pattern_based.py`, `PatternBasedCleaner`), which the `transcript_cleaner` hook returns
for every strategy when no extension overrides it. See
[Preprocessing profiles](PREPROCESSING_PROFILES_GUIDE.md).

---

## Deepgram

```bash
export DEEPGRAM_API_KEY=your-key
python -m podcast_scraper.cli https://example.com/feed.xml \
  --transcription-provider deepgram \
  --deepgram-model nova-3
```

Config: `transcription_provider: deepgram`, `deepgram_api_key`, `deepgram_model` (default `nova-3`).
SDK ships in **`[llm]`** extra (`deepgram-sdk`).

Deepgram returns utterance-level speaker labels in the API response, separate from a local pyannote
diarization pass.

---

## Self-hosted GPU profile

[`cloud_with_dgx_primary.yaml`](https://github.com/chipi/podcast_scraper/blob/main/config/profiles/cloud_with_dgx_primary.yaml)
sets `transcription.primary: tailnet_dgx_whisper` with Deepgram as the fallback, plus
`screenplay: true`, `diarize: true` and `diarization_provider: tailnet_dgx`. With the default
`diarization_provider: local`, diarization would instead run on the pipeline host after
transcription, not on the remote GPU service.

`diarization_provider: tailnet_dgx` sends diarization to the GPU host (a diarize service on
`dgx_diarize_port`, model `dgx_diarize_model`). The
[`prod_dgx_full.yaml`](https://github.com/chipi/podcast_scraper/blob/main/config/profiles/prod_dgx_full.yaml)
profile uses it together with `tailnet_dgx_whisper` transcription. As above, the diarization pass
itself is the extension's.

---

## Deployment profiles (diarize / screenplay)

Local and DGX deployment profiles under `config/profiles/` set **`screenplay: true`** and
**`diarize: true`** by default. `cloud_quality` and `cloud_thin` transcribe with OpenAI and diarize
with a separate pass (`diarize: true`, `diarization_provider: deepgram`). `cloud_balanced`
transcribes with Deepgram, which labels speakers in the transcription call itself, so it sets
`diarize: false` and keeps the native screenplay.

Details: [Profiles README](https://github.com/chipi/podcast_scraper/blob/main/config/profiles/README.md) in the repository.

---

## Progress indicator

Per-episode nested Whisper `"Transcribing"` progress was removed. The batch-level transcription bar
in the processing stage is the sole progress indicator during local Whisper runs.

---

## Troubleshooting quick reference

| Symptom | Likely cause | Action |
| ------- | ------------- | ------ |
| `diarize: true` but no speaker labels, no diarization sidecars | No extension installed that provides the diarization stage | Install it; this repository does not run pyannote |
| Diarization skipped with warning (extension installed) | No usable `HF_TOKEN`: both the default `community-1` and the `3.1` fallback are gated | Export `HF_TOKEN` for an account that accepted the model's terms; a fine-grained token also needs "public gated repositories" |
| Config error `diarize=true requires a transcription_provider…` | `diarize: true` with `gemini` or `mistral` | Use an eligible provider (`whisper`, `tailnet_dgx_whisper`, `openai`, `groq`, `deepgram`, `moss`) or drop `diarize` |
| Deepgram validation error | Missing API key | Set `DEEPGRAM_API_KEY` |
| Oversize API upload still fails | File exceeds chunk merge limits | Check ffmpeg; see chunker logs; lower preprocessing bitrate |

Full guide: [Troubleshooting](TROUBLESHOOTING.md).

---

## Overlap with the LLM stage

Audio work (Whisper, and diarization and screenplay formatting when present) does NOT block LLM
work (metadata, summary, KG) at the pipeline level. The orchestration
starts a dedicated **ProcessingProcessor** thread that consumes a
`processing_jobs` queue populated by the transcription thread the moment a
transcript is saved. So `transcript(ep1) → LLM(ep1)` runs concurrently with
`transcript(ep2)`.

- **Knob:** `transcription_parallelism` (default 1) drives how many episodes
  are transcribed simultaneously. **Whisper local is safest at 1** (GPU/CPU
  contention). **API providers** (OpenAI, Deepgram, self-hosted GPU) can safely go higher.
- **Knob:** `processing_parallelism` (default 2) drives how many episodes are
  in LLM work simultaneously. Rate-limit-bounded for API summarizers.

**MPS exclusive mode** is the one place we deliberately serialize. When both
Whisper AND local summarization run on Apple MPS (single GPU), the pipeline
holds LLM work until all transcription is done (`should_serialize_mps`, computed in
`workflow/orchestration.py` and honoured in `workflow/stages/processing.py`). On dedicated hardware
(Whisper on GPU + LLM via API/GPU host) the two overlap in full.

**How to check whether overlap is actually happening** — every run's summary
JSON reports `processing_overlap_ratio`, `processing_thread_busy_ratio`,
`processing_thread_queue_idle_seconds`, `inline_processed_episodes_count`,
`safety_net_processed_episodes_count`, and
`handoff_latency_seconds_per_episode`. Cross-referenced in
[Pipeline and workflow](PIPELINE_AND_WORKFLOW.md#parallelism-observability-1180).

## Related documents

- [DEPENDENCIES_GUIDE.md](DEPENDENCIES_GUIDE.md) — extras and what they install
- [PREPROCESSING_PROFILES_GUIDE.md](PREPROCESSING_PROFILES_GUIDE.md) — text cleaning profiles (includes sponsor step)
