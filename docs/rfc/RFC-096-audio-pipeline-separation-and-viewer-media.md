# RFC-096: Audio Pipeline Separation and Corpus Media for Viewer Playback

- **Status**: Completed
- **Authors**: Architecture Review
- **Stakeholders**: Core Pipeline, Server API, GI/KG Viewer
- **Related PRDs**:
  - `docs/prd/PRD-020-audio-speaker-diarization.md`
- **Related RFCs**:
  - `docs/rfc/RFC-040-audio-preprocessing-pipeline.md`
  - `docs/rfc/RFC-058-audio-speaker-diarization.md`
- **Related Issues**:
  - [#414](https://github.com/chipi/podcast_scraper/issues/414) — Audio pipeline separation
  - [#547](https://github.com/chipi/podcast_scraper/issues/547) — Viewer synced audio playback
- **Related Documents**:
  - `docs/guides/AUDIO_PIPELINE_GUIDE.md`
  - `docs/wip/WAVE-3-PLAN.md`

## Abstract

Wave 3 splits the monolithic run into explicit **pipeline stages** (`full`, `audio_only`,
`enrich_only`), hardens **transcript cache keys** with transcription provider + model, persists
episode MP3 under corpus **`media/`**, exposes **`GET /api/corpus/media`** for local playback, and
adds an optional **HTML audio player** in the transcript viewer with seek-to-quote support. RSS
enclosure proxy is out of scope.

## Problem Statement

After Wave 2, transcription and enrichment still run in one coupled process. Re-running summarization
or GI with a new LLM profile re-enters the full pipeline unless operators manually disable
transcription. The viewer shows GI quote timestamps as text only — no way to hear the referenced
audio when corpus files are local.

## Goals

1. **Stage modes** — `audio_only` stops after transcript + segments + media; `enrich_only` skips
   download/transcribe when transcripts exist.
2. **Cache correctness** — transcript cache entries keyed by audio hash **and** STT provider/model.
3. **Corpus media** — copy downloaded audio to `<run>/media/`; metadata records `audio_relpath`.
4. **Viewer playback** — local `/api/corpus/media` only; seek using `timestamp_start_ms`.

## Design

### 1. Pipeline stages (`pipeline_stage`)

| Value | Transcribe | Metadata / GI / KG / summaries |
| ----- | ---------- | ------------------------------ |
| `full` (default) | Yes (when configured) | Yes (when configured) |
| `audio_only` | Yes | Coerced off |
| `enrich_only` | Coerced off | Yes (when configured) |

Implemented as config coercion (`mode="before"`) so YAML/CLI cannot accidentally transcribe on
enrich-only reruns.

### 2. Transcript cache fingerprint

Cache file: `{audio_hash}_{provider_model_fp}.json` with legacy fallback to `{audio_hash}.json`.

### 3. Corpus `media/` layout

After successful transcribe (or cache hit with temp media present), copy bytes to:

```text
<output_dir>/media/<transcript_stem><original_ext>
```

Metadata field: `content.audio_relpath` (POSIX relpath from corpus root).

Convention helper: `media/<stem>.mp3` from `transcripts/<stem>.txt` for viewer fallback.

### 4. Server API

`GET /api/corpus/media?path=&relpath=` — allowlist prefix `media/`; `FileResponse` with
`Accept-Ranges: bytes`; audio MIME from extension.

### 5. Viewer

Extend `TranscriptViewerDialog` with optional `<audio controls>` when media URL resolves; expose
`seekToMs(ms)` for quote rows. No external RSS URLs in Wave 3.

## Acceptance criteria

- [ ] `pipeline_stage=enrich_only` skips transcription on a corpus with existing transcripts
- [ ] `pipeline_stage=audio_only` writes transcript + `media/` without GI/summary
- [ ] Cache miss when provider/model changes, hit when unchanged
- [ ] Viewer plays local media and seeks to quote timestamp in E2E/fixture corpus

## Out of scope

- RSS enclosure proxy / hotlinking publisher URLs
- Separate OS processes or job queues for stages (#414 stretch)
- Wave 3 delivered as **one PR** closing #414 and #547
