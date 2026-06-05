# Wave 3 plan — pipeline separation (#414) + viewer audio (#547)

**Branch:** `feat/audio-wave-3`  
**Status:** Draft plan (2026-06-05)  
**Parent programme:** Audio waves 1–3 ([AUDIO_PIPELINE_GUIDE.md](../guides/AUDIO_PIPELINE_GUIDE.md))

Wave 1 (#850) and Wave 2 (#895) are merged on `main`. Wave 3 splits into two
tracks that can land as separate PRs but share one theme: **decouple expensive
audio work from downstream LLM/GI work**, and **close the loop in the viewer**
when quote timestamps exist.

---

## Issues in scope

| Issue | Title | Primary surface |
| ----- | ----- | ---------------- |
| [#414](https://github.com/chipi/podcast_scraper/issues/414) | Audio pipeline separation | Python workflow / orchestration |
| [#547](https://github.com/chipi/podcast_scraper/issues/547) | Viewer: audio playback synced to transcript | `web/gi-kg-viewer/` + server API |

**Parallel (not Wave 3):** #109 commercial fixtures, #111 unique TTS voices, #594
autoresearch — other worktrees.

---

## Baseline after Wave 2

Already on `main`:

- **Transcript hash cache** (`transcript_cache_enabled`, `.cache/transcripts/`) —
  skips re-transcription when audio bytes match; saves text + optional segments
  ([#402](https://github.com/chipi/podcast_scraper/issues/402) closed).
- **Diarization cache** (`.cache/diarization/`) — skips pyannote on re-run with
  same audio + diarization config (Wave 2 follow-up).
- **Monolithic run** — `run_pipeline()` still threads download → transcribe →
  metadata/GI/KG/summary in one process (`workflow/orchestration.py`).
- **Transcript viewer** — in-app dialog with char highlights + **read-only**
  `audioTimingLabel` when GI quotes have `timestamp_*_ms`
  (`TranscriptViewerDialog.vue`); no `<audio>` element yet.
- **Episode metadata** — `content.media_url` (RSS enclosure) is written; local
  corpus does **not** standardize a served audio relpath for the viewer today.
- **`/api/corpus/binary`** — artwork only (`.podcast_scraper/corpus-art/`), not
  episode MP3.

---

## Track A — #414 Audio pipeline separation

### Problem (from issue)

Transcription is sequential and GPU-bound; LLM stages rerun on every config
change even when audio and transcript are unchanged. Goal: **Stage 1 (audio)** produces
a durable transcript artifact; **Stage 2 (enrichment)** consumes it without
touching audio providers.

Target shape:

```text
Stage 1 (audio):  Download → Preprocess → Transcribe → [Diarize] → persist + cache
Stage 2 (LLM):    Load transcript (+ segments) → speakers → summary → GI → KG
```

### Gap vs today

| Capability | Today | Wave 3 target |
| ---------- | ----- | ------------- |
| Skip transcribe when transcript on disk | `skip_existing` (per run dir) | Same + **content-hash cache** (partial) |
| Skip transcribe across LLM-only reruns | No first-class mode | **`--enrich-only` / config flag** |
| Separate worker pools | Transcription thread + processing thread (coupled in one run) | Explicit **stage boundary** + status JSONL |
| Cache includes diarized segments | Yes when cache write path runs | Document + test matrix |
| Operator docs | AUDIO_PIPELINE_GUIDE Wave 1–2 | Stage 1 vs 2 operator guide |

### Prerequisites (#414 table) — status

| Prerequisite | Status |
| ------------ | ------ |
| #402 Transcript caching | Closed — implemented (opt-in `transcript_cache_enabled`) |
| #402.2 JSONL metrics | Closed — `jsonl_emitter` |
| #401 Degradation policy | Verify at implementation time (stage failure hooks exist) |
| #399 Provider hardening | Largely shipped; re-validate retry paths when splitting stages |

### Proposed phases (Track A)

**3A — Design + thin boundary (first PR, low risk)**

- Draft **RFC** (new id, e.g. RFC-090) sourced from #414 — stage contracts, cache
  keys, failure semantics; no numbered spec IDs in code.
- Add config: `pipeline_stage: full | audio_only | enrich_only` (names TBD) with
  validation matrix (e.g. `enrich_only` requires existing transcript path or
  episode id in corpus).
- Refactor entry: `run_pipeline()` delegates to `run_audio_stage()` /
  `run_enrichment_stage()` without changing default behaviour (`full`).
- Integration test: run audio stage → change `summary_provider` → enrich-only
  run skips `transcribe` mock / cache hit.

**3B — Cache + corpus as source of truth**

- On audio stage completion, always write/update `.segments.json` when segments
  exist (align with GI timing #543).
- Enrich-only loads transcript from run dir or explicit `--transcript-path` /
  corpus episode row (Library metadata).
- Metrics: JSONL events `audio_stage_finished`, `enrichment_stage_started` with
  `transcript_cache_hit`, `diarization_cache_hit`.

**3C — Parallel LLM experimentation (acceptance for #414)**

- Document pattern: same audio hash, N enrich runs with different profiles.
- Optional CLI: `podcast_scraper enrich --corpus … --episode-id … --profile …`
  (thin wrapper over enrich-only stage).
- Update `ARCHITECTURE.md` + `AUDIO_PIPELINE_GUIDE.md` Wave 3 row.

### Track A — files to touch (estimate)

- `workflow/orchestration.py`, `workflow/stages/transcription.py`,
  `workflow/stages/processing.py`, `workflow/episode_processor.py`
- `config.py`, `cli.py`, `service.py`
- `cache/transcript_cache.py` (read paths for enrich-only)
- Tests: `tests/integration/workflow/`, `tests/unit/.../test_episode_processor.py`
- Docs: new RFC, `ARCHITECTURE.md`, `PIPELINE_AND_WORKFLOW.md`

### Track A — out of scope for Wave 3

- Separate processes/machines for stage 1 vs 2 (queue workers / Celery) — design
  hooks only.
- Replacing threading model with async job registry (#626 overlap) — note in RFC,
  do not merge scopes.

---

## Track B — #547 Viewer synced audio playback

### Problem (from issue)

When GI quotes have `timestamp_start_ms` / `timestamp_end_ms`, the transcript
dialog shows a static **Audio:** line but cannot play or seek. Operators cannot
verify quotes against heard audio.

### Dependencies

| Dependency | Status |
| ---------- | ------ |
| Transcript dialog | Landed (`TranscriptViewerDialog.vue`, E2E in `search-to-graph-mocks.spec.ts`) |
| Quote timestamps | Require segment-capable transcription (#543 matrix); Wave 2 diarization helps local Whisper |
| Stable audio URL in viewer | **Not built** — main design work for 547 |

### Audio source options (decide in RFC or UXS)

1. **Corpus-local file** (preferred for `podcast serve` on fixture/prod output)
   - Pipeline persists MP3 under allowlisted subtree (e.g. `media/` or reuse
     download temp with `reuse_media`-style retention).
   - New API: extend `/api/corpus/binary` or add `/api/corpus/media` with
     path allowlist + same sanitization as `corpus_text_file`.
2. **RSS `media_url` proxy** (fallback)
   - Server streams or redirects to enclosure URL; CORS/range-request handling;
   - Platform/ToS and auth complexity — likely phase 2 of 547.
3. **No audio** — hide player; keep timing labels (current behaviour).

Recommendation: **Phase 547a = local corpus file + player**; RSS enclosure proxy is
**out of scope for Wave 3** (follow-up issue if needed).

**Accepted (operator):** persist MP3 under corpus **`media/`** (Option A); viewer
serves audio **locally only** via new allowlisted API route — no RSS proxy in v1.

### Proposed phases (Track B)

**547a — API + metadata contract**

- Episode metadata / corpus API exposes optional `audio_relpath` (or derive from
  convention documented in RFC).
- Server route serves `audio/mpeg` with `Accept-Ranges` for seek.
- Unit tests under `tests/unit/podcast_scraper/server/`; integration under
  `tests/integration/server/`.

**547b — Viewer player UX**

- `TranscriptViewerDialog`: optional `<audio controls>` when URL resolves;
  **Seek** button on quote rows (NodeDetail, ResultCard) calls
  `transcriptViewer.seekToMs(ms)` and scrolls segment list when `.segments.json`
  loaded.
- Optional playback-driven highlight (segment active while playing) — stretch goal.
- Update `e2e/E2E_SURFACE_MAP.md`, Playwright spec, `docs/uxs/` if visible
  contract changes.
- Validate with Chrome DevTools MCP per viewer rules.

**547c — E2E with fixture MP3**

- Stack-test or Playwright fixture: episode with `p01_e01_fast.mp3` + segments +
  GI quote timestamps; assert seek updates `currentTime`.

### Track B — files to touch (estimate)

- `src/podcast_scraper/server/routes/` (new or extended corpus media route)
- `workflow/episode_processor.py` or download path (persist media relpath)
- `web/gi-kg-viewer/src/components/shared/TranscriptViewerDialog.vue`
- `web/gi-kg-viewer/src/components/graph/NodeDetail.vue`
- `web/gi-kg-viewer/e2e/*.spec.ts`

### Track B — non-goals (from issue)

- Full transcript editing
- Offline-only edge cases in v1
- Auto-fetch arbitrary external URLs without allowlist

---

## Delivery shape

**Accepted:** one PR on `feat/audio-wave-3` closing **#414** and **#547** (RFC +
pipeline split + `media/` persistence + local media API + viewer player + docs).

Suggested **commit order inside the branch** (single squash or stacked commits):

1. RFC-090 (Draft) + plan updates  
2. Track A — stage boundary (`full` / `audio_only` / `enrich_only`; default `full`)  
3. Track B — persist `media/<episode>.mp3` + `/api/corpus/media` (allowlisted)  
4. Track B — `TranscriptViewerDialog` player + seek; E2E + UXS/E2E_SURFACE_MAP  
5. Cache fingerprint includes transcription provider/model (cache already default-on)  
6. Docs + `AUDIO_PIPELINE_GUIDE` Wave 3 row  

---

## Validation gates (before merge)

- Python: `make ci-fast`
- Viewer: `make ci-ui-fast` + MCP browser loop for player/seek
- Docs: `make docs` when mkdocs nav / cross-refs change

---

## Decisions (accepted)

| # | Question | Decision |
| - | -------- | -------- |
| 1 | `transcript_cache_enabled` default | **No change** — already `true` on `main`. At stage split, **extend cache key** with transcription provider + model so cached STT does not silently freeze old output. |
| 2 | Episode MP3 layout | **Option A** — persist under corpus **`media/`** (or `audio/`) at download/transcribe time; metadata records `audio_relpath`. |
| 3 | #547 audio source | **Local serve only** for v1 — player uses corpus media API; hide player when no local file. No RSS enclosure proxy in Wave 3. |
| 4 | PR count | **One PR** for all Wave 3 work. |

---

## Next actions on this branch

1. Create **RFC-090** (pipeline separation + corpus `media/` + local viewer audio) — Draft.
2. Implement stage split (#414) + `media/` persistence + media API + viewer player (#547).
3. Single PR when `make ci-fast` + `make ci-ui-fast` green.
