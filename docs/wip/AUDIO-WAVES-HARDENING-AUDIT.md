# Audio Waves 1–3 — Hardening Audit & Change List

**Status:** Draft / backlog — review findings, not yet scheduled.
**Base reviewed:** `main @ bdfc8b23` (Wave 1 #850 + Wave 2 #895 + #896 + Wave 3 #898 all merged).
**Author:** review pass over the Cursor-built audio waves, 2026-06-06.

This file consolidates a detailed multi-agent review of the audio pipeline shipped across
Wave 1 (#850), Wave 2 (#895), and Wave 3 (#898). It is the source list for a single combined
"audio hardening" PR. Each item carries a priority (P1/P2/P3) and rough effort (S/M/L).

## Progress (branch `feat/audio-waves-hardening`, not yet pushed)

**DONE** (each fix + tests committed, targeted suites green):

- P1: A1, A2, A3, C1, C2, B1, B2, D1, H1, G1, I1, I2, I3, G3.
- P2/P3: A4, A5, B6, B7, B8, C3, C4, D2, D4, H3, E1 (normalization module).
- G2 reclassified NOT-A-BUG (verified — feed-transcript episodes download no audio).

**DEFERRED to a follow-up** (lower-value robustness/cleanups, or blocked):

- Blocked by live callers/tests (not dead): E4 (`speaker_detectors/patterns.py` still called by
  `ml_provider.py:1061`), B9 (`summarizer.remove_sponsor_blocks` has tests), G7 (header asserted).
- Moderate effort, sensitive code: H2 (transcript-cache legacy-fallback tightening), B3 (config-
  driven commercial threshold), B4 (host-speaker inference hardening), B5 (removal audit sidecar),
  F1 (per-provider byte/duration limits), G4 (cross-feed stem collision), G5 (symlink realpath),
  D3 (cross-chunk speaker reconciliation).
- Minor: A6, D5, D6, E2, E3, E5, G6, H4, H5, I5, and remaining E1 modules.

**Validation:** full `make ci-fast` + `ci-ui-fast` run as the end gate before push (in progress).

## 🚨 BLOCKER — diarization does not load (dependency conflict, found via real eval)

The whole pyannote path is **non-functional in a fresh `[ml]` install**, undetected
because every unit test mocks pyannote. Found by actually trying to run it:

- `[ml]` pins `torch>=2.11` → pip pulls torch 2.12 / torchaudio 2.11. **pyannote.audio
  3.4** (what `>=3.1` resolves to) references `torchaudio.AudioMetaData`, **removed in
  torchaudio ≥2.9** → `AttributeError` on `import pyannote.audio`. Diarization can't load
  regardless of HF token.
- **Likely production too:** `docker/pipeline/Dockerfile` installs latest CPU `torch` +
  `[ml]` → same combo. Must be verified in that build.
- **pyannote.audio 4.0.4** fixes the torch issue (uses `torchcodec`, needs only
  `torch>=2.8`) **but** cascades `numpy→2.x`, which breaks spaCy's `thinc` (needs
  `numpy<2`) and `mistralai` (opentelemetry pin). So the real fix is a **coordinated**
  upgrade: pyannote 4.x **+** spaCy/thinc bumped to numpy-2-compatible **+** opentelemetry
  alignment — validated in the controlled Docker/nightly build, not piecemeal on a laptop.
- **HF model access:** the operator's token is valid but the account is **not authorized**
  for `pyannote/speaker-diarization-3.1` (got `GatedRepoError 403`). Must accept the user
  conditions on the model page (and `pyannote/segmentation-3.0`).

**Status:** dependency-pin resolution is a dedicated follow-up (own PR), validated in the
controlled env. This PR adds the real-pyannote integration test (skip-gated) + the
token-path fix so the validation is runnable once the pins are fixed and access is granted.

## How to read this

- **P1** — data integrity, broken/no-op features, or security-sink test gaps. Fix first.
- **P2** — robustness / correctness gaps with a real failure mode.
- **P3** — precision, cleanups, dead code, missing-but-low-risk tests.
- Groups **A–F** = Wave 1+2 subsystems. Groups **G–I** = Wave 3 new surface.
- The media path-contract bug (G1 ≡ I1) was independently flagged by three reviewers — it is
  one coherent fix spanning backend + frontend, not two.

## Cross-cutting themes

1. **Tests are breadth-1, depth-0 across every wave.** One happy-path test per module; edge
   cases, integration, and the *claimed feature behavior* untested. Several RFC-promised suites
   were never written, and several Wave 3 tests *encode the bug* (flat fixtures hide the media
   path mismatch; cache tests skip the legacy fallback) so they pass while locking in defects.
2. **"Diarization-aware" is partly a dead feature end-to-end** (Deepgram native diarization
   discarded; commercial Phase-2 signals run in the wrong coordinate space).
3. **Wave 3's headline modes are partly cosmetic** — `pipeline_stage` coerces flags but nothing
   in `workflow/` reads it; the media feature 404s for the realistic majority of episodes.

---

## GROUP A — Diarization (`providers/ml/diarization/*`)

- **A1 · P1 · M** — Cache cross-episode collision: `get_audio_hash` hashes only the first 1 MB,
  but diarization spans the whole file, so episodes sharing an intro serve each other's
  diarization. Add a full-file content hash (+ size) local to the diarization cache.
  `cache.py:45` (uses `transcript_cache.get_audio_hash:31-35`).
- **A2 · P2 · S** — Alignment: add a deterministic tie-break for tied/overlapping pyannote turns
  and epsilon-tolerant float comparison. `alignment.py:22,31-38`.
- **A3 · P3 · S** — Empty/zero-turn pyannote output → degrade to gap-based, not a phantom
  `SPEAKER_00`. `alignment.py:23`.
- **A4 · P3 · S** — Narrow the `use_auth_token` blanket `except TypeError` (signature inspect).
  `pyannote_provider.py:25-32`.
- **A5 · P3 · S** — Validate `num_speakers` vs `min/max` and `min <= max`. `pipeline.py:53-56`.
- **A6 · P3 · S** — Cleanups: rename the diarize-truthiness helper off `_raw_screenplay_requested`
  (`config.py`); single-source the cache-dir path (`episode_processor` vs `cache.py` const);
  decide on `diarization_device` in the cache fingerprint.
- **A7 · P1 · M** — Tests: pyannote provider import-error/`use_auth_token`/device branches;
  mapping fewer/more-names fallbacks; cache corrupt-JSON; the failure → gap-based fallback path.

## GROUP B — Commercial cleaning (`cleaning/commercial/*`)

- **B1 · P1 · M** — Diarization char-offset misalignment: `.segments.json` concat offsets ≠
  cleaned-transcript offsets → guest-disqualify + boosts effectively random. Carry a stable offset
  map OR run signals on raw segment text. `diarization_signals.py:33-52` vs `core.py:549-562`.
- **B2 · P1 · M** — Inline-CTA false positives ("check out X.com" / "sign up today" 0.5 + position
  0.15 + brand 0.1 = 0.75 > 0.65) delete up to 2000 chars of real content. Require corroboration
  before inline-only removal. `patterns.py:63-79`.
- **B3 · P2 · S** — Make the confidence threshold + signal weights config-driven (0.65 + boosts
  hardcoded; prod path never passes a threshold). `patterns.py:9`, `core.py:248`.
- **B4 · P2 · M** — Harden host-speaker inference (turn-count over first 15% misfires on cold
  opens; wrong host inverts the disqualify logic) + log the inferred host. `context.py:43-64`.
- **B5 · P2 · S–M** — Commercial removal audit log / `.sponsors` sidecar (removals silent +
  unrecoverable; RFC §Monitoring + Open Q#4 unmet). `detector.py:98-103`.
- **B6 · P3 · S** — Restore-or-formally-drop the −0.10 off-window position penalty (code returns
  0.0 vs RFC −0.10). `positions.py:24`.
- **B7 · P3 · S** — Tighten the over-broad `okay,? so` block_end; remove `lenny` from
  `BRAND_NAMES` (host name == brand → spurious boost). `patterns.py:90,123`.
- **B8 · P1 · M** — Tests: missing `test_patterns.py` + `test_positions.py`; a realistic
  `.segments.json` full-pipeline test (current tests encode the broken offset assumption).
- **B9 · P3 · S** — Cleanup: delete the `summarizer.remove_sponsor_blocks` delegate (RFC §9 said
  remove). `summarizer.py:526`.

## GROUP C — Audio chunking (`preprocessing/audio/chunker.py`)

- **C1 · P1 · M** — Segment-level overlap never deduped (only chunk free-text is) → doubled
  segments at every seam corrupt downstream (search index, GI timing, diarization). Add
  time-window segment dedup in merge. `chunker.py:150-199`.
- **C2 · P1 · M** — Stream-copy `-c copy` cuts mid-MP3-frame → garbled seam audio degrades
  transcription AND defeats the text-dedupe. Re-encode chunks / snap to silence / validate overlap
  covers decoder delay. `chunker.py:130-148`.
- **C3 · P2 · S** — Post-split size guard: re-split / fail loudly if a chunk > `max_bytes` (only
  guard is a 0.95 factor). `chunker.py:112`.
- **C4 · P3 · S–M** — Warn (don't silently drop) zero-byte chunks; bound `num_chunks`; use the
  decoded start for offset to avoid per-seam drift.
- **C5 · P1 · M** — Tests: multi-chunk seam-correctness fixtures (no lost/duplicated words; chunk
  size under `max_bytes`).

## GROUP D — Deepgram provider (`providers/deepgram/*`)

- **D1 · P1 · M** — Native diarization discarded: config coercion forces diarize/screenplay off
  for deepgram (`config.py:83` eligibility set; coercion `config.py:3064`) AND the
  `has_diarized_labels` gate checks `speaker_label` but the provider emits `speaker`
  (`episode_processor.py:400` vs `deepgram_provider.py:44,83`). Pick a coherent story: wire an
  API-diarization screenplay path OR stop requesting diarize and fix the docs.
- **D2 · P2 · S** — Add retry/backoff/timeout to the API call (every sibling has it; deepgram has
  none). Add `_safe_deepgram_retryable()` + `retry_with_metrics`, record retries.
  `deepgram_provider.py:186`.
- **D3 · P2 · M** — Cross-chunk speaker-ID reconciliation (chunk-local 0/1 not reconciled; latent
  until D1 lands).
- **D4 · P2 · S** — Harden response parsing: warn on unparsable `{}`; test malformed/empty
  payloads. `deepgram_provider.py:17`.
- **D5 · P3 · S** — Use `episode_duration_seconds` for precise cost (currently discarded). `:197`.
- **D6 · P3 · S–M** — Tests: error path, empty/malformed response, multi-speaker
  `_words_to_segments`, not-initialized/`FileNotFound` guards, experiment-mode factory,
  capabilities.

## GROUP E — speaker_detectors refactor (`speaker_detectors/*`) — refactor verified clean

- **E1 · P1 · M** — Add the RFC-059-promised per-module unit tests
  (`test_{entities,normalization,ner,hosts,guests,detection}.py`) — the refactor's headline
  benefit ("testable in isolation") is unrealized; modules are covered only via the facade.
- **E2 · P3 · S** — De-dup the `_extract_person_entities` trampoline (`detection.py:16-25` vs
  `hosts.py:14-21`).
- **E3 · P3 · S** — Reconsider the `_log` single-line logger trampoline routing inconsistency
  (`hosts.py:24-31`).
- **E4 · P3 · S** — Delete dead `patterns.py` no-op + `ml_provider.py:1061` caller.
- **E5 · P3 · S** — Add a facade `__all__` surface-guard test.

## GROUP F — Per-provider limits (`utils/audio_payload_limits.py` + routing)

- **F1 · P2 · M** — Encode real per-provider byte/duration limits for gemini/mistral/deepgram (they
  reuse OpenAI 25 MB, no duration cap → oversize/over-duration chunks can still fail or skip).
  `audio_payload_limits.py:11-19`, `episode_processor.py:1265`.

---

## GROUP G — Wave 3 corpus media: persist + serve (`utils/corpus_media.py`, `routes/corpus_media.py`)

- **G1 · P1 · M** — Audio extension + subpath contract mismatch (confirmed by 3 reviewers).
  Persist writes `media/<stem><source-ext>` and flattens subdirs; metadata-resolve and the viewer
  assume `media/<...>.mp3`. Non-mp3 episodes (common `.m4a`) and any nested transcript path 404.
  Fix: store the real persisted relpath in metadata `audio_relpath` and have consumers use it
  instead of re-deriving/guessing. `corpus_media.py:18-37,65`, `transcriptSourceDisplay.ts:38-42`.
- **G2 · NOT A BUG (verified)** — Direct-download (feed-transcript) episodes never download audio
  at all: `process_transcript_download` fetches only the transcript URL and returns before any
  media download (`episode_processor.py:1807-1821`); the audio-download path is the mutually
  exclusive `transcribe_missing` branch, which already persists media. So there is nothing to
  persist for these episodes. Giving them playable audio would require *adding* an audio download
  purely for playback — a feature/bandwidth tradeoff, not a hardening fix. Left as a possible
  future feature, intentionally not implemented here.
- **G3 · P1 · S** — Add path-traversal tests for the media route (`..`, absolute relpath, missing
  `media/` prefix, disallowed suffix, symlink-escape). Security-sink route has zero
  negative-confinement coverage. (Route itself verified SAFE; CodeQL FPs #352–355 already
  dismissed by the Wave 3 PR.) `routes/corpus_media.py`.
- **G4 · P2 · S–M** — Stem flattening → cross-feed collision: `transcripts/feedA/ep_01` and
  `transcripts/feedB/ep_01` both → `media/ep_01.mp3`; second copy silently overwrites. Preserve
  subpath or hash the relpath. (Also makes `corpus_media.py:23-25` dead code.)
- **G5 · P2 · S** — Symlink hardening: `safe_relpath_under_corpus_root` does string-normpath, not
  `realpath`, so a symlink *inside* `media/` could be followed out. Low severity (needs local FS
  write), but `realpath`-re-confirm closes it.
- **G6 · P3 · M** — Disk-footprint controls: default-on `copy2` duplicates every episode's full
  audio into the corpus (≈doubles audio footprint). Add hardlink/symlink mode and/or size cap.
- **G7 · P3 · S** — Drop the redundant `Accept-Ranges` header (Starlette sets it).
  `corpus_media.py:87`.

## GROUP H — Wave 3 pipeline stages (#414) + transcript-cache keying (`config.py`, `transcript_cache.py`)

- **H1 · P1 · M** — `pipeline_stage` (`full/audio_only/enrich_only`) has ZERO workflow effect
  beyond flag coercion — nothing in `workflow/` reads it. `audio_only` doesn't force transcription
  on; `enrich_only` doesn't enable transcript reuse (needs `skip_existing`) → both are no-ops with
  default flags. Either make the modes set the full implied flag-set or branch the workflow on
  `pipeline_stage`. Fix the `config.py:2751` docstring's reuse claim. `config.py:3118-3146`.
- **H2 · P2 · M** — Legacy transcript-cache read-fallback defeats the provider/model isolation it
  adds: bare-`{hash}.json` entries are returned for *any* provider/model after upgrade
  (`transcript_cache.py:113-115`), and the write side never migrates/cleans them (`:187-188`) →
  orphaned files + wrong-provider hits during migration. Tighten the fallback to check stored
  provider/model metadata, or gate behind an explicit migration flag.
- **H3 · P3 · S** — `_get_provider_model_name` returns `None` when `provider.model` is already a
  plain string (`episode_processor.py:1008-1036`) → all of that provider's models collapse to one
  cache key. Add a string branch + test.
- **H4 · P3 · S** — Coercion is silent (once-per-process log only) and overrides explicitly-set
  flags; also holds the lock across `logger.info` unlike the diarize gate it mirrors. Add a
  per-config warning (or don't suppress) and move logging outside the lock. `config.py:3132-3145`.
- **H5 · P3 · S** — Dead `getattr(provider, "name", None)` fallback (no provider defines `name`);
  derived cache names diverge from config provider ids (`MLProvider`→`"ml"` not `"whisper"`).
  Either add canonical `.name` attrs (accept one-time key churn) or drop the dead branch + document.

## GROUP I — Wave 3 viewer audio playback (`TranscriptViewerDialog.vue`, `transcriptSourceDisplay.ts`)

- **I1 · P1 · M** — Same path/extension contract break as G1, frontend side: the viewer re-derives
  `media/<full-path>/<stem>.mp3` (subdirs preserved, `.mp3` hardcoded) which only matches the
  backend for flat top-level mp3 transcripts → 404 for the realistic majority. Plumb the metadata
  `audio_relpath` through the payload instead of re-deriving. `transcriptSourceDisplay.ts:38-49`.
- **I2 · P1 · M** — No missing-media handling: `audioUrl` is set unconditionally, so a broken
  `<audio controls>` renders on every legacy (no-media) episode with no `@error`/"unavailable"
  state. Hide the player or show a message; gate on `audio_relpath` presence or a HEAD probe.
  `TranscriptViewerDialog.vue:181-182,331`.
- **I3 · P1 · S** — Audio not paused/torn down on dialog close → background playback + lingering
  media resource. Add a `@close`/`close()` hook calling `audioEl.value?.pause()` + clear
  `audioUrl`. `TranscriptViewerDialog.vue:82-91`.
- **I4 · P2 · M** — Tests encode the bug (flat-fixture e2e mocks the exact wrong path; util test
  asserts the wrong nested derivation) and never exercise the no-media/404 path. Correct the
  derivation tests, add nested-path + non-mp3 cases, add an e2e no-media assertion.
- **I5 · P3 · S** — Accessibility (`aria-label` on `<audio>`) + consistent `audioSeekStartMs`
  plumbing from the insight/supporting-quote open path. `NodeDetail.vue:1119-1137`.

---

## Priority rollup

- **P1 (15):** A1, A7, B1, B2, B8, C1, C2, C5, D1, E1, G1≡I1, G2, G3, H1, I2, I3.
- **P2 (12):** A2, B3, B4, B5, C3, D2, D3, D4, F1, G4, G5, H2, I4.
- **P3 (16):** A3, A4, A5, A6, B6, B7, B9, C4, D5, D6, E2, E3, E4, E5, G6, G7, H3, H4, H5, I5.

## Suggested execution order (one PR, commit per item/group)

1. **P1 data integrity & broken features first:** the media contract (G1+I1+I2+I3+G2 as one
   coherent feature fix), A1 (diarization cache), C1+C2 (chunk seams), B1 (commercial offset),
   B2 (CTA precision), D1 (Deepgram diarization), H1 (pipeline_stage).
2. **P1 tests** to lock the fixes: A7, B8, C5, E1, G3, I4.
3. **P2 then P3** by group.
4. **Final gate:** `make ci-fast` + `make ci-ui-fast` (viewer touched by I*/G*), CodeQL re-check,
   then PR.

## Notes / out of scope

- The media route is **security-safe**; CodeQL `py/path-injection` FPs #352–355 were already
  dismissed in the Wave 3 PR. G3 is about *tests*, not a vulnerability.
- The speaker_detectors refactor (RFC-059) is genuinely behavior-preserving — only debt is the
  missing isolated tests (E1).
