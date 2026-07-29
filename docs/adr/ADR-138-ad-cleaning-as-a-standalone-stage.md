# ADR-138: Ad-cleaning as a standalone pipeline stage (one clean transcript for labeling + summary)

- **Status**: Proposed
- **Date**: 2026-07-29
- **Authors**: Marko Dragoljevic
- **Related issues**: labeling mis-map incident (prod-v2.4-100ep, John Kim); patch in
  `fix(labeling): strip known ad segments from the speaker-resolver LLM input`
- **Design source**: `docs/guides/AUDIO_PIPELINE_GUIDE.md` (host/guest resolution flow diagram)

## Context & Problem Statement

Ad-cleaning is not a diarization concern — it just produces a clean transcript — but today it is
**smeared across three unshared mechanisms**, none of which is the single source of truth:

1. **Inside diarization** (`apply_diarization_to_result`): `_ad_intervals` → `excise_ad_regions`
   computes ad *time regions*, used **only** by `classify_voices` to *type* which voices are ads.
   It does **not** strip ad *text* from a real voice.
2. **After diarization** (`_maybe_produce_adfree`): writes the `.adfree.{txt,segments.json}` file.
3. **In summary** (`metadata_generation`): an LLM cleaner writes the `.cleaned` transcript that
   summarization consumes.

Two problems follow:

- **Correctness.** The speaker-resolver LLM (LEVEL 4 in the flow diagram) is handed each voice's
  *raw* per-voice text + intro block, which still contain sponsor reads. The prompt claims
  "ads/cameos removed", but only ad *voices* were filtered. A voice whose diarized cluster contains
  the pre-roll ad is shown reading "Ramp is the only platform…" and **mis-maps** — the John Kim
  incident: the model put the host's name on the guest voice, and the roster then dropped the
  guest's deterministic name. Patched by stripping the already-computed `ad_intervals` from the LLM
  input; that patch is correct but is a plaster over a structural gap.
- **Timing + duplication.** The ad-free *file* is written **after** labeling, so it does not exist
  when labeling needs it. Three mechanisms re-derive "what is an ad" independently. `relabel_only`
  re-derives cleaning from raw segments rather than reusing a durable clean artifact, which is one
  suspected contributor to relabel-vs-full labeling drift (to be confirmed separately).

## Decision

Introduce a **standalone CLEANING stage** immediately after transcription and **before**
diarization/labeling. It excises ad regions **once** and produces one durable artifact:
`clean transcript + clean segments (+ ad-map)`. Every downstream text consumer reads it.

```text
transcribe → 🧹 CLEAN (standalone: excise ads → clean transcript + clean segments + ad-map)
                       ├──────────────────────────► SUMMARY   (flat clean text)      ┐ may run
                       └─ diarize(audio) → align clean segments → roster/LLM naming    ┘ in parallel
```

Key points:

- **Diarization is audio, not text** — pyannote maps *time → speaker clusters* and never reads the
  transcript, so this stage does not change diarization. What it changes is the **text** fed to the
  roster/LLM naming and to summary. Per-voice text is still assembled *after* diarization (you need
  the clusters to know who said what), but assembled from the **already-cleaned** segments, so ads
  never reach the naming LLM by construction.
- **One source of truth.** The stage's output is *the* clean transcript. `classify_voices` still
  needs the ad-map/intervals to type ad voices, so the stage emits the ad-map alongside the clean
  text; it does not lose "where the ads were".
- **Enables parallelism.** With the clean transcript produced first, **summary can run in parallel
  with diarization+labeling** (both consume it), instead of strictly after.

## Alternatives considered

1. **Status quo + the LLM-input strip patch (current).** Keeps three mechanisms; the LLM input is
   now clean, but the duplication and the "ad-free file is produced after labeling" inversion
   remain. Acceptable short-term, not the target.
2. **Clean inside diarization.** Where it lives today — wrong layer; ties a transcript concern to
   the diarizer and makes it invisible to summary's own cleaner.

## Consequences

- **The LLM-input strip patch becomes redundant** once the naming stage reads pre-cleaned segments —
  the strip is then a property of the stage, not a special case in `apply_diarization_to_result`.
- **Relabel/reprocess consistency improves**: `relabel_only` reuses the durable clean transcript
  rather than re-deriving cleaning per run (helps the relabel-vs-full drift, pending confirmation).
- **A pipeline reorder** touching `episode_processor`, the diarization pipeline, and
  `metadata_generation` — non-trivial, hence this ADR rather than a silent refactor.

## Non-Goals

- Changing the ad **detector** itself (`gi/ad_regions.excise_ad_regions`) or the LLM cleaner's
  rubric — this ADR is about *where* cleaning runs and *who* consumes its output, not *how* ads are
  found.
- The narrator/third-person attribution work (separate labeling lever).

## Open questions (to resolve before implementation)

- **Which cleaner is the stage?** The heuristic `excise_ad_regions` (fast, already used for
  intervals) vs the summary LLM cleaner (deeper). Possibly heuristic for the labeling path, LLM
  cleaner retained for summary — or one shared pass.
- **Caching / relabel:** the clean transcript becomes a first-class cached artifact; define its
  path, versioning, and how `relabel_only` / `--no-transcribe-missing` consume it.
- **Parallelism scope:** whether summary || labeling lands in this ADR or a follow-up.
