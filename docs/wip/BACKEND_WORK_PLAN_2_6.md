# 2.6 Backend Work Plan

Backend-only work while UI work proceeds separately.
Deepgram (#597) and transcription autoresearch (#594) deferred to 2.7.

## Completed (PR #624)

### Step 1: Embedding loader compat audit — DONE

All 3 sentence-transformers callsites (embedding, NLI, model preload)
use `inspect.signature` introspection for `local_files_only` kwarg.
Thread-safe caches added to QA, embedding, and NLI model loaders.
Bridge builder reuses cached embedding model instead of new instance.

### Step 2: Speaker flow validation (#598) — DONE

- `DESCRIPTION_SNIPPET_LENGTH` 20→500 (guest names were truncated)
- `detect_speaker_names` simplified from 1389→935 lines (removed
  heuristic pattern learner + scoring system)
- Trace matrix: config→NER→GI/KG→bridge (see `SPEAKER_FLOW_TRACE_MATRIX.md`)
- 7 integration tests, 2 E2E tests with content assertions
- Acceptance runner: self-deriving `--assert-artifacts` mode (#622)

## Next (start of next session)

## Step 3: Eval datasets (~1 day)

Foundation for measuring KG and summary improvements.

- Score QMSum against existing silver refs (partially done)
- Establish cross-dataset baseline: synthetic fixtures vs QMSum vs production corpus
- Document which metrics each dataset covers (GI coverage, KG topic quality,
  summary quality)

**Deliverable:** scoring matrix across datasets, documented in eval report.

## Step 4: KG prompt tuning per provider (#590, ~1-2 days)

Uses eval datasets from step 3 to measure improvements.

- KG v2 prompt already deployed — measure per-provider quality on eval set
- Identify weak providers (Grok was fixed, check mistral, ollama models)
- Tune prompt variants per provider if needed, score against KG silver refs
- Same autoresearch ratchet methodology as GI/summarization

**Deliverable:** per-provider KG quality matrix, optimal prompt per provider.

## Step 5: SummLlama as proper ML mode (#571, ~1 day)

Wire SummLlama3.2-3B as a proper `summary_provider=summllama` option.

- No-daemon alternative to BART (no MPS contention)
- Benchmark against BART on eval datasets from step 3
- Profile preset integration

**Deliverable:** `summary_provider=summllama` works end-to-end, scored against baselines.

## Step 6: Research-powered registry (#593, ~1-2 days)

Capstone — pulls together findings from steps 1-5.

- Profile presets with code loader (cloud_balanced, cloud_quality, local, dev, airgapped)
- Autoresearch-backed defaults for each profile
- Registry integrates ML models, LLM providers, and all tuning results
- Single config surface: pick a profile, get optimal settings

**Deliverable:** `config/profiles/*.yaml` with research-backed defaults, loader code.

## Pre-requisites (first thing next session)

- Re-ingest production corpus with latest pipeline (multi-quote, provider-mode GI)
- Validate explore expansion on fresh data (see `EXPLORE_EXPANSION_VALIDATION.md`)
- Ensure all tests use dev profile (note from #598 work)

## Summary

| Step | Issue | Effort | Status |
| ---- | :---: | :----: | ------ |
| 1. Embedding compat | — | 2 hrs | **DONE** (PR #624) |
| 2. Speaker flow | #598 | 1 day | **DONE** (PR #624) |
| 3. Eval datasets | — | 1 day | Next |
| 4. KG prompt tuning | #590 | 1-2 days | Pending (needs step 3) |
| 5. SummLlama | #571 | 1 day | Pending |
| 6. Registry | #593 | 1-2 days | Pending (capstone) |
| **Remaining** | | **~4-5 days** | |

## Deferred to 2.7

| Item | Issue | Reason |
| ---- | :---: | ------ |
| Deepgram provider | #597 | Transcription strategy decision |
| Transcription autoresearch | #594 | Depends on Deepgram |
| Speaker profiles / consensus / temporal | #601 parked | Needs diarization (SPOKEN_BY edges) |
