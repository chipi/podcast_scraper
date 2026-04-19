# 2.6 Backend Work Plan

Backend-only work while UI work proceeds separately.
Deepgram (#597) and transcription autoresearch (#594) deferred to 2.7.

## Pre-requisites (first session back)

- Re-ingest production corpus with latest pipeline (multi-quote, provider-mode GI)
- Validate explore expansion on fresh data (see `EXPLORE_EXPANSION_VALIDATION.md`)

## Step 1: Embedding loader compat audit (~2 hrs)

De-risks everything that uses sentence-transformers.

Fixed `embedding_loader.py` and `nli_loader.py` for `local_files_only`
kwarg compat with sentence-transformers 2.x vs 3.x. Now audit all
remaining ML loaders:

- `summarizer.py` — uses `local_files_only` in multiple places
- Any other `SentenceTransformer` or `CrossEncoder` constructors
- Confirm all loaders work with both 2.x and 3.x

**Deliverable:** all ML loaders compatible with sentence-transformers 2.x and 3.x.

## Step 2: Speaker flow validation — backend (#598, ~1 day)

Validate end-to-end speaker pipeline integrity before tuning anything.

- Trace matrix: config settings → NER extraction → GI/KG artifact fields
- Gap list with severity (P0-P3)
- Fix or test any broken contracts in the pipeline chain
- Skip viewer/graph validation (separate UI work)

**Deliverable:** trace matrix, gap list, code + test fixes.

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

## Summary

| Step | Issue | Effort | Dependency |
| ---- | :---: | :----: | ---------- |
| 1. Embedding compat | — | 2 hrs | None |
| 2. Speaker flow | #598 | 1 day | None |
| 3. Eval datasets | — | 1 day | None |
| 4. KG prompt tuning | #590 | 1-2 days | Step 3 |
| 5. SummLlama | #571 | 1 day | Step 1 |
| 6. Registry | #593 | 1-2 days | Steps 1-5 |
| **Total** | | **~6-8 days** | |

## Deferred to 2.7

| Item | Issue | Reason |
| ---- | :---: | ------ |
| Deepgram provider | #597 | Transcription strategy decision |
| Transcription autoresearch | #594 | Depends on Deepgram |
| Speaker profiles / consensus / temporal | #601 parked | Needs diarization (SPOKEN_BY edges) |
