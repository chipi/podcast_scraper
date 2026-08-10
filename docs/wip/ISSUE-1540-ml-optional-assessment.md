# Issue #1540 — make the local-ML path opt-in: assessment + split

**Status:** assessment complete; scope split agreed (2026-08-10).
**Branch:** `feat/long-term-fixes` (unpushed).

Grounded in the code, not the issue's framing. #1540 as filed bundles two things
with very different blast radius: **CI/build-time wins** (small, mostly already
done) and a **torch-free runtime image** (large — Search is hard-coupled to a
locally-built embedding index). We split them.

## The split

- **#1540 (this) → the CI/build wins we can do now:** lever **F** (cache the ML
  model download for e2e) + the doc-vs-code divergence it surfaced.
- **New discussion issue → the design that needs to be understood first:** lever
  **A** (torch-free default image + gateway embeddings), lever **E** (CI
  path-narrowing), and lever **D** (NER prepass on cloud). See "Deferred" below.

## What actually drags torch into the "cloud" image (verified)

- The deploy image is `pipeline-llm` = `[llm,search]`. stack-test **already never
  publishes `pipeline-ml`** (`stack-test.yml:190`) → Whisper/spaCy/pyannote/Pegasus
  are already out of the shipped image. The weight is **`[search]` → torch-CPU +
  sentence-transformers** (`docker/pipeline/Dockerfile:113-118`).
- cloud_balanced sets `vector_search: true` + `vector_embedding_provider:
  sentence_transformers` (`cloud_balanced.yaml:161-162`) → builds a LanceDB index
  at pipeline finalize, loading `all-MiniLM-L6-v2` locally → the runtime HF-Hub
  pull. `[llm]` alone has no torch; `[search]` is the whole cost.

## Blast radius, per lever

| Lever | Blast radius | Disposition |
|---|---|---|
| F. Cache e2e ML download | Low / already done | **#1540** (verify + doc) |
| E. CI path-filters (cloud change skips ML e2e/publish) | Medium — re-opens #1527 | Discussion issue |
| D. NER prepass off on cloud | Entangled — see below | Discussion issue |
| A. Ship `[llm]`-only default (no torch) | **High** — breaks Search | Discussion issue |

## F — already implemented (this is the "win" the issue wanted)

`python-app.yml` already has a `preload-ml-models` job with `actions/cache@v5`
keyed on `preload_ml_models.py` + `config.py` (lines 719-735), handing models to
the heavy test jobs via a shared `ml-models` artifact (765-809). Warm cache →
preload finishes in seconds. The residual ~24-min tax (line 907 comment: a 24-min
artifact download seen on #1274) is the **9 GB artifact download + full
`.[dev,ml,llm,search]` pip install** in each heavy test job — and that is
lever-A-entangled (the tests exercise the ML stack), not a standalone cache win.
`app-e2e` (605-645) already shows the lean pattern: `.[dev,search]` + preload only
the embedding model (`SKIP_WHISPER/SPACY/TRANSFORMERS`).

**Conclusion:** no safe standalone F code change remains. The one thing this
surfaced is a divergence fix (below).

### Divergence fixed (safe, in this commit)

`kg/pipeline.py:151` docstring said `kg_extraction_use_ner_prepass` is
"False (default)". The Field default is **`True`** (config.py:3422). Corrected +
annotated with the cloud-image caveat.

## Deferred to the discussion issue (with rationale)

### E — CI path-narrowing re-opens the #1527 gap

After #1527 (a config-only litellm change broke acceptance because a gate didn't
run), the repo **added** an `acceptance` path-filter so cloud/profile/provider
changes run *more* gates (`python-app.yml:167-178`, comment names #1527).
Narrowing CI to skip ML on cloud-only changes fights that fix. A safe version
must preserve the acceptance gate — that's a design conversation, not a quick win.

### D — NER prepass on cloud is not a cleanup; it's a latent-quality question

- `kg_extraction_use_ner_prepass` defaults **True** (config.py:3422); it's a real
  quality lever (#1035: 0%→100% entity coverage on DGX models).
- cloud_balanced has it on, but `get_ner_model` is **spaCy-only**
  (`speaker_detection.py:55-57`), spaCy is `[ml]`-only (`pyproject.toml:101`), and
  the shipped cloud image is `[llm,search]` → the pass load-fails and silently
  downgrades to v4 (`kg/pipeline.py:170-175`).
- So cloud KG entity extraction **already runs the v4 baseline in prod**. Whether
  that materially hurts recall on the litellm cloud model (deepseek-v4-flash) —
  vs the DGX models #1035 measured — is **unmeasured**. Turning the flag off would
  lock in that state without answering the question; wiring cloud NER drags ML in
  (lever A). Needs measurement first.

### A — torch-free default image (the big one)

BM25 exists (`search_bm25` / FTS) but lives **inside the same LanceDB index** whose
build needs embeddings — so "no torch" loses keyword search too, and viewer Search
+ MCP `search_corpus` + player all hard-fail `no_index`
(`hybrid_search.py:12,217` — "there is no fallback"). **Shrinker:** a
gateway-embeddings path already exists — `vector_embedding_provider: "ollama"` +
`vector_embedding_endpoint` routes encoding to a remote OpenAI-compatible endpoint
(`embedding_loader.py:158-166`); `skip_auto_vector_index` is already a flag
(`indexer.py:577`). So torch-free-**and**-search-intact ≈ point cloud_balanced's
embeddings at the LiteLLM gateway + a new `[search-remote]` extra (lancedb without
torch) — **if** the gateway serves an `/embeddings` route.

## NOT verified / open (equal weight)

- **Build-time numbers** (~50 min pipeline-llm, ~28 min e2e) — not measured; the
  issue author's observation. Torch is architecturally the heavy install.
- **Does LiteLLM expose `/embeddings` today?** Load-bearing for lever A. It routes
  chat today; embeddings is a separate route not confirmed configured.
- **Is prod cloud KG actually degraded** by the v4 fallback? Unmeasured (lever D).
- **`audio_preprocessing_profile: speech_optimal_v1` on cloud** — sets encode
  params (bitrate) consumed by the openai/gemini providers (`openai_provider.py:716`,
  `gemini_provider.py:730`); did not confirm whether it pulls any torch on the
  deepgram path. Assumed no; not verified.

## Next step for lever A (discussion)

Confirm the LiteLLM `/embeddings` unknown first — it decides whether A is "wire
config + a lean extra" or "stand up an embeddings route first."
