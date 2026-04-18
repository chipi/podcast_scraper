# Technical Debt Registry

This document tracks recognised technical debt across the project -- items that
work correctly today but have a known path to a better solution when the time
is right.

Each entry records the current state (how we cope), the ideal state (what
"done" looks like), the two or three realistic options, and a rough priority
and trigger for when to revisit.

---

## TD-001: CodeQL `py/path-injection` false positives

**Tracking:** [#538](https://github.com/chipi/podcast_scraper/issues/538)

**Status:** Managed (dismissal process in place)

**Current state:**

CodeQL flags every filesystem call (`os.walk`, `os.path.isdir`, `open`, etc.)
that receives a value derived from a FastAPI query parameter, even when the
value has been sanitised via `os.path.normpath` + `str.startswith` in a shared
helper function.  CodeQL's taint-tracking state machine requires the guard to
appear inline in the same function as the filesystem call; our architecture
performs sanitisation in shared helpers (`resolve_corpus_path_param`,
`resolved_corpus_root_str`, `normpath_if_under_root`,
`safe_relpath_under_corpus_root`).

As of April 2026, approximately 60 alerts have been dismissed as false
positives.  The full inventory and dismissal process are documented in
[`docs/ci/CODEQL_DISMISSALS.md`](../ci/CODEQL_DISMISSALS.md).

**Why this is debt, not a bug:**

- Zero security risk -- the sanitisers are correct and thoroughly tested.
- Low operational cost -- dismissing a new alert takes ~30 seconds (one
  `gh api` call + one table row), and the agent `.cursorrules` rule 16
  automates classification.
- But the codebase carries a growing list of dismissed alerts, and every new
  server route that touches the filesystem will produce more.

**Options to eliminate it:**

| Option | Approach | Pros | Cons |
| --- | --- | --- | --- |
| A | Custom CodeQL query pack | Write a `.ql` model extension that teaches CodeQL's taint tracker that `resolve_corpus_path_param` and friends are sanitisers (`isSanitizer` override). All ~60 alerts disappear at source; no dismissals needed. | Niche skill (CodeQL QL language); pack must be maintained as CodeQL evolves; benefits only this repo. |
| B | Inline sanitisation | Restructure every route handler to do `normpath + startswith` inline before any filesystem call. CodeQL sees the guard and stops flagging. | Duplicates the same 3-line guard in every handler; worse code quality for the sake of a static analysis tool; ongoing maintenance burden on every new route. |
| C | Wait for CodeQL improvement | GitHub has discussed cross-function sanitiser support in CodeQL issues. If/when they ship it, the false positives disappear with no code changes. | No timeline; may never land. |

**Recommendation:** Option C (wait) as the default posture.  Option A is the
right investment if we ever do a dedicated security hardening sprint or if the
alert volume becomes annoying.  Option B is not recommended -- it makes the
code worse.

**Priority:** Low

**Trigger to revisit:**

- CodeQL ships cross-function sanitiser modelling (Option C resolves itself)
- A new, genuinely dangerous alert type appears that requires deeper CodeQL
  investigation (justifies learning the QL language for Option A)
- Alert count exceeds ~100 and the dismissal table becomes unwieldy
- A security audit or compliance review requires zero open/dismissed alerts

---

## TD-002: Local ML stack produces shallow GI artifacts (no Quotes, weak grounding)

**Status:** Managed (CI gate softened; quality gap documented)

**Current state:**

When the pipeline runs with the local ML stack (`transcription_provider: whisper`,
`summary_provider: transformers`, `speaker_detector_provider: spacy`) on short
transcripts, the GI (Grounded Insights) output is significantly weaker than
the LLM-backed stack (OpenAI, Anthropic, etc.):

- **No Quote nodes** -- BART/LED summarisers produce bullet-point summaries but
  do not attribute them to specific transcript spans, so the GI builder has no
  source material to create Quote nodes with `char_start`/`char_end` offsets.
- **No speaker attribution** -- spaCy NER detects named entities but does not
  produce the structured speaker-to-quote mapping that LLM providers return.
- **Shallow insight text** -- local summarisers tend to produce shorter,
  less nuanced bullets than LLM models, especially on short inputs (< 1 minute
  of audio).

This surfaces in multiple places:

1. **`verify-gil-offsets-strict` CI gate** (the immediate trigger) -- the
   offset verification expects Quote nodes to exist so it can check alignment
   with FAISS transcript chunks.  Runs from ML-only acceptance configs
   (`sample_acceptance_e2e_fixture_single.yaml`) produce `verdict: no_quotes`.
   Fixed (April 2026) by treating `no_quotes` as a pass in strict mode --
   nothing to misalign means no alignment failure -- but the underlying
   quality gap remains.
2. **Search lift** -- the FAISS lift layer (RFC-072 Phase 5) boosts insight
   hits whose Quotes overlap the matched transcript chunk.  With no Quotes,
   ML-stack insights get zero lift, making search results less relevant for
   ML-only corpora.
3. **Viewer quote panel** -- the GI/KG viewer shows supporting quotes under
   each insight.  ML-stack insights have none, so the panel is empty.
4. **Position tracker / person profile** (future PRD-028, PRD-029) -- these
   features depend on grounded quotes with speaker IDs and char spans.

**Why this is debt, not a bug:**

- The ML stack is intentionally offline-first and free of API keys.  It works
  correctly for its design point (transcription + basic summarisation).
- LLM-backed configs produce full GI with Quotes and the pipeline is healthy
  there.
- The gap is a quality/feature ceiling, not a correctness issue.

**Options to close the gap:**

| Option | Approach | Pros | Cons |
| --- | --- | --- | --- |
| A | Post-hoc Quote extraction pass | After BART/LED summarisation, run a lightweight span-matching heuristic (fuzzy search of each bullet against the transcript) to synthesise Quote nodes with char offsets. No LLM needed. | Heuristic quality may be poor on short or paraphrased bullets; adds pipeline latency; needs new code + tests. |
| B | Small local LLM for GI grounding | Use a small quantised model (e.g. Qwen 3.5, Phi-4-mini via Ollama) only for the GI grounding step while keeping Whisper + BART for transcription/summary. Hybrid stack. | Requires Ollama or similar runtime; not zero-dependency; model download on first run. |
| C | Accept the gap, document clearly | Keep ML stack as "basic tier" and LLM stack as "full tier".  Document in README, acceptance configs, and UXS that Quote-dependent features require an LLM provider. | No engineering cost; but users on ML stack get a visibly worse experience in search lift, viewer, and future features. |
| D | Sentence-level pseudo-quotes | For each summary bullet, find the best-matching sentence in the transcript (TF-IDF or embedding cosine) and emit a Quote node spanning that sentence. Lighter than Option A (no fuzzy char alignment). | Still heuristic; one sentence per bullet may miss multi-sentence evidence; but good enough for lift + viewer. |

**Recommendation:** Option D as a near-term pragmatic fix (gives Quote nodes
for lift and viewer with minimal complexity), with Option B as the long-term
path if local LLM runtimes become standard in the user base.  Option C is the
fallback if engineering bandwidth is tight.

**Priority:** Medium

**Trigger to revisit:**

- Position tracker (PRD-028) or person profile (PRD-029) implementation begins --
  both hard-depend on Quote nodes with speaker IDs
- User feedback indicates ML-stack search results are noticeably worse
- A lightweight local model (< 2 GB) becomes easy to bundle or auto-download
- Acceptance benchmark (`docs/wip/gil-ml-vs-openai-outcome-benchmark.md`)
  produces quantitative lift/quality comparisons that justify the investment

---

## TD-003: Multi-feed ML acceptance (historical: Whisper reload / PyTorch meta in one process)

**Tracking:** [#539](https://github.com/chipi/podcast_scraper/issues/539)

**Status:** Addressed (2026-04). Multi-feed batches defer shared ML singleton teardown between feeds;
`preload_ml_models_if_needed` reuses matching ML fingerprints; HF QA cache cleared between feeds;
QA pipelines pass `low_cpu_mem_usage=False`; `summarizer.unload_model` skips `.to("cpu")` on meta
shells. `sample_acceptance_e2e_fixture_multi_ml` is back in `FAST_CONFIGS.txt`.

**Historical (pre-fix):** five feeds in one process could leave HF QA / torch in a state where
feed 2’s Whisper load failed with meta-tensor errors; CI weights were present (not a cache miss).

**Local smoke:** `make test-acceptance CONFIGS=config/acceptance/sample_acceptance_e2e_fixture_multi_ml.yaml USE_FIXTURES=1`

**Trigger to revisit:**

- Regression on `make test-acceptance-fixtures-fast` or multi-feed ML acceptance
- Changes to `preload_ml_models_if_needed`, `_cleanup_providers`, multi-feed `service.run`,
  or HF QA / Whisper init paths

---

## TD-004: Bandit `B615` Hugging Face Hub download suppressions

**Status:** Managed (`# nosec B615` applied at every `from_pretrained()` call site)

**Current state:**

Bandit's `B615` rule flags every call to `AutoTokenizer.from_pretrained(...)` /
`AutoModelForX.from_pretrained(...)` / `BartForConditionalGeneration.from_pretrained(...)`
etc. that does not pin `revision=<sha>`. Without pinning, Hugging Face Hub
returns whatever is currently at `main` on the repo — so in principle a model
repo owner (or a Hub compromise) could change the downloaded weights or loader
code under us.

As of April 2026, the codebase has **18 `# nosec B615` suppressions** across:

- `src/podcast_scraper/providers/ml/hybrid_ml_provider.py` (×4)
- `src/podcast_scraper/providers/ml/model_loader.py` (×4)
- `src/podcast_scraper/providers/ml/summarizer.py` (×6)
- `scripts/eval/run_longt5xl_v2.py` (×2)
- `scripts/eval/run_summllama_v2.py` (×2)

Every production ML model load uses this pattern. Eval standalone scripts were
added in the v2 sweep (PR #568) and were adjusted to match.

**Why this is debt, not a bug:**

- We rely on HF Hub's integrity for the model names we ship (`facebook/bart-large-cnn`,
  `DISLab/SummLlama3.2-3B`, etc.). No evidence of any malicious updates on the
  specific models we use.
- Pinning `revision="main"` satisfies bandit but **does not actually fix the
  concern** — it just names the same moving target. True pinning requires a
  commit SHA per model, which adds maintenance burden and breaks users who want
  to pull updates.
- Loader code in HF transformers (`from_pretrained`) is the trusted path; we do
  not do arbitrary-code loading with `trust_remote_code=True` anywhere.

**Options to eliminate it:**

| Option | Approach | Pros | Cons |
| --- | --- | --- | --- |
| A | Per-model SHA pin | Maintain a `HF_MODEL_REVISIONS` map keyed by model name to commit SHA; thread through every `from_pretrained()` call. | Reproducible loads; removes all suppressions. Maintenance burden — SHA must be updated whenever we want newer weights; first-boot requires network even with local cache. |
| B | Trusted-model allowlist + cache-first loader | Centralise HF loads through `model_loader.py`, make it refuse to download models not in an allowlist, and pre-populate the HF cache in CI / Docker build. Suppressions stay but the risk surface is narrower. | Concentrates the suppressions in one place; CI/Docker builds become deterministic. Doesn't remove bandit findings. |
| C | Accept bandit suppressions as policy | Keep the current pattern; document here so the decision is legible to audit. | Zero extra work. Doesn't improve posture. |

**Recommendation:** Option C (this entry is the deliberate choice). Option B is
the right upgrade if we ever do a security hardening sprint or need to prove
supply-chain discipline to an auditor. Option A is the gold standard but
out-of-proportion for the current risk profile.

**Priority:** Low

**Trigger to revisit:**

- Any report of a malicious model on HF Hub with a name we depend on
- Compliance/audit review that requires documented supply-chain controls
- Introduction of `trust_remote_code=True` anywhere (then the threat model
  changes materially)
- A deployment target where we can't pre-cache models (e.g., short-lived
  serverless) — Option B becomes valuable there

---

<!-- Add new TD-NNN entries above this line, following the same template. -->
