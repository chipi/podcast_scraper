# Issue #382 — Transformers v5 upgrade: execution plan

Companion to `ISSUE-382-TRANSFORMERS-V5-DEEP-ANALYSIS-2026-07-05.md`
(architecture + rationale). This file is the **step-by-step execution
manual**: exact commands, exact edits, per-phase test harness updates,
docs updates, registry / `data/eval` integration, and the push gate.

- **Branch:** `feat/upgrade-transformers-v5` (created 2026-07-05 off local `main` = `c1249814`).
- **Author:** operator + Claude Code.
- **Do-not-push posture:** every phase ends in a commit on this branch;
  push and PR happen only after the operator says "push".

## Scope — Path C epic (2026-07-05 revision)

Per operator direction (2026-07-05): #382 is treated as an **epic**. Beyond
the mechanical `pipeline()` → `generate()` migration, this branch also
folds three architectural collapses so the local ML stack ends up
aligned with the AI-provider abstraction ("just different profiles and
providers"). No separate GH sub-issues; the branch delivers all of it
in one PR that closes #382.

Overview of the 13 phases (10 original + 3 architectural):

| # | Phase | Wall-time | Gate |
|---|---|---|---|
| 0 | Baseline capture (pre-upgrade) | 30–45 min | Baseline artifacts committed under `data/eval/baselines/…v5_pre` |
| 1 | Dependency pass (`pyproject`) + local install | 30 min | `pip install` resolves clean; `.venv` imports all four packages |
| 2 | Registry / revision sanity + cache-path cleanup | 30 min | `get_transformers_cache_dir()` still works; every pinned revision still resolves |
| 3 | Extractive QA rewrite (drop `pipeline("question-answering")`) | ~4 h | New QA unit tests pass; GI grounding tests unchanged |
| 4 | Hybrid reduce rewrite (drop `pipeline("text2text-generation")`) | ~2 h | `TransformersReduceBackend` unit tests + FLAN-T5 smoke |
| 5 | Summarizer rewrite (drop `pipeline("summarization")` + adopt `GenerationConfig`) | ~6 h | `module_summarization` marker green, `ml_models` marker green |
| **E** | **Unify evidence backends** — `HFEvidenceBackend` for QA/NLI/embedding | ~4 h | 3 modules collapse to 1 abstraction + thin heads; public API preserved |
| **F** | **Unify HF seq2seq loaders** — `HFSeq2SeqBackend` for map + reduce | ~4 h | `SummaryModel` and `TransformersReduceBackend` become thin profiles |
| **G** | **Unify HF download helpers** — one `_download_hf_model(kind, model_id)` | ~1 h | 3 `_download_*_for_cache` functions collapse to 1 |
| 6 | Test-harness updates + coverage add-ons | ~3 h | New parity tests + updated markers; `pytest-cov` no regression |
| 7 | Post-upgrade eval run + `run-compare` gate | 1–2 h | ROUGE-L / QA-EM parity vs Phase-0 baseline (`>=0.95` / `>=0.98`) |
| 8 | Docs sweep (ADRs / guides / registry doc / CHANGELOG / issue #382) | 2 h | `make docs` (mkdocs strict) green |
| 9 | Makefile CVE-ignore removal + `make ci-fast` | 1 h + wall | `make ci-fast` green; secrets-scan clean |
| — | (operator-gated) push + PR ready-for-review | — | rebase onto origin/main → force-with-lease → open PR |

Total: **~3–4 focused days** with the architectural additions. Original
mechanical migration (~2 days) + E/F/G (~1–2 days).

## Architectural target — after this branch

Before this branch, the local ML stack had three parallel shapes for
"load HF model, cache it, forward it":

- `SummaryModel` (summarizer.py) — map model, class-based, ~800 LOC of
  loading + generation + retry + validation.
- `TransformersReduceBackend` (hybrid_ml_provider.py) — reduce model,
  class-based, different shape.
- `extractive_qa` / `nli_loader` / `embedding_loader` — bare-function
  triplet, near-identical device/cache/threading-lock idioms.

After this branch:

- `HFSeq2SeqBackend` — one loader/generator for BART/LED/Pegasus/LongT5/
  FLAN-T5; `SummaryModel` (map profile) and `TransformersReduceBackend`
  (reduce profile) are thin configs over it.
- `HFEvidenceBackend` — one loader for the evidence stack; QA / NLI /
  embedding are heads over it. `extractive_qa.answer_candidates(...)`
  etc. remain the public surface (via thin module wrappers) so callers
  don't move.
- `_download_hf_model(kind, model_id, **hf_kwargs)` in model_loader —
  one function replaces the three parallel download helpers.

Net LOC delta: **negative** (~-400 LOC after all collapses land).

---

## Guard rails (non-negotiable — verify before every commit)

- Every phase is one commit. Message follows conv-commit
  `<type>(<scope>): <subject>` per repo history.
- **No push** until operator says "push" or "ship it" (rule #1). Rebase
  before push (rule #2).
- **No new deps beyond the pyproject range bumps.** Any transitive that
  pulls in something surprising → surface for approval.
- **No skipped tests, no `xfail` markers.** If a test breaks, fix root
  cause; if the test was invalid, delete it explicitly with note.
- Every phase re-runs its **own** `pytest` subtarget before commit — not
  `make ci-fast` (rule #9 & #18).
- **`make ci-fast` runs once**, at Phase 9, as the final green light.
- **Historical eval artifacts are frozen** — Phase 0 baseline writes to a
  NEW run id under `data/eval/baselines/`; nothing under
  `data/eval/runs/` is mutated (`[[feedback_never_mutate_historical_artifacts]]`).

---

## Phase 0 — Pre-upgrade baseline capture

**Why:** we need a numeric anchor to prove Phase 7's post-upgrade output
is at parity (or better). Doing this on the current 4.57.6 pin captures
the "authoritative" v4 behavior we're leaving.

### Steps

1. Verify local venv state:
   ```bash
   .venv/bin/pip show transformers sentence-transformers torch accelerate \
     | awk '/^Name:|^Version:/'
   ```
   Expected: `transformers 4.57.6`, `sentence-transformers 5.5.1`,
   `torch 2.12.0`, `accelerate 1.13.0`.

2. Warm the HF cache (the pre-upgrade baseline needs local weights;
   downloads MUST NOT happen mid-run per `[[project_overview]]` supply-chain rule).
   ```bash
   make preload-ml-models
   ```

3. Capture the **summarizer** baseline on the smoke dataset:
   ```bash
   make baseline-create \
     BASELINE_ID=baseline_ml_bart_authority_smoke_v5_pre \
     DATASET_ID=curated_5feeds_smoke_v1 \
     PREPROCESSING_PROFILE=cleaning_v4
   ```
   Same recipe used for the existing
   `baseline_ml_bart_authority_smoke_v1` — apples-to-apples.

4. Capture the **hybrid_ml** baseline (Pegasus retirement path) so
   Phase 4's rewrite has a matched anchor:
   ```bash
   make baseline-create \
     BASELINE_ID=baseline_ml_pegasus_retirement_smoke_v5_pre \
     DATASET_ID=curated_5feeds_smoke_v1 \
     PREPROCESSING_PROFILE=cleaning_v4
   ```

5. Capture the **extractive QA** span outputs on a fixed fixture. There
   is no existing eval harness for QA alone, so we tap into the GI
   grounding tests' fixture. Author a small script under
   `scripts/dev/capture_qa_baseline.py` (~40 lines) that:
   - Loads `deepset/roberta-base-squad2` via
     `extractive_qa.answer_candidates(...)`
   - Iterates a fixed list of `(context, question)` pairs drawn from
     `tests/fixtures/gi/` or a Phase-0 curated set
   - Emits JSONL with `{context_id, question, top_k_spans:
     [{start, end, score, text}], device, transformers_version}`
   - Output path: `data/eval/references/qa_baseline_v5_pre.jsonl` (
     `references/` is the right home per the folder's existing
     convention).

6. Commit the baselines:
   ```bash
   git add data/eval/baselines/baseline_ml_bart_authority_smoke_v5_pre \
           data/eval/baselines/baseline_ml_pegasus_retirement_smoke_v5_pre \
           data/eval/references/qa_baseline_v5_pre.jsonl \
           scripts/dev/capture_qa_baseline.py
   git commit -m "chore(eval): capture pre-v5 baselines for #382 parity gate"
   ```

**Gate:** three new artifacts on disk; commit created. If any baseline
run fails, halt — do NOT continue until the failure is diagnosed on
v4.57.6.

---

## Phase 1 — Dependency pass + local install

### Steps

1. Edit `pyproject.toml`:
   ```diff
    ml = [
      ...
   -  # sentence-transformers 5.x allows transformers 4.41–5.x; keep <5 until we vendor
   -  # extractive QA: transformers 5 removed the ``question-answering`` pipeline task (CI preload).
   -  # CVE-2026-1839 (Trainer rng_state / torch.load): fixed in 5.0.0rc3+ only; pip-audit ignore in Makefile until 5.x stable.
   -  "transformers>=4.57.6,<5.0.0",
   +  # transformers v5: pipeline() rewrites landed in #382 (2026-07); dropped `pipeline(...)` entirely for
   +  # direct `AutoModel*.from_pretrained` + `generate()` / QA-head forward. CVE-2026-1839 fixed in >=5.0.0rc3.
   +  "transformers>=5.0.0,<6.0.0",
      ...
   -  "sentence-transformers>=5.4.1,<6.0.0",
   +  # 5.6.0 fixes silent causal-LM reranker scoring bug + restores TSDAE on transformers v5.
   +  "sentence-transformers>=5.6.0,<6.0.0",
   ...
    search = [
      ...
   -  "transformers>=4.57.6,<5.0.0",
   +  "transformers>=5.0.0,<6.0.0",
      ...
   -  "sentence-transformers>=5.4.1,<6.0.0",
   +  "sentence-transformers>=5.6.0,<6.0.0",
   ```

2. Install with **the same command flow** the Makefile uses (`make init`
   would rebuild the whole venv):
   ```bash
   .venv/bin/pip install --upgrade -e '.[dev,ml,search,llm,compare]'
   ```
   Save full resolver output to `/tmp/v5-resolver.log`; scan for
   `ResolutionImpossible`, `Requires-Python`, or downgrade prints.

3. Verify:
   ```bash
   .venv/bin/python -c "
   import transformers, sentence_transformers, torch, accelerate, huggingface_hub, tokenizers
   print('transformers', transformers.__version__)
   print('sentence_transformers', sentence_transformers.__version__)
   print('torch', torch.__version__)
   print('accelerate', accelerate.__version__)
   print('huggingface_hub', huggingface_hub.__version__)
   print('tokenizers', tokenizers.__version__)
   "
   ```
   Expected: `transformers >= 5.0.0`, `sentence_transformers >= 5.6.0`,
   others unchanged. **If `huggingface_hub < 1.0` after upgrade,
   halt — v5 needs 1.0+ per release notes.**

4. Commit **`pyproject.toml`** only (no code changes yet):
   ```bash
   git add pyproject.toml
   git commit -m "chore(deps): raise transformers to >=5.0, sentence-transformers to >=5.6 (#382)"
   ```

**Gate:** clean resolve; imports succeed at correct versions.

---

## Phase 2 — Registry / revision sanity + cache-path cleanup

Two small orthogonal jobs; one commit.

### 2a. Delete `transformers.file_utils` fallback

`src/podcast_scraper/cache/directories.py`:

```diff
    # 3. Fall back to huggingface_hub constants
    try:
        # Try modern huggingface_hub API first (transformers 4.20+)
        from huggingface_hub import constants
        if constants.HF_HUB_CACHE:
            return Path(constants.HF_HUB_CACHE)
    except (ImportError, AttributeError, TypeError):
        pass

-    try:
-        # Fallback to transformers file_utils (older versions)
-        from transformers import file_utils
-        if hasattr(file_utils, "default_cache_path") and file_utils.default_cache_path:
-            return Path(file_utils.default_cache_path)
-    except (ImportError, AttributeError, TypeError):
-        pass

    # 4. Standard user cache as final fallback
    return Path.home() / ".cache" / "huggingface" / "hub"
```

Adjust the docstring priority list (lines 73-83) — drop step 4, renumber.

### 2b. Registry revision sanity check

Every pinned revision SHA in `src/podcast_scraper/config_constants.py`
must still resolve under v5. HF revisions are git SHAs, so this can
only break if HF removes a repo (rare). One-liner smoke:

```bash
.venv/bin/python - <<'PY'
from podcast_scraper import config_constants as C
from transformers import AutoConfig
for name in ("PEGASUS_CNN_DAILYMAIL_REVISION", "LED_BASE_16384_REVISION",
             "LED_LARGE_16384_REVISION", "FLAN_T5_BASE_REVISION",
             "FLAN_T5_LARGE_REVISION", "LONG_T5_TGLOBAL_BASE_REVISION",
             "LONG_T5_TGLOBAL_LARGE_REVISION"):
    rev = getattr(C, name)
    print(name, rev, "OK" if rev else "MISSING")
PY
```

No code change here — this is a **verification**. Any missing revision
becomes a Phase-2 sub-task, not a silent skip.

### 2c. Verification + commit

```bash
.venv/bin/python -c "from podcast_scraper.cache import get_transformers_cache_dir; print(get_transformers_cache_dir())"
.venv/bin/python -m pytest tests/unit/podcast_scraper/cache -x -q
git add src/podcast_scraper/cache/directories.py
git commit -m "refactor(cache): drop transformers.file_utils fallback (v5 removed the module) (#382)"
```

**Gate:** `get_transformers_cache_dir()` still returns the same path;
cache-module unit tests green; revision smoke shows every pin resolves.

---

## Phase 3 — Extractive QA rewrite

The riskiest of the three pipeline replacements because the QA
top-k scoring semantics are subtle. Do this first so equivalence testing
runs against the smallest surface.

### 3a. Rewrite `src/podcast_scraper/providers/ml/extractive_qa.py`

Public API unchanged: keep `QASpan`, `answer`, `answer_candidates`,
`answer_multi`, `get_qa_pipeline`, `clear_qa_pipeline_cache`.

Internal switch:
- Replace `load_qa_pipeline` → `load_qa_model` that returns a
  `(model, tokenizer)` tuple loaded via
  `AutoModelForQuestionAnswering.from_pretrained(...)` +
  `AutoTokenizer.from_pretrained(...)` with the same `cache_dir`,
  `local_files_only=True`, `low_cpu_mem_usage=False` kwargs we use today.
- Replace `_qa_pipeline_call` / `_qa_pipeline_call_top_k` with a single
  `_qa_forward(model, tokenizer, question, context, top_k)` that:
  - Tokenizes with `truncation="only_second"`,
    `return_offsets_mapping=True`, `stride=128`, `max_length=384`,
    `padding="max_length"`, `return_tensors="pt"` (matches
    transformers' internal QA pipeline defaults).
  - Runs `model(**enc)` under `torch.no_grad()`.
  - Walks the top-k `(start, end)` pairs by
    `start_logit + end_logit` score, filtering to (a) `start <= end`,
    (b) `end - start <= max_answer_length` (default 30), (c) both
    indices in the context segment (`sequence_ids == 1`).
  - Maps back to `(char_start, char_end, text)` via
    `offset_mapping`.
  - Returns `list[QASpan]` in descending score order.
- Windowed path (`_iter_context_windows`) unchanged.

### 3b. Update `src/podcast_scraper/providers/ml/model_loader.py`

- Delete `build_huggingface_qa_pipeline`,
  `_call_transformers_qa_pipeline`, `_download_qa_pipeline_for_cache`.
- Introduce `download_qa_model_for_cache(model_id: str) -> None` — one
  `AutoModelForQuestionAnswering.from_pretrained(model_id, cache_dir=...)`
  + one `AutoTokenizer.from_pretrained(...)`.
- Update `preload_evidence_models(qa_models=...)` and
  `is_evidence_model_cached(model_id)` to check for
  `pytorch_model.bin` / `model.safetensors` + `tokenizer.json` +
  `config.json` in the snapshot dir instead of running the pipeline.

### 3c. Test harness updates for Phase 3

**New file:** `tests/unit/podcast_scraper/providers/ml/test_extractive_qa.py`

- `test_load_qa_model_returns_model_and_tokenizer` — patches `AutoModel*`
  + `AutoTokenizer` at the module boundary; asserts the pair returned.
- `test_answer_returns_top1_span` — mocks the forward pass to return
  fixed logits; asserts `QASpan.start`, `.end`, `.text`, `.score` match
  the argmax pair.
- `test_answer_candidates_returns_topk_sorted` — asserts top-k output is
  scored-desc and non-identical.
- `test_answer_candidates_windowing` — long-context path falls back to
  single window + logs the debug message.
- `test_answer_candidates_filters_impossible_spans` — invalid
  `(start > end)` pair is skipped.
- Marker: `pytestmark = [pytest.mark.unit, pytest.mark.module_ml_providers]`.

**Existing tests that keep working:**
- `tests/unit/podcast_scraper/gi/test_grounding.py` — patches at
  `extractive_qa.answer_candidates` (public API); refactor is invisible.
- `tests/unit/podcast_scraper/test_config.py::test_extractive_qa_*` — no
  code change needed.

**Parity test (belt-and-suspenders):** a marked-nightly test
`tests/integration/test_qa_v5_parity.py` (marker: `nightly`,
`ml_models`, `slow`) that:
- Loads `deepset/roberta-base-squad2` real model
- Iterates `data/eval/references/qa_baseline_v5_pre.jsonl` (Phase 0)
- Asserts top-1 `(start, end)` **exact match**, top-1 score within
  `abs(delta) < 0.05` (small drift allowed for dtype/backend).

### 3d. Commands + commit

```bash
.venv/bin/python -m pytest tests/unit/podcast_scraper/providers/ml/test_extractive_qa.py -x -v
.venv/bin/python -m pytest tests/unit/podcast_scraper/gi/ -x -q
git add src/podcast_scraper/providers/ml/extractive_qa.py \
        src/podcast_scraper/providers/ml/model_loader.py \
        tests/unit/podcast_scraper/providers/ml/test_extractive_qa.py \
        tests/integration/test_qa_v5_parity.py
git commit -m "refactor(ml): replace QA pipeline() with AutoModelForQuestionAnswering forward (#382)"
```

---

## Phase 4 — Hybrid reduce rewrite

Small, contained. Same pattern as Phase 3 but for
`text2text-generation`.

### 4a. Rewrite `TransformersReduceBackend` (`hybrid_ml_provider.py:65-226`)

- Remove `pipeline("text2text-generation", ...)` build (line 183).
- Load model + tokenizer with the existing snapshot-first logic
  (`get_transformers_snapshot_path` lookup at lines 127-166 — keep).
- New `reduce()` body:
  ```python
  inputs = self._tokenizer(prompt, return_tensors="pt",
                           truncation=True, max_length=1024).to(self._device)
  gen_cfg = GenerationConfig(
      max_new_tokens=max_new_tokens,
      num_beams=num_beams,
      do_sample=do_sample,
  )
  with torch.no_grad():
      output_ids = self._model.generate(**inputs, generation_config=gen_cfg)
  text = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
  return HybridReduceResult(text=text.strip(),
                            backend="transformers",
                            model=self.model_name)
  ```
- Delete the `cast(Any, pipeline)` workaround (line 182 comment).
- Public `HybridReduceResult` shape unchanged.

### 4b. Test harness updates for Phase 4

**Existing:** `tests/unit/podcast_scraper/summarization/test_hybrid_provider.py`
uses the abstract `InferenceBackend` protocol — mocks
`TransformersReduceBackend.reduce()` at the class boundary. Refactor
invisible.

**New in same file:** `test_transformers_reduce_backend_uses_generate`
— mocks `AutoModelForSeq2SeqLM` + `AutoTokenizer` on the module, calls
`initialize()` + `reduce("notes", "instr")`, asserts `.generate()` was
called with `GenerationConfig(max_new_tokens=..., num_beams=..., do_sample=...)`.

**Manual smoke** (documented, not automated CI):
```bash
.venv/bin/python - <<'PY'
from podcast_scraper.providers.ml.hybrid_ml_provider import TransformersReduceBackend
b = TransformersReduceBackend(model_name="google/flan-t5-base",
                              device=None, cache_dir=None)
b.initialize()
r = b.reduce("Point 1. Point 2. Point 3.",
             "Summarize the following notes into one paragraph.")
print(r.text[:200])
PY
```

### 4c. Commands + commit

```bash
.venv/bin/python -m pytest tests/unit/podcast_scraper/summarization/test_hybrid_provider.py -x -v
git add src/podcast_scraper/providers/ml/hybrid_ml_provider.py \
        tests/unit/podcast_scraper/summarization/test_hybrid_provider.py
git commit -m "refactor(ml): replace hybrid_ml text2text pipeline() with generate() (#382)"
```

---

## Phase 5 — Summarizer rewrite

Largest single phase. Do it AFTER 3 and 4 so the pattern is proven.

### 5a. Introduce `_generate_summary` helper on `SummaryModel`

Signature and body per §2a of the deep-analysis doc. Adopt
`GenerationConfig`:

```python
def _generate_summary(
    self,
    text: str,
    *,
    max_length: int,
    min_length: int,
    num_beams: int,
    no_repeat_ngram_size: int,
    early_stopping: bool,
    length_penalty: float = 1.0,
    repetition_penalty: float = 1.0,
) -> str:
    inputs = self.tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=self._encoder_max_length_for_model,
    ).to(self.device)
    gen_cfg = GenerationConfig(
        max_new_tokens=max_length,
        min_new_tokens=min_length,
        num_beams=num_beams,
        no_repeat_ngram_size=no_repeat_ngram_size,
        early_stopping=early_stopping,
        length_penalty=length_penalty,
        repetition_penalty=repetition_penalty,
    )
    with self._summarize_lock, torch.no_grad():
        output_ids = self.model.generate(**inputs,
                                          generation_config=gen_cfg)
    return self.tokenizer.decode(output_ids[0],
                                 skip_special_tokens=True).strip()
```

### 5b. Delete `_load_model_move_to_device_and_pipeline`'s pipeline creation

- Rename the method → `_load_model_move_to_device` (drops "and_pipeline").
- Body keeps everything except lines 1046-1051 (the `pipeline(...)`
  call). `self.pipeline` attribute goes away entirely; `SummaryModel`
  now exposes `self.model` + `self.tokenizer` + `_generate_summary`.

### 5c. Replace call sites

- Line 1520 (`self.pipeline(input_text, **pipeline_kwargs)`) →
  `self._generate_summary(input_text, **effective_gen_kwargs)`. The
  `pipeline_kwargs` local converts to the helper's signature.
- Lines 1590-1603 (OOM CPU fallback) — replace pipeline rebuild with
  simple `self.model.to("cpu"); self.device = "cpu"` then re-call
  `_generate_summary`.
- Lines 1633 ("Already borrowed" retry loop) — same swap.
- Remove all `type: ignore[call-overload]` decorators the pipeline call
  required.

### 5d. Attempt to remove `_load_pegasus_without_fake_warning`

Load Pegasus like BART, log a WARNING once (not raise) if
`missing_keys - allowed_missing` is non-empty. If v5 still emits the
misleading warning, keep the validator; if not, delete both
`_load_pegasus_without_fake_warning` and its call site
(`summarizer.py:1195-1206`).

**Empirical test — do this in the shell before committing Phase 5:**
```bash
.venv/bin/python - <<'PY' 2>&1 | head -30
import warnings, logging
logging.basicConfig(level=logging.WARNING)
warnings.simplefilter("always")
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
m = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-cnn_dailymail",
                                          revision="d47d13...")  # PEGASUS_CNN_DAILYMAIL_REVISION
print("loaded, config.static_position_embeddings=",
      getattr(m.config, "static_position_embeddings", None))
PY
```
If nothing prints between "loaded" and "config.static..." — delete the
validator. If the "newly initialized" warning still appears, keep it.

### 5e. Test harness updates for Phase 5

**Existing files** (all continue to pass; internals swapped but public
`SummaryModel.summarize(text, max_length, min_length)` signature
unchanged):
- `tests/unit/podcast_scraper/test_summarizer.py`
- `tests/unit/podcast_scraper/test_summarizer_functions.py`
- `tests/unit/podcast_scraper/test_summarization_failure_handling.py`
- `tests/unit/podcast_scraper/test_summarizer_security.py`

**Edits needed:** any test that patches
`transformers.pipeline` directly (grep first — likely none, since the
existing tests already mock at the `SummaryModel.summarize` boundary).

**New tests to add:**
- `test_summarizer_uses_generation_config` (in `test_summarizer.py`):
  patches `AutoModelForSeq2SeqLM.from_pretrained`, verifies
  `SummaryModel.summarize(...)` calls `model.generate(**inputs,
  generation_config=GenerationConfig(...))` with the correct kwargs.
- `test_summarizer_oom_fallback_no_pipeline_rebuild`: raises
  `RuntimeError("out of memory")` from `model.generate` first call,
  asserts second call is on CPU device — no `pipeline()` in the mock's
  call list.
- `test_summarizer_generation_config_from_registry` (integration-fast):
  loads BART-base from cache, verifies `model.generation_config.num_beams`
  matches the value we passed vs the model's default.

**New end-to-end parity test:**
`tests/integration/test_summarizer_v5_parity.py` (marker: `integration,
ml_models, slow`) that:
- Loads BART-base from cache
- Runs `SummaryModel.summarize()` on a fixed 5-transcript fixture
- Compares to `data/eval/baselines/baseline_ml_bart_authority_smoke_v5_pre/predictions.jsonl`
- Asserts per-episode `ROUGE-L(pre, post) >= 0.95` (deterministic seeds,
  same checkpoint, near-identical output).

### 5f. Commands + commit

```bash
.venv/bin/python -m pytest -m "unit and module_summarization" -x -v
.venv/bin/python -m pytest tests/unit/podcast_scraper/test_summarizer.py -x -v
git add src/podcast_scraper/providers/ml/summarizer.py \
        tests/unit/podcast_scraper/test_summarizer.py \
        tests/integration/test_summarizer_v5_parity.py
git commit -m "refactor(ml): replace summarization pipeline() with generate() + GenerationConfig (#382)"
```

---

## Phase E — Unify evidence backends (QA / NLI / embedding)

Introduce `HFEvidenceBackend` as the shared shape for the three evidence
models. Migrate `extractive_qa.py`, `nli_loader.py`, `embedding_loader.py`
to it. Public API preserved via module-level thin wrappers.

### E.a. Design — `src/podcast_scraper/providers/ml/hf_evidence_backend.py`

```python
class HFEvidenceBackend(ABC):
    """Shared loader + cache + forward for GIL evidence models."""
    kind: ClassVar[str]  # "qa" | "nli" | "embedding"

    def __init__(self, model_id: str, device: Optional[str] = None) -> None: ...
    @classmethod
    def get_or_load(cls, model_id: str, device: Optional[str] = None) -> Self: ...
    @classmethod
    def clear_cache(cls) -> None: ...

    @abstractmethod
    def _load(self) -> None: ...        # AutoTokenizer + AutoModelFor{QA,SequenceClassification,None}
    @abstractmethod
    def _forward(self, *args, **kwargs) -> Any: ...

class QAEvidenceBackend(HFEvidenceBackend): ...
class NLIEvidenceBackend(HFEvidenceBackend): ...
class EmbeddingBackend(HFEvidenceBackend): ...
```

Shared machinery in the ABC:

- Device resolution (`_resolve_device` with MPS→CPU coercion for QA/NLI).
- Per-`(resolved_id, device)` cache with a threading lock (currently
  duplicated 3 ways).
- `local_files_only=True`, `low_cpu_mem_usage=False`,
  `trust_remote_code=False`, `cache_dir=get_transformers_cache_dir()` —
  the "standard load kwargs" applied to every from_pretrained call.
- `_safe_score_float()` helper (currently duplicated 2 ways).

### E.b. Thin wrappers preserve public API

`extractive_qa.py` shrinks to:

```python
from .hf_evidence_backend import QAEvidenceBackend
QASpan = QASpan  # re-exported
def answer(context, question, model_id, ...) -> QASpan:
    return QAEvidenceBackend.get_or_load(model_id, ...).answer(question, context)
def answer_candidates(...) -> list[QASpan]: ...
def clear_qa_pipeline_cache() -> None:
    QAEvidenceBackend.clear_cache()
```

`nli_loader.py` and `embedding_loader.py` follow the same pattern.
Callers (`gi/grounding.py`, `ml_provider.py`, tests) don't move.

### E.c. Test harness

- New: `tests/unit/podcast_scraper/providers/ml/test_hf_evidence_backend.py`
  — cache-hit, threading, device resolution, `_safe_score_float`.
- Existing `test_grounding.py` still mocks at `extractive_qa.answer_candidates`
  — refactor invisible.
- Add contract test: all three subclasses satisfy the ABC.

### E.d. Gate + commit

```bash
.venv/bin/python -m pytest tests/unit/podcast_scraper/providers/ml/ -x -q
.venv/bin/python -m pytest tests/unit/podcast_scraper/gi/ -x -q
git commit -m "refactor(ml): unify evidence backends (QA/NLI/embedding) under HFEvidenceBackend (#382)"
```

---

## Phase F — Unify HF seq2seq loaders

Same collapse for the seq2seq path. `SummaryModel` and
`TransformersReduceBackend` do the same thing (load an HF seq2seq
checkpoint, generate) with different default gen params. Introduce a
shared backend; both become thin profiles.

### F.a. Design — `src/podcast_scraper/providers/ml/hf_seq2seq_backend.py`

```python
class HFSeq2SeqBackend:
    """Shared loader + generator for HF seq2seq checkpoints."""
    def __init__(self, model_id: str, *, device: Optional[str] = None,
                 revision: Optional[str] = None, cache_dir: Optional[str] = None,
                 default_gen_config: Optional[GenerationConfig] = None,
                 use_safetensors: Optional[bool] = None) -> None: ...
    def load(self) -> None: ...  # snapshot-first, model-family-specific class dispatch
    def generate(self, text: str, *, gen_config: Optional[GenerationConfig] = None) -> str: ...
    def to(self, device: str) -> None: ...  # for OOM fallback
    def unload(self) -> None: ...
```

Model-family dispatch (Pegasus / LED / BART / AutoModel) lives here
once — the current triple copy in summarizer.py:1176-1301 collapses.
The Pegasus warning validator moves in as an opt-in kwarg
(`validate_missing_positional_embeddings=True`).

### F.b. Consumers

- `SummaryModel` becomes: hold an `HFSeq2SeqBackend` instance +
  MAP-specific default `GenerationConfig` + preprocessing + chunking
  + retry loop.
- `TransformersReduceBackend` becomes: hold an `HFSeq2SeqBackend` +
  REDUCE-specific default `GenerationConfig` + prompt formatting.

Both consumers keep their existing public surface.

### F.c. Test harness

- New: `tests/unit/podcast_scraper/providers/ml/test_hf_seq2seq_backend.py`
  — model loading, family dispatch, generation, device fallback.
- Existing `test_summarizer.py` and `test_hybrid_provider.py` continue
  to work (mock at `SummaryModel.summarize` / `TransformersReduceBackend.reduce`
  boundaries).

### F.d. Gate + commit

```bash
.venv/bin/python -m pytest -m "unit and module_summarization" -x -q
.venv/bin/python -m pytest -m "unit and module_ml_providers" -x -q
git commit -m "refactor(ml): collapse SummaryModel + TransformersReduceBackend onto HFSeq2SeqBackend (#382)"
```

---

## Phase G — Unify HF download helpers

Model_loader has three parallel warmers
(`_download_qa_model_for_cache`, `_download_nli_cross_encoder_for_cache`,
`_download_sentence_transformer_for_cache`). Collapse to one.

### G.a. Design

```python
def _download_hf_model(
    kind: Literal["seq2seq", "qa", "nli", "embedding", "sentence-transformer"],
    model_id: str,
) -> None:
    """Warm the HF cache for one model. Preload-only path (downloads allowed)."""
    ...
```

Dispatch on `kind` for the model class + secondary loads
(sentence-transformers needs `SentenceTransformer(...)`; CrossEncoder
needs its own class; QA/NLI use `AutoModelForQuestionAnswering` /
`AutoModelForSequenceClassification` + `AutoTokenizer`).

### G.b. Consumers

`preload_evidence_models` loops over
`[(kind, model_id) for … in embedding_models + qa_models + nli_models]`
instead of three separate loops with identical scaffolding.

### G.c. Gate + commit

```bash
.venv/bin/python -m pytest tests/unit/podcast_scraper/providers/ml/ -x -q
git commit -m "refactor(ml): collapse HF cache-warm helpers under _download_hf_model (#382)"
```

---

## Phase 6 — Test-harness updates + coverage add-ons

Phases 3/4/5 add tests as they go; Phase 6 sweeps for **gaps** and
records coverage.

### 6a. Coverage delta

Baseline coverage before push:
```bash
.venv/bin/python -m pytest \
  -m "unit and (module_summarization or module_ml_providers)" \
  --cov=podcast_scraper.providers.ml \
  --cov=podcast_scraper.cache \
  --cov-report=term-missing --cov-report=xml:coverage-v5-ml.xml
```

Compare against the previous coverage report under `.build/coverage/`
(if present). Target: **no regression**; expect a modest bump because
QA/hybrid tests are new.

### 6b. Test markers audit

Verify every new test file has correct markers per `pyproject.toml`
markers block:
- Unit: `pytest.mark.unit` + `module_*` module marker.
- Parity/integration: `pytest.mark.integration` + `ml_models` + `slow`.
- Neither `xfail` nor `skip` markers added anywhere.

### 6c. Preload sanity

The Docker `pipeline` image bakes the ML cache
(`PRELOAD_ML_MODELS=true`, `[[feedback_stack_pipeline_preload_rule]]`).
Verify the preload script still populates every model after Phase 3's
changes to `model_loader._download_*`:

```bash
CACHE_DIR=/tmp/hf-preload-v5 make preload-ml-models
ls -la /tmp/hf-preload-v5/huggingface/hub | grep -E "models--" | wc -l
```

Expected: ≥ 6 entries (BART/LED/Pegasus/FLAN-T5/QA/NLI).

### 6d. Test integration into registry (nothing to change)

The model registry (`model_registry.py`) declares capabilities per
`model_id`. None of the fields (`max_input_tokens`, `model_type`,
`model_family`, `memory_mb`, `default_device`) depend on the pipeline
API. **No registry code change required.**

However, add a note in the registry docstring (top of the file, ~L1-20)
that reads *"After v5 migration (#382): consumers of this registry no
longer instantiate `pipeline()` — they use `AutoModelFor*.from_pretrained`
+ `generate()` / forward. Capability fields still describe the underlying
checkpoint, not the pipeline wrapper."*

### 6e. Commands + commit

```bash
.venv/bin/python -m pytest -m "unit and not ml_models" -x -q --cov=podcast_scraper --cov-report=term
.venv/bin/python -m pytest -m "unit and ml_models" -x -v
.venv/bin/python -m pytest -m "integration and ml_models" -x -v  # includes parity tests
git add src/podcast_scraper/providers/ml/model_registry.py
git commit -m "test(ml): add v5-parity tests + coverage for QA/summarizer refactor (#382)"
```

---

## Phase 7 — Post-upgrade eval run + `run-compare` gate

The numerical anchor. This is where "we didn't break anything" gets
proved (or disproved) against Phase 0.

### 7a. Re-run baselines under v5

```bash
make baseline-create \
  BASELINE_ID=baseline_ml_bart_authority_smoke_v5_post \
  DATASET_ID=curated_5feeds_smoke_v1 \
  PREPROCESSING_PROFILE=cleaning_v4

make baseline-create \
  BASELINE_ID=baseline_ml_pegasus_retirement_smoke_v5_post \
  DATASET_ID=curated_5feeds_smoke_v1 \
  PREPROCESSING_PROFILE=cleaning_v4
```

### 7b. Re-capture QA span baseline under v5

```bash
.venv/bin/python scripts/dev/capture_qa_baseline.py \
  --out data/eval/references/qa_baseline_v5_post.jsonl
```

### 7c. Compare

Use the existing runs-compare tooling for the summarizer baselines:

```bash
# Streamlit UI (interactive; operator judgment):
BASELINE=baseline_ml_bart_authority_smoke_v5_pre make run-compare
```

For the automated gate (headless, CI-compatible), add a small helper
`scripts/eval/compare_v5_parity.py` that:
- Reads `predictions.jsonl` from `_v5_pre` and `_v5_post`
- Computes per-episode ROUGE-L
- Asserts min ROUGE-L across episodes ≥ 0.95 (identical checkpoints +
  deterministic seeds ⇒ this is tight)
- Writes `data/eval/runs/v5_parity_2026-07-XX.json` with summary

For QA:
```bash
.venv/bin/python scripts/eval/compare_qa_parity.py \
  --pre data/eval/references/qa_baseline_v5_pre.jsonl \
  --post data/eval/references/qa_baseline_v5_post.jsonl
```
Asserts top-1 (start, end) exact match on ≥ 98 % of pairs; score delta
< 0.05 mean absolute.

### 7d. If parity fails

- **Summarizer <0.95 ROUGE-L:** almost certainly a `GenerationConfig`
  arg mismatch (`length_penalty`, `repetition_penalty`, `early_stopping`).
  Compare `model.generation_config` before/after. Fix in Phase 5's
  file, re-run.
- **QA <98% exact match:** most likely the `max_answer_length` filter or
  `sequence_ids` boundary. Cross-check against transformers'
  `QuestionAnsweringPipeline.postprocess()` source at the pinned v5
  commit — mimic exactly.
- **Score delta > 0.05:** typically dtype (v5 respects saved dtype) —
  can force `torch_dtype=torch.float32` on load to isolate; if that
  restores parity, decide whether to keep the mild drift or pin dtype
  explicitly.

### 7e. Commit the artifacts

```bash
git add data/eval/baselines/baseline_ml_bart_authority_smoke_v5_post \
        data/eval/baselines/baseline_ml_pegasus_retirement_smoke_v5_post \
        data/eval/references/qa_baseline_v5_post.jsonl \
        data/eval/runs/v5_parity_2026-07-*.json \
        scripts/eval/compare_v5_parity.py \
        scripts/eval/compare_qa_parity.py
git commit -m "chore(eval): post-v5 baselines + parity gate results for #382"
```

**Gate:** parity check passes numerically; JSON parity report exists.

### 7f. Autoresearch scope note (explicitly no-op)

`autoresearch/PER_MODEL_OPTIMAL_PARAMS.md` and
`autoresearch/MODEL_PLAYBOOK.md` are for vLLM-on-DGX LLM cohorts (the
LLM candidates in Track-A/B sweeps). They **do not describe** local
BART/LED summarization behavior and therefore need no update for
this migration. The `data/eval/` baselines under
`data/eval/baselines/baseline_ml_*` and the new `_v5_post` set are the
authoritative record for local ML behavior.

If we want to record the v5 attention-backend (SDPA-on-MPS/CUDA)
observations, add a **new** section to
`docs/adr/ADR-068-bart-led-as-ml-production-baseline.md` post-impl
notes rather than to the autoresearch compendium.

---

## Phase 8 — Docs sweep

Every doc that references the old pipeline API, the CVE-2026-1839
ignore, or the `<5` cap gets updated.

### 8a. Files to edit

| File | Edit |
|---|---|
| `docs/adr/ADR-068-bart-led-as-ml-production-baseline.md` | Add "Post-impl 2026-07-05" section: v5 upgrade, `pipeline()` retired, `GenerationConfig`-based, SDPA-verified findings. |
| `docs/adr/ADR-069-hybrid-ml-pipeline-as-production-direction.md` | Same post-impl note: `TransformersReduceBackend` uses direct `generate()`. |
| `docs/adr/ADR-071-four-tier-summarization-strategy.md` | Same. |
| `docs/adr/ADR-005-lazy-ml-dependency-loading.md` | Reference: lazy imports unchanged; just note v5 in the "known versions" section. |
| `docs/guides/ml-troubleshooting.md` (if exists) or `.ai-coding-guidelines.md` § ML | Delete any mention of `pipeline()`; add "loading a summarization / QA model in v5" recipe. |
| `autoresearch/MODEL_PLAYBOOK.md` | **No change** (out of scope — see §7f). |
| `docs/wip/issue-382-transformers-v5-upgrade-plan.md` | Update Status → **Superseded** and delete once merged (retain in this branch's final commit for history). |
| `docs/wip/WIP_README.md` | Mark both `ISSUE-382-*` files' status = **Closeable** (post-merge). |
| `CHANGELOG.md` (or `docs/releases/`) | New entry under next unreleased version: `- Upgrade transformers to v5. Retire pipeline() API for summarization / hybrid reduce / extractive QA; adopt GenerationConfig. Fixes CVE-2026-1839 exposure. Closes #382.` |
| `README.md` § Local ML | If it mentions `pipeline()`, update to `AutoModel*.from_pretrained` + `generate()`. |
| GitHub issue **#382** body / final comment | Fold the deep analysis's §1 delta table + Phase-7 parity numbers as the closing comment before PR merge. |

### 8b. Docs build

```bash
make docs
```
Expected: mkdocs strict green. If unresolved refs, follow
`[[feedback_make_docs_before_push]]`.

### 8c. Commit

```bash
git add docs/ CHANGELOG.md README.md
git commit -m "docs(#382): fold v5 migration into ADR-068/069/071 + CHANGELOG + WIP index"
```

---

## Phase 9 — Makefile CVE-ignore removal + `make ci-fast`

### 9a. Edit `Makefile`

Remove the three pip-audit `--ignore-vuln` entries pointing at v4-line
transformers CVEs (they were held open only because we couldn't move to
v5). Concretely:

```diff
   # Ignore CVE-2026-1839: transformers Trainer loads rng_state via torch.load without weights_only; fixed in 5.0.0rc3+.
-  #   We pin transformers<5.0.0 (extractive QA / pipeline — see pyproject [ml]). Revisit when stable 5.x is adopted.
-  # TODO(CVE-2026-1839): Remove --ignore-vuln after bumping transformers to a fixed 5.x release.
   ...
-  # Ignore PYSEC-2025-211..218 (transformers X-CLIP / Trainer / conversion-script
-  #   ...)
-  # TODO(transformers PYSECs): drop ignores after bumping transformers to a
-  #   patched 5.x release (paired with the CVE-2026-1839 ignore above).
   ...
-		--ignore-vuln CVE-2026-1839 \
-		--ignore-vuln PYSEC-2025-211 \
-		--ignore-vuln PYSEC-2025-212 \
-		... (PYSEC-2025-213..218)
```

Keep unrelated ignores (pygments CVE-2026-4539, etc.).

### 9b. Verify security scans still pass

```bash
make security
```
Expected: `pip-audit` green with no new advisories in the transformers
5.x line. If a new v5-specific advisory surfaces, follow the standard
triage — pin around it if possible, document explicitly, do NOT
silently re-add an ignore.

### 9c. Full validation ladder

Run **in this order**, halt on first failure:

```bash
make format-check
make lint
make type
make test-unit         # -m unit
make test-unit-ml      # -m "unit and ml_models"   (see Makefile target)
make test-integration-fast
make docs
make ci-fast           # ONCE — final green light
```

If `ci-fast` fails, re-run **only the failing subtarget** per rule #18
— never re-run the umbrella target to check a fix.

### 9d. Secrets scan

```bash
# via secrets-scan skill
```
Runs pre-commit-adjacent scan; must be clean. New `capture_qa_baseline.py`
and `compare_*_parity.py` scripts must not carry HF tokens.

### 9e. Commit + branch state

```bash
git add Makefile
git commit -m "chore(security): drop pip-audit CVE-2026-1839 / PYSEC-2025-211..218 ignores now that transformers>=5 (#382)"
git log --oneline main..HEAD  # ~9 commits, one per phase
git status                    # clean working tree
```

---

## Push gate (operator-approved only)

Before the operator says "push":

1. `git fetch origin main`
2. `git rebase origin/main` — resolve conflicts if any.
3. If rebase touched an ML/eval file, re-run **only** that phase's
   subtarget.
4. Show `git status` + `git log --oneline origin/main..HEAD` + `git diff origin/main..HEAD --stat`.
5. **Wait.**
6. On explicit "push": `git push -u origin feat/upgrade-transformers-v5 --force-with-lease`.
7. On explicit "PR": `gh pr create --title "Upgrade transformers to v5 (#382)" --body "..."` with the deep-analysis + parity report folded in, `Closes #382` in the body (per `[[feedback_pr_link_closes_issues]]`).

---

## Rollback matrix

| At end of phase | If a later phase blows up | Rollback |
|---|---|---|
| Phase 0 (baselines) | keep | never rollback baselines |
| Phase 1 (deps) | rare — resolver clean means it's safe to leave the pin | `git revert <hash>` on the pyproject commit |
| Phase 2 (cache) | trivial — 4-line delete | `git revert` |
| Phase 3 (QA) | isolated failure surface | `git revert <phase 3 hash>` — pyproject at v5 still works because `pipeline("question-answering")` was called via the deleted `_call_transformers_qa_pipeline`; reverting Phase 3 restores that helper AND the pipeline call, but v5 will raise at runtime → revert Phase 1 too. |
| Phase 4 (hybrid) | isolated | Same pattern: revert Phase 4 AND Phase 1. |
| Phase 5 (summarizer) | isolated | Same. |
| Phase 6-9 (tests/docs/CVE) | trivial | `git revert`; deps + code untouched |

**Nuclear rollback:** `git branch -D feat/upgrade-transformers-v5` from
`main` (nothing merged yet, no infra state). Zero cost.

---

## What this plan explicitly does NOT do

- **Does not merge** — merge is a follow-up manual step after operator +
  reviewer approval.
- **Does not touch `autoresearch/*`** — those docs are for vLLM LLM
  cohorts, not local ML. See §7f.
- **Does not create GH issues** (per `[[feedback_never_open_gh_issues]]`).
  #382 is the sole issue; the plan closes it via `Closes #382` in the PR body.
- **Does not add `transformers serve`, WeightConverter, Kernels, or
  Trainer usage** — out of scope (see deep-analysis §2g/2h).
- **Does not touch pyannote's `use_auth_token`** — orthogonal to v5.
- **Does not push, does not open PR** — both operator-gated.

---

## Progress ledger (fill in as we execute)

| # | Phase | Committed? | SHA | Notes |
|---|---|---|---|---|
| — | Preamble docs | ✓ | b23ad14b | Deep analysis + execution plan on branch |
| 0 | Baseline capture | ✓ | df6385aa | 5-ep summarizer baseline + 8-pair QA reference |
| 1 | Deps pass | ✓ | 7ea02357 | transformers 5.13.0, ST 5.6.0, hf_hub 1.22.0 |
| 2 | Cache + registry | ✓ | 46f6aee7 | file_utils fallback dropped; all pinned revisions resolve |
| 3 | QA rewrite | ✓ | 36ed7dcc | AutoModelForQuestionAnswering; 7/8 top-1 parity vs v4 pipeline |
| — | Path C plan expansion | ✓ | 8a80b0f6 | Phases E/F/G added to plan |
| 4 | Hybrid rewrite | ✓ | 943c84ae | pipeline("text2text-generation") → generate() |
| 5 | Summarizer rewrite | ✓ | faba3f60 | pipeline("summarization") → generate() + GenerationConfig |
| E | Unify evidence backends | ✓ | a5f671d6 | HFEvidenceBackend + AI-provider audit doc |
| F.1 | Unify HF seq2seq — backend + hybrid | ✓ | 7ec9ad55 | HFSeq2SeqBackend + TransformersReduceBackend migrated |
| F.2 | Unify HF seq2seq — summarizer | ✓ | d6b1807d | SummaryModel routed fully through HFSeq2SeqBackend |
| G | Unify HF download helpers | ✓ | d486f35d | _download_hf_evidence_model(kind, model_id) |
| 6-9 | Parity + release + CVE cleanup + docs | ✓ | 40c3bee6 | v2.7.0 release note, ADR-068 post-impl, testing strategy, Makefile CVE cleanup |
| Closeout.1 | Fix integration-test regressions + 8/10 audit items | ✓ | 4ce8707b | 31 broken integration tests rewritten; QA helper rename; registry docstring; SDPA measured; Makefile typo |
| Closeout.2 | Final ci-fast gate green | ✓ | f845d844 | Spelling fix, test-policy fix, test_bridge_builder update |
| Closeout.3 | Cross-link #1142 (bundled dedup follow-up) | ✓ | 19955618 | Audit doc + release note point at follow-up issue |
| Closeout.4 | Deep-review self-audit fixes | ✓ | a7f77ce5 | F1-F4 corrections from adversarial review (backend coverage 58%→85%, testing-strategy nuance, ledger, release-note QA naming) |
| — | Rebase + push | ⛔ operator | | Pending push approval |

---

*Author: analysis by Claude for operator, 2026-07-05.*
*Branch: `feat/upgrade-transformers-v5` (off local `main = c1249814`).*
*Companion: `ISSUE-382-TRANSFORMERS-V5-DEEP-ANALYSIS-2026-07-05.md`.*
