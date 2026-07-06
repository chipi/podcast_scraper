# Issue #382 — Transformers v5 upgrade: deep analysis (2026-07-05)

Follow-up to `docs/wip/issue-382-transformers-v5-upgrade-plan.md` (2026-04-02).
Grounds the plan against **today's** code, closes gaps the earlier note left
open, and identifies modernization opportunities that come for free once the
pin is raised.

**Status of the tree:**

- `transformers` **4.57.6** pinned `<5.0.0` in `[ml]` and `[search]`.
- `sentence-transformers` **5.5.1** already ships transformers-v5 compat
  (dual v4/v5 CI since ST 5.2.1); ST 5.6.0 fixes a silent causal-LM
  reranker scoring bug + restores TSDAE on v5.
- `torch` **2.12.0**, `accelerate` **1.13.0**, `tokenizers` **0.22.2** —
  all comfortably above v5's stated minimums.
- Two known blockers the pin comments already record:
  1. `pipeline("question-answering")` removed in v5 (`pyproject.toml:111`).
  2. **CVE-2026-1839** (Trainer `rng_state`/`torch.load` without
     `weights_only`) fixed **only in ≥5.0.0rc3**. `Makefile:436-502`
     currently `--ignore-vuln CVE-2026-1839` to hold the `<5` cap.

Not blocked. This is a **medium-churn**, ~2-day engineering task with a
low-risk rollback path.

---

## 1 — What the v5 delta actually is, for *our* code

Cross-referenced from the official [v5 migration guide][mg] and the
[v5.0.0 release notes][rn] against every touchpoint in `src/`. Grouped by
action required, not by module.

[mg]: https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md
[rn]: https://github.com/huggingface/transformers/releases/tag/v5.0.0

### 1a. Load-bearing removals (must migrate)

| v5 change | Our call sites | Effort |
|---|---|---|
| `pipeline("summarization")` removed | `summarizer.py:1046,1593` (build + OOM rebuild) + 3 call sites at `:1520,1600,1633` | **High** |
| `pipeline("text2text-generation")` removed | `hybrid_ml_provider.py:183` build + `:208` call | Medium |
| `pipeline("question-answering")` removed | `model_loader.py:243-296` (`build_huggingface_qa_pipeline`) + all `extractive_qa.py` consumers via `get_qa_pipeline` | Medium |
| `transformers.file_utils` removed | `cache/directories.py:117` (fallback path only; hub path already primary) | Trivial |
| Legacy `model.config` generation params removed → `model.generation_config` | `summarizer.py:1098-1103` already touches both; drop the `model.config` branch | Trivial |
| `TRANSFORMERS_CACHE` env var removed → `HF_HOME` | not used in `src/` | None |
| `use_auth_token` → `token` | not on `transformers` calls; only pyannote fallback at `pyannote_provider.py:36` (pyannote-specific arg, orthogonal) | None for #382 |

### 1b. Behaviour changes (verify, may need code adjustment)

| v5 change | Impact | Action |
|---|---|---|
| `AutoModelFor*.from_pretrained` **respects saved dtype** (no more silent fp32 promote) | BART / Pegasus / LED / FLAN-T5 checkpoints are fp32 → no visible change. `summllama_provider.py:106-115` explicitly passes `torch_dtype=` → OK. | Sanity check after bump |
| Default cache class model-defined (was `DynamicCache`) | Only affects `AutoModelForCausalLM.generate` → `summllama_provider.py`. Neutral-to-positive. | None |
| Slow/fast tokenizers unified; `encode_plus` / `batch_decode` removed | We use `AutoTokenizer.from_pretrained` + `__call__` / `decode` — not the removed calls. | Verified clean |
| Safetensors default shard 5 GB → 50 GB | Loading-side only; download works either way. | None |
| Non-generative models lose `generation_config` attribute | QA head (`AutoModelForQuestionAnswering`) is non-generative. Our post-load `setattr(model.generation_config, ...)` at `summarizer.py:1098-1099` runs only on seq2seq (which stays generative). | Verified clean |
| Pipeline signature stricter overloads (`cast(Any, pipeline)` at `hybrid_ml_provider.py:182`) | Moot once we replace `pipeline("text2text-generation")`. | Auto-resolved |

### 1c. Nothing-to-do (already forward-compat)

- `set_seed` at `workflow/stages/setup.py:107` — still exported by
  `transformers` in v5 (relocated, top-level import stable per guide).
- `from transformers.utils import logging as hf_logging` at
  `summarizer.py:60,75` + `nli_loader.py:158,175` — module unchanged.
- `AutoModelForSeq2SeqLM` for **text** seq2seq (BART/Pegasus/LED/T5)
  stays. The migration guide's `→ AutoModelForImageTextToText` rewrite
  applies only to VL models previously bucketed under seq2seq.
- `AutoModelForCausalLM` unchanged for `summllama_provider.py`.
- `low_cpu_mem_usage=False` explicit override for the tied-weights meta
  bug (`summarizer.py:1150-1154`) — kwarg preserved in v5. Keep it.

---

## 2 — Modernization opportunities (adopt in the same PR)

The earlier plan is a mechanical port. Because half of the touch surface is
already the `pipeline()` layer, the migration is the **right moment** to
delete that layer entirely rather than rebuild it as a `generate()` wrapper
that mimics pipeline semantics. Recommendations, scored by ROI:

### 2a. **[ADOPT] Drop the `pipeline()` abstraction — go direct to `generate()`**

The `pipeline()` layer is thin sugar and it is fighting us:

- Forces the `_call_transformers_qa_pipeline` indirection
  (`model_loader.py:299-311`) to make monkey-patch tests work — the
  top-level `transformers` is `_LazyModule` since 4.40 (comment already
  documents Issue #677).
- Causes the "multiple values for `local_files_only`" TypeError dance in
  `build_huggingface_qa_pipeline` (`model_loader.py:280-296`).
- Requires **rebuilding the pipeline** on device fallback
  (`summarizer.py:1590-1598`) instead of just moving the model.
- `hybrid_ml_provider.py` already has to `cast(Any, pipeline)` because
  v5's typed overloads reject `"text2text-generation"` (line 181-183).

The rewrite is small and reads cleaner. For seq2seq summarization:

```python
# Before — summarizer.py:1046
self.pipeline = pipeline("summarization", model=self.model,
                         tokenizer=self.tokenizer, device=pipeline_device)
# ... later ...
result = self.pipeline(text, max_length=max_length, min_length=min_length)
summary = result[0]["summary_text"]

# After
def _generate_summary(self, text: str, *, max_length: int,
                      min_length: int, **gen_kwargs) -> str:
    inputs = self.tokenizer(
        text, return_tensors="pt", truncation=True,
        max_length=self._encoder_max_length,
    ).to(self.device)
    output_ids = self.model.generate(
        **inputs,
        max_new_tokens=max_length,
        min_new_tokens=min_length,
        num_beams=gen_kwargs.get("num_beams", 4),
        no_repeat_ngram_size=gen_kwargs.get("no_repeat_ngram_size", 3),
        early_stopping=gen_kwargs.get("early_stopping", True),
    )
    return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
```

For QA:

```python
# Before — extractive_qa.py, via build_huggingface_qa_pipeline
pipe = pipeline("question-answering", model=model_id, device=device,
                model_kwargs={"cache_dir": ..., "low_cpu_mem_usage": False})
result = pipe(question=q, context=ctx, top_k=k)
# → list of {"answer": str, "score": float, "start": int, "end": int}

# After
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
model = AutoModelForQuestionAnswering.from_pretrained(model_id, ...)
tokenizer = AutoTokenizer.from_pretrained(model_id, ...)

def _qa_forward(question: str, ctx: str, *, top_k: int = 1) -> list[QASpan]:
    enc = tokenizer(question, ctx, return_tensors="pt",
                    truncation="only_second", return_offsets_mapping=True,
                    max_length=384, stride=128, padding="max_length").to(device)
    offsets = enc.pop("offset_mapping")[0]
    with torch.no_grad():
        out = model(**enc)
    # score = start_logit + end_logit; walk top-k valid (start, end) pairs
    # where start <= end and both are in the context segment.
    ...
```

Delete `_call_transformers_qa_pipeline`, the multi-value dance, and the
pipeline-rebuild-on-OOM branch. Net LOC delta is **negative** — the raw
API is less clever than the pipeline scaffolding around it.

**Why the same PR:** we are already rewriting these three modules for the
mechanical port. Doing it *as* the pipeline replacement instead of
*around* it saves us a follow-up PR of pure churn.

### 2b. **[ADOPT] Delete the `transformers.file_utils` fallback**

`cache/directories.py:115-122` is a defensive fallback whose primary path
(`huggingface_hub.constants.HF_HUB_CACHE`) has been present since HF-Hub
0.14 (early 2024). v5 removes `transformers.file_utils` entirely. Drop
the try/except block — the standard user cache fallback at line 125
already catches the pathological case.

### 2c. **[ADOPT] Bump `sentence-transformers` floor to `>=5.6.0`**

ST 5.6.0 (2026-06-16) fixes a **silent scoring bug** in causal-LM
rerankers and restores TSDAE on transformers v5. Our NLI CrossEncoder
path (`nli_loader.py`) doesn't currently use a causal-LM reranker, but
`>=5.6.0` is the lowest floor that ships every v5 fix and costs nothing.
Current `.[ml]`/`.[search]` are `sentence-transformers>=5.4.1,<6.0.0` —
change to `>=5.6.0,<6.0.0`.

### 2d. **[ADOPT] Fold generation defaults into `GenerationConfig`**

v5 makes `model.generation_config` the sole source of truth for
generation defaults (v4 still read some from `model.config`). We already
compute `num_beams`, `no_repeat_ngram_size`, `early_stopping`,
`max_new_tokens` etc. as **per-call kwargs** at `summarizer.py:629-750`.
The clean v5 pattern is:

```python
from transformers import GenerationConfig

gen_cfg = GenerationConfig(
    num_beams=num_beams,
    no_repeat_ngram_size=no_repeat_ngram_size,
    early_stopping=early_stopping,
    max_new_tokens=effective_max_new_tokens,
    min_new_tokens=effective_min_new_tokens,
)
output_ids = self.model.generate(**inputs, generation_config=gen_cfg)
```

Same effect, better trace/log surface, and matches the v5-first pattern
maintainers use in HF examples. **Non-breaking** even if we defer.

### 2e. **[DEFER, VERIFY] Attention backend auto-selection**

v5's `AttentionInterface` picks SDPA on CUDA / MPS by default. Our
BART/LED/Pegasus checkpoints benefit ~10-20 % on GPU with no code change
(seen in HF benchmarks). Action: no code change; **measure** in the
post-upgrade smoke and record the delta. If it regresses on MPS (LED has
been historically finicky), pin `attn_implementation="eager"` for LED
only via a from_pretrained kwarg.

### 2f. **[DEFER] Simplify `_load_pegasus_without_fake_warning`**

The "missing positional embedding keys" false-warning was a v4 quirk.
v5's loading refactor may have fixed it. Test after upgrade — if the
warning is gone, delete `_load_pegasus_without_fake_warning` and load
Pegasus like BART. **Do not preemptively delete** — keep the validation
if the warning persists.

### 2g. **[SKIP] `transformers serve`, WeightConverter, Kernels**

- `transformers serve` (OpenAI-compat local endpoint) overlaps our
  Ollama / llama.cpp / vLLM stack — no adoption path.
- `WeightConverter` API is for authors converting checkpoints between
  layouts (MoE, TP splits). Irrelevant for consuming existing BART/QA.
- `Kernels` library is a runtime kernel registry — no wins for our
  workload size on M-series / GB10.

### 2h. **[SKIP for now] Trainer-based fine-tuning**

We don't use `transformers.Trainer`. The CVE-2026-1839 fix in v5 is
still the reason to move; nothing to migrate here. If a future task
introduces on-device fine-tuning (RFC-042 hybrid never landed a Trainer
path), reassess.

---

## 3 — Dependency pass (single resolver shot)

```diff
 # [ml] and [search] blocks
-  "transformers>=4.57.6,<5.0.0",
+  "transformers>=5.0.0,<6.0.0",   # v5 mandatory: pipeline() migration + CVE-2026-1839 fix
-  "sentence-transformers>=5.4.1,<6.0.0",
+  "sentence-transformers>=5.6.0,<6.0.0",  # 5.6.0: v5 causal-LM reranker bug + TSDAE fix
```

No change required for `torch`, `accelerate`, `tokenizers`,
`huggingface_hub` — all current pins exceed v5's floors:

| Package | Our pin | v5 minimum | Verdict |
|---|---|---|---|
| torch | `>=2.11.0,<3.0.0` | 2.4 | OK |
| accelerate | `>=1.13.0,<2.0.0` | 1.1.0 | OK |
| tokenizers | (via transformers) 0.22.2 | 0.21 | OK |
| huggingface-hub | (transitive) | 1.0 | verify at install |
| protobuf | `>=3.20.0,!=6.33.4,<8.0.0` | any | OK |
| sentencepiece | `>=0.2.1,<1.0.0` | any | OK |

**Makefile cleanup once green:**

- `Makefile:436-438` — remove `--ignore-vuln CVE-2026-1839` (fixed in ≥5.0.0rc3).
- `Makefile:477-484` — the `PYSEC-2025-211..218` transformers-4.x
  advisories block: all fixes live in the 5.x line, so this ignore block
  goes away entirely.
- `pyproject.toml:110-112` — replace the 3-line explanatory comment with
  the reason for the new floor (`v5 pipeline rewrite completed in #382`).

---

## 4 — Phased execution plan (refined)

Same order as the earlier plan, but each phase is now scoped to a
concrete file diff and a validation gate.

### Phase 0 — pyproject + local install (30 min)

1. Edit `pyproject.toml` per §3.
2. `.venv/bin/pip install --upgrade -e .[dev,ml,search,llm,compare]`
   — capture the full resolver output; look for `ResolutionImpossible`
   or version pins that trip.
3. `.venv/bin/python -c "import transformers; print(transformers.__version__)"`
   and same for `sentence_transformers`, `torch`, `accelerate`.
4. **Gate:** clean resolve, imports succeed.

### Phase 1 — cache/directories cleanup (15 min)

- Delete the `transformers.file_utils` fallback (`cache/directories.py:115-122`).
- Gate: `.venv/bin/python -c "from podcast_scraper.cache import get_transformers_cache_dir; print(get_transformers_cache_dir())"`.

### Phase 2 — extractive QA rewrite (~4 h)

- Rewrite `extractive_qa.py` around `AutoModelForQuestionAnswering` +
  tokenizer forward pass. Keep the `QASpan` dataclass and public
  functions (`answer`, `answer_candidates`, `answer_multi`) —
  unchanged API from callers' point of view.
- Delete `build_huggingface_qa_pipeline`, `_call_transformers_qa_pipeline`,
  `_download_qa_pipeline_for_cache` in `model_loader.py`; replace with a
  `_download_qa_model_for_cache` that just calls `from_pretrained` twice.
- Update `is_evidence_model_cached` if it introspects pipeline internals.
- Gate: `.venv/bin/python -m pytest -m "unit and (gi or qa)" -x`
  (existing GI/QA unit tests are already mocked at the
  `extractive_qa.answer_candidates` boundary, so most survive).

### Phase 3 — hybrid reduce rewrite (~2 h)

- Rewrite `TransformersReduceBackend` (`hybrid_ml_provider.py:65-226`)
  around `AutoModelForSeq2SeqLM.generate()`; drop the
  `pipeline("text2text-generation")` build at line 183 and the
  `cast(Any, pipeline)` workaround.
- `HybridReduceResult` public shape unchanged.
- Gate: `.venv/bin/python -m pytest -m "unit and hybrid_ml" -x` +
  short manual FLAN-T5 smoke on cached fixture.

### Phase 4 — summarizer rewrite (largest, ~6 h)

- Introduce `_generate_summary(text, ...)` helper (see §2a).
- Delete `_load_model_move_to_device_and_pipeline`'s pipeline creation;
  keep the device-fallback logic on `model.to(device)`.
- Replace all four call sites (`:1520,1600,1633` + the retry at
  `:1600-1603`) with `_generate_summary(...)`.
- OOM device fallback becomes: `self.model.to("cpu")` + re-call — no
  pipeline rebuild.
- Adopt `GenerationConfig` per §2d.
- Attempt to delete `_load_pegasus_without_fake_warning`; if the missing
  positional embedding warning persists on v5, keep it (see §2f).
- Gate: `.venv/bin/python -m pytest -m "unit and module_summarization" -x`
  + `-m "integration and ml_models"` for the sanity BART/LED runs.

### Phase 5 — full validation (~2 h + wall time)

Ordered by cost, per rule #8 ("run the correct validation, not the
heaviest"):

1. `make lint` + `make type` — style/type sweep.
2. `make test-unit` (`-m "unit and not ml_models"`).
3. `make test-unit -k ml_models` — pulls real fixtures.
4. `make test-integration-fast` — integration path with cached models.
5. `make docs` — WIP note + issue-382 body update.
6. `make ci-fast` — **once**, as the final green light before push (per
   memory: "ci-fast only at very end").
7. Manual smoke: one short episode through `make transcript-smoke` (or
   equivalent) with cloud-thin + local ml profiles, comparing summary
   text length + `EPISODE_METADATA.summary` field against a pre-upgrade
   baseline captured in Phase 0.

### Phase 6 — Makefile / docs cleanup

- Remove the three `pip-audit --ignore-vuln` entries pointing at v4 CVEs.
- Update `pyproject.toml` comment (§3).
- Bump `MODEL_PLAYBOOK.md` / dev-notes to say local ML no longer uses
  `pipeline()`; direct `AutoModel*.from_pretrained` + `generate()` /
  forward.

---

## 5 — Risk register + rollback

| Risk | Likelihood | Mitigation | Rollback cost |
|---|---|---|---|
| Pegasus static-position warning re-triggers a false-failure in `_load_pegasus_without_fake_warning` | Medium | Keep the validator; loosen `allowed_missing` if v5 renames keys. | 0 (keep validator) |
| SDPA on MPS regresses LED throughput | Low-Medium | Add `attn_implementation="eager"` only for LED. | 5 min |
| `AutoModelForQuestionAnswering` scoring differs from pipeline (post-processing subtleties) | Medium | Capture pre-upgrade top-k spans on a cached fixture in Phase 0; assert equivalence after Phase 2. | Revert phase-2 commit |
| `sentence-transformers` 5.6.0 requires `transformers>=5.x` (some future ST version may) | Very Low | Our floor is already 5.0.0. | N/A |
| `pip-audit` surfaces a new v5-only CVE | Low | Same triage as any advisory: pin + document, do not silently ignore. | Add explicit ignore |

**Rollback path.** Single feature branch, phased commits. If Phase 4
blows up on a real-corpus regression that the fixtures didn't catch,
revert Phases 4 → 3 → 2 in reverse order; Phase 0 (pin bump) can stay if
Phases 1-3 are stable, since `pipeline()` calls in the deleted paths are
gone by then. Full-rollback is `git reset --hard` on the branch — no
migrations, no infra state, no external deps.

---

## 6 — Test gates (what proves it worked)

Explicit; run in this order.

1. **Static:** `make lint`, `make type`.
2. **Unit (fast):** `.venv/bin/python -m pytest tests/unit -m "unit and not ml_models" -x`
3. **Unit (ml_models):** same target with `-m "unit and ml_models"` —
   loads real BART-base + a QA head from the local cache.
4. **Integration (fast):** `make test-integration-fast`.
5. **Docs:** `make docs` (mkdocs strict).
6. **Equivalence smoke** (custom, Phase 0 vs Phase 4):
   - Feed one cached transcript through the old and new summarizer;
     assert `ROUGE-L >= 0.95` between summaries (identical checkpoints
     + deterministic seeds ⇒ near-identical output).
   - Feed the QA context+question pair through old-pipeline and
     new-forward; assert top-1 span `start,end` match exactly.
7. **`make ci-fast`** — one shot, at the end. If it fails, run the
   failing subtarget in isolation per rule #18.

---

## 7 — Not doing (and why)

- **No follow-up "adopt WeightConverter" ticket** — we don't reshape
  checkpoints; the API is for model authors.
- **No `transformers serve` migration** — Ollama+llama.cpp+vLLM already
  fill that slot in the local stack.
- **No `Trainer` cleanup** — we don't use it; the CVE-2026-1839 fix is
  the migration driver, not the target.
- **No pyannote `use_auth_token` unification** — pyannote 4.x owns that
  kwarg semantics; orthogonal to transformers v5.
- **No Kernels library adoption** — no measurable win on our workload
  sizes.

---

## 8 — Concrete next step

Not yet actionable — this is analysis. When operator approves:

1. Create the branch `feat/upgrade-transformers-v5` (per issue #382).
2. Execute Phases 0-6 in order on that single branch, one commit per
   phase for bisectability.
3. Open PR with `Closes #382` in the body (per `feedback_pr_link_closes_issues`).

**Est. total wall time:** ~1.5–2 focused days including baseline capture
and integration test time.

---

*Author: analysis by Claude for operator, 2026-07-05.*
*Supersedes: docs/wip/issue-382-transformers-v5-upgrade-plan.md (kept for reference).*
