# Issue #429: Re-analysis and Internal Plan

**Issue:** [chipi/podcast_scraper#429](https://github.com/chipi/podcast_scraper/issues/429) – Follow-up: Complete remaining shipping readiness items from #379

**Approach:** Work in phases as suggested in the issue (Phase 1 → 2 → 3). This doc re-analyses current code vs issue and defines the internal plan.

---

## Re-analysis: Current Code vs Issue

### 1. Reproducible Runs (Section 1)

| Item | Issue says | Current code | Gap |
|------|------------|---------------|-----|
| **1.1 Pin Python in CI** | Pin exact version (e.g. 3.11.8) | CI uses `python-version: "3.11"` (floating) | Pin to exact (e.g. 3.11.8) in all workflows |
| **1.2 Pin ML deps** | Pin exact versions for torch, transformers, etc. | pyproject.toml uses ranges (`torch>=2.0.0,<3.0.0`) | Still ranges; optional to pin exact in CI/lockfile |
| **1.3 HF model revisions** | Pegasus ✅; LED and others need revision pinning | Pegasus + LED revisions in config_constants; summarizer pins them. Run manifest has revision fields but **does not populate** them | Populate `whisper_model_revision`, `summary_model_revision`, `reduce_model_revision` in `create_run_manifest()` |
| **1.4 Run manifest** | Git SHA, config SHA, OS/CPU/GPU, model revisions, device, generation params | `run_manifest.py` has RunManifest with most fields; created in orchestration; **model revisions not set**; no schema_version in create_run_manifest output for revisions | Add revision fields to `create_run_manifest()`; confirm device/generation params present |
| **1.5 Set seeds** | torch.manual_seed, numpy.random.seed, transformers.set_seed; document MPS non-determinism | **No seed setting in code.** Run manifest has `seed` but Config has **no `seed` field**; getattr(cfg, "seed", None) always None | Add `seed: Optional[int]` to Config; set seeds at pipeline start; document MPS |

### 2. Hardening Failure Modes (Section 2)

| Item | Issue says | Current code | Gap |
|------|------------|---------------|-----|
| **2.1 Per-episode status** | ok\|failed\|skipped; error_type, error_message, stage, retry_count | `workflow/metrics.py`: EpisodeStatus has status, error_type, error_message, stage, retry_count. `workflow/run_index.py`: EpisodeIndexEntry has status, error_type, error_message, error_stage. Pipeline records statuses. | Verify end-to-end: status/error_type/error_message/stage flow into run index and run.jsonl |
| **2.2 Timeouts** | Transcription 30 min, summarization 10 min; retry for transient HF/IO | Config: `transcription_timeout`, `summarization_timeout` (defaults 1800, 600). `episode_processor`: transcription wrapped in `timeout_context`. **Summarization**: no process-level timeout_context around summarization call | Add timeout enforcement for summarization (e.g. wrap in `timeout_context` where summarization runs per episode) |
| **2.3 Device fallback** | Log device fallback events | MPS OOM fallback exists; need to confirm logging of fallback events | Verify/add log when falling back to CPU |
| **2.4 Model-load fallback** | If from_pretrained fails: clear cache folder, re-download, retry once | No clear-cache-and-retry in summarizer/model_loader | Implement one retry after clearing model cache on load failure |

### 3. Supply-Chain & Security (Section 3)

| Item | Issue says | Current code | Gap |
|------|------------|---------------|-----|
| **3.1 trust_remote_code** | Default False; document when True needed | summarizer.py uses `trust_remote_code=False` | Done; add short doc note if missing |
| **3.2 Safetensors** | Prefer safetensors; document | Not explicitly enforced | Document; optional to prefer safetensors in load |
| **3.3 Model allowlist** | Configurable allowlist of allowed models | `ALLOWED_HUGGINGFACE_MODELS` in config_constants; summarizer validates in `_validate_model_allowlist` | Done |
| **3.4 Cache path validation** | HF cache inside designated dir; sanitize model names | `path_validation.py`: validate_cache_path, sanitize_model_name; summarizer uses them | Done |

### 4–8 (Performance, Output contract, Tests, CLI, Packaging)

Deferred to Phase 2/3 per issue. Not re-analysed in detail here.

---

## Internal Plan (Phased)

### Phase 1 (Critical – Before Shipping)

**1. Reproducibility**

- 1.1 **CI Python:** Pin exact Python version (e.g. 3.11.8) in all workflows that use setup-python.
- 1.2 **ML deps:** Leave as ranges in pyproject.toml for now; optional: add note in issue for future lockfile/CI pin.
- 1.3 **Run manifest revisions:** In `run_manifest.create_run_manifest()`, set `whisper_model_revision`, `summary_model_revision`, `reduce_model_revision` from config (or from constants where we pin).
- 1.4 **Run manifest:** Already has Git SHA, config hash, OS/CPU/GPU, devices, temperature; after 1.3 it will have model revisions.
- 1.5 **Seeds:** Add `seed: Optional[int]` to Config (default None). In orchestration/setup, if seed is set: call `torch.manual_seed(seed)`, `numpy.random.seed(seed)`, and `transformers.set_seed(seed)` before any non-deterministic work. Document MPS non-determinism in docs.

**2. Failure hardening**

- 2.1 **Per-episode tracking:** Audit that `record_episode_status` / `update_episode_status` are called with error_type, error_message, stage on failure; run index/build_run_index use pipeline_metrics.episode_statuses so index has status, error_type, error_message, error_stage. Fix any missing wiring.
- 2.2 **Summarization timeout:** Locate the per-episode summarization call (e.g. in processing/metadata_generation) and wrap it with `timeout_context(cfg.summarization_timeout, "summarization for episode ...")`.
- 2.3 **Device fallback logging:** Confirm we log when switching to CPU after MPS OOM; add one log line if missing.
- 2.4 **Model-load fallback:** In summarizer (or central load path), on first `from_pretrained` failure: clear that model’s cache dir, retry once; then re-raise if still failing.

**3. Security**

- 3.1 trust_remote_code: Already False; add one-line note in docs if not present.
- 3.3–3.4 Allowlist and cache path validation: Already done; no code change.

### Phase 2 (Important – Quality)

- Output contract (versioned schemas, run.json, per-episode details).
- Tests (golden, chaos, offline).
- CLI (exit code policy, --fail-fast, --max-failures, --json-logs).

### Phase 3 (Polish)

- Performance (bounded queues, warm cache, run index).
- Packaging (pipx/uv, dependency checks, doctor command).

---

## Execution Order (Phase 1)

1. **Reproducibility**
   - Add Config field `seed`, then set seeds in pipeline start (orchestration/setup).
   - Populate model revision fields in `create_run_manifest()`.
   - Pin Python version in CI to exact (e.g. 3.11.8).
   - Document MPS non-determinism.
2. **Failure hardening**
   - Enforce summarization timeout (timeout_context).
   - Verify/fix per-episode error_type/error_message/stage → run index.
   - Add/verify device fallback log.
   - Implement model-load clear-cache-and-retry once.
3. **Security**
   - Doc note for trust_remote_code.

Starting with **Phase 1.1 (Reproducibility)**: add `seed` to Config and set seeds, then run manifest revisions, then CI Python pin.

---

## Phase 1.1 Completed (2026-02-10)

- **Config:** Added `seed: Optional[int]` with env `SEED` and CLI `--seed`.
- **Seeds:** `set_reproducibility_seeds(cfg)` in setup.py; called from orchestration after `initialize_ml_environment()`. Sets torch, numpy, transformers seeds when `cfg.seed` is set.
- **Run manifest:** `_revision_for_summary_model()` added; `create_run_manifest()` now sets `whisper_model_revision` (None), `summary_model_revision`, `reduce_model_revision` using same pinning as summarizer.
- **CI:** All workflows pin Python to `3.11.8` (docs, nightly, python-app, snyk).
- **MPS:** Config description and set_reproducibility_seeds docstring note MPS may remain non-deterministic; dedicated doc deferred.

---

## Phase 1.2 & 1.3 Completed (2026-02-10)

### Phase 1.2 (Failure hardening)

- **2.1 Per-episode failure recording:** When a processing job fails (exception or summarization timeout), we now call `pipeline_metrics.update_episode_status(episode_id, status="failed", stage="metadata"|"summarization", error_type=..., error_message=...)` so the run index gets `error_type`, `error_message`, and `error_stage`. Same for transcription failures in both sequential and concurrent paths in `workflow/stages/transcription.py`.
- **2.2 Summarization timeout:** In `workflow/stages/processing.py`, each `call_generate_metadata` is wrapped in `timeout_context(cfg.summarization_timeout, "summarization for episode {idx}")`. On timeout we record episode status as failed with stage "summarization" and error_type "TimeoutError".
- **2.3 Device fallback logging:** Already present in `providers/ml/summarizer.py` (warning when falling back from MPS/CUDA to CPU, info when move succeeds). No change.
- **2.4 Model-load fallback:** Already implemented in summarizer `_load_with_retry`: on load failure we clear that model's cache and retry once. No change.

### Phase 1.3 (Security)

- **3.1 trust_remote_code:** Added a short doc note in `docs/guides/DEVELOPMENT_GUIDE.md` under Security notes: HuggingFace model loading uses `trust_remote_code=False`; only enable `trust_remote_code=True` if a model's documentation explicitly requires it and the source is trusted (Issue #429).
