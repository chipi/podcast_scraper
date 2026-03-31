# PR #476 follow-up: test plan for Codecov / coverage gaps

This document prioritizes **high-value** tests to raise **patch** and **project**
coverage after the GI/KG/eval/viz merge, without duplicating the full pyramid.

**Principles** (see `.cursor/rules/testing-strategy.mdc`, `docs/guides/TESTING_GUIDE.md`):

- Prefer **unit** tests with mocks for I/O and LLM calls.
- Add **integration** tests only where multiple real modules must interact.
- Reuse fixtures under `tests/fixtures/gil_kg_ci_enforce`, `tests/fixtures/kg/`.

**Why Codecov can show 0% on files that have tests:** uploads are keyed to CI
flags/commits; `tests/unit/evaluation/test_eval_gi_kg.py` must run under the same
job that produces the uploaded XML. If reports are stale or from an older SHA,
ignore the UI until the latest green run is selected.

---

## Phase 1 — Correctness & regression (do first)

### 1.1 `evaluation/scorer.py` (GIL/KG / vs_reference paths)

**Goal:** Cover new branches that aggregate metrics and handle partial runs.

| Test case | Idea |
|-----------|------|
| Partial predictions vs reference | Same as existing scorer test idea: `missing` reference episodes → log path uses `sorted(..., key=str)`; assert no crash and expected keys. |
| Empty `predictions.jsonl` / wrong task | `score_run` with `task=grounded_insights` and empty file → structured error or empty intrinsic (match current contract). |
| vs_reference error blob | Reference side has `"error"` key → scorer skips and continues (if implemented). |

**Where:** extend `tests/unit/podcast_scraper/evaluation/test_scorer.py` or
`tests/unit/evaluation/test_eval_gi_kg.py` depending on where `score_run` tests
already live.

### 1.2 `evaluation/experiment_config.py`

**Goal:** Lock validator behavior for GIL/KG eval YAML shapes.

| Test case | Idea |
|-----------|------|
| Invalid `task` + `backend` combo | Beyond existing stub tests: wrong `dataset_id` format if validated. |
| Optional fields defaults | Load minimal dict → defaults match RFC/docs. |

**Where:** `tests/unit/podcast_scraper/test_config.py` or dedicated
`tests/unit/podcast_scraper/evaluation/test_experiment_config_gi_kg.py`.

### 1.3 `evaluation/schema_validator.py` (if GIL/KG metrics schemas)

**Goal:** Reject malformed `metrics_gil_*` / `metrics_kg_*` payloads early.

| Test case | Idea |
|-----------|------|
| Missing required top-level keys | `ValidationError` or clear error. |
| Valid minimal payload | Passes. |

---

## Phase 2 — GIL support modules (pure / mockable)

### 2.1 `gi/compare_runs.py`

**Goal:** Stats and reporting without touching real run directories beyond `tmp_path`.

| Test case | Idea |
|-----------|------|
| `collect_gil_stats_from_run_root` | `tmp_path` with one fake `metadata/*.gi.json` (minimal valid shape). |
| `paired_episode_rows` | Two runs with same episode ids → row count. |
| `summarize_agreement` | Controlled inputs → expected counts/rates. |
| `format_text_report` | Snapshot or substring assertions (stable headings). |

**Where:** `tests/unit/podcast_scraper/gi/test_compare_runs.py` (extend).

### 2.2 `gi/deps.py`

**Goal:** Provider wiring and validation without loading transformers.

| Test case | Idea |
|-----------|------|
| `_provider_field_str` | `MagicMock` attribute → default string (already partially covered by pipeline tests). |
| `validate_gil_grounding_dependencies` | `generate_gi=False` → no-op; `entailment_provider` not local → no-op; local + missing `sentence_transformers` → `ProviderDependencyError` (mock import). |

**Where:** extend `tests/unit/podcast_scraper/gi/test_deps.py`.

### 2.3 `gi/provenance.py` / `gi/corpus.py` / `gi/io.py`

**Goal:** Thin file I/O and parsing.

| Test case | Idea |
|-----------|------|
| Round-trip or read failure | Use `tmp_path`; assert exceptions or empty corpus behavior per contract. |

**Where:** extend existing `test_gi_corpus.py` / new small tests next to them.

---

## Phase 3 — KG surface (CLI handlers vs `cli.py`)

### 3.1 `kg/cli_handlers.py`

**Goal:** Execute handler functions used by `cli.main` with argv mocks.

| Test case | Idea |
|-----------|------|
| `validate` / `inspect` code paths | If not fully covered by `test_kg_cli.py`, call handlers directly with `tmp_path` fixtures. |
| Error exit codes | Invalid path → non-zero (assert documented code). |

**Where:** `tests/unit/podcast_scraper/kg/test_kg_cli.py` or
`tests/unit/podcast_scraper/kg/test_kg_cli_handlers.py`.

### 3.2 `kg/pipeline.py` / `kg/llm_extract.py`

**Goal:** Stub provider responses; no network.

| Test case | Idea |
|-----------|------|
| Extract with invalid JSON from stub | Retry or error path. |
| Pipeline skip flags | `generate_kg=False` style branches if present. |

**Where:** extend `tests/unit/podcast_scraper/kg/test_kg_pipeline.py`,
`test_kg_llm_extract.py`.

---

## Phase 4 — Autoresearch track A (`evaluation/autoresearch_track_a.py`)

**Goal:** Cover judge helpers without API keys.

| Test case | Idea |
|-----------|------|
| `resolve_judge_openai_key` / `resolve_judge_anthropic_key` | `monkeypatch` env: dedicated key wins; `ALLOW_PRODUCTION_KEYS` + prod key; missing → `AutoresearchConfigError`. |
| `rouge_weight_from_env` | Invalid string → error; bounds. |
| `load_judge_config` | Non-dict YAML → `ValueError`. |
| `call_openai_judge` / `call_anthropic_judge` | `@patch` client → fixed message → `parse_judge_score_json` path. |
| `judge_one_episode` | Both providers mocked → `JudgeOutcome` + contested flag. |
| `summary_text_from_prediction` / `transcripts_by_episode_id` | Small dict / `tmp_path` materialized files. |

**Where:** extend `tests/unit/podcast_scraper/evaluation/test_autoresearch_track_a.py`.

---

## Phase 5 — Pricing & metrics helpers

### 5.1 `pricing_assumptions.py`

**Goal:** Edge cases beyond `test_pricing_assumptions.py`.

| Test case | Idea |
|-----------|------|
| Missing provider / model in YAML | `lookup_*` returns `None` or raises per docstring. |
| Integration with `workflow.helpers` | One test that `estimate_cost` (or equivalent) uses assumptions when file present (mock). |

### 5.2 `gi/quality_metrics.py` / `kg/quality_metrics.py`

**Goal:** Already partially hit by `gil_quality_metrics` / `kg_quality_metrics`
scripts; add **direct** unit tests on dataclass methods.

| Test case | Idea |
|-----------|------|
| `avg_*` with empty lists | Returns `0.0`. |
| `to_dict` | Keys stable for CI scripts. |

**Where:** `tests/unit/podcast_scraper/gi/test_gil_quality_metrics.py` and KG
counterpart.

---

## Phase 6 — Large files (cherry-pick only)

Use **surgical** tests; do not try to cover whole files.

| Module | Suggested focus |
|--------|-----------------|
| `cli.py` | New subcommands/flags: parse_args + one `main([...])` smoke per branch (pattern from `test_kg_cli.py`). |
| `metadata_generation.py` | New GI/KG call sites: mock provider, assert `build_artifact` / KG extract invoked with expected config flags. |
| `providers/*/...` | Single test per provider for **new** method only (mock HTTP). |
| `workflow/helpers.py` | Functions touched for episode id / metrics; mirror `test_workflow_helpers.py` style. |
| `gi/explore.py` | One CLI or entry function test with `tmp_path` + mock corpus. |

---

## Execution order (suggested sprints)

1. **Sprint A (1–2 sessions):** Phase 1 + 2.1 + 2.2 + 4 (autoresearch mocks).
2. **Sprint B:** Phase 3 + 5.2.
3. **Sprint C:** Phase 5.1 edges + Phase 6 as needed for Codecov comments.

After each sprint: `make test-unit` (or `make ci-fast` if you touched integration).

---

## Markers

Use `@pytest.mark.unit` on new tests; reserve `@pytest.mark.integration` for
any test that loads real ML or hits network (should be rare in this plan).

---

## References

- Existing GI/KG eval tests: `tests/unit/evaluation/test_eval_gi_kg.py`
- GIL pipeline: `tests/unit/podcast_scraper/gi/test_pipeline.py`
- KG CLI: `tests/unit/podcast_scraper/kg/test_kg_cli.py`
- Autoresearch: `tests/unit/podcast_scraper/evaluation/test_autoresearch_track_a.py`
