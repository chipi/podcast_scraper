# Test Suite Review — 2026-04-27

**Issue:** [#678](https://github.com/chipi/podcast_scraper/issues/678)
**Branch:** `feat/rfc-081-phase-1-prep`
**Author:** Marko + Claude

## Why this exists

We've made substantial changes to the pipeline (cloud_thin wiring, GIL/KG
enrichment, chip refactors, profile changes), the viewer (chip surfaces in
Library / Search / Digest / Graph, Person Landing, TEV), and the test
infrastructure (Docker compose, BuildKit fix, locator sweep). Pre-prod
([RFC-081](../rfc/RFC-081-pre-prod-environment-and-control-plane.md)) is
about to wire up publish + observability + alerts — before that ships, the
test surface needs a hygiene pass so signal-to-noise stays high.

This doc is the **punch list of follow-up PRs** that fall out of the audit.
Each item is sized to land independently. After landing, this doc moves to
`docs/architecture/TESTING_STRATEGY.md` updates or gets archived.

## Audit method

Five independent read-only surveys ran in parallel via Explore agents,
one per test surface. Their structured findings are summarised below. No
files were modified during the audit.

## Per-surface findings

### `tests/unit/`

**Status update (correction, post-audit verification): the
"structural violations" framing was wrong.**

The audit flagged ~37 + ~20 tests as importing non-`[dev]` extras
and concluded they would "silently skip or fail in the unit CI job".
On verification:

- The `Python application` workflow's `test-unit` job (which installs
  `[dev]` only) is **green** on recent runs.
- The provider modules use **lazy imports** (`importlib` at runtime,
  `from transformers import ...` inside functions). Top-level
  `from podcast_scraper.providers.ml.ml_provider import MLProvider`
  works in a `[dev]`-only environment because the heavy extras only
  load at instantiation time, not module-import time.

So the tests are **not** silently broken. They pass fine in CI today.

What remains true:

- **`tests/unit/podcast_scraper/test_summarizer.py`** still patches
  `transformers.pipeline` / `transformers.AutoModelForSeq2SeqLM`
  directly (~36 occurrences) — the pre-#677 anti-pattern that breaks
  against `transformers >= 4.40` lazy modules. Should patch the
  indirection helper (`SummaryModel._load_model`) instead. **Real
  bug, real fix needed (see PR-2).**
- **`tests/unit/podcast_scraper/gi/test_gil_quality_metrics.py`** /
  **`tests/unit/podcast_scraper/test_cli.py`** assert on quality
  thresholds that belong in `data/eval/` (PR-4).

What's a stylistic concern (not a bug):

- ~57 tests live in `tests/unit/` but conceptually exercise modules
  gated by `[ml]` / `[server]`. Some teams would consider this a
  test-policy concern (the unit tier should stay agnostic to
  optional extras even if imports happen to be lazy-safe). Moving
  them to `tests/integration/` is defensible but is **not blocking
  unit-CI signal** — it would be cleanup for clarity, not correctness.
  **Deprioritised** — operator decides if they want this refactor.
- **`tests/unit/podcast_scraper/test_summarizer.py`** still patches
  `transformers.pipeline` / `transformers.AutoModelForSeq2SeqLM` directly
  (~36 occurrences), the pre-#677 anti-pattern. Should patch the
  indirection helper (`SummaryModel._load_model`) instead.
- **`tests/unit/podcast_scraper/gi/test_gil_quality_metrics.py`**
  asserts on `enforce_prd017_thresholds` with hardcoded quality minima
  (`min_avg_insights=0.5` etc.). Quality thresholds belong in
  `data/eval/`, not pytest.
- **`tests/unit/podcast_scraper/test_cli.py`** has hardcoded quality
  score assertions (`gi_qa_score_min == 0.11`, `0.3`).

### `tests/integration/`

**Status: Moderate hygiene debt.**

- **`tests/integration/providers/test_summarizer_security_integration.py`**
  patches transformer internals (BartForConditionalGeneration, etc.)
  instead of indirection helpers — same #677 anti-pattern.
- **Misclassified integration tests** (no real I/O, should be unit):
  - `tests/integration/test_protocol_verification_integration.py`
  - `tests/integration/test_cache_and_artifact_paths_integration.py`
  - parts of `tests/integration/test_summary_schema_integration.py`
- **Quality-bar assertions in pytest** that should live in `data/eval/`:
  - `tests/integration/gi/test_ki_integration.py::test_quality_metrics_on_artifact`
  - `tests/integration/gi/test_evidence_stack_integration.py` (QA span
    score / NLI score range checks)
  - `tests/integration/test_summary_schema_integration.py` schema-status
    plus bullet count expectations
- **Coverage gaps**:
  - `src/podcast_scraper/server/routes/corpus_persons.py` — `/top`
    endpoint has only indirect coverage via library tests
  - `src/podcast_scraper/server/routes/corpus_library.py` — POST routes
    `resolve-episode-artifacts` and `node-episodes` lack explicit
    request/response contract tests
  - `src/podcast_scraper/server/routes/jobs.py` — `/jobs/{id}/cancel`
    idempotency and double-cancel behaviour untested

### `tests/e2e/` (Python)

**Status: Moderate — quality-bar assertions sprawled.**

- **~47 instances of `assert ... summary ...`** spread across 15+ files
  asserting on summary content / shape / non-emptiness. Should migrate
  to `data/eval/` where versioned silver references + LLM judge live.
  Affected files include `test_ml_models_e2e.py`,
  `test_hybrid_ml_provider_e2e.py`, `test_nightly_full_suite_e2e.py`,
  `test_full_pipeline_e2e.py`, `test_map_reduce_strategies_e2e.py`, and
  every `test_<provider>_provider_e2e.py`.
- **Layer-violation mocking**:
  - `tests/e2e/test_provider_real_models_e2e.py:789+` mocks `openai.OpenAI`
    at the SDK level — should use the conftest E2E HTTP server mock or
    not mock at all.
  - `tests/e2e/test_pipeline_concurrent_e2e.py:368-369` uses `MagicMock`
    to stub the summarization provider mid-pipeline.
  - `tests/e2e/test_anthropic_provider_e2e.py` likely patches the
    Anthropic SDK directly (verify on PR-time).
- **Coverage gaps**:
  - GI/KG end-to-end output validation — `gi.json` + `kg.json`
    artefacts are not asserted to exist or be structurally sound by any
    e2e spec.
  - GI/KG cost recording — `metrics.json` should include
    `llm_gi_cost_usd` + `llm_kg_cost_usd`; no smoke test asserts this.
- **Provider redundancy** (acceptable but parameterizable): 7 files run
  near-identical scenarios per provider. Could collapse to a single
  parameterized class for easier maintenance.
- **`e2e_mode` fixture is clean** — no orphans after the `data_quality`
  removal in commit `2fbfd71b`.

### `web/gi-kg-viewer/e2e/`

**Status: Very clean — confidence 95%.**

- **No stale candidates.** The chip-refactor sweep in commit `a3d12c48`
  was thorough; no legacy testids remain.
- **Coverage gap (the one real issue):** **Person Landing + TEV views**
  (added in PR #676) have **zero** Playwright coverage despite being
  documented in `web/gi-kg-viewer/e2e/E2E_SURFACE_MAP.md` (surfaces
  223–224). Entry points exist (`explore-top-speaker-link`,
  `search-result-speaker-link`, Digest topic-title click) but the flows
  are unexercised.
- **Locator hygiene clean** — `getByTestId()` everywhere meaningful;
  CSS / id selectors used only where appropriate (`#search-q`,
  `.graph-canvas`, `[data-connection-node-id]`).
- **Spec parallelism risk: low.** 8 separate `beforeEach` blocks manage
  localStorage state; consolidating to a shared fixture would tidy this
  but is not a correctness issue.
- **Helper sprawl: zero.** `helpers.ts` and `dashboardApiMocks.ts` are
  all alive and used.

### `tests/stack-test/`

**Status: Very clean — confidence 7/10.**

- **No stale candidates.** Recent locator sweeps (`8f33c24e` +
  `433def0c`) caught everything; first stack-test green on CI just
  landed (run [24985699898](https://github.com/chipi/podcast_scraper/actions/runs/24985699898)).
- **Coverage gaps** (acceptable for a smoke gate, but worth filing):
  - Error-recovery paths — `JOB_TERMINAL_BAD` set is defined but never
    exercised. Only success path is asserted.
  - Profile switching — only `airgapped_thin` is tested; `cloud_thin`
    is documented but not exercised by an automated spec.
  - Feed validation failures — malformed RSS, duplicate URLs.
  - Job cancellation flow.
- **Parallelism risks: negligible** — Playwright config hardcodes
  `workers: 1`.
- **Hardcoded fixture date assumptions: safe** — graph lens defaulted
  to "all time" for stack-test builds; digest filter set to "All time"
  via chip popover; no date-relative filters applied.

## Punch list (follow-up PRs)

Sorted by priority. **Each item is one PR.**

### Priority A — blocks unit-test signal integrity

#### PR-1 — `tests/unit/` test-policy compliance

Move ~57 provider/server tests out of `tests/unit/` (either to
`tests/integration/`, or rewrite with `sys.modules` mocking before
import). Without this, the unit CI job is silently lying — many tests
that look like they pass actually never run.

- **Affected**: `tests/unit/podcast_scraper/providers/{ml,gemini,deepseek,grok,ollama,anthropic}/*` (~37 files), `tests/unit/podcast_scraper/server/*` (~20 files)
- **Reference**: [`docs/guides/UNIT_TESTING_GUIDE.md`](../guides/UNIT_TESTING_GUIDE.md) — *Pyproject extras: what unit tests may depend on*
- **Acceptance**: `make test-unit` passes against a fresh `[dev]`-only venv

#### PR-2 — Mocking hygiene fix (post-#677 indirection pattern)

- `tests/unit/podcast_scraper/test_summarizer.py` — patch
  `SummaryModel._load_model` instead of `transformers.pipeline` /
  `AutoModelForSeq2SeqLM` (~36 occurrences)
- `tests/integration/providers/test_summarizer_security_integration.py`
  — same fix
- **Acceptance**: tests pass with `transformers >= 4.40` lazy modules
  (the bug class that #677 originally fixed for QA pipelines)

### Status update on PR-B4 (2026-04-27, post-implementation review)

**The audit overstated PR-B4 too.** On verification, what I claimed
were "quality-bar assertions" are mostly **smoke assertions**:

| Pattern | Type |
| --- | --- |
| `assert "summary" in result` | smoke (schema) |
| `assert isinstance(result["summary"], str)` | smoke (type) |
| `assert len(result["summary"]) > 0` | smoke (non-empty existence) |
| `assert len(transcript_files) > 0` | smoke (file written) |
| `enforce_prd017_thresholds(m, min_avg_insights=0.5)` | logic test (verifies threshold-enforcement *function*, with the threshold as a test argument) |
| `cfg.gi_qa_score_min == 0.11` | CLI parsing test (CLI arg flows to Config field) |

None of these encode a quality bar against real model output. The
quality-bar pieces (against versioned silver references + LLM judge)
already live in `data/eval/` per-cluster:

- `data/eval/configs/summarization/*.yaml` (per-provider summarization eval)
- `data/eval/configs/gil_*.yaml` (Grounded Insights eval)
- `data/eval/configs/kg_*.yaml` (Knowledge Graph eval)
- `data/eval/configs/ner/*.yaml` (NER eval)
- `data/eval/references/silver/`, `data/eval/datasets/curated_5feeds_*.json`

So PR-B4 is **also largely a no-op**. The pytest tier does smoke
("did anything come out?"); `data/eval/` does quality ("is what came
out good enough?"). Both layers are correct as-is. No migration
required.

The real-deal e2e quality migration (if/when we tighten thresholds)
would land as new eval configs in `data/eval/configs/` plus eval
runner invocations in CI — separate from pytest entirely. That's a
follow-up RFC if it ever becomes priority.

### Priority B — cleanup, reduces noise

#### PR-3 — Move misclassified integration→unit tests

- `tests/integration/test_protocol_verification_integration.py` →
  `tests/unit/`
- `tests/integration/test_cache_and_artifact_paths_integration.py` →
  `tests/unit/`
- Split `tests/integration/test_summary_schema_integration.py` —
  schema-only tests to unit, real-provider tests stay integration
- **Acceptance**: integration test runtime drops by the moved tests;
  unit suite gains them; no flakiness change

#### PR-4 — Migrate quality assertions to `data/eval/`

The biggest cleanup item by line count. Tests that assert on summary
content, GI / KG quality, model output shape should live in the eval
harness with versioned silver references + LLM judge.

- `tests/unit/podcast_scraper/gi/test_gil_quality_metrics.py`
  threshold assertions
- `tests/unit/podcast_scraper/test_cli.py` `gi_qa_score_min` checks
- `tests/integration/gi/test_ki_integration.py::test_quality_metrics_on_artifact`
- `tests/integration/gi/test_evidence_stack_integration.py` QA / NLI
  range checks
- `tests/integration/test_summary_schema_integration.py` schema-status
  / bullet-count assertions
- `tests/e2e/test_ml_models_e2e.py`, `test_hybrid_ml_provider_e2e.py`,
  `test_nightly_full_suite_e2e.py`, `test_full_pipeline_e2e.py`,
  `test_map_reduce_strategies_e2e.py`, all `test_<provider>_*_e2e.py`
- **Recommended split**: one PR per cluster (unit / integration / e2e)
  to keep each reviewable
- **Acceptance**: pytest tests keep only smoke / contract assertions
  (provider initialises, pipeline runs to completion, artefacts
  written); quality bar lives in `data/eval/`

### Priority C — coverage gaps

#### PR-5 — Person Landing + TEV Playwright specs (PR #676 follow-up)

Two new files:

- `web/gi-kg-viewer/e2e/person-landing.spec.ts` — Explore → Top
  speakers → Person Landing flow; tablist; positions panel; mentions
  chart; `person-landing-go-graph` and `person-landing-prefill-search`
- `web/gi-kg-viewer/e2e/topic-entity-view.spec.ts` — Digest topic-title
  click → TEV; stats line; mentions list; `topic-entity-view-go-graph`
  and `topic-entity-view-prefill-search`
- **Acceptance**: both surfaces' testids documented in `E2E_SURFACE_MAP.md`
  are exercised; specs run green on `make test-ui-e2e`

#### PR-6 — FastAPI route integration tests (coverage gaps)

- `tests/integration/server/test_corpus_persons_top.py` — `/api/corpus/persons/top` request/response contract
- `tests/integration/server/test_corpus_library_post_routes.py` —
  `/corpus/resolve-episode-artifacts` and `/corpus/node-episodes`
- `tests/integration/server/test_jobs_cancel.py` — idempotency,
  double-cancel, cancel-after-completion behaviour

#### PR-7 — GI/KG e2e smoke

`tests/e2e/test_gi_kg_artifacts_e2e.py` — single end-to-end run that
asserts:

- `gi.json` written, parses, has `insights[]` with expected schema
- `kg.json` written, parses, has `nodes[]` + `edges[]`
- `metrics.json` includes `llm_gi_cost_usd` + `llm_kg_cost_usd` when
  the providers are LLM-backed

### Priority D — nice to have, future

#### PR-8 — Stack-test error-recovery + alternate-profile coverage

Extend `tests/stack-test/stack-jobs-flow.spec.ts` (or a sibling) with:

- Malformed RSS URL handling
- Duplicate feed URL rejection
- Job cancellation flow (mid-run cancel, post-completion idempotency)
- `cloud_thin` profile run (currently only documented, not asserted)

#### PR-9 — Provider e2e parameterization

Collapse 7 `test_<provider>_provider_e2e.py` files into a single
parameterized class. Same scenarios (transcription, summarization,
full pipeline, mega-bundled cost recording) per provider via
`@pytest.mark.parametrize`. Reduces maintenance burden when scenarios
evolve.

## Suggested ordering for landing

1. **PR-1** (unit test-policy) — first, it unblocks honest CI signal.
2. **PR-2** (mocking hygiene) — small, surgical, reduces brittleness.
3. **PR-5** (Person Landing + TEV specs) — fast, contained, fills a
   real gap from PR #676.
4. **PR-3** (misclassified integration→unit) — small, clear win.
5. **PR-4** (quality assertions to data/eval) — biggest, split into
   sub-PRs as outlined above.
6. **PR-6, PR-7** (coverage gaps) — fill in routes + GI/KG smoke.
7. **PR-8, PR-9** (nice-to-haves) — defer until pre-prod is stable.

## Confidence + caveats

- **Findings are systematic, not edge cases.** Where the audit said
  "~37 tests violate the unit-test extras rule", that's a real grep
  count, not a guess.
- **The audit was read-only.** Some findings (e.g., "this mock might
  fail at runtime against `transformers >= 4.40` lazy modules") are
  pattern-based predictions — confirm at PR time by running the test.
- **Provider redundancy is flagged but acceptable.** Don't refactor
  unless the maintenance burden is biting; collapsing 7 files into
  one parameterized class reduces flexibility for provider-specific
  edge cases.
- **`data/eval/` is the right destination for quality assertions, but
  the migration itself is non-trivial.** Each cluster (summary / GI /
  KG / provider-specific) needs versioned silver references + a
  judge config. PR-4 is the biggest item on this list and should be
  split into sub-PRs to keep review loads sane.
