# Tests and docs audit — 2026-08-18

Scope: every read-only gate the repo already ships, plus four measurements the gates do not
make. Branch `fix/cleaner-and-open-items`, rebased on `1444ebc6`.

The point of this pass was to find **gaps, issues and drifts** — not to restate that the suite
is green. Where something is fine, the number is given so the claim can be re-checked; where
something is not, it is named with the command that produced it.

---

## 1. Existing gates: what is red

Every `*-check` / `lint-*` target was run. `strip-*` targets were skipped because they mutate
files rather than report.

| Gate | Before | After |
|---|---|---|
| `check-unit-imports` | PASS | PASS |
| `check-test-policy` | PASS | PASS |
| `profile-drift-check` | PASS | PASS |
| `check-pricing-assumptions` | PASS | PASS |
| `verify-stack-profiles` | PASS | PASS |
| `lint-markdown-docs` | PASS | PASS |
| `spelling-docs` | PASS | PASS |
| `docstrings` | PASS | PASS |
| **`check-doc-structure`** | **FAIL** | PASS — fixed, `efca9e94` |
| **`docs-check`** | **FAIL** (same cause) | PASS |
| **`lint`** | **FAIL** | PASS — fixed, `efca9e94` |

### 1.1 `check-doc-structure` was red for everyone, on nobody's fault

16 broken links reported, **all 16 inside `.venv-dev/lib/python3.11/site-packages`** — a
vendored PHP parser, deepgram, nltk — and **0 in our own docs**. `SKIP_DIRS` listed the literal
`.venv`, but the dev environment is `.venv-dev`, so the walk descended into third-party
packages.

This matters more than a one-word typo suggests. The script was added on 2026-08-17
specifically so documentation drift would fail loudly. It shipped permanently red, and a gate
that can only fail on files nobody can edit is a gate that gets ignored — the exact failure it
was written to prevent. Now: `Documentation structure OK (14 required READMEs, 947 markdown
files linked correctly)`.

### 1.2 `make lint` was red — and `make format-check` cannot see it

Two errors, both introduced on this branch. The important one is process, not the fix:
`format-check` runs black and isort only. **F401 (unused import) is invisible to it** — only
`make lint` (flake8) catches it. I had been running `format-check` before each commit and not
`lint`, so `d816c8a1` shipped an unused `pathlib.Path` import.

**Recommendation:** the pre-commit habit should be `make lint && make format-check`, or the two
should be folded into one target. Right now it is possible to be "green" by the check you
remember to run.

---

## 2. Measurements the gates do not make

### 2.1 Tests that cannot fail — 219 of 11,713 (1.9%)

Scanned every `test_*` function with an AST walk for an `assert`, a `raise`, `pytest.raises`,
or any `assert*` call. 219 have none.

**Most are legitimate.** A "should not raise" smoke test is a real assertion — if the call
raises, the test fails. 49 of the 219 are cache/cleanup/no-op/idempotency checks of exactly
that shape, and they are correct.

**Three are not.** These wrap the call in `try/except`, swallow the exception, and assert
nothing — they pass whether the code works or not:

| File | Test |
|---|---|
| `tests/e2e/test_pipeline_error_recovery_e2e.py:448` | `test_pipeline_handles_invalid_config_gracefully` |
| `tests/integration/providers/test_provider_error_handling_extended.py:98` | `test_cleanup_after_failed_initialization` |
| `tests/integration/providers/test_provider_error_handling_extended.py:467` | `test_provider_cleanup_on_exception` |

The first says so in its own comments — *"If it doesn't raise, that's okay"*, *"except … that's
good"*. It accepts both outcomes by construction.

**Not fixed in this pass.** Each needs a decision about what the test was meant to prove, which
is a change to intent, not a cleanup. Filed here rather than guessed at.

### 2.2 A test I wrote today had this exact defect

Worth recording because it is the same failure mode and it nearly shipped. My first regression
test for the transcript cleaner used a synthetic transcript; it passed **with the fix disabled**,
because the synthetic never reproduced the span-merge chain. It pinned nothing.

The committed version (`c6893892`) uses the fixture that does reproduce it, and I verified it
red-then-green:

```
fix disabled -> AssertionError: cleaner kept only 418/3427 chars (12.2%)
fix restored -> 12 passed
```

**Recommendation:** for any regression test, disabling the fix and watching it go red is the
only proof the test has teeth. Nothing in CI can check this for you.

### 2.3 Skip / xfail hygiene — clean

- **0** unconditional skips (`pytest.mark.skip`, `@unittest.skip`). Nothing is silently off.
- **1** real `xfail`, and it is `xfail(strict)` against #1669, so it fails if it starts passing.
  The other four `xfail` mentions are prose in docstrings.
- 54 `skipif`/`skipUnless`, which are environment guards (no `lancedb`/`transformers`/spaCy on
  this Intel Mac) and behave correctly.

### 2.4 Pytest markers — 12 dead declarations

**0** markers used but undeclared (strict-markers is doing its job). **12** declared and never
used:

```
analysis, app, chaos, data_quality, golden, infrastructure, module_cache,
module_evaluation, module_exceptions, module_groq_providers, multi_episode, offline
```

Harmless, but they advertise selectors that select nothing — `pytest -m golden` silently runs
zero tests rather than erroring. Worth pruning or wiring up.

### 2.5 ADR/RFC/PRD/UXS citations in code — clean

Cross-checked every `ADR-nnn` / `RFC-nnn` / `PRD-nnn` / `UXS-nnn` cited in `src/` and `tests/`
against the docs on disk.

| Kind | Docs on disk | Cited in code | Cited but missing |
|---|---|---|---|
| ADR | 152 | 54 | 0 |
| RFC | 116 | 49 | 0 (see below) |
| PRD | 46 | 14 | 0 |
| UXS | 16 | 4 | 0 |

The scan initially flagged 8 missing RFCs. All 8 are **IETF** RFCs — 1918 (private addresses),
7636 (PKCE), 8058, 8252, 8259 (JSON), 8414, 8707, 9728 — cited correctly as internet standards.
Zero real drift. Noted so the next person running this check does not re-investigate them.

---

## 3. What this pass did NOT cover

Equal weight, per the coverage rule.

- **Whether documentation prose is TRUE.** No tool checks this, and `check_doc_structure.py`
  says so explicitly. Structure and links are verified; claims are not.
- **The e2e and stack-test suites were not executed.** Only collected (12,781 tests, 0 import
  errors). Runtime behaviour is unverified here.
- **Search / ML-dependent tests did not run** — no `lancedb`, `transformers`, `sklearn`, `PIL`
  on this Intel Mac. Those skips are counted above but their subject matter is untested locally.
- **Docker-dependent gates were not run at all** — `stack-test-*`, `index-two-tier-docker`,
  `app-docker-build`. The daemon has been dead since 2026-08-17 23:38:59 (task #43).
- **The eval gate and corpus audit are not in CI**, so a green PR still proves nothing about
  ranking quality or fixture soundness. Pre-existing, unchanged by this pass.
- **Coverage percentages were not re-measured.** The e2e floor was last moved to 39.40% against
  a 39.00 floor; I did not re-run it.
- **No review of `web/` (Playwright/Vitest) tests.** Python only.

---

## 4. Actions

Done:

- `efca9e94` — doc-structure gate no longer walks virtualenvs; `lint` green.

Open, needing a decision rather than a cleanup:

1. Fix or delete the 3 unconditionally-passing tests (§2.1).
2. Prune or wire up the 12 dead pytest markers (§2.4).
3. Decide whether `make lint` joins `format-check` in the pre-commit habit (§1.2).
