---
id: c5_outcome_hook_matched_rise
class: light
status: open
issue: none (0043's false FAIL on prod, 2026-11-05)
touches:
  - src/podcast_scraper/upgrade/outcome_hook.py
  - src/podcast_scraper/upgrade/cli_handlers.py
  - scripts/measure/plan_diff_ro.py
  - scripts/measure/write_digest_ro.py
  - tests/unit/podcast_scraper/upgrade/
  - tests/unit/scripts/measure/
---

# C5 — the outcome hook passes an accepted episode whose rise is the clone's own

Method: [CORPUS_CHANGE_METHOD.md](../../guides/CORPUS_CHANGE_METHOD.md) §4. Ids only.

## 0. Why light

No stored file changes: the hook decides whether an apply's outcome is the measured one and
writes only its receipt. No migration is open (0043 to 0045 landed on `1d00b5fa`), so no outcome
goes stale.

## 1. Intake

- 2026-11-05, prod, 0043 (records `0043_non_participant_names_removed`, step 11): every plan row
  differed only by the identity ledger's hash (refreshed by a nightly after the snapshot); the
  write digest proved the apply's bytes equal the gate's; the operator accepted the 101 rows. The
  apply's one measured change, `S2_hidden` +1 on one episode, then failed `compare()`, which fails
  any rise on an accepted (`differing`) episode without looking at the clone's delta. Receipt
  FAIL, exit 5, though prod's deltas equal the clone's; the run stopped before its ledger refresh.
- Asked by the prod session at the operator's request: (1) match an accepted episode whose delta
  equals the clone's; (3) the two read-only checks as committed tooling; (4) the closing ledger
  refresh after exit 5, or its command; (5) the dry run's hook comparison in the runbook. (2), an
  accept file carrying the write-digest proof, is a format change left for the next arc.

## 2. Outcome and baseline

- **Outcome:** for an accepted episode, a gating rise equal to the clone's measured delta for that
  episode passes; a rise the clone did not measure, or a different one, still fails; a fall still
  passes. After an outcome failure, `upgrade run` prints the exact ledger-refresh command.
  `scripts/measure/plan_diff_ro.py` and `write_digest_ro.py` exist and are tested.
- **Baseline:** `compare()` at `1d00b5fa` fails every rise on an accepted episode.

## 3. Diagnosis by loss

One rule: section 1.

## 4. Prediction

Dated: 2026-11-05T12:10

- 0043's prod case replayed through `compare()`: 1 failure before, 0 after.
- Accepted episode, rise the clone did not measure (a `prod_only` row): still 1 failure. A rise of
  2 where the clone measured 1: still 1 failure. Matched (not accepted) episodes: unchanged rule.
- Every existing outcome-hook test passes except the one line that asserts the old rule for a
  rise the clone did measure (none of them does; checked when writing the test).
- No stored file and no migration's plan changes.

### 4b. Rows added by the acceptance read (2026-11-05), and the rule they settle

**The rule, tightened by row 6:** an accepted `differs` entry whose differing fields are exactly
`["inputs"]` (the plan's counters equal the gate's; 0043's 101 rows) is compared like a matched
episode: every gating metric's prod delta must equal the clone's. Every other diff entry
(`prod_only`, `clone_only`, `differs` with other fields) keeps today's rule (no gating rise).

1. Per metric: clone `{S2_hidden: 1}`, prod `{S2_hidden: 1, S12_quotes_off_text: 1}` on an
   inputs-only episode: exactly one failure, for S12.
2. Smaller rise (prod +1, clone +2) on an inputs-only episode: fails.
3. A fall where the clone rose (prod -1, clone +1) on an inputs-only episode: fails (exact rule);
   on another differing entry: passes, as before.
4. Clone's delta not made on prod (prod none, clone +1), inputs-only: fails (as a matched episode).
5. A rise the clone did not measure (clone 0 or absent) on an inputs-only episode: fails.
6. `differs` with fields other than `inputs` only, with a rise equal to the clone's: fails (old
   rule; an equal rise there may be coincidence).
7. `prod_only` with a rise: fails; `clone_only` (absent from prod's plan) with a prod rise: fails.
8. Keys: tests use real-shaped meta paths (`feeds/f/run_1/metadata/ep.metadata.json`), the format
   of both `plan_diff` meta and `score_by_episode` keys.
9. S1c total still enforced after an inputs-only match: a `measure()`-level test.
10. Mixed call: an inputs-only matched rise plus a matched-episode mismatch: only the mismatch fails.
11. The receipt names the inputs-only episodes compared exactly (`inputs_only_matched`).
12. Non-gating metrics are never compared (unchanged).
13. The existing test's line asserting a rise on a differing episode (S1n on `e1`, where the clone
    measured only S1_hidden_unknown) stays a failure under the new rule.

## 5. Acceptance

Prediction read 2026-11-05 by a fresh agent (reviewer, Sonnet; not the builder), 4 minutes, before
any code in `src/`. Verdict: **rows added** (4b, 1 to 13; row 6 tightened the rule). Step-8 spot
checks: replay 0043's case through the old and the new `compare()` (1 failure before, 0 after);
mutation (the old line back, `==` weakened to `>=`); the full `test_outcome_hook.py` count; the
S1c check reached after the match; the two scripts read-only and tested.

## 6. Build

- `upgrade/outcome_hook.measure`: an accepted plan-diff entry that `_inputs_only` (kind `differs`,
  fields exactly `["inputs"]`) leaves the `differing` set, so `compare()` holds it to the clone's
  delta exactly as a matched episode; the receipt names those episodes (`inputs_only_matched`).
  `compare()` itself is unchanged.
- `upgrade/cli_handlers._cmd_run`: after `HookOutcomeFailed` (exit 5) it prints the ledger refresh
  the run did not reach, `identity-ledger --output-dir <root> --dry-run`, then without.
- `scripts/measure/plan_diff_ro.py` (the hook's own plan diff, read-only, `--out` outside the
  corpus) and `scripts/measure/write_digest_ro.py` (the would-be writes' hash, read-only).
- Tests: `test_outcome_hook.py` (the 0043 replay; exact match per metric for an inputs-only row:
  another metric, smaller, a fall against a rise, not made, unmeasured; other diff kinds keep the
  no-rise rule; a mixed run; S1c still checked; a non-gating metric; the exit-5 message) and
  `tests/unit/scripts/measure/test_plan_and_write_checks.py` (the scripts' output and that they
  write nothing). The repro (0043's case) was written first and failed on `1d00b5fa`'s code:
  `ep: S2_hidden rose by 1`.
- Mutation (scratchpad): the old rule back, any `differs` as inputs-only, every entry as
  inputs-only, inputs-only passing whatever prod shows, the receipt field dropped: 5 of 5 killed.

## 7. Measure

- `test_outcome_hook.py`: 65 passed with the new tests (64 before the exit-5 test was added).
- The scripts on snapshot-prod-20261103: `plan_diff_ro.py` 0043: 101 and 101 episodes, no
  difference; `write_digest_ro.py` 0043: 101 episodes, 1,155 files, `writes_sha256 6fa784209bfc…`,
  the value prod printed on 2026-11-05. `digest_matches` reads false now, as it must: this change
  edits modules inside 0043's code digest, and 0043 is landed (its outcome is no longer compared).
- No stored file and no plan changes: the hook only judges and writes its receipt.
- **Builder: GO.**

## 8. Verdict

Verdict read 2026-11-05 by a fresh agent (reviewer, Sonnet; not the builder, not the step-5
reader), inputs as file paths, ran the two test files only.

Review budget: minutes 5, tokens 60000

```text
REVIEW c5_outcome_hook_matched_rise 2026-11-05 commit f2e5dff3 reviewer Sonnet (fresh) minutes 4
PREDICTION vs MEASURE: matches; 68 passed (the two test files); mutation and snapshot figures are the builder's, not re-run
REQUIRED ITEMS: 4b rows 1-13 each covered by a named test (rows 3b and 13 at the compare() level, which is unchanged)
AFTER-STATE READ: compare() untouched; no accept file or non-inputs-only rows: unchanged; the change only tightens or equals outside the inputs-only class; exit-5 message only on HookOutcomeFailed; the scripts write nothing in the corpus, but --out was not guarded
OPEN CALLS: --out could point inside the corpus; digest_matches reads false by design after this change; the accept file does not carry the write-digest proof (next arc)
DOUBTED NUMBER: none
VERDICT: GO WITH CONDITIONS (1) write_digest_ro.py on the gate clone and on prod for any migration whose rows are accepted as inputs-only; (2) guard --out against paths under --corpus, or document it
```

Conditions answered: (2) built: `plan_diff_ro.py` refuses an `--out` inside `--corpus`
(`test_an_out_path_inside_the_corpus_is_refused`; the plan-report test now writes outside its
corpus). (1) carried as a condition below.

Verdict: GO WITH CONDITIONS

- **Condition 1:** an accept file whose rows are `differs` on `inputs` only is written only after
  `scripts/measure/write_digest_ro.py` printed the same `writes_sha256` on the gate's clone and on
  prod; the hook does not check the bytes itself until the accept file carries that proof.
- **Condition 2:** after this change, `plan_diff_ro.py`'s `digest_matches` reads false for 0043 to
  0045 (their outcomes were built on the previous hook); they are landed and not re-applied, so it
  is expected, not a stop.

## 9. Operator review

- Operator, 2026-11-05, in the ingestion session (typos corrected): "can you do the hook fixes now
  and also send to main asap?".

## 10. Before apply / deploy

_TODO_

## 11. After apply / deploy

_TODO_

## 12. Retrospective

_TODO_

## Not covered

_TODO_
