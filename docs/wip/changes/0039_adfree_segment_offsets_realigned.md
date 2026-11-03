---
id: 0039_adfree_segment_offsets_realigned
class: full
status: landed
issue: none (RFC-131 D2; found by 0041's offset guard, 2026-11-01)
touches:
  - src/podcast_scraper/upgrade/migrations/m0039_adfree_segment_offsets_realigned.py
  - src/podcast_scraper/upgrade/registry.py
  - tests/unit/podcast_scraper/upgrade/test_m0039_adfree_segment_offsets_realigned.py
---

# 0039 — every ad-free segment's offsets point at its own words again (text only)

Method: [CORPUS_CHANGE_METHOD.md](../../guides/CORPUS_CHANGE_METHOD.md) §4. Names are kept out of this
record; the list is in eval-data `data/eval/name-consolidation/d2i9/adfree_offsets_snapshot.jsonl`.

## 1. Intake

- **Found by 0041.** m0023's in-place prefix rename refuses an episode when an ad-free segment would
  not slice to its text ("a segment would not slice to its text"). On snapshot-prod-20261030 one
  episode is refused, and the refusal predates 0041: the episode's files are already inconsistent.
- **Measured** (`d2i9/adfree_offsets.py`, every served episode):
  - 2,948 episodes carry ad-free segment offsets;
  - **1** has rows that do not slice to their own text in its `.adfree.txt`: 187 of 258, from row 71
    on, off by 1 growing to 8 characters (one more per shortened prefix);
  - the untouched snapshot (before 0038) has the same 187, so 0038 did not cause it.
- **The cause:** a one-off hand repair on 2026-10-06 (receipts
  `speaker-org-china-daily-20261006.receipts.jsonl`, "name-first tail matched 'reported'"). It
  rewrote the episode's `.adfree.txt` and `.adfree.segments.json`, removing a speaker prefix, without
  moving the offsets after it. No tool in the repo writes those receipts (searched `scripts`, `src`,
  `docs`); 0012, which shares the receipt shape, never rewrites transcripts. Nothing to fix at the
  source; this migration's check is what catches the shape if it recurs.
- **Readers of the offsets:** the player's transcript (each segment's span in the ad-free text) and
  m0023's offset carry, which every speaker rewrite since 0023 relies on.

## 2. Outcome and baseline

Rows of served ad-free segments that do not slice to their own text: **187, in 1 episode.**

## 3. Diagnosis

A text rewrite that changes lengths must move every offset after the change; the hand repair moved
none.

## 4. Prediction

Dated: 2026-11-01

The change, migration `0039_adfree_segment_offsets_realigned` (registered after 0038, before 0041):
- **For each served episode whose ad-free segments do not all slice to their text:** walk the rows
  in order. Each row's text is searched for in `.adfree.txt` near its stored `char_start` (the
  window and its checks in 4b), and the row then takes that span.
- **All or none per episode:** one row not found that way refuses the episode, untouched and listed.
- Rows that already slice are unchanged. So are the text, the transcript, every other file, and
  every other field of a row.
- **Undo and safety:** a backup and a receipt, an atomic write, the corpus owner's ownership,
  verify (no row left not slicing, none refused) and undo, through `upgrade/file_rewrite`.
- **The ad map:** the ad-free transcript is not changed, so the episode's `.adfree.admap.json` (its
  offsets are into the raw transcript) is not touched.

Predicted on the clone (snapshot-prod-20261030 with 0038 and the D2 ledger):
- 1 episode written, 187 rows realigned, 0 refused;
- after: 0 rows not slicing corpus-wide;
- 0 other files changed;
- **0041's refusal of that episode goes away**; its gate is rerun after.
- A second run writes nothing.

### 4b. Rows added after the fresh read (the rule tightened before any code)

- **The window is narrow, not 64 characters.** A row's text must start within `MAX_DRIFT` (16)
  characters of its stored `char_start`, twice the largest drift measured. A repeated short line
  ("Right.", "Yeah.") inside the window is refused by the found-exactly-once rule.
- **A correction:** section 1 said the rows drift by "1, then 2, then 3 characters", and 4b first set
  the window to 8 as "the measured largest drift, 3, plus a margin". That measurement read only the
  first 50 rows that do not slice. Over all 187 the drift runs from 1 to 8 (`-8: 15, -7: 20,
  -6: 37, -5: 24, -4: 14, -3: 45, -2: 18, -1: 14`): one more character per shortened prefix. Found
  when the build's check on the real episode moved the last row by 8, exactly the window's edge.
- **Found exactly once in that window.** The text occurring twice there refuses the episode, as does
  not finding it.
- **The drift is monotone and spans never overlap or invert.** A rewrite that removed text can only
  move later offsets further one way. A row whose drift reverses direction, or whose span starts
  before the previous row's end, refuses the episode.
- **Only `char_start` and `char_end` change.** Every other key of every row is byte-identical,
  checked on the real episode, as are `.adfree.txt` and `.adfree.admap.json`.
- **`upgrade verify --only 0038` does not select 0039** (`--only` matches `id.startswith(o + "_")`);
  `--only 0039` does. A test registers 0039 and runs the landing and state checks on its
  non-numeric id.
- **Step 8:**
  - rows 70 to 72 and the last row, before and after;
  - all 187 slice to their own text after;
  - 0041 on that episode, refusing no longer;
  - a second run writes nothing.

### 4c. Revision after the measurement

Dated: 2026-11-01T10:56

**A correction to this section's first version:** it said `upgrade/registry.py` gained only 0042's
registration, and that 0039's module was unchanged since the measurement. Both were wrong, and the
fresh reader found it.
- The registry also gained 0041's registration.
- 0039's code digest (`outcome_hook.migration_digest`: the module and every module it imports)
  moved from `198b9929` at the measurement to `8476aa89`.

So the measurement is stale, and prod's outcome hook would refuse it. 0039 is re-measured with the
final code, in one batch with 0041 and 0042 after the last code change. Predicted: as section 4 (1
episode, 187 rows, 0 refused).

## 5. Acceptance

Fresh reader, 2026-11-01, a sonnet reviewer subagent with read-only tools, before any code: "not
complete"; five rows, all taken in 4b.

Revision (4c), fresh reader (a sonnet reviewer subagent with read-only tools, about 20 s):
Dated: 2026-11-01T10:55
"Not complete": the registry also gained 0041; 0039's file changed after the measurement. Both are
taken: 4c is corrected, and the change is re-measured.

## 6. Build

Tests (`tests/unit/podcast_scraper/upgrade/test_m0039_adfree_segment_offsets_realigned.py`):
- registered between 0038 and 0041;
- rows left early by a shortened prefix take their text again;
- each of these refuses the episode:
  - a repeated short line near the row;
  - a drift that shrinks;
  - a span that would overlap the row before;
  - a drift that reverses;
  - a row that slices again after the drift began;
- an episode whose rows slice is untouched.

Code: `upgrade/migrations/m0039_adfree_segment_offsets_realigned.py`, `MAX_DRIFT` 16, through
`upgrade/file_rewrite` (backup, receipt, atomic write, undo).

**Mutants** (`mutate.py`, run 2026-11-02 after the landing check found this section missing):
- killed 6 of 6: several occurrences accepted; the direction check off; the shrink check off; the
  overlap check off; an unbounded window; and "slices again after the drift began" off.
- The last one survived the first run: no test covered that branch. The test above was added, and
  then it was killed.

## 7. Measure

Gate (`0039_adfree_segment_offsets_realigned.gate.json`, the final batch, code digest `e0bedfef5a62`, 2026-11-01; each gate on its own base: snapshot-prod-20261030 with 0038, the D2 ledger built with the final rules (`--no-respell`), and the migrations before it applied):
- computed class full; gating moves none; **1 file changed**;
- `upgrade verify` 37 ok; failing 0025 alone, no regression (the one ledger-joined pair, failing on
  the base too, which 0041 folds);
- the real episode's check (`d2i9/check_0038a.py`): 258 rows of 258 slice; every other key, the
  ad-free text and the ad map byte-identical; a second run writes nothing.
- **Drift measured in full:** 1 to 8 characters (15 rows at 8), inside `MAX_DRIFT` 16.

### Final rerun on D2's final ledger (s5d through 4e, s6g, s6h; 2026-11-02)

Code digest `bc3c9bb670fd` (the shared digest moved with s6h's `identity/ledger` and `identity/display`). Chain eval-data `d2out/scripts/chain5.sh`, each gate on its own base: snapshot-prod-20261030 with 0038, the ledger rebuilt `--no-respell` with the final rules, the migrations before it applied. The gate report is from this run. Record s6h section 7 has the whole chain.
- 187 segment rows realigned in 1 episode, 0 refused: unchanged from the final batch.
- `upgrade verify` ok ("every served ad-free segment slices to its own text"), no regressions,
  no gating moves.

## 8. Verdict

**Builder's verdict, before the review (2026-11-02): GO WITH CONDITIONS.** 187 rows realigned in 1 episode, 0 refused, as predicted; verify ok; second apply 0; undo byte-identical. Gate report on committed code `b64e08e04` (clean, code digest `bc3c9bb670fd`, worktree run eval-data `d2out/scripts/chain7.sh`), equal in outcome to the final rerun in section 7. Second apply and full-copy undo: record d2_name_consolidation, Undo.
Conditions: `make ci-fast` and `make test-integration` green on the committed tree (plan step 13);
at the apply, prod's dry-run plan diffed against the clone's (method step 10: prod has 43 episodes
the snapshot lacks).

Verdict read 2026-11-02 by a fresh agent (reviewer, Sonnet; not the builder, not a step-5 reader), inputs as file paths (brief eval-data `d2out/scripts/step8_prompt.txt`), about 10 minutes for all four, ran nothing over the corpus. Its block, condensed:

```text
REVIEW 0039_adfree_segment_offsets_realigned 2026-11-02 commit b64e08e04 reviewer Sonnet (fresh) minutes 2
PREDICTION vs GATE: 7 rows compared, 7 equal, 0 differ
REQUIRED ITEMS: refusals 0; code clean b64e08e04; undo recorded (1 restored, byte-identical); second apply recorded (0); census vs gate suffixes equal (.segments.json 1); not-covered present, thin; ci-fast log missing
AFTER-STATE READ: the record's reads (258 of 258 rows slice); the 4b excerpt rows not shown as excerpts, the all-rows verify accepted
OPEN CALLS: none blocking
DOUBTED NUMBER: none
VERDICT: GO WITH CONDITIONS  outcome equals the dated prediction; ci-fast and test-integration not yet run on the committed tree
```

Conditions, each answered:

1. ci-fast and test-integration on the committed tree: plan step 13, next. **Open.**

Verdict: GO WITH CONDITIONS

- **Condition 1:** `make ci-fast` and `make test-integration` green on the committed tree (plan step 13).
- **Condition 2:** At the apply, prod's dry-run plan diffed against the clone's (method step 10).

Outcome: `src/podcast_scraper/upgrade/outcomes/0039_adfree_segment_offsets_realigned.json`, sha256
865d318a77cbd0de2d4be81cc913068fe15a95b619bd77f9ae91058b676ba3df (`scripts/measure/build_outcome.py`, from the gate report measured on
b64e08e04 and this verdict).

## 10. Before apply / deploy

- **Prod's plan against the clone's:** dry run "would realign 187 row(s) in 1 episode(s); 0 refused";
  the outcome hook's plan diff: 0 episodes.
- **Run steps, as run on prod (2026-11-02, each on the operator's go typed in prod's session):**
  1. deploy-all-prod on `sha-8906696` (run 41489217128, both live smokes).
     - The operator waived Docker Build & Test and CodeQL Advanced, which did not run on 890669690:
       their path filters skip a test-only commit.
     - `deploy.py verify`: VERIFY PASS, 10 of 10 containers.
  2. The ledger rebuilt with D2's rules, `identity-ledger --no-respell` (pipeline-llm,
     18:53–18:56Z): 644 joins (the clone 631), 40 blocked (the clone 40). Its 5 canonical changes
     are the clone's.
  3. 0039, 0040, 0041 and 0042, one corpus-migration job each, in order.
     - One pre-apply snapshot, before 0039; the other three ran with `--no-snapshot`.
     - The ledger refresh after each run deferred the respelling until 0041 was applied
       (`identity/refresh_guard`).
  4. `enrich`, then the incremental `index` (not `index-two-tier`, which drops the index), both in
     pipeline-llm.
  5. The bounded verify of 0039–0042, the near-spelling report, and spot checks through the API.

## 11. After apply / deploy

Prod, 2026-11-02, each step on the operator's go typed in prod's session.

- **Apply** 19:05:13–19:13:51Z, exit 0: "realigned 187 row(s) in 1 episode(s); 0 refused", as on
  the clone.
  - The snapshot `.corpus-upgrade-backup-2.7.32-20261102T190554Z` was taken (removed 2026-11-03 on
    the operator's go).
  - Now at 2.7.33.
- **Verify** `--only 0039`: OK. Ownership: PASS.
- The ledger refresh after it: 644 joins, respell deferred (0041 pending).
- **Derived surfaces:**
  - `enrich` 21:12:54–21:29Z: exit 0, 9 enrichers, insight_density 481 of 481; the refresh after it
    respelled 0;
  - incremental `index` 21:30:39–22:44:27Z: exit 0;
  - the 2026-11-03 nightly ran after it, exit 0.
- **Bounded verify** (`upgrade verify --only 0039,0040,0041,0042`, capped container, after the
  nightly): all four OK.
- **Near-spelling report:** 15 lines, 11 pairs, each read as one person in prod's transcripts. The
  0 expected was the clone's: prod's ledger has joins the clone did not. The verdicts went into the
  reviewed list for the next deploy.
- **Spot checks** (the API, not the player UI):
  - a respelled episode's search hits and speaker labels read right;
  - an empty `person_profile` (`person:robert-caro`) is empty on the D1 snapshot too: a separate
    bug, not D2.
- **Rollback:** this migration's receipts and backups under `.podcast_scraper/upgrade-backups/`. The
  pre-apply snapshot and the rollback image holds were removed on the operator's go, 2026-11-03.
- **Operator accepted:** 2026-11-03.

Status `landed` (2026-11-03): applied on prod, so its outcome is no longer compared with later code.

## Not covered

- Raw-transcript (`.txt`) segment rows: they carry no offsets of their own
  (`format_diarized_screenplay_with_offsets` renders them), so they are not checked here.
