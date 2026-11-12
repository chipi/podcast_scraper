---
id: 0040_gi_quote_offsets_realigned
class: full
status: landed
issue: none (the person-identity design D2; found by 0041's after-checks; operator decision 4A, 2026-11-01)
touches:
  - src/podcast_scraper/upgrade/migrations/m0040_gi_quote_offsets_realigned.py
  - src/podcast_scraper/upgrade/registry.py
  - tests/unit/podcast_scraper/upgrade/test_m0040_gi_quote_offsets_realigned.py
---

# 0040 — a GI quote's offsets point at its own words, located by its time (text only)

Method: [CORPUS_CHANGE_METHOD.md](../../guides/CORPUS_CHANGE_METHOD.md) §4. Names are kept out of this
record; the list is in eval-data `data/eval/name-consolidation/d2i9/`.

## 1. Intake

- **Found by 0041's after-checks** (`d2i9/quote_spans.py`): 24 GI quotes in 11 served episodes do not
  slice to their own text in the transcript they reference. The count is the same before and after
  0041, so 0041 does not cause it. The operator chose to fix it in D2 (decision 4A, 2026-11-01).
- **What they are** (`d2i9/quote_drift.py`, `quote_ref_check.py`, `quote_by_time.py`):
  - all 24 reference the episode's `.adfree.txt`;
  - none slices at the same offsets in the raw or the cleaned transcript either;
  - the stored spans point at unrelated words, thousands of characters away;
  - every quote's text is short and occurs 0 to 9 times in the transcript, so the text alone cannot
    place any of them;
  - every quote carries `timestamp_start_ms` / `timestamp_end_ms`.
- **Readers:** the player's quote highlight, search's quote rows (`indexer`, offsets), 0042's
  cluster copy, and m0023's offset carry.

## 2. Baseline

24 quotes in 11 episodes do not slice. By time (`d2i9/quote_by_time.py`): **18** have their text
exactly once inside the ad-free segments whose time span overlaps the quote's time window, **2**
have it twice there, and **4** have none. Two of the four quote texts contain a speaker label
("Speaker 3:"), so the quote text itself is malformed.

## 3. Diagnosis

The offsets were left pointing into an older ad-free text after a cut changed it (most likely the
ad-cut rebuild of 0030 or the hand repair that 0039 fixes). Not established per quote; the fix
does not depend on it.

## 4. Prediction

Dated: 2026-11-01

The change, migration `0040_gi_quote_offsets_realigned` (after 0039, before 0041):
- For each GI quote that does not slice to its text in its referenced transcript, collect the
  transcript's segments (that transcript's own `.segments.json`) whose time span overlaps the
  quote's `[timestamp_start_ms, timestamp_end_ms]`, widened by 1 second.
- Find every occurrence of the quote's text that starts inside one of those segments' character
  spans. **Exactly one:** the quote takes that span. **None or several:** the quote is left as it is
  and counted.
- Only `char_start` and `char_end` change. Text, times, speaker, ids, edges and every other quote
  are unchanged, and no transcript or segments file is written.
- A backup and a receipt, an atomic write, the corpus owner's ownership, undo, and verify through
  `upgrade/file_rewrite`. **Verify** passes when no quote that time can place is left. The quotes
  time cannot place are listed in the plan, not failed: there is no evidence to fix them with.

Predicted on the clone (snapshot-prod-20261030 with 0038 and 0039):
- **18 quotes realigned** in the episodes of section 2;
- **6 left and listed** (2 ambiguous, 4 not found);
- after: those 18 slice to their text; 0 other quotes change;
- a second run writes nothing.

### 4b. Rows added after the fresh read

- **The time base matches:** a quote's times come from the segments it overlaps
  (`gi/pipeline._timestamps_for_quote`), and the ad-free build drops segments and re-offsets
  characters without shifting times (`adfree_transcript`). So the quote's window and the ad-free
  segments share one timeline.
- **The segments file:** for a quote referencing `<base>.adfree.txt`, `<base>.adfree.segments.json`.
  Its rows must slice to their own text (0039 guarantees it, and 0040 checks it). An episode whose
  segments do not is refused, not guessed.
- **Order:** registered after 0039 and before 0041. m0023's carry, and so 0041, refuses a quote whose
  text is not its slice (`m0023._move_quote`). After 0040 on the clone: 0 such refusals in 0041.
- **The search index:** two-tier hits carry no character offsets (`hybrid_search._to_search_result`).
  The offsets the transcript lift reads come from the index's metadata sidecar, which every reindex
  rewrites from the GI, skipped episodes included (`two_tier_indexer._fingerprint_skip_unchanged`).
  D2's run steps end with a reindex, so no index row is edited.
- **0042's cluster copy** takes the realigned offsets (0042 runs after 0040).

## 5. Acceptance

Fresh reader, 2026-11-01, a sonnet reviewer subagent with read-only tools, one pass over this record and two others, before any code:
Dated: 2026-11-01T12:35
"Not complete". Every row is taken in 4b.

## 7. Measure

Gate (`0040_gi_quote_offsets_realigned.gate.json`, the final batch, code digest `e0bedfef5a62`, 2026-11-01; each gate on its own base: snapshot-prod-20261030 with 0038, the D2 ledger built with the final rules (`--no-respell`), and the migrations before it applied):
- computed class full; **8 files changed**;
- gating moves: `S12_quotes_off_text` **-18**, the prediction exactly;
- `upgrade verify` 38 ok; failing 0025 alone, no regression;
- after 0041 (`d2i9/quote_spans.py`): the 6 quotes time cannot place, unchanged in 4 episodes.

### Final rerun on D2's final ledger (s5d through 4e, s6g, s6h; 2026-11-02)

Code digest `bc3c9bb670fd` (the shared digest moved with s6h's `identity/ledger` and `identity/display`). Chain eval-data `d2out/scripts/chain5.sh`, each gate on its own base: snapshot-prod-20261030 with 0038, the ledger rebuilt `--no-respell` with the final rules, the migrations before it applied. The gate report is from this run. Record s6h section 7 has the whole chain.
- 18 quotes realigned in 8 episodes, 6 left: unchanged from the final batch.
- `upgrade verify` ok, no regressions, no gating moves.

## 8. Verdict

**Builder's verdict, before the review (2026-11-02): GO WITH CONDITIONS.** 18 quotes realigned in 8 episodes, 6 left and listed, as predicted; verify ok; second apply 0; undo byte-identical (8 files). Gate report on committed code `b64e08e04` (clean, code digest `bc3c9bb670fd`, worktree run eval-data `d2out/scripts/chain7.sh`), equal in outcome to the final rerun in section 7. Second apply and full-copy undo: record d2_name_consolidation, Undo.
Conditions: `make ci-fast` and `make test-integration` green on the committed tree (plan step 13);
at the apply, prod's dry-run plan diffed against the clone's (method step 10: prod has 43 episodes
the snapshot lacks).

Verdict read 2026-11-02 by a fresh agent (reviewer, Sonnet; not the builder, not a step-5 reader), inputs as file paths (brief eval-data `d2out/scripts/step8_prompt.txt`), about 10 minutes for all four, ran nothing over the corpus. Its block, condensed:

```text
REVIEW 0040_gi_quote_offsets_realigned 2026-11-02 commit b64e08e04 reviewer Sonnet (fresh) minutes 2
PREDICTION vs GATE: 6 rows compared, 6 equal, 0 differ
REQUIRED ITEMS: 6 quotes left, listed in the plan; code clean b64e08e04; undo recorded (8 restored, byte-identical); second apply recorded (0); census vs gate suffixes equal (.gi.json 8); not-covered present, thin; ci-fast log missing
AFTER-STATE READ: the 2 quote spans off their text are among the 6 left, byte-identical before and after 0041; no per-quote excerpts for the 18
OPEN CALLS: the 6 left stay (no evidence on disk to place them); accepted
DOUBTED NUMBER: none
VERDICT: GO WITH CONDITIONS  every gate number equals the prediction; ci-fast and test-integration not yet run on the committed tree
```

Conditions, each answered:

1. ci-fast and test-integration on the committed tree: plan step 13, next. **Open.**

Verdict: GO WITH CONDITIONS

- **Condition 1:** `make ci-fast` and `make test-integration` green on the committed tree (plan step 13).
- **Condition 2:** At the apply, prod's dry-run plan diffed against the clone's (method step 10).

Outcome: `src/podcast_scraper/upgrade/outcomes/0040_gi_quote_offsets_realigned.json`, sha256
1220975cafcf0ef0ef50550a85f62bbaa62c69443a6ac9ac21e426b38676f1d6 (`scripts/measure/build_outcome.py`, from the gate report measured on
b64e08e04 and this verdict).

## 10. Before apply / deploy

- **Prod's plan against the clone's:** dry run "would realign 18 quote(s) in 8 episode(s); 6 left";
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

- **Apply** 19:28:18–19:34:49Z, exit 0, `--no-snapshot`: "realigned 18 quote(s) in 8 episode(s); 6
  left", as on the clone. Now at 2.7.34.
- **Verify** `--only 0040`: OK (6 left, listed). Ownership: PASS.
- The ledger refresh after it: respell deferred.
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

- The 6 quotes time cannot place: listed for the operator; no fix without evidence.

## 6. Build

Tests (`tests/unit/podcast_scraper/upgrade/test_m0040_gi_quote_offsets_realigned.py`):
- a quote takes the occurrence inside its own time, from a repeated phrase;
- a quote that sliced is unchanged;
- one time cannot place is left and listed;
- segments that do not slice refuse the episode;
- undo; a second run writes nothing.

Code: `upgrade/migrations/m0040_gi_quote_offsets_realigned.py`, registered after 0039, `to_version`
2.7.34 (0041 moved to 2.7.35, 0042 to 2.7.36).

Mutants, all killed: the time window ignored; several occurrences accepted; unsliced segments not
refused.

Dry run on the clone (snapshot-prod-20261030, 0038 and 0039, the D2 ledger): **18 quotes realigned in
8 episodes, 6 left, 0 refused**, as predicted.

