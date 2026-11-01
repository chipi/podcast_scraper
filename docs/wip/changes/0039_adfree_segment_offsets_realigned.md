---
id: 0039_adfree_segment_offsets_realigned
class: full
status: open
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

## 7. Measure

Gate (`0039_adfree_segment_offsets_realigned.gate.json`, the final batch, code digest `e0bedfef5a62`, 2026-11-01; each gate on its own base: snapshot-prod-20261030 with 0038, the D2 ledger built with the final rules (`--no-respell`), and the migrations before it applied):
- computed class full; gating moves none; **1 file changed**;
- `upgrade verify` 37 ok; failing 0025 alone, no regression (the one ledger-joined pair, failing on
  the base too, which 0041 folds);
- the real episode's check (`d2i9/check_0038a.py`): 258 rows of 258 slice; every other key, the
  ad-free text and the ad map byte-identical; a second run writes nothing.
- **Drift measured in full:** 1 to 8 characters (15 rows at 8), inside `MAX_DRIFT` 16.

## Not covered

- Raw-transcript (`.txt`) segment rows: they carry no offsets of their own
  (`format_diarized_screenplay_with_offsets` renders them), so they are not checked here.
