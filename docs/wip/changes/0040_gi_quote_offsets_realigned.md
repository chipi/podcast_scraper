---
id: 0040_gi_quote_offsets_realigned
class: full
status: open
issue: none (RFC-131 D2; found by 0041's after-checks; operator decision 4A, 2026-11-01)
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

