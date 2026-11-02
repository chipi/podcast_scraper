---
id: 0042_insight_cluster_quotes_resynced
class: full
status: open
issue: none (RFC-131 plan V3, D2 item 5; and 0041's derived surfaces)
touches:
  - src/podcast_scraper/upgrade/migrations/m0042_insight_cluster_quotes_resynced.py
  - src/podcast_scraper/upgrade/registry.py
  - tests/unit/podcast_scraper/upgrade/test_m0042_insight_cluster_quotes_resynced.py
---

# 0042 — each insight cluster's quotes match their GI quotes again (text only)

Method: [CORPUS_CHANGE_METHOD.md](../../guides/CORPUS_CHANGE_METHOD.md) §4. Plan V3, D2 item 5, and a
derived surface of 0041. Names are kept out of this record.

## 1. Intake

- **D2 item 5** (prod's D1 step-5 count, 2026-10-31): one quote entry in prod's
  `search/insight_clusters.json` keeps the raw-label `speaker_id` of a Person that 0038 removed.
- **0041** carries every GI quote's `char_start` / `char_end` when a speaker prefix changes length
  (m0023's offset carry). `insight_clusters.json` copies each supporting quote's `quote_id`, `text`,
  `speaker_id`, `char_start` and `char_end` from the GI at build time
  (`search/insight_clusters.collect_insight_rows`), so 0041 leaves its offsets stale.
- **Readers:** the operator CLI (`gi clusters`, `search/insight_cluster_context`) prints each quote's
  text and `speaker_id`; explore and briefing use the cluster context. No listener surface reads the
  file.
- **Rebuilding it** (`insight-clusters`) re-embeds every insight with a sentence-transformer and
  re-clusters, which can change membership. The prod rule is text first: the copied fields are
  resynced from the GI on disk, the ladder's step 2, and membership is not touched.
- The snapshot has no `search/` directory. The measurement needs prod's file (requested from prod's
  session, read-only, 2026-10-31).

## 2. Baseline

Prod's file (`search/insight_clusters.json`, copied read-only on the operator's go, 2026-11-01: 24,555,352
bytes, sha256 `797679ed…ffb96`, built 2026-10-19) over the clone (snapshot-prod-20261030, 0038
applied, the D2 ledger), before 0041, 0042's own dry run:
- 33,162 supporting quotes; 0 without a GI match; 13 keys in more than one run;
- **1,636 already differ from their GI quote:** `char_start` 1,563, `char_end` 1,561, `speaker_id`
  336, `text` 5.
- **A correction to section 1 and the prediction:** they described the stale entries as item 5's one
  `speaker_id` plus 0041's offset shifts. The file was built on 2026-10-19, and the naming migrations
  since (0029 to 0038) rewrote GI quotes: their offsets, and their speaker ids where a rename moved
  an id. So the file has drifted all along; item 5's entry is the one that was counted.

## 3. Diagnosis

The file is a copy of GI fields with no resync. A migration that rewrites GI quotes (0038 removed a
speaker id; 0041 moves offsets) leaves it stale.

## 4. Prediction

Dated: 2026-10-31

The change, migration `0042_insight_cluster_quotes_resynced`:
- for each cluster member's supporting quote, by `(episode_id, quote_id)`, the copied fields
  (`text`, `speaker_id`, `char_start`, `char_end`) take the GI quote's current values;
- a quote whose GI quote is gone keeps its entry and is counted, never invented or dropped;
- membership, order, ids, centroids, scores and every other field are untouched;
- a backup and a receipt, an atomic write, the corpus owner's ownership, verify (nothing left to
  resync), and undo, all through `upgrade/file_rewrite`;
- no file: nothing to do, and verify passes.

Predicted, on the clone with prod's file and 0041 applied: the 1,636 above plus the quotes 0041
shifts are resynced, with the per-field counts measured at step 7; 0 unmatched quotes change; every
non-quote field is identical; a second run writes nothing.

### 4b. Rows added after the fresh read

- **The key is `(episode_id, insight_id, quote_id)`, not `(episode_id, quote_id)`.** The builder
  walks every `*.gi.json` with no newest-run selection (`insight_clusters.py`, `corpus_rglob`), and
  `episode_id` falls back to the file stem. So one episode's quote can come from several runs. The
  resync reads the GI files that contain the member's own insight and quote. Where several runs
  match, the newest served run (`select_served_artifacts`) is the tie-break. The duplicates across
  runs are counted.
- **A quote whose GI quote has no `speaker_id`** (0038's removal) takes `null`, the value a fresh
  build writes (`qprops.get("speaker_id")`); the key is kept.
- **Must not move:** `insight_id`, `insight_type`, `grounded`, `similarity_to_centroid`,
  `canonical_insight`, `member_count`, `episode_ids`, the member order, and the number and order of
  quotes per member. Insight text is not resynced: 0038 and 0041 do not change it (0038 changes
  Quote nodes only), and the record says so.
- **The payload has no person names** (no `speaker_name`), so 0041's spellings do not reach it.
- **Readers:** only `gi clusters --format json` exposes the offsets. The CLI and the cluster context
  read `text` and `speaker_id`.
- **Step 8:** three or more resynced members compared by eye with their GI quotes, item 5's entry
  among them.

## 5. Acceptance

Fresh reader, 2026-10-31, a sonnet reviewer subagent with read-only tools, before any code: "not
complete". Every row is taken in 4b.

## 6. Build

Code: `upgrade/migrations/m0042_insight_cluster_quotes_resynced.py` (renumbered from 0040; it walks the
corpus with `corpus_rglob`). Tests (`tests/unit/podcast_scraper/upgrade/test_m0042_insight_cluster_quotes_resynced.py`):
- a quote takes its GI quote's fields, and nothing else moves;
- the newest served run wins;
- no clusters file means nothing to do.

Mutants, both killed: the served-run tie-break off; an absent `speaker_id` not written as null.

## 7. Measure

Gate (`0042_insight_cluster_quotes_resynced.gate.json`, the final batch, code digest `e0bedfef5a62`, 2026-11-01; each gate on its own base: snapshot-prod-20261030 with 0038, the D2 ledger built with the final rules (`--no-respell`), and the migrations before it applied, with prod's
`search/insight_clusters.json` copied in):
- **computed class light**, against this record's `full`. A record may be stricter than its
  measurement; noted, the record stays full;
- 1 file changed (the clusters file); gating moves none;
- `upgrade verify` **41 ok, none failing**.
- Dry run over prod's file after 0041 (`dry40.py`): 33,162 quotes, **1,896 resynced** (`char_start`
  1,821, `char_end` 1,819, `speaker_id` 338, `text` 5), 0 unmatched, 13 keys in several runs.


**Rerun on D2's final ledger (s5d through 4e, s6g; 2026-11-02):** the gate report is from this run.
- **Invalid earlier reruns:** two reruns between these (the s5d ledger, then s6g's) ran without
  prod's clusters file. My chain script's copy into `search/` failed, because the folder did not
  exist on those bases, and the script did not stop on the error.
  - They reported "resynced 0 cluster quote(s)", and the first was committed (`00f7534c5`) as if
    valid.
  - Both are void. The chain now creates the folder, checks the file's sha256 before the gate, and
    stops on any error.
- **This run:** computed class light, 1 file changed, `upgrade verify` 41 ok, none failing.
  - 33,162 quotes, **1,885 resynced**: `char_start` 1,810, `char_end` 1,808, `speaker_id` 338,
    `text` 5. 0 unmatched; 13 keys in several runs.
- **Against the batch above** (1,896 resynced): 11 fewer offset changes. `speaker_id` and `text`
  are unchanged.
  - A likely cause, not traced by id: 0041 writes 501 episodes instead of 573 on this ledger, so
    fewer GI quotes move before 0042.

### Final rerun on D2's final ledger (s5d through 4e, s6g, s6h; 2026-11-02)

Code digest `bc3c9bb670fd` (the shared digest moved with s6h's `identity/ledger` and `identity/display`). Chain `scratchpad/chain5.sh`, each gate on its own base: snapshot-prod-20261030 with 0038, the ledger rebuilt `--no-respell` with the final rules, the migrations before it applied. The gate report is from this run. Record s6h section 7 has the whole chain.
- 33,162 quotes, **1,835 resynced**: `char_start` 1,760, `char_end` 1,758, `speaker_id` 338, `text`
  5. 0 unmatched; 13 keys in several runs. Computed class light, 1 file changed.
- `upgrade verify` 41 of 41 ok, none failing; no gating moves. Prod's clusters file was checked by
  sha256 before the gate (`CLUSTERS_FILE_OK`).
- **Against the run above** (1,885): 50 fewer offset changes. `speaker_id` and `text` are
  unchanged. This fits 0041 respelling 467 episodes instead of 501, but it is not traced per
  quote.

## 8. Verdict

**Builder's verdict, before the review (2026-11-02): GO WITH CONDITIONS.** 1,835 of 33,162 cluster quotes resynced, 0 unmatched; 41 of 41 verify ok; second apply 0; undo byte-identical. The count moved from 1,885 after s6h without an integer prediction (section 7, the final rerun). Gate report on committed code `b64e08e04` (clean, code digest `bc3c9bb670fd`, worktree run `scratchpad/chain7.sh`), equal in outcome to the final rerun in section 7. Second apply and full-copy undo: record d2_name_consolidation, Undo.
Conditions: `make ci-fast` and `make test-integration` green on the committed tree (plan step 13);
at the apply, prod's dry-run plan diffed against the clone's (method step 10: prod has 43 episodes
the snapshot lacks).

Verdict read 2026-11-02 by a fresh agent (reviewer, Sonnet; not the builder, not a step-5 reader), inputs as file paths (brief `scratchpad/step8_prompt.txt`), about 10 minutes for all four, ran nothing over the corpus. Its block, condensed:

```text
REVIEW 0042_insight_cluster_quotes_resynced 2026-11-02 commit b64e08e04 reviewer Sonnet (fresh) minutes 2
PREDICTION vs GATE: 5 rows compared, 4 equal, 1 without an integer prediction (resynced 1,835; 1,896 -> 1,885 -> 1,835 across reruns, the cause not traced by id)
REQUIRED ITEMS: 0 unmatched; code clean b64e08e04; undo recorded (1 restored, byte-identical); second apply recorded (0); census vs gate suffixes equal (1 file); not-covered present, thin; ci-fast log missing
AFTER-STATE READ: the 4b by-eye check of 3 or more resynced members not recorded; verify passes
OPEN CALLS: accept 1,835 as measured, not predicted (operator told); the by-eye read
DOUBTED NUMBER: 1,835, only as untraced (a gate row comparing 0042's offset changes with 0041's moved GI quotes would settle it)
VERDICT: GO WITH CONDITIONS  invariants hold (0 unmatched, 1 file, 41 of 41 verify, no-op second apply); 1,835 had no integer prediction and the by-eye read is not recorded
```

Conditions, each answered:

1. ci-fast and test-integration on the committed tree: plan step 13, next. **Open.**
2. The by-eye read of 3 or more resynced members (4b): **done** after the review (eval-data
   `d2out/s42_read.py` → `.out`, base-0042 against d2-final). Five entries, each equal to its GI
   quote on text, `speaker_id`, `char_start` and `char_end`, and each after-span slicing to the
   quote's text:
   - item 5's own entry (`https://iono.fm/e/1726016`, `quote:18dce0dab1c49e18`): `speaker_id` was
     the raw label `person:speaker-…-09` and is now null, as in its GI quote;
   - an unresolved id set to null;
   - a text change (a "Name: " prefix dropped);
   - two offset-only changes in two episodes.
   - The script counts 1,832 changed by key (episode, insight, quote) against the gate's 1,835;
     the 13 keys in several runs collapse duplicates.
3. 1,835 with no integer prediction: measured, not predicted. The 50 fewer offset changes than the
   run before s6h are not traced by id. **Operator's call**: accept as measured, or trace it.

Verdict: GO WITH CONDITIONS

- **Condition 1:** `make ci-fast` and `make test-integration` green on the committed tree (plan step 13).
- **Condition 2:** The operator's call on 1,835 resynced: accepted as measured, or traced.
- **Condition 3:** At the apply, prod's dry-run plan diffed against the clone's (method step 10).

Outcome: `src/podcast_scraper/upgrade/outcomes/0042_insight_cluster_quotes_resynced.json`, sha256
42e6b3c73efe314adaa9a98a934c5241a966ce16cd73b0394c018b998b8705c8 (`scripts/measure/build_outcome.py`, from the gate report measured on
b64e08e04 and this verdict).

**Condition 2, answered after the verdict:** the operator accepts 1,835 as measured, not traced
(2026-11-02: "I take your recommendations").

## Not covered

- Cluster membership built from an older corpus; a rebuild decides that, not this.
