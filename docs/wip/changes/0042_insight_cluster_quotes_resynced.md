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

## Not covered

- Cluster membership built from an older corpus; a rebuild decides that, not this.
