---
id: d2_name_consolidation
class: full
status: landed
issue: none (the person-identity design; plan in podcast-scraper-eval-data docs/wip/NAME-CONSOLIDATION-ARC-V3-2026-10-30.md)
touches:
  - src/podcast_scraper/identity/*.py
  - src/podcast_scraper/kg/entity_clusters.py
  - src/podcast_scraper/upgrade/migrations/m0039_adfree_segment_offsets_realigned.py
  - src/podcast_scraper/upgrade/migrations/m0040_gi_quote_offsets_realigned.py
  - src/podcast_scraper/upgrade/migrations/m0041_joined_names_respelled.py
  - src/podcast_scraper/upgrade/migrations/m0042_insight_cluster_quotes_resynced.py
---

# d2_name_consolidation — the name-consolidation arc's second deploy: one spelling per person, and the joins that rest on evidence

Method: [CORPUS_CHANGE_METHOD.md](../../guides/CORPUS_CHANGE_METHOD.md) §5, the report template.
This record holds D2's outcome table: D1's after-state against D2's final state, on one snapshot.
Each part has its own record, which this one cites:

- ledger rules: s5a–s5d and s5o;
- display rules: s6a–s6h;
- migrations: 0039–0042;
- the bounded verify: d2_verify_bounded.

Person names are kept out of this public record. The pairs and names are in eval-data
`data/eval/name-consolidation/`.

**States compared,** both on snapshot-prod-20261030 (3,069 episodes):

- **before:** D1's after-state. 0038 applied, D1's ledger (eval-data `d1/scoreboard_after_branch.*`,
  `d1/gold_after_branch.json`).
- **after:** D2 final. The ledger rebuilt with D2's rules (`--no-respell`), then 0039, 0040, 0041
  and 0042 applied in order, prod's `search/insight_clusters.json` copied in (eval-data
  `d2out/*_s6h.*`, corpus `/Users/claude/snapshots/d2-final`).
- **Code:** branch `feat/name-consolidation` at `1f1bdd84e`, code digest `bc3c9bb670fd` (all four
  gates).

## Outcome table

```text
OUTCOME (snapshot-prod-20261030, clone)          before → after        predicted          pass?
OWN  joins, all (ledger id map)                     949 → 631            631 (s5d 4e)        yes
OWN  person joins: full-name + scoped           720 → 349 + 173      349 + 173 (s5d 4e)  yes
OWN  org joins                                     229 → 109            109 (s5o part 2)    yes
OWN  blocked joins                                   9 → 40             40 (s5d 4e)         yes
OWN  unreviewed near-spelling joins                  — → 0              0 (s5d 4e)          yes
OWN  person groups by display rule (publisher / frequency / tie / publisher_tie / plurality)
                                                       → 158/104/75/4/8   the same (s6h 4b)   yes
OWN  0039 segment rows realigned                     — → 187 in 1 ep    the same (0039)     yes
OWN  0040 GI quotes realigned / left                 — → 18 in 8 / 6    the same (0040)     yes
OWN  0041 episodes / files respelled                 — → 467 / 2,289    467 / 2,289 (s6h)   yes
OWN  0041 names refused / folds                      — → 179 / 1        179 / 1 (s6h 4b)    yes
OWN  0042 cluster quotes resynced (of 33,162)        — → 1,835          not predicted       measured, see below
S1n named but hidden                                 7 → 7              7                   yes*
S1  hidden by our defect                        17,860 → 17,860         17,860              yes*
S1c consistency (must be 0)                          0 → 0              0                   yes*
S12 quotes off their text                           24 → 6              6 (0040: 6 left)    yes
S7  dev_v1 correct / wrong / unnamed     2,452/380/1,313 → 2,440/392/1,313   wrong +12 (s6h 4c)  accepted loss
S7  val_v1 correct / wrong / unnamed  11,496/2,128/4,254 → 11,496/2,128/4,256 unchanged       +2 unnamed, see below
S7  val_v2 correct / wrong / unnamed   3,707/439/1,109 → 3,707/439/1,109    unchanged          yes
S7  strict (fold_exact) dev_v1 / val_v1 / val_v2
                               2,248/10,513/3,338 → 2,251/10,684/3,355    not predicted as a total   up on every set
S6  gold regressions                              — → 12 (dev_v1)       12 (s6h 4c)         accepted loss
S2  all hidden                                  31,474 → 31,474         31,474              yes*
S3  principal unidentified                       9,219 → 9,219          9,219               yes*
S9  unattributed talk share                    not measured: corpus_scoreboard has no S9 row
S10a / S10b zero visible (attributed / panel)  194 / 2 → 194 / 2        the same            yes*
S11 non-person names on shown insights         not measured: no row in corpus_scoreboard
S4  named quotes share                         not measured: no row in corpus_scoreboard
S5  episodes with no named voice               not measured: prod /metrics only
S8  undecidable dev_v1 / val_v1 / val_v2       77/926/651 → 77/930/651   unchanged          +4 val_v1, see below
A1  sponsor-marker hits / episodes           396 / 297 → 396 / 297     the same            yes*
```

\* No record predicts these scoreboard rows against D1 as integers. Each D2 gate measured "gating
moves: none" (0039–0042 gate reports), and record 0041 states that the scoreboard's gating numbers
do not move. A row marked `*` meets that, not an integer prediction. Only S12's 6 has a record's
integer (0040: 6 left).

**The rows that need a sentence:**

- **S7 dev_v1, +12 wrong:** all 12 are `dara-obrey`, episode `urn:bbc:podcast:p0pd1vwf`. The group
  is all ASR spellings of the feed's stated co-host. The operator accepted the loss on 2026-11-02:
  "Accept for now and take a note" (record s6h, Not covered).
- **val_v1 S7_unnamed +2 and S8 +4:** 6 quotes were unplaceable in D1 and are placed now
  (`quote_not_placeable` 30 → 24), because 0040 realigned them.
  - The 6 are: `insight:56b3722ef8dd6623`, `68076f05df37ad56`, `a6c36d43422b1c79` and
    `db005f2ae300ba1a` (now S8), and `68a3f02648983321` and `f218096111413b90` (now S7_unnamed).
  - Source: eval-data `d2out/gold_moves_by_insight.out`.
  - The moves are not regressions: each insight was unscored before and is now scored as unnamed
    or undecidable.
- **The strict tier rises on every set** (+3, +171, +17), because spellings now match the gold
  label exactly. No set-wide total was predicted, but each part was: s6h's 50 back, and the
  respelling records' moves.
- **0042, 1,835 resynced, not predicted:** the run before s6h gave 1,885. The difference is
  entirely in offsets (`char_start` 1,810 → 1,760). It fits fewer respellings, but it is not
  traced per quote (record 0042, the final rerun).
- **The gold-move script undercounts.** `d2out/gold_moves.py` keys rows by (set, insight, voice), so
  it misses a row whose voice changed, here the 6 above. `gold_moves_by_insight.py` keys by
  (set, insight). 360 insights differ in some row: dev_v1 40, val_v1 258, val_v2 62. Besides the
  12 and the 6, every other difference is a stored spelling inside the same bucket.

## Population

- **Reached:**
  - 0039: 1 episode;
  - 0040: 8 episodes;
  - 0041: 467 episodes, 2,289 files;
  - 0042: 1 file, 1,835 of 33,162 cluster quotes;
  - the ledger: every served episode's Person ids.
- **Not reached:** the other served episodes. 3,069 − 467 = 2,602 are not respelled.
- **Clone vs prod plan diff: not done yet.** Prod had 3,112 episodes on 2026-11-02, 43 more than
  the snapshot. The diff is taken at the apply: each migration's dry run on prod against the
  clone's plan, per the method's step 10.

## Commands

| What | Command (eval-data `d2out/scripts/`, measurement scripts in `d2out/`) | Output |
| --- | --- | --- |
| ledger rebuild, group diff | `chain5.sh` step ledger; `d2out/s6h_group_diff.py` | "predicted 12, changed as predicted: 12", "other changes: 0" |
| gates 0039–0042 | `chain5.sh` → `gate_run.sh` → `scripts/measure/corpus_change_gate.py` | four EXIT=0; verify 40 and 41 of 41 ok |
| after-checks 0041 | `d2i9/after_checks.py base-0041 base-0042` | 0 must-not-move breaches (verdict line "FAIL 0": the 2 quote spans among 0040's 6) |
| gold | `scripts/measure/gold_insights.py --corpus d2-final … --set dev_v1 --set val_v1 --set val_v2` | `d2out/gold_after_s6h.txt` |
| scoreboard | `scripts/measure/corpus_scoreboard.py --corpus d2-final` | `d2out/scoreboard_after_s6h.txt` |
| outcome numbers | `d2out/d2_outcome_numbers.py` | `d2out/d2_outcome_numbers.out` |
| near-spelling report | `scripts/measure/near_spelling_joins.py --corpus clone-pre0038a` | "0 pairs, 0 members" |
| second apply, undo | `chain6.sh` → `d2out/idem_undo.py` | `d2out/idem_undo.out`, four EXIT=0 |

## Undo

Second apply and full-copy undo, each on a copy of its own after-state (`d2out/idem_undo.out`):

| | second apply | undo | files differing from the before-state after undo |
| --- | --- | --- | --- |
| 0042 | resynced 0; receipted files changed 0 | restored 1, refused 0 | 0 |
| 0041 | rewrote 0 episodes; changed 0 | restored 2,289, refused 0 | 0 |
| 0040 | realigned 0 quotes; changed 0 | restored 8, refused 0 | 0 |
| 0039 | realigned 0 rows; changed 0 | restored 1, refused 0 | 0 |

- `verify` after each undo reports the migration as still to apply, which is the before-state.
- **The ledger's undo:** `restore_previous`, tested in D1 (record d1, section 7).
  `identity_ledger.previous.json` is kept by every write.
- **The before-state copy is not pristine:** `clone-pre0038a` holds a receipts file from an apply of
  0039 under its earlier id `0038a` (2026-11-01 10:48). The one file it names is byte-identical to
  the prod snapshot (sha256 `81b809f2…`, checked), so the undo comparison holds. Prod has no such
  file.

## Derived surfaces

- **Enrichment outputs** for the respelled episodes: the run step `enrich` (step 14).
- **Search index rows:** 84 episodes with changed chunks (1,273 of 2,136, 0041 after-checks), plus
  the names in metadata rows. The run step `reindex` (step 14).
- **Bridge:** rewritten by 0041 in the same pass (`bridge_names_respelled` 424 and others, in its
  gate message). No separate step.
- **Insight clusters:** 0042 resyncs prod's `search/insight_clusters.json` from the GI quotes.
- **The served ledger:** rebuilt first on prod (`identity-ledger --no-respell`), before 0041 reads
  it.

## Paths exercised

- 0041: in-place episodes 92 (5,764 lines), names refused 179, folds 1, bridge identities merged 1.
- 0042: speaker id changed 338, text changed 5, keys in several runs 13.
- Display rules: all five (publisher, frequency, tie, publisher_tie, plurality), each with groups
  on this snapshot.

## Read

The step-8 fresh reviewer read 0039–0042 (2026-11-02): GO WITH CONDITIONS on each; the conditions and their answers are in each record's section 8.

## Not covered / not verified

- **The prod plan diff** (Population): not done until the apply. 43 episodes ingested since the
  snapshot are not measured here.
- **S9, S11, S4 and S5 are not measured:** the scoreboard instrument has no rows for them.
  The method's template lists them, and D1's table omitted them too.
- **The accepted loss,** `dara-obrey`'s 12 dev_v1 insights. Two more ASR-spelling groups of the
  same shape stay as they are: `luisa-lemi` (5 val_v1 insights leave `fold_exact`) and
  `john-glenn-hill`'s other episodes. Both need the deferred feed-host rule.
- **Strict-tier losses D1 → D2 not addressed** (record s6h, Not covered): `sarah-gonzalez`-shaped
  (16), `erika-barris` (8), `david-sanger` (8), one other (1).
- **0042's 50 fewer offset changes:** not traced per quote.
- **The 2 bare speaker labels** kept by 0041 (record 0041, the final rerun): read right; the earlier
  count of 4 is not reconciled by id.
- **`known_hosts` is mixed evidence** (record s6h 4b): correct self-introductions, ASR spellings
  and parse artifacts. Joins, the display rule and m0041's and m0025's stated-name reads take it
  as stated. Not fixed here.
- **Bare names in a show are no longer pooled without voice evidence** (record s5c). That stays
  until the voice-fingerprint validation; it is raised then.
- **`david-krakauer`'s card loses a borrowed bio** (record s6h).
- **The 12 undecided groups and s6g's 75 ties** show differing spellings across episodes, as prod
  does today. One name per person is not reached for them.
- **The gold-move script's blind spot** (rows whose voice changed) affects the D1 record's moves
  too. I have not rechecked D1's with the corrected script.
- **The integration suite and ci-fast** have not run on `1f1bdd84e` (plan step 13).
- **Organisation joins** (s5o): measured by count (229 → 109) in this table. The org reads are in
  record s5o. No gold set covers organisations.
- **Read-time surfaces on the player** (cards, chips, the people page) are not exercised in a
  browser for D2.

## On prod (2026-11-02 and 11-03)

The run steps and each migration's prod numbers are in records 0039–0042, steps 10 and 11. In short:

- **Deploy:** `sha-8906696` (890669690).
- **Ledger:** 644 joins (the clone 631), 40 blocked (the clone 40).
- **0039 and 0040:** equal to the clone, plan diff 0.
- **0041:** 479 episodes (the clone 467); its 14 plan-diff entries were each given a verified cause.
- **0042:** 1,838 quotes (the clone 1,835); its 1 plan-diff entry was given a cause.
- **Outcome hook:** every verdict PASS, with empty deltas.
- **After the applies:** enrich and the incremental index; the bounded verify, all OK.
- **The near-spelling report:** 11 pairs, all one person, added to the reviewed list (2026-11-03).
- **Operator accepted** 2026-11-03.
- **Rollback:** each migration's receipts and backups. The pre-apply snapshot and the image holds
  were removed on the operator's go.

Not covered on prod:

- the player UI (spot checks went through the API);
- the gold sets and the scoreboard (measured on the clone only);
- which rule step joins `person:robert-carrow` on prod.

## Corrections sent

- To the operator, 2026-11-02: the after-checks verdict line "AFTER CHECKS: FAIL 0" had been
  reported as a pass in record s6g and in session reports. Corrected in records s6g and 0041.
- Record 0041's "Not covered" heading, deleted by commit `00f7534c5`, is restored.
