---
id: 0036_bridge_identities_restored
class: full
status: landed
issue: none (the operator opens issues; found by 0034's derived-surface census, 2026-10-26)
touches:
  - src/podcast_scraper/upgrade/migrations/m0036_bridge_identities_restored.py
  - src/podcast_scraper/upgrade/registry.py
---

# 0036 — every published GI / KG person has its bridge identity

Method: [CORPUS_CHANGE_METHOD.md](../../guides/CORPUS_CHANGE_METHOD.md). Operator, 2026-10-26:
"Build extra migration for gap mentioned" (the bridge gap found while measuring 0034).

## 1. Intake

- **What the bridge is for.** `<episode>.bridge.json` joins an episode's GI and KG persons; the
  cross-layer read surfaces read it (`server/cil_queries.py`: person profile, position arc, topic
  timeline, perspectives, episodes for a person). A person GI / KG hold with no bridge identity is
  missing from those surfaces for that episode.
- **How it was found.** 0034's census (step 7) showed one alias-matched identity dropped by the
  shared rewrite (fixed in a1698c214). A corpus-wide check then found older gaps on
  snapshot-prod-20261025 that no current migration causes.
- **Reading list:** `builders/bridge_builder.py` (`build_bridge`, `_collect_identity`: the pipeline's
  own identity shape), `upgrade/migrations/m0025_one_person_one_entry.py` (`_graph_people`: the
  rewrite's bridge step), the records of 0015, 0017, 0020, 0025 (the migrations whose backups last
  held most of the missing identities), `server/cil_queries.py` (the readers).

## 2. Outcome and baseline

- **The outcome:** every GI / KG Person whose name the pipeline would publish has a bridge
  identity (by id or alias) in its episode.
- **Baseline, snapshot-prod-20261025** (`scratchpad/m0036/gaps.py`, to be committed as
  `scripts/measure/bridge_gaps.py`): **115 persons in 45 episodes** carry no bridge identity.
  Layers: KG only 55, GI and KG 42, GI only 18. By the pipeline's own rule
  (`is_publishable_speaker_name`): **103 publishable, 12 not**.

## 3. Diagnosis by loss

- **By the last migration backup that still held an identity for the person:** 0025 84 (83
  publishable, 1 not), 0015 11 (all unpublishable), none 20 (publishable; no backup ever held an
  identity: persons a later rewrite added to GI / KG without a bridge entry).
- **Unpublishable (12):** 0015 removed these names (junk labels) from the bridge, and GI / KG kept
  the node. The bridge is right there; restoring it would expose the junk name on the person
  surfaces. Not in 0036's write set; listed by key as NOT COVERED with the KG-side repair proposed.

## 4. Prediction and design

Dated: 2026-10-26T10:40

- **Design.** Deterministic, text only, no list: for each served episode, every GI / KG Person id
  that no bridge identity carries (by id or alias), whose name the pipeline would publish
  (`is_publishable_speaker_name`, with the episode's language), gets the identity the pipeline's
  own builder makes (`bridge_builder._collect_identity` over the GI node then the KG node: id,
  type, display name, aliases, sources; no fuzzy pass, which needs an embedding model). Nothing
  else in the bridge or any other file is written. Undo from `.podcast_scraper/upgrade-backups/0036/`
  through receipts. `verify`: no publishable GI / KG person lacks a bridge identity. Registered
  after 0035, `to_version` 2.7.30; runs on whatever the corpus holds then (no frozen list: the
  rule reads no judgement).
- **Prediction** on snapshot-prod-20261025 with 0034 (a1698c214) and 0035 (2080d1cb2) applied
  (`scratchpad/gate35b`, gaps counted by `scratchpad/m0036/gaps.py`):
  - identities added **102** in **34** episodes; bridge files written **34**; no other file;
  - after the apply: publishable gaps **0**; unpublishable gaps **12** in 12 episodes, untouched;
  - every scoreboard number unchanged (the scoreboard does not read the bridge): S1, S2, S3,
    S1c, S1n, S10a, S10b, S12, commercial, A1, insights, quotes all equal before and after;
  - computed class `full` (a `*.bridge.json` is a content file).
- **Not reached:** the 12 unpublishable persons (KG / GI keep junk labels 0015 meant to remove;
  the KG-side repair is a separate change, proposed in step 12); fuzzy merges the builder would
  add with an embedding model.

Dated: 2026-10-26T10:53

- **Restated before any measurement (D15), the design changed:** 15 of the 34 episodes have an
  EMPTY bridge (the file is `{}`: no `identities`, `episode_id` or `schema_version`); 74 of the 102
  gaps sit there. The snapshot's backups show 0025 wrote them empty (each held 21 to 30
  identities before 0025; 0017 and 0020 backups for two of them). An empty bridge also loses every
  organisation, topic and object identity, so those 15 episodes are missing from the topic readers
  too, not only from the person readers.
- **Design, restated:** an EMPTY bridge (no `identities` list) is replaced by the bridge the
  pipeline's builder makes from the episode's GI and KG (`build_bridge(..., fuzzy_reconcile=False)`),
  with the same publishable filter on its persons; every other bridge only gets the missing
  publishable persons APPENDED (existing identities keep their place). Persons are read GI first,
  as the builder reads them; display name `name`, else `label`; layer prefixes stripped; an
  unreadable metadata or bridge is skipped with a counted reason. Superseded runs are not read.
- **Prediction, restated:** 15 bridges rebuilt with **490** identities (74 person, 56 org, 271
  topic, 89 object), 1 unpublishable person left out; 19 bridges get **28** persons appended, 11
  unpublishable left out; in all **102** person identities in **34** episodes, **12** left out;
  publishable gaps after 0; 34 bridge files written and no other file; every scoreboard number
  unchanged; idempotent; undo byte-identical.
- **What it does not deliver** (acceptance-A finding): most cross-layer readers read `sources.gi`
  only (`_bridge_gi_ids`); the persons restored from KG alone show only in the node-episodes
  listing (`_bridge_all_ids`), not in person profiles. Counted before the run in step 7.

Dated: 2026-10-26T11:29

- **A defect in shared code, found by 0036's tests (D1):** `rewrite_bridges_m0007.apply_to_payload`
  returns the SAME dict it was given when the id map is empty; m0017's `_rewrite_bridge` (run by
  m0025's rewrite, so by 0034 and 0035) then does `bridge.clear(); bridge.update(out)`, which
  empties the bridge to `{}`. This is how 0025 left the 15 empty bridges prod serves. Repro test
  first (`test_the_shared_rewrite_never_writes_a_bridge_without_its_identities[renamed]`, red).
- **The fix:** `apply_to_payload` returns a copy, as `rewrite_ids` does (its contract: "in a copy").
- **Prediction for the re-measure** (before it runs): on the clones 0034 and 0035 wrote no new empty
  bridge (empty bridge files: snapshot 17 incl. superseded runs, after 0034 16, after 0035 16), so
  0036's prediction (10:53) stands. Also (D2, from the same tests): an EMPTY bridge whose episode lacks a readable GI or KG is skipped with a counted reason, not rebuilt from part of its inputs (acceptance-B R8); prediction: none of the 15 is such an episode, so 15 rebuilt as before.

Dated: 2026-10-26T12:34

- **Restated before the clone (D15), the base moved:** 0034's round 3 (149 removals, ffaae4ae0)
  and 0035 applied first. On that base (`scratchpad/gate35c`): **101** person identities in **34**
  episodes; **14** empty bridges rebuilt with **455** identities (70 person, 49 org, 251 topic, 85
  object), 1 unpublishable person left out there; **12** unpublishable left out in all; 34 bridge
  files written and no other file; every scoreboard number unchanged; idempotent; undo
  byte-identical.

Dated: 2026-10-26T14:18

- **After the step-8 NO-GO (section 8):** (D3) 0034's shared rewrite turned 2 served `{}` bridges into
  `{"identities": []}`, which 0036 did not treat as empty, so they stayed without `episode_id` /
  `schema_version` and most identities while verify passed; (D4) appended identities were not
  kept in id order, against acceptance criterion 3.
- **The fix:** a bridge without `episode_id` is damaged and is rebuilt like `{}` (both GI and KG
  readable, as D2); missing persons are inserted in id order when the existing identities were in
  id order (the builder's), so existing identities keep their relative order.
- **Prediction on the 12:34 base** (before any run): bridges rebuilt **16** with **523** identities;
  persons **101** in 34 episodes (81 in rebuilt bridges, 20 appended in 18); 12 left out; 34 files.
  The base moves again with 0034's round 4, so this is restated on that base before the gate.

Dated: 2026-10-26T14:46

- **Restated on the final base before the clone** (0034 round 4, a859902ab, and 0035 applied;
  counted read-only by the migration's own `find_gaps` on `scratchpad/gate35d`): persons **101** in
  **34** episodes; **16** damaged bridges rebuilt with **523** identities; 18 bridges appended to; 12
  unpublishable left out; 34 files; scoreboard unchanged. Equal to the 14:18 prediction.

## 5. Acceptance

Dated: 2026-10-26T10:44

Written by a fresh step-5 reviewer, without seeing code:
[0036_bridge_identities_restored.acceptance.md](0036_bridge_identities_restored.acceptance.md).
Integer pass bars on snapshot-prod-20261025 with 0034 and 0035 applied: 102 identities added in 34
episodes (only those 34 bridge files written), publishable gaps 0, the 12 unpublishable untouched,
each identity byte-equal to `bridge_builder._collect_identity` over the GI then the KG node, every
scoreboard number unchanged, a read of 25 added identities, the reader effect, idempotence, the
undo cycle, verify semantics, rule tests with mutation bars, the prod plan diff. Its Not covered
section lists 14 items.

Dated: 2026-10-26T10:55

A fresh reviewer's addendum for the 15 empty-bridge rebuilds, with its integer pass bars and Not covered section: [0036_bridge_identities_restored.acceptance-B.md](0036_bridge_identities_restored.acceptance-B.md).

Dated: 2026-10-26T11:31

Fresh-reviewer acceptance addendum for D1 and D2, integer bars before the code fix: [0036_bridge_identities_restored.acceptance-C.md](0036_bridge_identities_restored.acceptance-C.md).

Dated: 2026-10-26T14:19

Fresh-reviewer acceptance addendum for D3 and D4 and the step-8 record corrections, integer bars before the code fix: [0036_bridge_identities_restored.acceptance-D.md](0036_bridge_identities_restored.acceptance-D.md).

## 6. Build

- Commits (branch `feat/m0034-list`, local): 7fd0eff36 (migration, registration, first tests),
  ffaae4ae0 (D2 guard; the tester's rule / mutation table: 44 tests; the D1 fix in shared code),
  c8a8b2585 (restatement).
- Tests: `tests/unit/podcast_scraper/upgrade/test_m0036_bridge_identities_restored.py`; whole
  upgrade suite 668 passed on ffaae4ae0. Mutations (the acceptance table and acceptance-B, run in a
  copy by a fresh tester): 76 run, 74 caught, 2 argued equivalent (the `find_gaps` yield condition;
  `_display`'s strip, the builder strips again).

## 7. Measure

Clone measurement on c8a8b2585 (clean tree), base snapshot-prod-20261025 with 0034 round 3 and
0035 applied and in the ledger; report `0036_bridge_identities_restored.gate.json`:

- **Apply:** 101 person identities in 34 episodes; 14 empty bridges rebuilt with 455 identities;
  12 unpublishable left out; verify OK. Every number of the 12:34 restatement met.
- **Files:** 34, all `*.bridge.json`; no other file written; no scoreboard number moved
  (`gating_moves` empty). After: publishable gaps 0; the 12 unpublishable remain, listed by key in
  `scratchpad/m0036/gaps_after36.json` (private: names); 1 empty bridge file remains, in a
  superseded run (verify, which counts served empty bridges, passes).
- **Idempotence:** a second apply on the after-state adds 0, rebuilds 0, every file unchanged.
- **Undo cycle** (`scratchpad/m0036/undo_cycle.py`): undo restored 34, refused 0; corpus content
  equal to the base (0 of 145,031 files); apply again 34 episodes, 101 added; undo again the same;
  no other migration's backups changed. UNDO-CYCLE PASS.
- **Derived surfaces:** none built from the bridge is stored (the CIL readers read it at request
  time, `server/cil_queries.py`, cached by mtime). On prod the bridges' mtimes refresh those caches.
- **What it delivers to readers, on the final clone** (the round-2 step-8 reviewer's measurement,
  section 8b, which supersedes the round-1 figures): the CIL readers read each feed's latest run
  only, 405 of 3,504 bridges; of 0036's 34 written bridges 12 are read, 10 of the 16 rebuilt.
  `episodes_for_bridge_node_id` lists 46 of the 101 restored (episode, person) pairs after (22
  GI-sourced, 24 KG-only), 0 before. The 10 read rebuilt bridges bring back 177 topic identities
  (100 GI-sourced, 99 distinct ids); `topic_person_ids` is non-empty for 99 of 99 after, 15 of 99
  before. `person_profile` for the two persons of feed aef1e4c0 item 0002 gains 22 and 17 quotes
  (0 before); the profile delta equals an independent GI walk for 14 of 14 GI-sourced restored
  persons. The rest are correct on disk and reach readers if their run becomes the latest.
- **The 12 unpublishable persons** (left out, all KG-only): 10 are one junk KG label in two feeds,
  1 a role word, 1 a letter-spaced garble; listed by (episode, person id) in the private eval repo
  (`data/eval/migrations/0036_bridge_identities_restored.unpublishable.json`; the ids carry names).
- **Near-duplicates:** 11 of the 101 added persons have a near-identical other Person in the same
  episode (difflib ratio >= 0.78); listed in the private eval repo
  (`0036_bridge_identities_restored.near_duplicates.json`, names). 0036 adds what the builder adds,
  so these mirror GI / KG as they are; merging them is a GI / KG question, not a bridge one.
- **KG-side repair, proposed (not built, a separate change under the method):** the 12 junk Person
  nodes are KG-only (10 one label in two feeds, 1 a role word, 1 a letter-spaced garble; keys in
  `0036_bridge_identities_restored.unpublishable.json`). The repair: a migration that removes from KG
  every Person node whose name fails `is_publishable_speaker_name` (with the episode's language),
  its edges re-pointed or dropped as m0025's `_edges_moved_off` does, with receipts and undo, a
  frozen list, a fresh reviewer's verdicts, and the full step 1-12 record (KG content: class full).
  Expected reach on the final base: 12 nodes in 12 episodes; insight / scoreboard moves: 0 (KG
  persons carry no insight visibility). Until then the bridge correctly does not expose them.

- **Re-measure on 0034 round 5 + 0035** (a5a9a6909, clean tree; base `gate35e`; this record's
  `.gate.json` is this run): 101 identities restored in 34 episodes, 16 damaged bridges rebuilt with
  523 identities, 12 left out, 34 files, no gating move; plan, files and verify equal to the
  6100d6b7b run. Idempotent (a second apply adds 0, rebuilds 0, files unchanged). Undo cycle
  (`undo_cycle_r3.log`): restored 34, refused 0, 0 of 145,208 files differ, apply again 34 (101
  added), undo again the same: UNDO-CYCLE PASS.

- **Re-measure on 0034 round 6 + 0035** (eea65f01f, clean tree; base `gate35f`; this record's
  `.gate.json` is this run): plan, files (34), verify and every number equal to the previous run:
  101 identities in 34 episodes, 16 bridges rebuilt with 523 identities, 12 left out, no gating
  move. Idempotent. Undo cycle (`undo_cycle_r4.log`): restored 34, refused 0, 0 of 145,234 files
  differ, apply again 34 (101 added), undo again the same: UNDO-CYCLE PASS.

- **Re-measure after 0034's round-6 conditions** (d94e4d13a, clean tree; base `gate35g`; this
  record's `.gate.json` is this run): everything equal to the previous run, plan differences 0;
  idempotent. Undo cycle (`undo_cycle_r5.log`): UNDO-CYCLE PASS.

## 8. Verdict

Dated: 2026-10-26, by a fresh step-8 reviewer (not the builder), on commit 9e39accc5 (code unchanged since
c8a8b2585, the commit the gate report measured). Base `gate35c` (snapshot-prod-20261025 with 0034 round 3 and
0035), after `gate36`; reviewer code in the session scratchpad `r8a36/` (private: names). Every integer below
was produced by the reviewer's own scripts, none copied from the gate report. Ids only; the 25-item read is in
`r8a36/verdicts_0036.tsv` (private).

### Re-derived against the 12:34 prediction and the acceptance files

| Criterion | Prediction / bar | Reviewer's number | Result |
| --- | --- | --- | --- |
| Baseline gaps (own count, layer prefixes stripped) | 101 + 12 | 113 gaps in 45 episodes; 101 publishable (34 episodes), 12 unpublishable (12 episodes); layers KG 55, GI+KG 40, GI 18 | met |
| Written files (sha256 of all 144,996 base files vs 145,031 after) | 34 `*.bridge.json`, 0 other | 34 changed, all bridges; 0 deleted; 35 new = 34 backups + the receipts file | met |
| Person identities in 34 episodes | 101 | 101 (70 in rebuilt bridges + 31 appended) | met |
| Empty bridges rebuilt | 14, 455 identities (70/49/251/85) | 14; 455 = 70 person, 49 org, 251 topic, 85 object; 1 unpublishable person left out | met |
| Each rebuilt file equals `build_bridge(..., emitted_at=REBUILT_EMITTED_AT, fuzzy_reconcile=False)` minus unpublishable persons | 14 of 14 | 14 of 14 equal as JSON text (all fields incl. `emitted_at`); no `fuzzy_merges` key | met |
| Each appended identity equals `_collect_identity` (GI then KG) | 20 of 20 | 0 mismatches over the 18 appended bridges that held identities before; existing identities unchanged in place and in value; non-identity keys unchanged | met |
| Publishable gaps after / unpublishable | 0 / 12 untouched | 0 publishable; the 12 (episode, id) pairs equal before and after | met |
| Scoreboard (committed `corpus_scoreboard.py`, run by the reviewer on both trees) | all unchanged | 51 JSON keys compared, only the `corpus` path differs | met |
| Idempotence (copy of the after-clone) | 0 written | second apply: 0 added, 0 rebuilt, 0 files differ from the after-state | met |
| Undo cycle (own script, copy, sha256 of every file) | byte-identical | undo 1: restored 34, refused 0, 0 files differ from base; verify reads False (101 gaps, 14 empty); apply 2: 101 added, 14 rebuilt, 0 bridge files differ from apply 1; undo 2: restored 34, refused 0, 0 differ from base; other migrations' backups changed 0 | met |
| Cause of the empties | 0025 | 14 of 14 served `{}` bridges are in 0025's receipts and 0025's last `sha_after` equals the file's sha today; each held 21 to 30 identities in 0025's backup | met |
| D1 | repro red before, green after | mutation "apply_to_payload returns its input" turns `test_the_shared_rewrite_never_writes_a_bridge_without_its_identities[renamed]` red | met |
| D2 | guard tested | mutation "partial-input guard removed": 4 tests red | met |
| Tests | mutation bars, 0 survivors | 52 tests green on a clean `git archive` of 9e39accc5; `tests/unit/podcast_scraper/upgrade` + `tests/unit/upgrade`: 785 passed; 40 mutations (rule table of acceptance, acceptance-B, my own additions) run in the copy: 38 killed, 2 survived, both argued equivalent: removing `bool(name)` (the gate returned False for "", "   " in None / en / de: 6 of 6 calls), and `_display`'s strip (the builder strips again) | met |

### Misses

1. **Two served bridges are still half-built after 0036 (blocker).** On the base, 16 served bridges are
   damaged, not 14: 14 are `{}`, and 2 are `{"identities": []}` (feed `aef1e4c0`, run `ed9ab40e`, item 0002;
   feed `e739a158`, run `0d12a48d`, item 0001). Both were `{}` in the snapshot; 0034 round 3 wrote
   `{"identities": []}` over them (its receipts: `sha_before` is the sha of `{}\n`, `ca3d163bab`, `sha_after`
   `43679bcfa3`, both files). 0036 detects an empty bridge as `bridge == {}`, so it classes these two as
   non-empty and only appends persons (8 and 3). After 0036 they hold persons only, with no `episode_id`
   and no `schema_version`, and lack what the builder gives (10 org, 7 object, 40 topic identities over the two,
   counted with fuzzy off). `verify` passes. Reader consequence measured with the app's own code
   (`person_profile`, which returns early on a bridge with no `episode_id`): the two restored GI persons of item
   0002 have 22 and 17 quotes spoken in that episode in the GI, and their profile gains 0 and 0; for the other
   12 of 14 GI-sourced restored persons in the readers' set the profile delta equals the independent GI walk.
   This is the defect the 10:53 block set out to remove (an episode missing from the topic and person readers),
   left standing in 2 of 16 episodes by a detection rule that the pipeline's own shared rewrite defeats. The
   prediction (14) was met; the population (16) was wrong, and the record's "15 became 14" was not explained.
2. **Acceptance 3 not met: the appended bridges are no longer sorted by id.** 18 of 18 appended bridges were
   sorted before and 0 of 18 are after (the new identities are placed at the end; the builder sorts). The
   record's design says "appended" on purpose but never records the departure from the acceptance.
3. **The reader effect in the record is overstated.** The CIL readers read the latest run per feed only: 405
   of 3,504 bridges in the clone. Of the 34 written, 12 are read; of the 14 rebuilt, 9. Measured on the
   after-clone: `episodes_for_bridge_node_id` lists 46 of the 101 restored (episode, person) pairs (0 before),
   not 101; 22 are GI-sourced, 24 KG-only; the 9 read rebuilt bridges bring back 157 topic identities (90
   GI-sourced), not 251; `topic_person_ids` returns a speaker for 90 of those 90 GI topics after, 16 of 90
   before (the 5 rebuilt bridges outside the readers' set bring nothing to them). Section 7's last bullet should
   say so.
4. **The 25-person read is 24 confirmed, 1 not confirmed.** Seeded draw (seed 20261026): 5 KG-only, 5 GI+KG, 3
   GI-only by lot, 2 more by lot, plus the 10 longest / one-token / non-ASCII names. 24 are real people of the
   episode (host, guest, audience speaker who gives their name, or a named mention); of those, item 4 is a real
   colleague spelled wrongly by the ASR, and items 18 and 20 are real public figures named in passing. Item 21
   (KG-only, unresolved-`...72061239-e77d-45a9-936d-b4ab7ff7daf6`, run `20261008-041529` of feed `0d35b453`) is
   one token in one ASR sentence, and the KG's own description says "possibly a collaborator": not
   confirmed as a person. Acceptance 10 says one non-person fails the change. The pipeline's own builder
   writes such identities (it has no publishable gate), so I report it as a bar miss rather than a filter bug,
   for the operator to rule on. Screen of all 101: none is an organisation or topic shape; 5 are one-token
   names; 8 of the 101 have no full-name hit in the transcript (3 are GI speakers with 3, 26 and 156 quotes).
5. **Acceptance 9 and 5 artefacts are missing from the record.** (a) Near-duplicate pairs: 11 of the 101 added
   persons have another Person in the same episode whose name is within difflib ratio 0.78 (10 GI-sourced, 1 KG-only;
   in 7 of the 11 the other node is a KG-only spelling variant that the bridge already holds); the record states no count and
   stores no list. (b) The 12 unpublishable keys are in a scratchpad file, not in the record; the KG-side
   repair is "proposed in step 12", which does not exist. What the 12 are (reviewer's census): 10 are one KG
   label id in 2 feeds, 1 a role word (`person:unresolved-host-bd678cc4-...`), 1 a letter-spaced garble; all 12
   are KG-only. A repair that fits the ladder of CLAUDE.md: a deterministic KG migration removing, in the 12
   keys' episodes, Person nodes and their edges whose names fail the same gate, with the same two-layer check
   0015 used; it is a separate change.
6. **Acceptance 22, 23, 24 are not done and the record does not claim them.** No prod plan diff (step 10), no
   derived-surface census beyond "read at request time" (I checked `kg/entity_clusters.py`: the entity id map
   is built from GI/KG and cached in process; it does not read bridges), and no `make ci-fast` log attached;
   I did not run ci-fast either.

### Not verified by the reviewer

- The `make ci-fast` gate (not run; the repo-wide rules the upgrade suite does not cover).
- Prod: the 34 / 101 / 12 are the clone's; prod's 17 empty bridge files and its own 0034 / 0035 outcomes are
  unknown here, and the 2 half-built bridges depend on 0034 touching those two episodes on prod.
- Whether other served bridges lack topics, orgs or objects for reasons other than emptiness: on the after-clone
  the builder (fuzzy off) would still add 13 org identities (1 each in 13 further episodes) besides the two
  half-built bridges' 10 org, 7 object and 40 topic; fuzzy merges and aliases may explain the 13. Out of
  0036's person scope, counted not judged.
- The rebuilt bridges are not the pre-damage ones: 0025's backups held 7 to 10 fuzzy merges each (topics merged
  GI to KG); the rebuild (fuzzy off) keeps both ids, so each of the 14 rebuilt bridges holds 5 to 10 more topic
  identities than the backup, and 1 lacks a person the gate refuses. Declared in the record's Not covered; the
  effect on topic counts per episode was not judged.
- Search-index rows and the app in a browser; the viewer was not opened.
- Any LLM or relabel path: none used.

### Conditions to clear the NO-GO

1. Repro first, then make a bridge the builder would never write (no `episode_id`, or no non-empty `identities`)
   an empty bridge in 0036 (or stop the shared rewrite writing `identities: []` over `{}`); restate the
   prediction on this base (16 rebuilt, at least 57 more builder identities) and re-measure on a clone with
   the whole census, the undo cycle and the reader check (`person_profile` delta equal to the GI walk for the
   two persons of item 0002, 39 quotes).
2. Sort the appended identities by id (as the builder does) or have the acceptance's author drop criterion 3,
   in the record.
3. Store in the record the 12 unpublishable keys, the near-duplicate count (11 by my threshold) and the KG-side
   repair proposal; correct section 7's reader claims to 46 of 101, 9 of 14 rebuilt, 157 topic identities.
4. Rule on item 21 of the read (one unconfirmed one-token KG person), then attach the `make ci-fast` log.

Verdict: NO-GO two served bridges (feed aef1e4c0 item 0002 and feed e739a158 item 0001) are left without episode_id, org, topic and object identities because 0034 round 3 turned their {} into {"identities": []} and 0036 detects only the {} literal (person_profile still skips one of them: 22 and 17 quotes unreachable), the appended bridges are unsorted against acceptance 3, the reader effect in section 7 is overstated (46 of 101, 9 of 14, 157 topics), the 25-person read has 1 unconfirmed person, the record lacks the 12 keys, the near-duplicate count and the KG repair, and ci-fast is not attached

## 8b. Verdict (round 2)

Dated: 2026-10-26, by a fresh step-8 reviewer (not the builder, not the round-1 reviewer), on commit 0e9f4454d
(m0036 and its tests unchanged since a859902ab; the gate report measured 6100d6b7b, whose src and tests equal
a859902ab). Base `gate35d` (snapshot-prod-20261025 with 0034 round 4 and 0035), after `gate36b`; reviewer code in
the session scratchpad `r8b36/` (derive.py, readers.py, topics.py, undo.py, mut.py; ids only, no names are printed
in this section). Every integer below was produced by the reviewer's own scripts in this round, none copied from the
gate report or the builder's blocks. Mutations ran only in a `git archive 0e9f4454d` copy.

### Round-1 conditions

| Round-1 condition | Result | Reviewer's evidence |
| --- | --- | --- |
| 1a. Damaged bridges rebuilt (the 2 `{"identities": []}` ones included) | closed | Base: 16 served bridges without `episode_id` (14 `{}`, 2 `{"identities": []}`: feed `aef1e4c0` item 0002, feed `e739a158` item 0001). After: 0. All 16 carry `episode_id` and `schema_version` "1.0". |
| 1b. Each rebuilt bridge equals `build_bridge(..., emitted_at=REBUILT_EMITTED_AT, fuzzy_reconcile=False)` minus unpublishable persons | closed | 16 of 16 equal as JSON text (all fields, key order, `emitted_at`), no `fuzzy_merges` key; 523 identities = 81 person, 59 org, 291 topic, 92 object. The 2 formerly half-built bridges hold 34 identities each (68, of which 11 persons: 57 more than the persons-only state, as predicted). |
| 1c. Reader check on the 2 persons of item 0002 | closed (measured; not yet in the record) | `person_profile` quotes before 0 and 0, after 22 and 17, each equal to the independent walk of the GI (quotes with a SPOKEN_BY edge to the person): 22 and 17. For all 14 GI-sourced restored persons in the readers' set (22 pairs) the profile delta equals the GI walk: 14 of 14, 0 mismatches. |
| 2. Appended identities in id order | closed | 18 appended bridges (20 identities): sorted before 18 of 18, after 18 of 18; existing identities a subsequence of the written list in 18 of 18 (0 moved), existing values changed 0, non-identity keys changed 0, duplicate ids 0 (0 over all 34 written). |
| 3a. The 12 unpublishable keys stored | partly closed | 12 (episode, person id) pairs are in the private eval repo, `podcast-scraper-eval-data-naming/data/eval/migrations/0036_bridge_identities_restored.unpublishable.json` (ids carry names, so not in this repo): 12 distinct pairs, equal to my own count of the after-clone's persons with no bridge identity (12 of 12), all KG-only; 10 are one label id in 2 feeds. The file is UNTRACKED in that repo (`git status`: `??`), and acceptance-D 17 asked for it under `docs/wip/changes/`. |
| 3b. Near-duplicate count | partly closed | Section 7 states 11 of 101 at difflib 0.78; the list of the 11 (acceptance-D 18) is stored nowhere I can find. Not re-derived by me. |
| 3c. KG repair proposal | partly closed | Section 7 has a 3-line proposal. Acceptance-D 19 asked for the 12 episodes, the two-layer check, dry-run, verify, undo, ownership, the derived surfaces to re-derive and "awaiting the operator's go"; most of that is absent. |
| 3d. Section 7 reader claims corrected | NOT closed for the final state | Section 7's reader bullet carries the 14-bridge numbers (9 of 14 read, 157 topics, 90 of 90, 16 of 90). On the final 16-bridge clone I measured: CIL readers read 405 of 3,504 bridges; 12 of the 34 written are read; 10 of the 16 rebuilt are read (the 10th is the item 0002 bridge); `episodes_for_bridge_node_id` lists 46 of 101 restored pairs (22 GI-sourced, 24 KG-only), 0 before; the 10 read rebuilt bridges bring back 177 topic identities (100 GI-sourced, 99 distinct ids); `topic_person_ids` is non-empty for 99 of 99 of those ids after, 15 of 99 before. 7b says it supersedes the numbers above but does not restate the reader paragraph. |
| 4a. Ruling on read item 21 (one unconfirmed one-token KG person) | NOT closed | The record has no ruling and no line saying none was given. |
| 4b. `make ci-fast` log attached | NOT closed | No log in the record or beside it; I did not run ci-fast either. |

### Re-derived against the 14:18 / 14:46 prediction (101 / 34 / 16 / 523 / 12 / 34 / scoreboard)

| Number | Prediction | Reviewer's number | Result |
| --- | --- | --- | --- |
| Baseline gaps, own count on `gate35d` | 101 + 12 | 113 gaps in 45 episodes; 101 publishable in 34 episodes, 12 unpublishable in 12 episodes; layers KG-only 55, GI+KG 40, GI-only 18 | met |
| Person identities restored | 101 in 34 episodes | 101 in 34 (81 in rebuilt bridges + 20 appended in 18) | met |
| Damaged bridges rebuilt / identities | 16 / 523 | 16 / 523 | met |
| Unpublishable left out | 12 untouched | after: 12 gaps, the same 12 (episode, id) pairs as before, 12 of 12 KG-only | met |
| Files written (sha256 of 145,030 base files vs 145,065 after) | 34 `*.bridge.json`, nothing else | 34 changed, all bridges, all served; 0 other changed; 0 deleted; 35 new = 34 backups + the receipts file | met |
| Scoreboard (committed `corpus_scoreboard.py`, both trees) | unchanged | 52 flattened keys compared, only `corpus` (the path) differs | met |
| `verify` | False on base, True after | base: False, "101 ... still lack a bridge identity; 16 empty bridge(s)"; after: True | met |
| Dry run on base | = prediction | "identities_added=101, bridges_rebuilt=16, rebuilt_identities=523, unpublishable_left_out=12" | met |
| Idempotence (copy of after-clone) | 0 written | second apply: added 0, rebuilt 0, episodes 0, skipped {}; whole tree sha-equal | met |
| Undo cycle (copy, sha256 of every file) | byte-identical | undo 1: restored 34, refused 0, 0 content files differ from base (35 differ, all 0036 bookkeeping), verify False; apply 2: added 101, rebuilt 16, 523, 34 episodes, content equal to apply 1 (bridges and everything else); undo 2: restored 34, refused 0, 0 content differ; other migrations' backups changed 0 | met |
| Tests | green | `test_m0036_bridge_identities_restored.py`: 52 passed on the archive copy | met, but see below |

### Tests and mutations

- Acceptance-D 15 asked for at least 14 more tests than the 52 of 9e39accc5. The 0036 test file still has 52 (the
  diff since 9e39accc5: one assertion re-ordered for D4, one input given an `episode_id`, one test renamed and
  turned from "skipped" to "rebuilt"); the D1 tests are in a new file (`test_bridge_rewrite_empty_id_map.py`). Bars 6
  (3 over-reach tests), 7 (4 guard tests for the no-`episode_id` rule), 8 (2 verify tests), 12 and 13 (5 D4
  insertion tests) are not met as written: there is no test for a `{"identities": []}` bridge without `episode_id`
  being rebuilt, none for an unsorted list keeping its order with two insertions, none for insertion at start, middle
  and end.
- Mutations I ran in the copy (acceptance-D 14, a to h plus two of mine): 10 run, 9 killed by the existing tests, 1
  survived. Killed: (a) back to `bridge == {}`; (b) rule widened to "empty identities"; (c) widened to "no
  schema_version"; (d) partial-input guard removed; (e) `verify` ignores damaged bridges; (f) append at the end;
  (g) sort all identities always; (h) sort even when the list was unsorted; (j) sort reversed. Survivor: (i) `not
  bridge.get("episode_id")` in place of `"episode_id" not in bridge`, which differs only for an empty-string
  `episode_id`; I did not count such bridges on the corpus, so I call it unargued, not equivalent. The 9 kills show
  the code is guarded; they do not make the missing tests exist.

### Not verified by the reviewer

- `make ci-fast` (not run; no log exists).
- Prod: every integer is the clone's; prod's empty-bridge files, whether 0034 touched the same 2 episodes there, and
  the step-10 plan diff are not known here.
- The 25-person read of round 1 was not repeated; the unconfirmed item 21 stands without a ruling.
- The near-duplicate list (11) was not re-derived; the 13 episodes where the builder would add one org identity
  each, and the fuzzy merges of 0025's backups, remain out of scope as the record states.
- Search-index rows and the app in a browser.

### Conditions (all before the apply is handed to prod)

1. Rewrite section 7's reader bullet on the final clone with the integers above (405 of 3,504 read; 12 of 34 written
   read; 10 of 16 rebuilt read; 46 of 101 pairs listed, 22 GI-sourced and 24 KG-only; 177 topic identities, 100
   GI-sourced, 99 distinct ids; `topic_person_ids` 99 of 99 after and 15 of 99 before; `person_profile` 22 and 17
   quotes for the two persons of item 0002) and remove the 9 / 14, 157, 90 of 90 and 16 of 90 claims.
2. Commit the unpublishable-keys file in the private eval repo and link it from section 3; store the list of the 11
   near-duplicates (episode key, person id, other person id) beside it; expand the KG repair proposal to what
   acceptance-D 19 lists, marked as awaiting the operator's go.
3. Add the acceptance-D tests (over-reach 3, no-`episode_id` guard 4, verify 2, D4 insertion and unsorted order 5 or
   more) so the test count rises by at least 14, or have the acceptance author drop those bars in the record.
4. The operator rules on read item 21 (or the record says none was given), and the `make ci-fast` log is attached to
   the committed tree before landing.

Verdict: GO WITH CONDITIONS Conditions: (1) rewrite section 7's reader bullet on the final clone (405 of 3,504 read, 12 of 34 written read, 10 of 16 rebuilt read, 46 of 101 pairs listed, 177 topic identities of which 100 GI-sourced, topic_person_ids 99 of 99 after and 15 of 99 before, person_profile 22 and 17 quotes for item 0002) and drop the 9/14, 157 and 16-of-90 claims; (2) commit the 12 unpublishable keys in the private eval repo, store the 11 near-duplicate list, expand the KG repair proposal; (3) add the acceptance-D tests (test count 52 unchanged, bar was +14) or have the bars dropped in the record; (4) operator ruling on read item 21 and the make ci-fast log attached before landing

Operator accepted: 2026-10-26: the sampled KG person "Walla" (round-1 read item 21, a garbled mention name) keeps its bridge identity: the bridge mirrors KG; garbled KG names belong to the KG-side repair.

**Outcome (condition 4, 2026-10-26):** `scripts/measure/build_outcome.py --migration 0036_bridge_identities_restored` on the gate
report measured at c2c0dcad2 (clean tree; the final gate after the format / lint / type /
docstring fixes, every number and plan equal to the d94e4d13a run): `upgrade/outcomes/0036_bridge_identities_restored.json`, outcome 14d782ea8f9a062bba4dc07a3d1886653eaaebb682d7f23d92ace627b571100a.
`make ci-fast` on the commit holding it: below.

## 10. Before apply / deploy

By the prod session, on the operator's direct go (2026-10-27 ~10:20Z: "0036 --to 2.7.30"),
after 0034 and 0035, on sha-2914e56. Dry run 10:16Z: `restored 101 bridge identit(ies) in 34
episode(s)`; identities_added 101, bridges_rebuilt 16, rebuilt_identities 523,
unpublishable_left_out 12, as the gate; **outcome hook: plan diff 0 episode(s)**.

## 11. After apply / deploy

- **Apply** 2026-10-27 10:17:41-10:23:53Z, `--to 2.7.30`, EXIT=0; receipt equal to the dry run;
  ledger at 2.7.30, 0 pending. Undo recorded: `m0036_bridge_identities_restored.undo`.
- **`upgrade verify`** 10:57Z (after the 0030 re-run): 34 [OK] incl. 0030, 0034, 0035, 0036
  ("every published GI / KG person has a bridge identity"); the only FAIL is 0025 on the 5 known
  episodes.
- **Ownership:** chowned with 0035's (5 paths); OWNERSHIP PASS.
- **Derived surfaces** (once, after 0034, 0035, 0036 and the 0030 re-run): `enrich --only
  insight_density,insight_sentiment` 248 ok, 0 failed; incremental `index` scanned 3004, skipped
  2756, reindexed 248, EXIT=0. No rebuild.
- **Snapshots:** the `upgrade-backups/` snapshots were deleted on the operator's go, after the
  ingestion session validated them (2026-10-27). The undo above therefore no longer has its
  backup: forward-only from here.
- **Not verified by the prod session:** the corpus-scope enrichers after the next nightly (open
  item 7, 2026-10-28); the search-index rows of the written episodes, read one by one; the app in
  a browser on a written episode.
