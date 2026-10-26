# 0036: step-5 acceptance criteria

<!-- markdownlint-disable MD029 -- the criterion numbers are identifiers the record cites -->

Written by a fresh step-5 reviewer (not the builder), 2026-10-26T10:44 UTC, before any 0036 code exists, from
the record's steps 1 to 4, `builders/bridge_builder.py` (`_collect_identity`, `build_bridge`),
m0025's bridge step, `server/cil_queries.py` and `hosts.is_publishable_speaker_name`. No
transcript text and no person names appear here; people are cited by key or episode path only.

Base: `snapshot-prod-20261025` with 0034 (a1698c214) and 0035 (2080d1cb2) applied, the clone the
prediction names (`scratchpad/gate35b`). Every number comes from a stored script; the gap counter
is the committed `scripts/measure/bridge_gaps.py` (the record says it will be committed; criterion
1 fails while it is only in a scratchpad). The reviewer re-derives criteria 1 to 6 with its own
code, not the builder's script.

## Definitions fixed before the build

- **Gap:** a GI or KG node whose id (layer prefixes stripped by `_strip_layer_prefixes`) starts
  with `person:` and is carried by no bridge identity of the same episode, neither as `id` nor in
  `aliases`.
- **Publishable:** `is_publishable_speaker_name(name, language=<episode language>)` with the
  default `require_person_shape=True`, the pipeline's own call. The name is the node's
  `properties.name`, else `properties.label`, as `_collect_identity` reads it. The record must
  state the default and the language source; a migration using `require_person_shape=False` or
  English for all episodes fails criterion 5.
- **Added identity:** an element of `identities` in the after-bridge whose `id` is absent from
  the before-bridge's ids.

## A. Own-outcome numbers (all integers, prediction restated)

1. **Baseline re-derived on the base:** gaps 115 persons in 45 episodes (KG only 55, GI and KG 42,
   GI only 18); publishable 103, unpublishable 12. If the base clone gives other integers (0034 and
   0035 were applied, so the record's 115 is a pre-check), the builder restates the prediction
   before the build; it is never explained after.
2. **Identities added: exactly 102, in exactly 34 episodes.** Counted by diffing every
   `*.bridge.json` before and after by identity id, not by a counter the migration prints. FAIL on
   101 or 103, and on 33 or 35 episodes.
3. **Only those 34 bridge files written.** A sha256 census of every file in the corpus tree
   (every extension, including `.podcast_scraper/` outside the 0036 backup tag) shows exactly 34
   changed files, all `*.bridge.json`, and 0 files created or deleted outside
   `.podcast_scraper/upgrade-backups/0036/` and the receipts file. In each of the 34 files the
   only difference from the before-file is added elements of `identities`: `schema_version`,
   `episode_id`, `emitted_at`, `fuzzy_merges` and every pre-existing identity are byte-identical,
   and `identities` stays sorted by `id` as `build_bridge` sorts it. (The record counts 103 publishable gaps as persons and predicts 102 identities. The builder states
   the cause of the 1 difference before the build; the reviewer checks it. An unexplained difference
   fails.)
4. **Publishable gaps after: 0.** Re-run `bridge_gaps.py` on the after-state: publishable 0 in 0
   episodes.
5. **Unpublishable untouched: 12 gaps in 12 episodes.** The after-state still lists exactly the 12
   keys (episode path plus person id) the before-state lists; the list is stored in the record
   (keys only), and a set diff before vs after is empty. None of the 12 appears in any added
   identity. The 12 bridges are among the 34 written files only if they also hold a publishable gap;
   then only the publishable identity is added, shown by the file diff.
6. **Scoreboard unchanged, every number:** S1, S2, S3, S1c, S1n, S10a, S10b, S12, commercial, A1,
   insights, quotes, episodes, from the committed `corpus_scoreboard.py`, before and after, equal
   to the integer. Own-outcome rows not in the scoreboard: hidden insights, S7, S8 (if the
   scoreboard prints them) equal too. FAIL on any move; 0036 does not touch what they read.

## B. The added identities are the pipeline's own

7. **Byte-level check by a script** (reviewer's, stored in the record's folder): for each of the
   102 added identities, rebuild the expected identity by calling
   `bridge_builder._collect_identity` into an empty mapping with the episode's GI node, then the
   KG node of that id (the order `build_bridge` uses), and compare
   `json.dumps(x, sort_keys=True, ensure_ascii=False)` of the migration's identity with it. Pass:
   102 of 102 equal, including key order as stored on disk. FAIL on one difference. The script also
   counts, as stated integers before the run: identities with both sources (gi and kg true), GI
   only, KG only, with a non-empty `aliases`, with an empty `display_name`. Expected: empty
   `display_name` 0. The alias count is predicted by the builder and checked against the node
   aliases by the same script.
8. **Alias carried both ways:** (a) every alias on the GI node and on the KG node of an added
   identity is in the identity's `aliases`, sorted and de-duplicated as the builder does; (b) a gap
   whose id is already an alias of an existing identity is not added (the by-id-or-alias test).
   Both appear as test cases (criterion 13) and as a count on the clone (how many added identities
   carry aliases; how many gaps skipped by alias).
9. **No fuzzy pass:** no added identity has a `fuzzy_merges` entry; no existing file gains a
   `fuzzy_merges` key; no two ids merged. A gap id that is a near-duplicate of an existing
   identity (same person, differently spelled id) is added as its own identity, and a list of such
   near-duplicate pairs in the 34 episodes is stored (count stated) as a known cost, not merged.

## C. The reviewer's read of a sample

10. **Sample: 25 added identities, drawn by the reviewer:** a seeded random draw (seed recorded in
    the read file) of 15 over all 102, stratified so that at least 5 are KG only, 5 GI and KG, and 3
    GI only (the three strata exist in the base; if a stratum has fewer than its quota, all of it is
    read), plus the 10 added identities with the longest or most unusual names by
    `is_publishable_speaker_name`-edge character (non-ASCII, one-token, three-or-more-token), so the
    rule's edges are read. For each, the reviewer opens the episode's GI and KG: is the node a
    person of that episode (speaker with insights or quotes, or a named guest/host/mention in the KG
    with a role), not a topic, an organisation or a label that survived the gate? Verdict per
    identity in a table keyed by episode path and person id. Pass: 25 of 25 real people of their
    episode. One non-person fails the change and sends it to step 3 (the filter is then wrong);
    it is never excused as a sample miss. If any of the 102 is found to be a non-person by any
    other reading, the same.
11. **Whole-population screen beyond the sample:** all 102 display names are listed (by key, in the
    verdicts file, not in the record) and scanned once by the reviewer for non-person shapes; the
    count read is 102, not the 25.

## D. The readers

12. **Reader effect, named and checked** (`server/cil_queries.py`; the clone served by the app's
    own code, no reimplementation):
    - `episodes_for_bridge_node_id` (`corpus_library.py` node-episodes listing) reads every bridge
      id (`_bridge_all_ids`). For each of the 102 restored ids, the call returns the episode's
      `bridge_relative_path` after and does not before (a person with no other episode: 0 rows
      before, 1 after). Pass: 102 of 102, listed. Also the 34 episodes' rows for a person who already
      appeared elsewhere gain exactly +1 row per restored episode.
    - `person_profile`, `position_arc` and the topic readers read `sources.gi` only
      (`_bridge_gi_ids`). So the effect is for the GI-bearing restored persons only (GI only and GI
      and KG strata: predicted count = identities with `sources.gi` true, stated by the builder
      before the run). Check: for one restored person with a `SPOKEN_BY` quote in the episode and
      a topic edge, `person_profile(root, anchor, id)` returns that episode's quotes after and not
      before; the count of added insights and quotes in the profile is stated and equals the
      count read from the GI by an independent walk. KG-only restored persons do not change
      `person_profile`; that is predicted, not hidden: the number of KG-only restored identities
      (their only effect is in the node-episodes listing and `_bridge_all_ids` readers) is
      stated.
    - `topic_person_ids` for a topic of such an episode lists the restored GI person after. One
      example is shown with its key.
    - `person_topic_ids` equals the keys of that person's `person_profile` topics.
13. Total effect stated as integers: persons whose `person_profile` was empty or smaller before and
    is larger after; quotes and insights reachable through the CIL surfaces before and after
    (whole-corpus sums computed by the reader functions on the clone); the S7-style outcome
    "insights of published people reachable from a person surface" if the scoreboard has it.

## E. Tests (they must fail without each rule)

14. Scenario tests on real-shaped fixture bridges, GI and KG (made by `build_bridge`, not
    hand-written), with these cases. Each rule has a mutation; the reviewer runs all of them at
    step 8, and a surviving mutation (every test green) is an unguarded path that sends the work
    back. Baseline: zero failures on the unmutated code. Bars: every row killed, 0 survivors.

    | Rule | Test | Mutation that must turn it red |
    |---|---|---|
    | publishable filter | an unpublishable-name GI person and KG person (a name the gate refuses) with no bridge identity: not added, listed, bridge file byte-identical | filter removed; filter inverted; filter called with `require_person_shape=False`; language fixed to English for a non-English episode (the test uses one episode with a language-specific refusal) |
    | gate per layer | a person publishable by the GI name but not by the KG name (or the reverse): the decision follows the node that is read first (GI then KG), stated in the record, one test per direction | name read from KG first; either layer's name accepted |
    | alias carried | a GI node with aliases and a KG node with other aliases: the added identity holds the sorted union | aliases dropped; only GI or only KG aliases kept; not sorted or not de-duplicated |
    | by-id-or-alias | a gap id listed as an alias on an existing identity: not added; a gap with an id equal to an existing identity id: not added | alias check removed; id check removed |
    | no fuzzy merge | two persons with near names: both stay separate; no `fuzzy_merges` key | fuzzy pass invoked (with a stub embedder returning high similarity) |
    | nothing else written | the 34-style bridge: every key except the new identities is byte-identical; `emitted_at` and `schema_version` unchanged; an episode with no gap writes no file; GI, KG and all other files in the episode directory unchanged | `emitted_at` refreshed; bridge rebuilt wholesale with `build_bridge`; identities re-sorted wrongly; file rewritten with no change |
    | non-person ids | an `org:` / `topic:` / `object:` node with no identity: untouched by 0036 (person only) | person-only guard widened to all `_CIL_PREFIXES` |
    | layer prefixes | a node id written `g:person:x` or `k:person:x` / `kg:person:x`: the identity id is stripped, matching `_strip_layer_prefixes` | strip dropped |
    | display name | a node with `name` blank and `label` set: the label is the display name; both blank: not added and listed | label fallback removed |
    | missing files | an episode with no GI or no KG, a malformed bridge, a bridge with no `identities` key: skipped with a counted reason, not a crash, not written | guard removed |
    | superseded runs | a bridge under a superseded `feeds/<feed>/run_*` sibling: handled as the migration states (written or skipped) and tested; the record states which | n/a (one test each way) |

15. A test that compares the migration's identity with `_collect_identity`'s over a corpus of at
    least 20 GI/KG pairs drawn from the clone's real files (fixture copies, no names in the test),
    equal 20 of 20; it fails if the migration builds the identity itself instead of calling the
    builder.
16. A `verify` test for each semantics below.

## F. Apply, idempotence, undo, verify

17. **Idempotence:** a second apply on the after-state writes 0 files, 0 backups, 0 receipts, and
    reports 0 identities added; the corpus sha256 census is identical to the after-state of the first
    apply. A third run after a no-op undo refusal is the same.
18. **Undo cycle:** undo, apply, undo on the clone. After the first undo: corpus byte-identical
    (sha256 census, every file) to the base, restored 34, refused 0, verify reads pending, the
    receipts file reflects the undo. After the re-apply: the 34 files byte-identical to the first
    apply's (same sha256), 102 added. After the second undo: byte-identical to the base again. The
    `emitted_at` rule matters here: a re-apply that stamps a new time fails this criterion, which is
    why 0036 must keep the existing `emitted_at`.
19. **Verify semantics:** `verify` is PASS on the after-state; FAIL on the base (reports the 102
    publishable gaps or the episodes, with the count); FAIL if one added identity is removed from
    one bridge (1 gap); PASS when the only remaining gaps are unpublishable ones (12 remain, PASS);
    PASS on an empty corpus; and does not read an unpublishable person as a failure. It reads the
    same `is_publishable_speaker_name` call and the same by-id-or-alias test as the plan; a
    verify using its own copy of either fails (a test pins that the migration and verify call the
    same function object). Full `upgrade verify` output: every other migration's line is the same
    before and after, line for line (0015, 0019, 0022 to 0025, 0029, 0031, 0033 to 0035 named), and
    only 0036's line changes (pending/FAIL to PASS).
20. **Receipts isolated:** own id, receipts file and backup tag `0036`; other migrations' receipts
    and backups byte-identical; undo of 0036 restores only 0036's files.
21. **Registered after 0035, `to_version` 2.7.30**, the registry test and the landing test
    (record, gate report, steps 1 to 8) pass; class `full` computed by `corpus_change_gate.py`
    from the apply, not declared.

## G. Prod (step 10)

22. **Prod's read-only plan diffed against the clone's:** the same 34 episodes, the same 102
    identities (the keys), the same 12 unpublishable, every counter equal. Prod differs from the
    snapshot taken 2026-10-25: any episode ingested since, and any difference in the gaps, is listed
    with its cause (an episode new since the snapshot; an episode edited by 0034 / 0035 on prod only
    if those ran there). A difference with no cause stops the run (D15). The plan also shows 0035
    and 0034 as applied on prod before 0036 is planned, or the plan states which base it describes.
23. Prod's search-index rows and any other surface built from the bridge: the census (D17) states
    what reads the bridge beyond `cil_queries` (the person pages `corpus_persons.py`,
    `app_relational.py`, the enrichers, the entity resolver `identity/resolver.py`, which
    aggregates bridge identities across the corpus) and whether any is cached or stored (an entity
    resolution output, a graph cache) that must be re-run for the 34 episodes. Each is listed with
    the deterministic non-LLM step that refreshes it, or "reads the bridge live".
24. `make ci-fast; echo MAKE_EXIT=$?` on the committed tree, log attached, MAKE_EXIT=0 before the
    step-8 verdict except the landing test for step-8 reasons.

## Not covered (the criteria above do not reach, and no number here will say so)

1. The 12 unpublishable persons: their bridge identity is absent and stays so; the KG-side repair
   of the junk names 0015 meant to remove is a separate change. Until it lands, those 12 episodes
   still lack the person on every reader, and the GI / KG still hold the junk names. Criterion 5
   only proves they are untouched.
2. The population is the 45 episodes on one snapshot (2026-10-25). Episodes ingested after, and
   gaps created by a later pipeline run or migration, are not measured; the migration is a rule, so
   the verify catches them but no number here predicts them. A reprocess of an episode with the
   current builder is not tested for creating the same gap again (the cause for the 20 persons that
   "no backup ever held" is not diagnosed in the record, only that no bridge entry exists);
   a future migration that drops a bridge identity (as 0034's alias bug did) is not guarded.
3. Fuzzy-reconcile merges the builder would add with an embedding model: not run, so a restored
   identity can duplicate an existing one under a different spelling. Criterion 9 counts the
   near-duplicates but does not judge them.
4. Whether the 102 identities are the right people: the sample is 25, plus a screen of the display
   names of all 102. The gate is a name filter, and a name that passes it but is not a person (an
   organisation shape, a mis-split name) can exist outside the 25. The screen is one reviewer's
   reading of names, not of each episode.
5. Alias correctness upstream: criterion 7 proves the identity equals what the builder gives for
   the node as stored; it does not prove the node's aliases are right (a wrong alias on a GI or KG
   node is copied into the bridge).
6. Language: the filter takes the episode language; a wrong or missing language field on an
   episode (the default `TARGET_LANGUAGE`) changes the publishable decision, and no criterion
   here audits that field's accuracy. Only a language-specific test case checks the call passes it.
7. Reader effect beyond `cil_queries`: the app's person card, the viewer, the relational graph,
   search-index rows and the entity resolver are not run end to end; criterion 23 lists them but
   the pass bar is a listing, not a render. The app is not opened in a browser or on a phone.
8. Prod itself: the 34 / 102 / 12 integers are the clone's; prod's are known only at step 10, and
   the post-apply re-measure (step 11) is outside this file. No prod data was read to write it.
9. The cause of the gaps: the diagnosis names the last backup that held each identity, not the
   code that dropped it; 84 from 0025, 11 from 0015, 20 from no backup. These criteria do not
   prove that no current migration or the pipeline recreates the gaps; the unprotected recurrence
   is a step-12 item.
10. Position, topic and perspective readers that look at `sources.gi` only: a restored KG-only
    person helps none of them. This is predicted, so it is a coverage limit, not a defect; the
    count of such identities is stated, and the person-surface gain is smaller than 102.
11. Search-index rows, the embeddings of the added display names, and any LLM-using path: nothing
    is rebuilt, by the operator's rule of 2026-10-01; no criterion proves they need no refresh.
12. Concurrency: an apply while a reprocess or ingest writes a bridge for the same episode is not
    tested (the migration reads then writes a whole file). The preflight of the runner is
    assumed to refuse it.
13. Non-served trees: bridges under superseded `run_*` directories are not counted in the
    34 / 102 unless the record's definition includes them; criterion 14's last row pins the
    behaviour but no number covers the superseded files on prod.
14. Mutations list the rules above; a mutation of the by-id-or-alias test against an alias with
    differing case or whitespace is not specified (the builder strips whitespace only).
