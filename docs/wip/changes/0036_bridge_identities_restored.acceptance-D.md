Dated: 2026-10-26T14:19

# 0036 acceptance addendum D: damaged bridges without episode_id (D3), id order of inserted identities (D4), and the step-8 record corrections

<!-- markdownlint-disable MD029 -- the criterion numbers are identifiers the record cites -->

Method step 5, by a fresh reviewer (not the builder), before the code fix, for the block dated
2026-10-26T14:18 in the record and the step-8 NO-GO of section 8. It adds to acceptance, B and C;
nothing in them is replaced. Read: the record (section 4 blocks 11:29, 12:34, 14:18; sections 7 and 8),
acceptance, B, C, and `m0036_bridge_identities_restored.py` (`bridge == {}` is the only empty test;
the append path is `list(bridge["identities"]) + add`). The reviewer measured and ran nothing: every
integer is a bar chosen by the reviewer, for the operator to accept. Ids only: no names.

Base: the 12:34 base (`gate35c`), predicted 16 rebuilt, 523 identities. 0034's round 4 moves the
base; if it moves it, the builder restates every integer in the record BEFORE the gate and the bars
below are read on the restated base (same rules, new integers). A bar met on a base the builder did
not restate is not met.

## Rule D3: a bridge with no episode_id is damaged

1. **Rebuilt count: 16 bridge files** (14 `{}` plus 2 `{"identities": []}`: feed `aef1e4c0` run
   `ed9ab40e` item 0002, feed `e739a158` run `0d12a48d` item 0001), each damaged in the base (no
   `episode_id`) and carrying `episode_id` and `schema_version` after. FAIL on 15 or 17.
2. **Identities in the 16 rebuilt files: 523**, counted by a script of the reviewer's from the after
   files by id prefix, with the per-prefix split stated in the record by the builder (person, org,
   topic, object; the four sum to 523); FAIL on any component off by one. Of the 523, the 2 formerly
   half-built bridges hold **57 more** builder identities than before (10 org, 7 object, 40 topic).
3. **Equality, 16 of 16:** each equals `build_bridge(episode_id, gi, kg, emitted_at=REBUILT_EMITTED_AT,
   fuzzy_reconcile=False)` with every person that `is_publishable_speaker_name` refuses removed,
   compared as JSON text in every field (`schema_version`, `episode_id`, `emitted_at`, identities in
   order, every key, nested `sources` and `aliases`); no `fuzzy_merges` key. One difference fails.
4. **Persons:** **101** person identities in **34** episodes in all (81 in rebuilt bridges, 20 appended
   in 18 episodes); **12** unpublishable left out and still listed as gaps, the same (episode, id)
   pairs before and after; publishable gaps after **0**.
5. **Files:** exactly **34** `*.bridge.json` differ from the base, **0** other files differ, **0**
   deleted; scoreboard: **0** of the 51 keys differ (only the `corpus` path).
6. **The rule must not over-reach (test):** a bridge that HAS `episode_id` and a non-empty
   `identities` list is never rebuilt: **at least 3** tests, each asserting `bridges_rebuilt == 0`
   and the file byte-equal: (a) a complete bridge with persons missing (append path, not rebuild);
   (b) a complete bridge with `identities: []` and an `episode_id` (the rule is "no episode_id", so
   this one is NOT damaged and is not rebuilt; the builder states in the test name what it does
   with it); (c) a complete bridge whose `episode_id` differs from the metadata's (not rebuilt; its
   id is not rewritten). All **3** green after.
7. **Guard as D2:** a damaged bridge (no `episode_id`) whose GI or KG is missing, or unparsable,
   is skipped, the file byte-equal, with a counted reason in `details["skipped"]` and in the apply
   message: **4** tests (no GI; no KG; unreadable GI; unreadable KG), each asserting
   `bridges_rebuilt == 0` and the reason count **1**; red before, green after. On the clone,
   damaged bridges skipped for missing or unreadable input: **0** of 16; if any is skipped the run is
   not a pass and the record names the ids and why.
8. **verify fails on a damaged bridge:** **2** tests (a `{}` bridge; a `{"identities": []}` bridge
   without `episode_id`) each make `verify` return False and name the count of damaged bridges;
   after the apply on the clone, `verify` returns True. On the base, `verify` FALSE with **16**
   damaged (the run ledger shows 14 plus 2); a `verify` that returns True on the base is a fail.
9. **Reader check on the two formerly half-built bridges** (the step-8 condition 1): with the app's
   own `person_profile`, the two restored GI persons of item 0002 (feed `aef1e4c0`) have a profile
   delta of **22** and **17** quotes (**39** together), each equal to the independent GI walk; before
   the change **0** and **0**. FAIL on any other integer.
10. **Idempotence:** a second apply on the after-state writes **0** files, adds **0**, rebuilds **0**.
    **Undo cycle:** undo restores **34**, refuses **0**, **0** of the corpus files differ from the
    base; apply again gives **34** files and the same **101**, **16** rebuilt; undo again the same;
    other migrations' backups changed **0**.

## Rule D4: inserted identities keep id order

11. **Sorted stays sorted (test and clone):** where the existing `identities` list was in id order,
    the written list is in id order. On the clone, of the **18** appended bridges, **18** of **18**
    are sorted by id after (step 8 found 0 of 18 after, 18 of 18 before). FAIL on 17.
12. **Existing relative order unchanged otherwise (test):** where the existing list was NOT in id
    order, the existing identities keep their relative order and the new ones are placed without
    reordering them; **at least 2** tests (an unsorted list with one inserted; an unsorted list with
    two inserted at the first and last position); the check is that the existing ids, in the written
    list, are a subsequence equal to the original list: **0** moved. Existing identities equal in
    value to the base: **0** changed, over every appended bridge on the clone.
13. **Insertion tests:** **at least 3** more: insert at the start, in the middle and at the end of a
    sorted list; two inserted persons whose ids sort around an existing one; an inserted id that
    equals none of the existing (no duplicate ids in any written list: **0** duplicates over the 34).
    A bridge whose list is empty gets its persons in id order.

## Mutations (each new rule caught)

14. Run in a copy by a fresh tester; **at least 8** mutations, **8** of **8** caught (a survivor is
    argued equivalent, in writing, or fails):
    (a) remove the "no episode_id = damaged" rule (back to `== {}`): caught by items 1, 6(none), 8;
    (b) rule widened to "empty identities list = damaged" (ignores `episode_id`): caught by 6(b);
    (c) rule widened to "any bridge missing `schema_version` = damaged": caught by 6(a) or 6(c);
    (d) remove the D3 input guard (rebuild from partial input): caught by the 4 tests of item 7;
    (e) `verify` ignores damaged bridges: caught by the 2 tests of item 8;
    (f) append at the end again (no sorting): caught by items 11 and 13;
    (g) sort ALL identities, including the existing ones, always: caught by item 12;
    (h) sort the written list when it was not sorted before: caught by item 12.
    Plus the earlier tables: acceptance, B and C mutations still all caught (**0** survivors beyond
    the **2** already argued equivalent: `bool(name)` and `_display`'s strip).
15. Tests: whole `tests/unit/podcast_scraper/upgrade` plus `tests/unit/upgrade` green with **0**
    newly failing; the test count rises by **at least 14** over the 52 of 9e39accc5.

## The record corrections the step-8 reviewer asked for (section 7 and the Not covered section)

16. Section 7's reader paragraph is rewritten with these integers, from the after-clone: the CIL
    readers read the latest run per feed only (**405** of **3,504** bridges); of the 34 written,
    **12** are read; of the 14 rebuilt, **9** are read (restated for the 16 if the two formerly
    half-built bridges are read; the builder counts it); `episodes_for_bridge_node_id` lists
    **46** of the **101** restored (episode, person) pairs (**0** before; **22** GI-sourced, **24**
    KG-only); the read rebuilt bridges bring back **157** topic identities (**90** GI-sourced), not
    251; `topic_person_ids` returns a speaker for **90** of **90** GI topics after, **16** of **90**
    before. The sentence "brings back 251 topic identities" is gone. FAIL on any integer off.
17. The record stores the **12** unpublishable keys, one per line as (episode key, person id), ids
    only, no names, in a file under `docs/wip/changes/` linked from section 3. 12 lines, 12 distinct
    pairs, equal to the 12 pairs of the gap census after the change (**0** differences). Their
    census in the record: **10** one KG label id in 2 feeds, **1** a role word, **1** a letter-spaced
    garble; **12** of **12** KG-only.
18. The record states the **near-duplicate count: 11** of the 101 added persons have another Person
    in the same episode within difflib ratio 0.78 (10 GI-sourced, 1 KG-only; in **7** of the 11 the
    other node is a KG-only spelling variant that the bridge already holds), the threshold written
    next to the number, and a list of the 11 as (episode key, person id, other person id), ids only.
    A different count with the same threshold is a fail and is explained.
19. The record contains the **KG repair proposal**, as a numbered section or a block (step 12 does
    not exist, so it is not deferred there): a deterministic KG migration that removes, in the **12**
    keys' **12** episodes, the Person nodes and their edges whose names fail the same publishable
    gate, with 0015's two-layer check, dry-run, verify, undo and corpus-owner ownership; it names the
    surfaces to re-derive (bridge, enrichment outputs, search rows) and states it is a separate change
    awaiting the operator's go. No LLM, relabel or DGX step in it.

## NOT COVERED by this addendum (stated, not hidden)

- Item 21 of the step-8 read (one unconfirmed one-token KG person): a ruling for the operator, not a
  code bar; the record gets the operator's ruling or a line saying none was given.
- The `make ci-fast` log: not a bar here; attached once at the end of the branch, as the standing
  rule says.
- The prod plan diff (step 10), the prod empty-bridge census (17 files) and whether 0034 touched the
  two episodes on prod: unknown on this clone.
- The 13 further episodes where the builder (fuzzy off) would add one org identity each, and fuzzy
  merges of the backups (7 to 10 per rebuilt bridge): out of 0036's person scope.
- Search-index rows and the app in a browser.
- Prod: the 16, 523, 101, 34 and 12 are clone integers.
