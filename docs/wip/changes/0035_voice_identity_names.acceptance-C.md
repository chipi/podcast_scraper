Dated: 2026-10-26T11:31

# 0035 acceptance addendum C: the empty-id-map bridge aliasing fix

<!-- markdownlint-disable MD029 -- the criterion numbers are identifiers the record cites -->

Method step 5, by a fresh reviewer (not the builder), before the code fix, for the block dated
2026-10-26T11:29 in the record. It adds to the earlier acceptance files; nothing in them is replaced.
The reviewer measured and ran nothing: every integer below is a bar chosen by the reviewer, for the
operator to accept. Read: the 11:29 block, `rewrite_bridges_m0007.apply_to_payload` (returns `payload`
itself when `mapping` is empty), m0017 `_rewrite_bridge` (`bridge.clear(); bridge.update(out)`), the
repro `test_the_shared_rewrite_never_writes_a_bridge_without_its_identities`. Ids only: no names.

## D1 pass bars

1. The re-measure (same snapshot, clean tree at the fix commit, same method as acceptance-B) equals the last measurement on every number and the same file set: every integer in the record's section 7 table unchanged, the sorted list of written paths identical (**0** added, **0** removed), verify **OK**, refused unchanged, must-not-move numbers **0** moved. Only the code digest may differ. Any changed integer fails; the builder quotes the section 7 numbers before the run.
2. The repro goes green: **3 of 3** parametrised paths of `test_the_shared_rewrite_never_writes_a_bridge_without_its_identities` pass after the fix and at least **1** (the `renamed` path) fails on the unfixed code, shown on both trees.
3. Empty bridge files on the clone after 0035: **16**, the same as after 0034 (**0** new from 0035).
4. Mutation: revert the fix (return the input object when the map is empty) and the repro fails: **1 of 1** killed.
5. Users of m0017's bridge rewrite (`_Episode._rewrite_bridge`, the only caller of `apply_to_payload`; `grep -rn apply_to_payload src` finds nothing else), listed by the reviewer from a grep of `src/podcast_scraper/upgrade/migrations`:

- m0017 itself (`test_m0017_speaker_names_canonicalised.py`): its own `_Episode`.
- m0020 (`test_m0020_titled_person_ids_remerged.py`): imports m0017's `_Episode` directly.
- m0025 (`test_m0025_one_person_one_entry.py`): `_RunRenameEpisode(_RenameEpisode)`, so 0034, 0035 and m0032 through m0025's machinery (m0032 per its docstring, line 31; the builder confirms the import path).
- m0015 builds on m0012's `_Episode`, not m0017's; it is argued out, not covered, and the builder confirms with `grep -n "_rewrite_bridge" m0012*.py m0015*.py` returning 0.

Bar: each user above either runs the repro (`test_the_shared_rewrite_never_writes_a_bridge_without_its_identities`, 3 parametrised paths: renamed, voices, unnamed) or has a test written for it that feeds an episode whose id map is empty and asserts the bridge keeps its `identities` list. m0017 and m0020 have no such test today: **2** tests to add, or the builder argues in one sentence per migration why its id map is never empty on a bridge that has identities. Any user left with neither fails.
