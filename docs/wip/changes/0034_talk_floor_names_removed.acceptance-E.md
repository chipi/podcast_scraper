Dated: 2026-10-26T11:31

# 0034 acceptance addendum E: the empty-id-map bridge aliasing fix

<!-- markdownlint-disable MD029 -- the criterion numbers are identifiers the record cites -->

Method step 5, by a fresh reviewer (not the builder), before the code fix, for the block dated
2026-10-26T11:29 in the record. It adds to the earlier acceptance files; nothing in them is replaced.
The reviewer measured and ran nothing: every integer below is a bar chosen by the reviewer, for the
operator to accept. Read: the 11:29 block, `rewrite_bridges_m0007.apply_to_payload` (returns `payload`
itself when `mapping` is empty), m0017 `_rewrite_bridge` (`bridge.clear(); bridge.update(out)`), the
repro `test_the_shared_rewrite_never_writes_a_bridge_without_its_identities`. Ids only: no names.

## D1 pass bars

1. The re-measure (same snapshot, clean tree at the fix commit, same method as acceptance-C and -D) equals the last measurement on every number: episodes written **126**, voices changed **140**, verify **OK**, S1 / S2 **+56**, S10a **+1**, refused **0**, `bridge_identities_removed` **112**, files written **1,499**; every must-not-move number **0** moved; the sorted list of written paths identical (**0** added, **0** removed). Only the code digest may differ. Any other changed integer fails. (The reviewer quotes the numbers from acceptance-D and the record; the builder checks them against `0034_talk_floor_names_removed.gate.json` before the run and states any that differ.)
2. The repro goes green: **3 of 3** parametrised paths of `test_the_shared_rewrite_never_writes_a_bridge_without_its_identities` pass after the fix and **3 of 3** (or the red subset the record names, at least **1**, the `renamed` path) fail on the unfixed code, shown by running it on both trees.
3. Empty bridge files after the re-run on the clone: **16**, the same as after 0034 now (**0** new). Bridges with an empty id map and a non-empty `identities` list in, `identities` list unchanged out: **all** of those on the clone, counted (not sampled).
4. Mutation: revert the fix (`apply_to_payload` returns the input object when `mapping` is empty) and the repro fails: **1 of 1** mutant killed. A second mutant, returning a shallow `dict(payload)`, is **killed or argued** (it is safe for `_rewrite_bridge`'s clear/update only if `identities` is not mutated in place; the builder says which).
5. Users of m0017's bridge rewrite (`_Episode._rewrite_bridge`, the only caller of `apply_to_payload`; `grep -rn apply_to_payload src` finds nothing else), listed by the reviewer from a grep of `src/podcast_scraper/upgrade/migrations`:

- m0017 itself (`test_m0017_speaker_names_canonicalised.py`): its own `_Episode`.
- m0020 (`test_m0020_titled_person_ids_remerged.py`): imports m0017's `_Episode` directly.
- m0025 (`test_m0025_one_person_one_entry.py`): `_RunRenameEpisode(_RenameEpisode)`, so 0034, 0035 and m0032 through m0025's machinery (m0032 per its docstring, line 31; the builder confirms the import path).
- m0015 builds on m0012's `_Episode`, not m0017's; it is argued out, not covered, and the builder confirms with `grep -n "_rewrite_bridge" m0012*.py m0015*.py` returning 0.

Bar: each user above either runs the repro (`test_the_shared_rewrite_never_writes_a_bridge_without_its_identities`, 3 parametrised paths: renamed, voices, unnamed) or has a test written for it that feeds an episode whose id map is empty and asserts the bridge keeps its `identities` list. m0017 and m0020 have no such test today: **2** tests to add, or the builder argues in one sentence per migration why its id map is never empty on a bridge that has identities. Any user left with neither fails.
