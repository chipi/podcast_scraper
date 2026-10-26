Dated: 2026-10-26T11:31

# 0036 acceptance addendum C: the empty-id-map aliasing fix (D1) and the partial-input rebuild guard (D2)

<!-- markdownlint-disable MD029 -- the criterion numbers are identifiers the record cites -->

Method step 5, by a fresh reviewer (not the builder), before the code fix, for the block dated
2026-10-26T11:29 in the record. It adds to the earlier acceptance files; nothing in them is replaced.
The reviewer measured and ran nothing: every integer below is a bar chosen by the reviewer, for the
operator to accept. Read: the 11:29 block, `rewrite_bridges_m0007.apply_to_payload` (returns `payload`
itself when `mapping` is empty), m0017 `_rewrite_bridge` (`bridge.clear(); bridge.update(out)`), the
repro `test_the_shared_rewrite_never_writes_a_bridge_without_its_identities`. Ids only: no names.

## D1 pass bars

1. Every D1 bar of 0034 acceptance-E and 0035 acceptance-C holds on the same clone, and the user list of m0017's bridge rewrite is the same list (cross-reference, not repeated): re-measure of 0034 and 0035 equal to their last measurements on every number and file set; repro **3 of 3** green and the `renamed` path red before; mutation returning the input object caught (**1 of 1**); every other user covered by the repro or by a test, or argued.
2. 0036's own numbers on the clone with 0034 and 0035 applied are unchanged from acceptance (10:44) and acceptance-B: **102** identities added in **34** episodes, only those **34** bridge files written, publishable gaps **0**, the **12** unpublishable untouched, idempotent (**0** files written on the second run), undo byte-identical (**0** differing files).

## D2 pass bars

3. An EMPTY bridge whose episode has no GI, no KG, or an unreadable (unparsable) GI or KG is skipped, the bridge file byte-equal, with a counted reason in `details["skipped"]`: **both** repro tests (`test_an_empty_bridge_with_no_gi_and_no_kg_is_skipped_with_a_counted_reason`, `test_an_empty_bridge_with_an_unreadable_gi_is_skipped_with_a_counted_reason`) green after, red before. Two further tests are added by the builder, **2**: unreadable KG with a good GI, and a missing KG with a good GI, each asserting `bridges_rebuilt == 0` and the skipped count **1**.
4. On the clone: empty bridges rebuilt **15 of 15** (the predicted none-is-partial-input holds), skipped for missing or unreadable input **0**; if any is skipped, the run stops being a pass and the record states which ids and why.
5. Mutation: remove the guard (rebuild from whatever input is readable) and the D2 tests fail: **1 of 1** killed, by at least **2** tests.
6. A non-empty bridge (the append path) with a missing or unreadable GI or KG keeps its present behaviour, unchanged: the existing tests for it stay green (**0** newly failing); the reviewer's reading of acceptance-B R8 is that the guard is for the empty rebuild only, and the builder says so if the code differs.
