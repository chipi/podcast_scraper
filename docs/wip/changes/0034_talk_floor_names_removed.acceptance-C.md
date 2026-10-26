Dated: 2026-10-26T09:50

# 0034 acceptance addendum C: the bridge-alias fix in the shared rewrite

<!-- markdownlint-disable MD029 -- the criterion numbers are identifiers the record cites -->

Method step 5, written by a fresh reviewer (not the builder) for the fix named in the record's
section 4d ("A defect found by the census of that clone", dated 2026-10-26T09:50). It adds to
`0034_talk_floor_names_removed.acceptance.md` (the 31) and `...acceptance-B.md` (replacing 6, 24,
25; adding 32, 33); nothing in those is replaced here. Written from the record's section 4d, the
method's step 5, the shared bridge step in `m0025_one_person_one_entry.py` (the loop that bumps
`bridge_identities_removed`) and the repro test
`test_a_bridge_identity_kept_by_an_alias_survives_the_rewrite`. The reviewer did not read the
fix and measured nothing: every integer below is either the 09:27 measurement quoted from the
record (clean tree, db37f8b1c, `0034_talk_floor_names_removed.gate.json`) or a bar chosen by the
reviewer, for the operator to accept. Ids only: no person names, no transcript text.

## 1. The defect, as the criteria read it

The bridge step keeps a `person:` identity only when its own id is in the episode's GI or KG
person ids; it never reads `aliases`. An identity whose id is an older unresolved form, with the
person's GI / KG id only among its aliases, is dropped although GI and KG keep the person. On the
clone this removed one identity that should have stayed: `bridge_identities_removed` was 99, of
which 98 are the removed voices' persons. The fix keeps an identity when its id OR any alias is a
GI / KG person id, and sets `sources.gi` / `sources.kg` from whichever matches.

## 2. What must equal the 09:27 measurement exactly (re-measure on the fix, same snapshot, same clean-tree method)

Rerun `scripts/measure/corpus_change_gate.py` for 0034 on a fresh clone of
`snapshot-prod-20261025`, from a clean tree at the fix commit. Pass bars, integers, all equal:

1. Episodes written: **126**. Voices changed: **140**. Verify: **OK**.
2. S1 and S2: **+56** each (the 09:27 figure, itself restated against the record's +48 to +55 and
   explained by insight id in criterion 12; the fix must not move it). S10a: **+1**.
3. Files written: **1,499**, and the **set** of written paths identical to the 09:27 gate file's
   (compared as a sorted list; 0 paths added, 0 removed).
4. Every must-not-move number in the gate file equal to its 09:27 value (all of them, 0 moved).
5. Refused: **0**. Names off, names kept and written-episode counts as in acceptance-B criterion
   10 and the 09:27 gate file: no change.
6. Insights shown to hidden `unknown`: **56**; shown under a name another voice keeps: **2**.

## 3. What may change, and nothing else

The only permitted differences from the 09:27 gate file:

7. `bridge_identities_removed`: **99 -> 98** (exactly one fewer; the other 98 are the removed
   voices' persons and stay removed).
8. The bridge file of **exactly one** episode differs from the 09:27 clone: it keeps the one
   identity, with `sources.gi` and `sources.kg` set from the matching alias. The sha256 of every
   other written bridge equals the 09:27 value (**0** other bridge diffs; reported as a count over
   the 1,499 files, expected 1 of 1,499).
9. Because that one bridge differs, its file sha256 and any counter that sums bridge bytes or
   identities may move by that one identity. Any other changed counter in the gate file fails.
10. The kept identity is not a removed voice's person: its alias that matches is a person id still
    present in the same episode's GI and KG after the rewrite (checked by id, 1 of 1).

## 4. Mutation bars for the fix (step 8 reviewer runs these against the tests)

Each mutation must turn at least one test red. A mutation that leaves all tests green is an
unguarded path and sends the work back.

11. **Aliases ignored again** (the match is on `ident_id` only): the repro test must fail. Bar:
    red count **>= 1**, and it is the repro test.
12. **Alias match keeps a removed voice's person** (match on aliases, but without excluding the
    persons the rewrite removed): a test must fail that proves the 98 removed-person cases still
    leave the bridge. If no existing test fails, a test is added first (the repro's second
    identity, the removed voice's person with `aliases: []`, only guards the empty-alias case; a
    removed person whose alias list names another kept id, and one whose alias names its own
    removed id, must each be a case). Bar: red count **>= 1** per mutation.
13. **Alias match on any alias, including the display name**: the repro lists the display name as
    an alias beside the id; a mutation that keeps an identity because an alias equals a
    non-person string (a name, an empty string, `None`) must fail a test. Bar: **>= 1** red.
14. **`sources` not set from the match** (identity kept, sources left as read): a test must assert
    `sources.gi` / `sources.kg` on the kept identity. Bar: **>= 1** red.
15. **`bridge_identities_removed` not bumped for a dropped identity, or bumped for a kept one**:
    a test must assert the counter (expected 1 in the repro: the removed voice's person only).
    Bar: **>= 1** red per direction.
16. Mutation count reported as `killed / run`; bar **5 of 5** (criteria 11 to 15), 0 survivors.

## 5. The test bar

17. The repro `test_a_bridge_identity_kept_by_an_alias_survives_the_rewrite` **fails on
    db37f8b1c** (the commit before the fix) and **passes on the fix commit**. Both runs are
    shown, the first with the failing assertion (the kept identity is absent from the bridge), the second green.
    Bar: fail count **1 -> 0** for that test; no other test in
    `tests/unit/podcast_scraper/upgrade/test_m0034_talk_floor_names_removed.py` changes state.
18. The unit tests of every migration that shares the step (m0012, m0029, m0030, m0031, and
    m0025 itself) pass unchanged on the fix: failures **0**. A test that had to change is listed
    with its reason, and a changed expectation about a dropped identity is a finding, not a pass.
19. `make ci-fast` passes once on the committed tree before landing (method step 8), log
    attached.

## 6. Exposure of 0031, 0033 and 0035 to the same defect

The same step serves other migrations. 0031 is applied on prod (receipt PASS, 2026-10-19); the
imported shared code means each applied migration may have dropped identities that GI / KG still
hold. This is measured, not assumed, and read-only.

20. **Method, per applied migration** (0031 first; 0033 and 0035 for as far as they are applied
    or share the step; the migration list is stated with the evidence for each, including any
    that does not call the step, which counts as 0 by that evidence):
    - Take the snapshot's backups, `.podcast_scraper/upgrade-backups/<migration id>/`, for the
      bridge files that migration touched: the **before** bridges.
    - Take the **served** bridges of the same episodes (the snapshot's corpus, or prod's, read
      only).
    - For each episode: list every identity id in the before bridge that is absent from the
      served bridge, and keep those whose `aliases` contain a person id that is present in the
      served episode's **GI or KG** person nodes. That is the defect's signature: dropped by id,
      still held by alias.
    - Report, per migration: episodes with a bridge in the backup (denominator), episodes with
      at least one signature hit, identities hit in total, distinct hit identities. Integers.
21. **Control:** the same query on 0034's own 09:27 clone must report **1** hit identity in
    **1** episode (the known case) before the fix and **0** after. A query that does not find the
    known case is not run on the others.
22. **What result triggers a separate repair change:** any migration with **>= 1** hit identity
    in **>= 1** episode. A separate change record is opened for it (a repair, the shape of 0031
    for 0029), with its own steps 1 to 8; it is not folded into 0034's apply. The 0034 apply
    itself proceeds only on criteria 1 to 19. A result of **0** across all three, with the
    denominators stated, closes the exposure; a migration whose backups are missing from the
    snapshot is reported as **NOT MEASURED** and is not counted as 0.
23. Each hit is classed by id, never by name: (a) the identity's alias is the GI / KG person id of
    a person still shown (repair needed), (b) the alias id is a removed voice's person (correctly
    dropped, not a hit), (c) other. The repair trigger counts class (a) only; classes (b) and (c)
    are listed with counts.

## 7. Not covered

- The fix commit and the m0025 code were not read by this reviewer; the criteria are written from
  the record and the repro. Whether the fix is the minimal one is the step-8 reviewer's call.
- The re-measure has not run; sections 2 and 3 are bars for it, not results. The 09:27 numbers
  are quoted from the record, not recomputed here.
- Identities dropped for a reason other than the alias defect (an id in neither GI nor KG with no
  alias in either) are intentionally still dropped; whether any of those were wrongly dropped by
  another route is not measured here.
- Sources that are not bridges: whether the same alias blindness exists anywhere else in the
  shared rewrite (other files that key a person by id only) is not covered; criterion 20 reads
  bridges only.
- Prod itself: the exposure measure runs on the snapshot's backups and served files. Prod's
  state after 2026-10-25 is not covered, and prod is not reachable by SSH from here; a hit on the
  snapshot is a lower bound for the repair, not its final count.
- Derived surfaces fed by a dropped bridge identity (search-index rows, enrichment outputs)
  are not measured here. A repair change under criterion 22 owns that census (method D17).
- The bars in sections 4 to 6 are the reviewer's choice, not measured: for the operator to
  accept before the re-measure runs. Criterion 12's extra cases are a request, not a check that
  they already exist.
- No person was read, and no verdict was changed: acceptance-B's 80 of 89, 2 of 120 and 5 of 9
  bars are untouched and not re-asserted.
