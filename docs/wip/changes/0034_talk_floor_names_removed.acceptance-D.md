Dated: 2026-10-26T10:09

# 0034 acceptance addendum D: the bridge-identity correction and the verdicts-loader basis set

<!-- markdownlint-disable MD029 -- the criterion numbers are identifiers the record cites -->

Method step 5, by a fresh reviewer (not the builder), for the revision dated 2026-10-26T10:08 in
the record's section 4d ("Correction" and its prediction). It adds to acceptance (the 31), -B and
-C; nothing in those is replaced. The reviewer measured nothing and did not run the fix: the
09:27 numbers are quoted from the record (clean tree, db37f8b1c,
`0034_talk_floor_names_removed.gate.json`); every other integer is a bar chosen by the reviewer,
for the operator to accept. Read: the record's 4d, acceptance-C, the bridge step in
`m0025_one_person_one_entry.py` (the `taken_off` set), and `reviewer_verdicts.py` (`BASES`,
`parse`, `load`). Ids only: no person names, no transcript text.

## 1. The two changes, as the criteria read them

- Bridge: the 9a7118eb4 code drops an identity whose own id is in `taken_off` even when that id
  is still a GI person. The correction keeps an identity whose OWN id is still a GI / KG person;
  an identity matching only by alias is kept unless its own id is in `taken_off`.
- Loader: `load` / `parse` check each row's basis against the module-level `BASES`. The revision
  lets a migration pass its own basis set; 0034 passes none.

## 2. Re-measure bars (same snapshot, clean tree at the fix commit, same method as acceptance-C)

1. The re-measure equals the 09:27 measurement on every number and every file set: episodes
   written **126**, voices changed **140**, verify **OK**, S1 / S2 **+56**, S10a **+1**, refused
   **0**, insights shown to hidden `unknown` **56**, shown under another voice's name **2**, files
   written **1,499**, and the sorted list of written paths identical (**0** added, **0**
   removed). Every must-not-move number equal (**0** moved). Any other changed integer fails.
2. `bridge_identities_removed` is **98** (09:27: 99; 9a7118eb4 run: 112).
3. Against the 9a7118eb4 run (`gate_0034_r4`), exactly **14** bridges differ, each in a different
   episode and each keeping identities that run dropped. Against the db37f8b1c run, exactly
   **1** bridge differs (the alias-matched identity kept). Reported as counts over the 1,499
   files by sha256; **0** other bridge diffs in either comparison.
4. For each of the 14, the kept identity's own id is a person still in that episode's GI or KG
   after the rewrite (checked by id, **14 of 14**); none is a removed voice's person.
5. The 98 identities removed are exactly the removed voices' persons: **98 of 98** have their
   own id in `taken_off` and **0** is still a GI / KG person id of a shown voice.

## 3. Mutation bars for the bridge step

Acceptance-C's five mutants (its criteria 11 to 15: aliases ignored, alias keeps a removed
person, any alias including non-person strings, `sources` not set from the match, counter wrong
in either direction) stay killed, **5 of 5**, **0** survivors, rerun on the new code.

6. A sixth mutant: `taken_off` is applied even when the identity's own id is still a GI / KG
   person (the 9a7118eb4 behaviour, `if ident_id in taken_off` with no own-id check) must turn a
   named test red. Bar: red count **>= 1**, and that test builds an identity whose own id is
   both in `taken_off` and a GI person id and asserts it is kept. Total reported
   `killed / run` = **6 of 6**.
7. The same test file also holds the opposite case, so the correction cannot become "never apply
   `taken_off`": an identity whose own id is in `taken_off` and in neither GI nor KG is dropped
   (counter +1), and an identity matched only by alias whose own id is in `taken_off` is dropped.
   Bar: both asserted, **>= 1** red when `taken_off` is ignored altogether.
8. The repro `test_a_bridge_identity_kept_by_an_alias_survives_the_rewrite` still passes; the
   unit tests of every migration sharing the step (m0012, m0025, m0029, m0030, m0031) pass
   unchanged: failures **0**; a changed expectation is listed with its reason and is a finding.

## 4. Loader bars

9. Default unchanged: 0034's verdicts file, loaded with the default set, parses to the same
   result as before the change: same rows, same per-verdict counts, same `list_sha256` check,
   same errors. Bar: a test (or recorded diff) shows **0** differences over its **546** rows.
10. Scope: a migration-supplied set is used only by the migration that passes it. A call without
    one still sees `BASES`, whatever another migration passed earlier in the same process; no
    module-level state is mutated (`BASES` is equal before and after a supplied-set call).
    Bar: a test calls with a supplied set, then without, and asserts the second uses the default.
11. One test shows, on the same file text: the default set **refuses** a `WRONG_PERSON` row with
    basis `undecidable` (`VerdictError`, message naming the allowed bases), and a supplied set
    that lists `undecidable` under `WRONG_PERSON` **allows** it. Bar: 1 refusal, 1 allowance,
    the same row.
12. A supplied set does not weaken the other checks: with it, an unknown verdict, a basis outside
    the supplied set, a missing `segment_idx` / `line_sha256` on any basis other than
    `undecidable`, and a wrong `list_sha256` are each still refused. Bar: **4 of 4** refused.
13. Mutants of the loader: (a) the supplied set ignored, (b) the supplied set replacing `BASES`
    for later calls, (c) the default also allowing `WRONG_PERSON` + `undecidable`. Each must turn
    a test red. Bar: **3 of 3** killed.

## 5. The test bar

14. The full upgrade test directories pass: `tests/unit/podcast_scraper/upgrade`,
    `tests/unit/upgrade` and `tests/integration/upgrade`, failures **0**, errors **0**, run once on
    the committed tree with the counts of passed tests shown (no skipped test that was not
    skipped at 9a7118eb4). `make ci-fast` once before landing, per method step 8.

## 6. Not covered

- The fix and the loader change were not read for correctness by this reviewer; the criteria are
  written from the record and the pre-change code. Whether the correction is minimal is the
  step-8 reviewer's call.
- Nothing has run: sections 2 to 5 are bars, not results. The 09:27 figures are quoted, not
  recomputed; the 14 and the 112 are quoted from the record's 10:08 block and not recounted.
- 0035 is the stated reason for the supplied set, but 0035's own use of it, and its verdicts, are
  not covered. Nothing here shows a migration other than 0034 passes a set today.
- Exposure of 0031, 0033 and 0035 to the alias defect: acceptance-C criteria 20 to 23 stand
  unchanged; the correction can change their exposure counts (an identity kept by its own id
  where 9a7118eb4 dropped it), and that re-measure is not covered here.
- Whether the 14 episodes' kept identities are right to keep beyond the own-id-is-still-a-person
  test (criterion 4) is not judged; the old code kept them and the record calls that right.
- Derived surfaces fed by a bridge identity (search-index rows, enrichment outputs) are not
  measured here.
- Prod's state after 2026-10-25 is not covered; no SSH to prod from here.
- The bars are the reviewer's choice, not measured: for the operator to accept before the
  re-measure runs. Acceptance-B's 80 of 89, 2 of 120 and 5 of 9 are untouched and not re-asserted.
