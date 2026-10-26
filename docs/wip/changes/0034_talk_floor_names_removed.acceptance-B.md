Dated: 2026-10-26T08:32

# 0034 revision B: step-5 acceptance criteria (replaces the verdict rule of the 31)

<!-- markdownlint-disable MD029 -- the criterion numbers are identifiers the record cites -->

Written by a fresh step-5 reviewer (not the builder), after the step-8 NO-GO (record section 8)
and the operator's ruling of 2026-10-26: **keep the name unless the text shows someone else.**
Companion to `0034_talk_floor_names_removed.acceptance.md` (the original 31, "the 31").
Precedent for the shape: `0031_named_voices_shown.acceptance-B.md`. Written from the record's
sections 4d and 8, the original criteria and the method. The reviewer did not read the m0034
code or its tests, did not open the readers' output files (`reread/reader_*.tsv`), and recomputed
nothing on a corpus: every integer below that is not a bar comes from section 8 of the record.
Snapshot: `snapshot-prod-20261025`. Definitions of the 31 (voice key, talk share, K, W, Iw, Qw)
stand. K and W are now the sets under the revised verdicts.

## 1. What stands, what is replaced

Stands unchanged, text and bars as in the 31: 1, 4, 8, 9, 11, 14, 15, 16, 18, 19, 22, 26, 27, 28,
29, 30, 31.

Stands with its numbers restated from the revised verdicts, and its bar kept: 5 (rows = the 546
applied keys, closed verdict set, 0 unreviewed, 0 duplicate, 0 outside the list), 10 (names off =
exactly |W|; kept byte-unchanged = exactly 546 - |W|; episodes written = exactly the episodes
holding a W key; refused 0; the dated prediction is 4d's: off 166, kept 380, written 146, and a
different W count is restated by D15 before the clone is measured, never explained after), 12 to
13, 17, 20, 21, 23 (see section 4: they failed and stay as written, with the closing evidence
named).

Replaced: **6, 24, 25** (below). Added: **32, 33** (the two-reader reread, the carried nine).

Criterion 7 stands as written; its closing evidence is named in section 4.

## 2. Replaced criteria

6. **Each verdict carries a basis and the line that decides it, and the default is keep.**
   - The verdicts file gains a `basis` column (closed set below), a `segment_idx` column and a
     `line_sha256` column (sha256 of the deciding line's text, so the public file proves the
     evidence without carrying text). The header keeps migration id, `reviewed_commit`,
     `list_sha256`, reviewer and date.
   - `CORRECT` / `CORRECT_MISSPELLED` bases: `self_intro`, `addressed`, `host_read`,
     `continuity`, `gold`, `clip_own_person`, **`undecidable`**.
   - `WRONG_PERSON` bases: `another_person`, `ad_or_promo_not_own`, `clip_other_person`,
     `not_a_person`, `merged_voices`. Each needs a deciding line in which the text itself shows
     the voice is someone other than the listed name (it is addressed or introduced as another
     person, it reads another person's lines, it is a clip or ad of a stated other source, the
     label is not a person, or two people share the voice).
   - **The absence of evidence is never a basis for `WRONG_PERSON`.** "No evidence it is them",
     "never introduces themselves", "cannot confirm" are not removal reasons any more. A voice
     the reader cannot decide is `CORRECT` with basis `undecidable`, and the name is kept.
   - The loader refuses (named refusals, each a test that fails without the check): a row with no
     basis; a basis outside its verdict's set; `WRONG_PERSON` with basis `undecidable`; a row
     whose `segment_idx` / `line_sha256` is empty.
   - Pass bars, integers:
     - rows without a basis: **0** of 546;
     - `WRONG_PERSON` rows whose evidence text begins "no evidence", or states only an absence:
       **0**;
     - `WRONG_PERSON` rows without a stated line: **0**;
     - rows with `undecidable`: counted and listed by key (expected from 4d: about 25 to 39 of
       the 89 re-read; the count is reported, not fitted);
     - `segment_idx` and `line_sha256` in the shipped file equal the private evidence file
       (`...verdicts-with-evidence.tsv`, eval repo) for **546 of 546** rows.
   - The shipped file still carries no transcript text and no person's name beyond the listed
     name field it already has.

24. **All of K is read, and a wrong keep is a keep where the text shows someone else.**
    - K is now about 380 (4d). The next step-8 reviewer reads: every row whose verdict moved
      from `WRONG_PERSON` to `CORRECT` in the revision (all of them, including every
      `undecidable`), and the two rows it flagged on the previous pass
      (`1_2766128` SPEAKER_12, `519b7338` SPEAKER_03). The 341 earlier keeps stand on its earlier
      read, with those two as the only open items.
    - A wrong keep is a K row where the text shows the voice is someone other than the name:
      another person's lines, a character, a wrong first name. A K row the reader merely cannot
      confirm is not a wrong keep (that is `undecidable`, kept by the rule).
    - Pass bar: **0** wrong keeps. A wrong keep is corrected to `WRONG_PERSON` with a stated
      basis, the list sha and the verdicts are re-bound, and step 7 re-runs.
    - The two flagged rows are each settled in writing: kept with the argued basis, or flipped.

25. **The removal sample is read under the same rule; a removed right name is a removal where
    the text does not show someone else.**
    - W is the final W set after criterion 32 (expected 166). It is frozen and hashed (verdicts
      sha256 recorded) before the sample is drawn. The old sample (seeds 77 and 20261026) is
      void: the W set changed, and a stratum of the old set cannot be reused.
    - The reviewer records its seed before it opens the verdicts file; it must differ from 77 and
      20261026. Stratified by feed: at least 1 row per feed holding a W key, the rest
      proportional, total **120**, or all W rows if |W| <= 120. The 9 carried rows (criterion
      33) are excluded from the draw so no row counts twice.
    - Always read, in addition and outside the draw: every host-role row that ends in W; every
      row of 2(e) and 7; every row of W that came out of the 89 (criterion 32), in full, with the
      third reader's line where the readers split.
    - **Under the new rule an unreadable W row is a wrong removal**: the reviewer judges "does
      the text show someone else", and "cannot tell" counts against the removal. (In the NO-GO
      "cannot tell" counted for the removal; that is reversed.)
    - Tolerated: **at most 2** wrong removals in the 120 sample (the 31 allowed 3, under a rule
      where removal was the default; the removal now needs a positive line, so the bar tightens
      from 3 to 2). 3 or more sends the whole verdicts file to a full re-read by two fresh
      readers under criterion 32. Each wrong removal found is corrected to `CORRECT` (basis
      stated), the file is re-bound, and step 7 re-runs; the sample is then redrawn with a new
      seed over the new W, up to the same 120.
    - Wrong removals outside the draw (host-role, 2(e), 7, the 89-derived rows) are not
      tolerated: each is corrected and counted, and the count is reported. More than **5** across
      all of these also sends the file back.

## 3. Added criteria

32. **The reread of the 89, two readers, one settler.**
    - Population: the 89 `WRONG_PERSON` rows whose evidence began "no evidence it is them",
      identified by key from the private evidence file. The 116 other `WRONG_PERSON` rows keep
      their cited deciding line and are not re-read here (record 4d), except the carried nine
      (criterion 33). The reader packets are built by someone other than the readers.
    - Readers A and B: fresh agents; not the builder, not the previous step-8 reviewer, not any
      of the four agents who wrote the first verdicts, not the third reader. Each reads
      independently, from the transcript, blind to: the builder's verdict and evidence line, the
      step-8 reviewer's reading (its 50 W / 14 C / 25 U of the 89), the other reader, and the
      list's stratum labels. They apply criterion 6's rule and the closed sets, and quote the
      deciding line for every `WRONG_PERSON`.
    - **Agreement bar: A and B give the same verdict (`CORRECT`, undecidable included, versus
      `WRONG_PERSON`) on at least 80 of the 89 voices** (about 90 percent). Fewer than 80 means
      the operator's rule is not readable from the text; the work stops and goes to the operator
      with the disagreements, it is not patched by editing the rule or by a third reader.
    - Settling: every voice on which A and B differ (at most 9) goes to a third fresh reader C,
      blind to both readers' verdicts and quotes, to the builder and to the step-8 reviewer. The
      majority of A, B, C decides. A `WRONG_PERSON` that stands needs a quoted line showing
      someone else; if C's line is absent the row is `CORRECT` `undecidable`. The builder never
      settles and never edits a verdict.
    - Record the integers: agreed `WRONG_PERSON` (expected 50, 4d), agreed `CORRECT`, split,
      settled each way. Final W = 116 + W from the 89 - the carried flips (criterion 33). If it
      is not 166 the population is restated (D15) before the clone is re-measured.
    - Pass bar: every one of the 89 has a final verdict with a basis; the next step-8 reviewer
      reads every settled split and every 89-derived `WRONG_PERSON` itself (criterion 25).
      The previous step-8 reviewer's reading may be shown to it afterwards, never before.

33. **The carried nine.** Nine removals whose evidence is a real deciding line were read by the
    previous step-8 reviewer as right (1) or undecidable (8). They are not in the 89 and the
    builder does not re-judge them. The next step-8 reviewer:
    - reads each of the nine first, from the whole transcript, blind to the builder's verdict
      and to the previous reading;
    - records for each: stays `WRONG_PERSON` (text shows someone else, with the line) or flips
      to `CORRECT` (right, or undecidable);
    - flips the ones it finds not shown, in the verdicts file, re-binds the sha, and the clone
      gate re-runs;
    - bar: all 9 read and each settled in writing. **If 5 or more of the 9 flip**, the cited
      lines of the 116 cannot be trusted as a class: the 116 go through criterion 32's two
      readers too. A flip count of 4 or fewer is reported and the 116 stay.
    - The nine, by episode id and voice:
      1. `tag:soundcloud,2010:tracks/2135080263` SPEAKER_02
      2. `f004446b-a024-4773-bd04-b497017ec344` SPEAKER_03
      3. `d867d354-76e2-11f1-84e8-ff42fe6fab7b` SPEAKER_00
      4. `0e910a81-d26f-4e7f-852a-b4ec0142c272` SPEAKER_03
      5. `urn:bbc:podcast:m002c70n` SPEAKER_07
      6. `0308f04c-c356-11f1-835c-fbffaff6b369` SPEAKER_02
      7. `eae61d33-b7f8-4944-b34c-fe24e145f603` SPEAKER_06
      8. `730396fd-79a1-4bb7-b008-ea64e75d4fa9` SPEAKER_02 (the previous reviewer's "right"; the
         gold labels also call this voice a right person, so it is expected to flip)
      9. `https://iono.fm/e/1728203` SPEAKER_08

## 4. The failed criteria, and the evidence that closes each

Each closes only on the stated artefact, produced by the builder, and re-derived by the next
step-8 reviewer with its own code (D1). "Restated" means the integer is re-derived on the final
W, expected near 146 episodes.

2. **Not reached, as integers.** Evidence: a "Not reached" section in the record with five
   integers, each from a stored script with its snapshot id:
   - (a) named voices at or above the floor, and under-floor voices already unnamed (section 8.1
     found 5,456 and 5,565; restate);
   - (b) placeholder-labelled under-floor voices (8.1: 0);
   - (c) the 14 post-snapshot episodes: the count, and a statement that their floor check is
     criterion 27 (prod, read-only), not claimed here;
   - (d) the D4 upper bound: (episode, removed name) pairs in final W where, after the apply, no
     voice carries that name. Script path and output committed;
   - (e) voices whose name another voice also carries (8.1: 42, 25 of them in W; restate), and
     the `ef0fb53c-56a7-11f1-823b-6b194b39cee8` case listed.
   Bar: five integers present and equal to the next reviewer's own.

3. **The list carries numbers.** Evidence: each of the 546 entries holds the voice's talk seconds
   and the episode's total talk seconds as numbers (a second recomputation of the share from the
   entry alone). Command: `jq` over the list counting entries with a non-numeric seconds field:
   **0** of 546. The list's sha256 is equal in three places (verdicts header `list_sha256`,
   the migration digest `list_file`, the receipts header): `sha256sum` output and the three
   strings pasted equal. The edited-entry refusal test fails when the check is removed (a mutant
   that drops it is killed).

6. See section 2. Closes on the `basis` / `segment_idx` / `line_sha256` columns, the loader
   tests, and the integers 0 / 0 / 0 / 546 of 546 above.

7. **The 92 labelled names against the verdicts.** Evidence: a committed script (path stated)
   and its output file that joins the 92 labels of the 10-17 measurement to the final verdicts,
   for those still served and in the list (8.1 found 87 listed; the 5 not listed are named with
   cause), prints every disagreement, and shows each settled. The three voices proved right by
   the gold labels (`715ff48a` SPEAKER_01, `730396fd` SPEAKER_02, `e32aa079` SPEAKER_06) are
   disagreements until they are `CORRECT`. Bar: **0** disagreements left unexplained, and 0 of
   those three still `WRONG_PERSON` (a gold-named right person removed is not an explanation).

12. **The insights traded, with lists.** Evidence:
   - a gate-report file listing insight ids for Iw_vis, Iw_unid, Iw_unk, Iw_com and Iw_none, on
     the final W, produced by the S7 raw-voice join;
   - the reconciliation of the S1 delta to the id-level diff: the delta equals the moved ids;
     the difference between |Iw_vis| and the delta is listed id by id with its cause (the
     shared-name guard, an insight with no `speaker` field, no placeable quote). On the NO-GO
     clone this was 94 against 87: bar **0** unexplained ids;
   - the commercial guard: code that does not retype a `commercial` W voice, and a unit test
     with a commercial-typed W voice that fails when the guard is removed. Bar: on the clone,
     Iw_com moved to `unknown` = **0**; the guard mutant is killed;
   - the upper-bound integers (shown, hidden-by-type insights and quotes on all 556 voices)
     stated in the record before the verdicts are final;
   - the prediction 4d: S1 and S2 each rise by 50 to 57, S10a by 0 to 2; a value outside is
     restated, not passed.

13. **People who lose their last voice.** Evidence: a list file by (episode_id, person id) of
    every GI Person, KG person node and bridge identity whose only voice was a final W voice,
    each with one stated rule (removed, or kept because other insights mention the person or
    another voice speaks as them). The NO-GO found residue to classify: 25 GI Person nodes (12
    kept by mentions, 12 spoken by other voices' quotes, 1 other), 31 bridge identities, 9 KG
    labels, 1 unplaced roster entry. Restate on the final W; bar: every item in one class,
    **0** unclassified, and the roster / `context.json` facts lost listed.

17. (Not asked to restate, but it closes with the revision.) S7-correct falls on 0 gold-named
    voices and S6 = 0: the three gold voices of criterion 7 are `CORRECT`; `gold_insights.py
    --snapshot snapshot-prod-20261025`, before and after, gives the same S7-correct totals
    (11,498 and 3,706 in the NO-GO's two sets, dev_v1 unchanged) and the reviewer's own S6
    script (path stated) prints 0.

20. **Derived-surface census.** Evidence: the output of `python
    scripts/measure/derived_surface_census.py --corpus <after> --receipts <0034 receipts>`
    committed, with every surface ticked: written, unaffected (with a grep for each removed
    name in that surface showing **0** occurrences), or stale with its ladder rung. The NO-GO
    found stale on 180 episodes: `.insight_density`, `.insight_sentiment`, `.manifest`; also
    `.anon.txt` (37), `.context.json` (30), `.adfree.admap` (6) and the search-index rows.
    Bar: **0** surfaces stale without a rung or an argued, grepped reason; the search-index rows
    counted by episode (the repair itself is criterion 28, prod).

21. **Undo on the full clone.** Evidence: the log of a fresh copy of the after-clone showing
    restored N and refused **0**, where N is criterion 18's integer on the final W (the NO-GO
    clone had 2,137 content files in 180 episodes plus the ledger; restate for about 146
    episodes); `sha256` census of the copy equal to snapshot-prod-20261025 (**0** differing
    files); `upgrade verify` for 0034 reads pending; the scoreboard JSON equals the before JSON
    field for field; apply, undo, apply, undo ends refused 0 and identical; backups only under
    `.podcast_scraper/upgrade-backups/0034/`.

23. **Mutations.** Evidence: the next step-8 reviewer's own re-run of the 42 mutants of 8.2
    against the whole `tests/unit/podcast_scraper/upgrade/` directory, plus new mutants for this
    revision (verdict mapping that removes on `undecidable`; a loader that accepts a
    `WRONG_PERSON` without a basis; a `CORRECT` row without a basis accepted). Bar: **0**
    survivors, except at most **2** equivalent mutants each argued in writing (the unsorted
    `write_list` and the negative-span clamp on this data are the only ones argued so far). The
    other 14 survivors each need a test that kills them: the `<` versus `<=` boundary with a
    voice at exactly 5%; the 0.06 floor; the denominator (named voices only; unattributed
    segments counted); served name from the last segment; `CORRECT_MISSPELLED` that removes;
    the receipts `verdicts_sha256` header; `verify` ignoring refused episodes; the detector over
    every copy; no `.txt` guard; the backup tag; the shared-name refusal; `speaker_voice_type`
    not set `unknown`; the commercial guard.

Partial in the NO-GO, closing on the same kind of evidence at the next review, bars as in the 31:
4 (the other migrations' `verify` lines compared line for line), 11 (the removed name on the
GI / KG / bridge / roster residue, by class), 15 (S11 measured), 16 (S4, S5, S8, S9 reported), 19
(the metadata GI / KG count recount), 22 (a `commercial` W voice path with a clone instance or a
test).

## Not covered

What this acceptance list does not cover, or covers only by statement:

- Prod: criteria 27 to 29 (plan diff, the 14 post-snapshot episodes, search-index rows, apply
  counts) cannot be evidenced on the clone. Prod access here is HTTP, not SSH; filesystem work
  on prod is a handover.
- The 116 removals with a cited deciding line are read by nobody new unless 5 of the carried 9
  flip (33). A wrong removal among the other 107 is found only if it falls in the 120 sample or
  the always-read sets.
- The 8 episodes of the 10 dropped entries (the shared-label refusal): their names stay served.
  The previous reading calls 4 of the 10 wrong persons, so 4 wrong names remain on disk. This
  revision does not reach them; they are an open item, not a pass.
- The D4 shape of #2094: removing a wrong name does not give it to the right voice; those voices
  stay unnamed. Criterion 2(d) counts them; nothing here fixes them.
- 58 K voices carry a misspelled or garbled form of the name; only 6 are marked
  `CORRECT_MISSPELLED`. A garbled or wrong first name is a wrong keep only if the reader reaches
  it (24); there is no sweep of the 58.
- The 341 earlier keeps are accepted on the previous reading except the two flagged rows; a
  wrong keep among them that the previous reviewer missed is not searched for.
- Under "keep unless the text shows someone else", more wrong names stay served than before.
  This acceptance bounds wrong keeps by reading (24), it does not measure the rate of wrong keeps
  among `undecidable`, which is unknown.
- Whether two readers agreeing on 80 of 89 is a high enough bar is a judgement, not a measured
  quantity; there is no kappa and no held-out check of the rule on voices outside this list.
- Reprocess durability: a kept guest under the floor loses the name on a later reprocess, because
  the pipeline floor exempts hosts only. Served and new episodes still follow different rules.
- T3, the introduction-cue rule, is not part of this change and nothing here measures it.
- S11, S4, S5, S8, S9 on prod: only reported on the clone (criterion 15 and 16 cross-checks).
- Per-language rows: all values on this snapshot were `en` at the earlier review; nothing here
  re-checks them.
- The 3,504 metadata files against 2,966 served episodes were not reconciled by anyone.
- Two served episodes that name an absent transcript file: the detector cannot reach them.
- The `MISSED` order (acceptance written after the build) was accepted by the operator in 4c; the
  same shape recurs here, since this revision was written after the NO-GO, and it needs the
  operator's line if the method's template requires one for a second miss.
- The decision trace (history) keeps the removed names; no criterion removes them.
- The clone's 28 no-op rewrites and the 299 un-normalised receipt paths are cosmetic and not
  gated.
- I did not read the migration code, the reader outputs, or any transcript. I did not recompute
  any share, count, or hash; every corpus integer here is a section 8 figure to be re-derived.
- The sha256, `jq` and script paths in section 4 are what the evidence must be; none of them
  exists yet in the record, and whether the census script accepts the receipts as written is
  unchecked.
- The 5 of 9 flip trigger in 33 and the 2-of-120 and 5-outside bars in 25 are my integer choices,
  not derived from a measurement; the operator may move them before step 8.
