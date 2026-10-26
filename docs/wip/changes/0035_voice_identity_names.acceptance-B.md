Dated: 2026-10-26T09:21

# 0035 revision B: step-5 acceptance criteria (replaces the verdict rule and the left-rename rule)

<!-- markdownlint-disable MD029 -- the criterion numbers are identifiers the record cites -->

Written by a fresh step-5 reviewer (not the builder), after the step-8 verdict GO WITH CONDITIONS
(record section 8) and the operator's decisions of 2026-10-26 (record 4d). Precedent for the
shape: `0034_talk_floor_names_removed.acceptance-B.md` (the same rule, applied there to removals).
Written from the method, the record's sections 4b, 6, 7 and 8, 0034's revision B and the
builder's text of 4d. The reviewer did not read the m0035 code or tests, did not open any
reader output or transcript, and recomputed nothing on a corpus: every corpus integer below
that is not a bar is a section 8 or 4d figure, to be re-derived by the next step-8 reviewer
with its own code. No transcript text and no person's name appears here; voices are named by
episode id and voice label only.
Snapshot: `snapshot-prod-20261025`, with the revised 0034 (275e76cc6) applied first.

## 1. The revision, in the operator's terms

- (a) An UNNAMING applies only when the text shows the voice is someone else. An unnaming
  with no evidence is kept (the name stays).
- (b) A RENAME applies only with positive evidence in the text (unchanged from 4b; now enforced
  by a basis).
- (c) The 78 left renames and the 4 unnamings applied on "no evidence" (82 voices) are re-read
  blind by two fresh readers; splits by a third.
- (d) `Buzzsprout-20005821` SPEAKER_00 is dropped from the list: the name is right, the role
  taken from the feed's group is wrong (it reads as a standing-in host, the entry would write
  guest), and a review cannot correct a role.
- (e) 0035's verdicts gain `basis`, `segment_idx`, `line_sha256` as 0034's did, and 0035's
  loader requires them.
- (f) 0035 is re-measured on `snapshot-prod-20261025`, with the revised 0034 applied first.

## 2. What stands, what is replaced

Step-8 conditions (record section 8):

- **Condition 1 (landable record) stands.** Section 4 gets its `Dated:` line with integers; the
  integers are now 4d's (below), not the +196 / -93 / -226 of the first clone, which are
  superseded and kept only as history. Sections 6 and 7 are rewritten for the revised run.
  Stale text removed (the "not in the registry yet" text). The outcome built with
  `build_outcome.py` and its sha256 named in section 8. `make ci-fast` on the committed tree,
  log attached, the landing test green: **0** failing tests, **1** run.
- **Condition 2 (the 23 surviving mutants) stands, bar tightened to 0034's:** 0 survivors,
  except at most **2** equivalent mutants each argued in writing (the `<=` at the floor is the
  only one argued so far). New mutants for this revision are added in criterion 36.
- **Condition 3 (the operator sees the table first) stands, narrowed.** The operator sees the
  4d prediction against the measured table before anything is applied. Of its three rulings:
  the role of `Buzzsprout-20005821` SPEAKER_00 is settled by (d); the 11 voices the step-8
  reviewer would apply, and the partner renames behind the 12 insights shown under a wrong
  name, are no longer ruled by the operator one by one: they are in the 78 and are decided by
  the readers (criterion 34). The operator rules only on what the readers' agreement bar sends
  back.
- **Condition 4 (deciding line in the shipped rows; refuted vs unconfirmed split) is replaced**
  by criterion 32: `basis`, `segment_idx`, `line_sha256` on every row. The `undecidable:` vs
  `refuted:` split of the 74 + 4 "no evidence" rows is carried by `basis` (`undecidable` vs a
  showing basis).

Step-8 criteria K1 to K9:

- **K1 (population) is replaced** by criterion 31 (162, one named exclusion).
- **K2 (coverage) is replaced** by criterion 32.
- **K3 (own outcome numbers) is replaced:** same method (own tree diff, voices whose segment
  label / role move equal the applied keys exactly, 0 outside), new integers (criterion 35).
- **K4 (gating numbers) stands**, with 4d's integers as the bars (criterion 35).
- **K5 (applied changes are right) stands and widens:** every applied change is read, rename
  needs positive identity evidence, unnaming needs the text to show another person. Bar: **0**
  wrong applied (criterion 37).
- **K6 (left changes) is replaced** by the two-reader reread (criterion 34); the next reviewer's
  own left-rename sample is criterion 38.
- **K7 (mutation) stands**, see condition 2 and criterion 36.
- **K8 (safety of the apply) stands and is no longer "not verified":** criterion 39.
- **K9 (landing) stands:** condition 1.

## 3. Criteria

31. **The population.**
    - The detector on the snapshot (pre-0034) returns 163 keys; the frozen list holds **162**
      (138 renames, 24 unnamings). The only difference between detector and list is
      `Buzzsprout-20005821` SPEAKER_00, and the exclusion is a named, tested rule in the code (a
      mutant that removes it is killed: criterion 36), not a hand edit of the list file. The
      list's sha256 is equal in three places (verdicts header `list_sha256`, the migration
      digest `list_file`, the receipts header).
    - No voice shared with 0034's list: **0** voices (episodes sharing entries are counted and
      listed; the 10 of record 4b are restated on the revised 0034).
    - The detector on the clone that has the revised 0034 applied is re-run and reported: its
      key count, and the keys it adds or drops against the frozen 162 listed one by one with
      cause (the 3 extra keys of the first run, in one feed, are 0036 material and stay out of
      the list). Bar: **0** entries of the 162 refused on that clone; any entry that is a no-op
      because 0034's revision changed the served name it relies on is listed by key (the count
      is reported, not fitted).

32. **Each verdict carries a basis and the line that decides it.**
    - The verdicts file gains `basis`, `segment_idx` and `line_sha256` (sha256 of the deciding
      line's text, so the public file proves the evidence without carrying text), as 0034's file
      did. The header keeps migration id, `reviewed_commit`, `list_sha256`, reviewer and date.
      The loader refuses a row with an empty `segment_idx` or `line_sha256`. For a row with basis
      `undecidable` the line is the one the reader examined most closely, and the basis says
      that no line decides.
    - **Renames.** `CORRECT` / `CORRECT_MISSPELLED` (the name is added) need a positive basis:
      `self_intro`, `addressed`, `host_read`, `continuity`, `gold`, `clip_own_person`. The
      loader refuses a rename `CORRECT` or `CORRECT_MISSPELLED` with basis `undecidable`
      (no positive evidence, no rename). `WRONG_PERSON` on a rename means the name is not added
      and **may** carry `undecidable`, or `another_person`, `ad_or_promo_not_own`,
      `clip_other_person`, `not_a_person`, `merged_voices`.
    - **Unnamings.** `WRONG_PERSON` (the name comes off) needs a showing basis:
      `another_person`, `ad_or_promo_not_own`, `clip_other_person`, `not_a_person`,
      `merged_voices`, with a deciding line in which the text shows the voice is someone other
      than the name. The loader refuses an unnaming `WRONG_PERSON` with basis `undecidable`, and
      refuses one whose basis is outside that set. `CORRECT` on an unnaming (the name stays)
      carries a basis from the positive set or `undecidable`; an unnaming with `CORRECT` is
      never applied.
    - The loader's named refusals are each a test that fails without the check: no basis; basis
      outside its verdict's set (renames and unnamings tested separately); rename `CORRECT` with
      `undecidable`; unnaming `WRONG_PERSON` with `undecidable`; empty `segment_idx`; empty
      `line_sha256`.
    - **Absence of evidence is never a basis to take a name off.** A `WRONG_PERSON` unnaming
      whose evidence text begins "no evidence" or states only an absence is refused as an
      evidence cell; **0** such rows.
    - Pass bars, integers: rows without a basis **0** of 162; rows with a basis outside its
      set **0**; unnamings applied without a stated line **0**; `segment_idx` and `line_sha256`
      in the shipped file equal the private evidence file (eval repo,
      `0035_voice_identity_names.verdicts-with-evidence.tsv`) for **162 of 162** rows; and, from
      the snapshot's own segment files, sha256 of the segment text at `segment_idx` equals
      `line_sha256` for **162 of 162** rows (a committed script, its output committed). The
      `undecidable` rows are counted and listed by key; the count is reported, not fitted.
    - The 80 carried rows (60 applied renames, 20 applied unnamings) were written without these
      columns. Their `segment_idx` and `line_sha256` are located from the private deciding
      quote, and their `basis` is assigned by a fresh agent (not the builder, not a reader of
      criterion 34), who reads the quote against the transcript; a carried row it cannot give a
      positive (rename) or showing (unnaming) basis is reported as a defect, not defaulted.
    - The shipped file still carries no transcript text and no person's name beyond the name
      field it already has.

33. **Kept stays kept, and a removal needs a showing line.** This criterion is the semantics
    of 32 on the apply path, tested on a clone, not only in the loader:
    - an unnaming `CORRECT` (basis `undecidable` included) writes **0** changes to that voice:
      segments, roster, diagnostics byte-identical to the before-clone;
    - a rename `WRONG_PERSON` writes **0** changes to that voice;
    - the receipts and the plan name the kept unnamings and the left renames by key, so a kept
      or left voice is visible rather than silently absent.

34. **The reread of the 82, two readers, one settler.**
    - Population: the 78 left renames and the 4 unnamings applied with "no evidence it is them",
      identified by key from the private evidence file: **82** voices. The other applied rows
      (60 renames, 20 unnamings) keep their cited line and are not re-read here (read by the
      next step-8 reviewer, criterion 37). The reader packets are built by someone other than
      the readers; a packet carries no fingerprint score (a fingerprint is not text evidence).
    - Readers A and B: fresh agents; not the builder, not the step-8 reviewer, not any of the
      agents who wrote the first verdicts, not the third reader. Each reads independently, from
      the whole transcript (not only the packet), blind to: the builder's verdict and evidence
      line, the step-8 reviewer's reading (its 11 of 48), the other reader, and the list's
      stratum labels. They apply criterion 32's rule and closed sets, and quote the deciding
      line for every rename they would add and every unnaming they would apply.
    - **Agreement bar: A and B give the same apply decision (for a rename, add vs do not add;
      for an unnaming, remove vs keep) on at least 74 of the 82 voices** (about 90 percent; 0034
      used 80 of 89, 89.9 percent). Fewer than 74 means the operator's rule is not readable from
      the text; work stops and goes to the operator with the disagreements. It is not patched
      by editing the rule or by a third reader.
    - Settling: every voice on which A and B differ (at most 8) goes to a third fresh reader C,
      blind to both readers' verdicts and quotes, to the builder and to the step-8 reviewer. The
      majority of A, B, C decides. A rename that is added, or an unnaming that is applied, needs
      a quoted line; if C's line is absent the row is "not added" (rename) or `CORRECT`
      `undecidable` (unnaming). The builder never settles and never edits a verdict.
    - Record the integers: agreed add / agreed not add (renames); agreed remove / agreed keep
      (unnamings); split; settled each way. The previous reading of the 4 unnamings (3 read
      undecidable by the step-8 reviewer, 12 insights on them) is shown to a reader only
      afterwards.
    - Pass bar: every one of the 82 has a final verdict with a basis; the next step-8 reviewer
      reads every settled split (criterion 37).

35. **The outcome numbers, 4d, as integers.** All measured on `snapshot-prod-20261025` with the
    revised 0034 applied first, by `scripts/measure/corpus_change_gate.py`, on clean code.
    4d's prediction is dated before any reader reported (2026-10-26T09:20) and stands as the
    dated prediction; a value outside a range is restated by D15 before the clone is measured,
    never explained after.
    - list **162**; unnamings in the list 24; renames in the list 138;
    - renames applied **71 to 76** (60 carried + 11 to 16 of the 78);
    - unnamings applied **20 to 21** (the 24 minus the 3 or 4 kept without a showing line);
    - voices rewritten = renames applied + unnamings applied (**91 to 97**); roster entries
      unnamed = unnamings applied; episodes written = exactly the episodes holding an applied
      voice (no integer was predicted for episodes; the builder states one before the clone,
      and it is restated, not fitted); refused **0**;
    - voices whose segment label / role differ between the 0034-only clone and the 0035 clone =
      exactly the applied keys: **0** extra, **0** missing;
    - S1 rises by **+150 to +184**; S2 **-105 to -93**; S3 **-260 to -226**;
    - must not move: S1c **0**, S12 **24**, S10b **2**, commercial **94**, A1 **382** hits in
      **290** episodes; S1n does not rise. Insights and quotes counts unchanged. S10a is
      reported; its 0035 delta is expected within **-2 to +2** and each episode that moves is
      named by id (my integer, not 4d's);
    - the S1 move is reconciled to insight ids: the shown-to-hidden-`unknown` set, the
      hidden-`unknown`-to-shown set and the hidden-`unidentified`-to-shown set, with their sizes
      (on the first run 403 / 207 / 289), the delta equals the moved ids, and **0** ids are
      unexplained. Insights hidden by an applied unnaming sit only on voices where the text
      shows someone else (criterion 37 reads each of those voices).

36. **Mutations.**
    - The next step-8 reviewer re-runs its 43 mutants of record 8 (K7) against the whole
      `tests/unit/podcast_scraper/upgrade/` directory, plus new mutants for this revision:
      unnaming applied on `CORRECT`; unnaming applied on basis `undecidable`; rename applied on
      basis `undecidable`; loader that accepts a rename `CORRECT` or an unnaming `WRONG_PERSON`
      with `undecidable`; loader that accepts an empty `segment_idx` or `line_sha256`; a row
      with no basis accepted; the dropped entry's exclusion removed; the kept unnaming still
      rewriting its roster.
    - Bar: **0** survivors, except at most **2** equivalent mutants each argued in writing. The
      tests the 23 survivors of the first run named are written (the detector's unnaming
      branch; `_reviewed` for an unnaming; `CORRECT_MISSPELLED`; a respelling pair that
      `same_person` accepts; `BACKUP_TAG` and `RECEIPTS_FILE` distinct from 0034's; the role
      fallback; the rung constants).

37. **What the next step-8 reviewer reads.** Its own code, its own seed (recorded before it
    opens the verdicts file; it must differ from 8035 and from every seed of 0034's review),
    blind to the builder's verdicts until it has read.
    - **A sample of applied renames:** every rename newly applied from the 78 (11 to 16, in
      full, with the third reader's line where the readers split); the 9 carried renames the
      step-8 reviewer rated thin (`05598a71` SPEAKER_01, `1_2766234` SPEAKER_02, `20ebeb5b`
      SPEAKER_01, `41ca1334` SPEAKER_01, `4ae8ea8d` SPEAKER_00, `4e743da6` SPEAKER_01,
      `75b67139` SPEAKER_01, `a353187e` SPEAKER_01, `iono.fm/e/1729071` SPEAKER_05), each
      settled in writing; and a seeded draw of **20** from the other 51 carried renames. The 9
      thin rows have no name in the text, only a fingerprint plus a solo or host-read context;
      under (b) each needs a positive basis (`host_read` or `continuity`) with a quoted line, or
      it is flipped to `WRONG_PERSON` `undecidable` (the name is not added), the file re-bound
      and the clone re-run. That flips the 4d count (60 kept) and is restated by D15, not
      explained. Every applied rename that changes a role is read (11 guest to host on the
      first run); **0** may go host to guest.
    - **Every applied unnaming** (20 to 21), in full: the text shows the voice is someone other
      than the name, with the line. The step-8 reviewer read 21 of 24 as such and 3 as
      undecidable (`ce8e9a37` SPEAKER_04, `substack:post:220234130` SPEAKER_01,
      `youtube:MWX36ZYnsm0` SPEAKER_00): those 3 must be among the kept, or a showing line
      is quoted for them.
    - **Every kept unnaming** (3 to 4): the reviewer reads whether the text shows someone else.
    - **Every split** of criterion 34 (at most 8).
    - **The dropped entry** (`Buzzsprout-20005821` SPEAKER_00): the role defect is shown by the
      text; the key is absent from the list, verdicts, plan and receipts; its segments, roster
      and diagnostics are byte-identical to the 0034-only clone; its partner SPEAKER_01 (the
      applied unnaming, 34 insights) is separately confirmed as an unnaming with a showing line.
    - Pass bars: **0** wrong applied renames (the text shows someone else, or no positive
      identity line exists); **0** wrong applied unnamings (an unnaming where the text does not
      show someone else); **0** wrong keeps among the kept unnamings (text shows someone else).
      A wrong one is corrected, the file re-bound, the clone re-run. One wrong rename in the
      20-draw sends the other 51 carried renames to two fresh readers under criterion 34's
      method (my integer).

38. **The left renames, sampled.** After the readers' flips the renames still left are a
    population of 62 to 67. The next reviewer reads a seeded draw of **30**, whole transcript,
    and counts each voice where it would add the name with a positive line. A missed rename is
    a hidden or wrongly named voice that stays as it is, not a wrong name; it is counted, not
    gated, up to **3** of 30 (about 10 percent, against the first run's 11 of 48). More than 3
    sends the left renames to a third full read under criterion 34's method.

39. **Safety of the apply.** Evidence from the clone with the revised 0034 applied first:
    - **Plan:** per-episode rows with counts and input sha256s; `plan_sha256` recorded. Prod's
      dry run must reproduce it; differences are allowed only on episodes 0034 also writes,
      listed by id.
    - **Derived-surface census:** `scripts/measure/derived_surface_census.py --corpus <after>
      --receipts <0035 receipts>` output committed. Every surface ticked: written, unaffected
      (with a grep for each name removed by an applied unnaming and for each old name replaced
      by a rename in that surface, showing **0** occurrences), or stale with its ladder rung.
      The first clone wrote 991 files in 72 episodes (metadata, gi, kg, bridge, diagnostics,
      segments, turns, text, cleaned, context, admap) and left each episode's
      `insight_density` / `insight_sentiment` sidecars and the search-index rows stale; restate
      on the final apply set. Bar: **0** surfaces stale without a rung or an argued, grepped
      reason; the search-index rows counted by episode (the repair itself is a prod step).
    - **Undo cycle:** on a fresh copy of the after-clone: apply, undo, apply, undo ends refused
      **0** and identical; after the undo of 0035 alone the copy equals the 0034-only clone
      (sha256 census, **0** differing files), which differs from the snapshot on the 10-odd
      episodes both migrations write; after undoing 0035 then 0034 the copy equals
      `snapshot-prod-20261025` (**0** differing files). `upgrade verify` for 0035 reads pending;
      the scoreboard JSON equals the before JSON field for field; backups only under
      `.podcast_scraper/upgrade-backups/0035/`.
    - **Ledger:** the first run's undo left the migration's entry in `upgrade_ledger.json`. Either
      the undo removes it (with a test that fails without it) or the record states the
      two-step prod undo (`undo`, then remove the ledger entry, then `upgrade verify`) and that
      step is run and shown on the clone. Bar: ledger equal to the before ledger after the
      cycle, or the written procedure demonstrated to make it so.
    - **Outcome:** `upgrade/outcomes/0035_voice_identity_names.json` built with
      `build_outcome.py`, its sha256 named in section 8; the outcome table (4d prediction vs
      measured, criterion 35) and the revised-0034 sha are in the record.

40. **The gold and labelled voices.** The 92 labels of the 10-17 measurement are joined to the
    final 0035 verdicts for the keys in the list (a committed script, its output committed).
    Bar: **0** disagreements left unexplained; **0** applied unnamings on a voice whose gold
    label says the name is right; S6 = 0 from the reviewer's own script, and
    `gold_insights.py --snapshot snapshot-prod-20261025` S7-correct before and after 0035 are
    equal or the difference is listed by insight id with its voice and basis.

41. **Reporting order.** The outcome table (criterion 35) is the first thing in the record's
    section 7, before mechanics, and the verdict states the outcome (voices named, voices
    unnamed, insights shown and hidden, voices left) before the passes.

## Not covered

What this acceptance list does not cover, or covers only by statement:

- Prod: the plan diff, apply counts, the 14-odd post-snapshot episodes and the search-index
  repair cannot be evidenced on the clone. Prod access here is HTTP, not SSH; filesystem work
  on prod is a handover (step 10).
- The 80 carried rows (60 renames, 20 unnamings) keep their original reading. Of those, only
  the 9 thin renames, the role-changing renames, the 20-draw and every applied unnaming are
  read again; a wrong carried rename among the other 31 is found only if the 20-draw hits it.
- A wrong rename that the text can neither confirm nor refute, applied on a positive-looking
  but wrong line (a name said about someone else), is bounded only by the reader's judgement,
  not measured. The rate of wrong names among `undecidable` is unknown.
- Under "keep unless the text shows someone else", more wrong names stay served than before for
  unnamings. This acceptance bounds wrong keeps by reading the kept unnamings (3 to 4); the
  wrong names that stay on the voices the rung never listed are outside the list's reach.
- The dropped entry's wrong role stays on prod: this revision does not correct a role, and the
  partner voice's separately applied unnaming does not give the right name to the dropped
  voice's partner. The role cannot be fixed by a review; it needs another rung or a hand edit
  and neither is claimed here.
- The 12 insights shown under a name the step-8 reviewer read as wrong, on partner voices of
  applied unnamings (11 swap pairs), are fixed only where the readers add the partner
  rename. The count that stays is reported (criterion 38 sample), not removed.
- The 3 extra keys the detector finds once 0034 is applied (one feed's two-label renames) are
  0036 material; nothing here lists, reads or applies them.
- The 38 of the 78 left renames the step-8 reviewer did not read are now read by two readers
  (34); the packets those readers see are built by a person who may mis-pick which segments to
  show, and the reader searching the whole transcript depends on the reader's diligence, not
  on a check.
- Whether 74 of 82 is a high enough bar is a judgement, not a measured quantity; there is no
  kappa and no held-out check of the rule on voices outside this list.
- The integers 20 (draw), 51 trigger, 30 (draw), 3 of 30, 8035-differs and the S10a range are
  my choices, not derived from a measurement; the operator may move them before step 8.
- Whole-corpus S4, S5, S8, S9, S11 are not required by this list; the gate measured none of
  them on the first run and a later reviewer is not asked here to.
- The basis for the 80 carried rows is assigned afterwards by a fresh agent reading the old
  quote; a carried row whose quote was thin gets a thin basis, and `segment_idx` may point at
  a line that is a near neighbour of what the first reviewer meant. The sha equality with the
  snapshot (32) proves the line is the line, not that it is the deciding one.
- For `undecidable` rows the stored line is "the closest line", not evidence; the column does
  not make that row auditable beyond the basis label.
- Reprocess durability: a re-run of the pipeline can rename or unname these voices again, as
  the pipeline's voice identity rung and the migration's reading differ.
- Per-language rows: nothing here re-checks language; the first snapshot was all `en`.
- Mutants outside the upgrade test directory (the rung itself in the pipeline) are not run.
- T3, the introduction-cue rule, is not part of this change and nothing here measures it.
- The decision trace (history) keeps the names this migration removes; no criterion removes
  them.
- The `MISSED` order for 0035 (acceptance after the build) was accepted by the operator in 4b;
  this revision is written after a verdict and a second miss of the same shape needs the
  operator's line if the method's template requires one.
- I did not read the migration code, the verdicts, the reader outputs or any transcript, and I
  recomputed no share, count or hash; every corpus integer not marked "bar" or "my integer" is
  a section 8 or 4d figure to be re-derived by the next reviewer.
- Whether `derived_surface_census.py` accepts 0035's receipts as written, and whether the
  segment indexing for `segment_idx` is the one 0034's file uses, are unchecked.
