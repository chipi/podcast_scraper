# Loss ranking: step-5 acceptance criteria

Written by a fresh step-5 reviewer (Opus advisor), 2026-10-20T08:13 UTC, before any code, from its
own reads and scratch measurements of `lost/gateworkB6c` (snapshot-prod-20261016 + 0030 + 0031)
and the gold's `labels_v1`. Stored as written, reformatted; examples that quoted gold label names
are replaced by descriptions (gold-derived text is never committed).

<!-- markdownlint-disable MD029 -- the criterion numbers are identifiers the record cites -->

- **A0.** Step 4 is restated, dated, before any code, with the integers below; any difference from
  them is listed by insight id with its cause and goes back to step 3.

## Gold: wrong names

- **A1. Population:** every row the committed S7 code (`quality/gold_insights.score_case`,
  unchanged) puts in `S7_wrong`, on cases whose `meta_relpath` is served: 2,947 (380 / 2,128 /
  439), all shown. In `S7_wrong` the label and stored names are never empty (0 of 2,947); a label
  with no name stays `scored_no_name`, an empty or `SPEAKER_NN` stored name stays `S7_unnamed`, an
  S8 label, an unplaceable quote or no grounded quote stays in its S7 bucket.
- **A2. Wrong names outside `S7_wrong`,** reported as separate worklist rows: `name_without_support`
  (`scored_no_name` rows whose stored name is real) 81; `undecidable_named` (`S8_undecidable` with a
  real stored name) 887, a count only; `unplaceable_named` (`quote_not_placeable` with a real
  stored name) 31, a count only. All shown.
- **A3. Classes for `S7_wrong`,** exactly one, in this order:
  1. `binding`: another labelled voice in the same case has a label name whose `_fold` equals
     `_fold(stored)` (the owner). Any labelled voice with a name may own, whatever its confidence;
     rows where an owner's label is low-confidence or `role: unknown` are flagged
     `owner_label_weak` (32). Two or more owners (117 insights): all are used, the sub-class
     independent of their order. Matching is fold-exact; what `same_person` would add (83) is a
     sensitivity count, not a class.
  2. `spelling`: not binding, and, with `_fold` and `_drop_honorifics`, the stored name has no more
     tokens than the label and each stored token pairs with a different label token, equal or
     matching under `same_person`'s own `FUZZY_THRESHOLD`. No new thresholds. (A surname spelled
     differently in a way the matcher does not accept, and a first name with a greeting fused to
     it, both fail.)
  3. `other_person`: the rest. Flags: `shares_label_token` (a folded token shared with the label:
     101); `stored_on_another_voice_unnamed_label` (the stored name's roster entry holds a different
     voice whose label has no name: 22).
- **A4. Binding sub-classes.** A voice's published names are the folded real stored names on all
  its S7 rows, any bucket (`SPEAKER_NN` and empty are not names). `swapped`: an owner publishes the
  insight's own label name; else `duplicated`: an owner publishes the stored name; else
  `rotated/third_name`: an owner publishes another name; `rotated/owner_unnamed`: every owner has
  S7 rows and publishes no name; `rotated/owner_no_insight`: no owner has an S7 row. Roster-based
  owner names cannot be the definition (for 1,072 of 1,897 binding insights the roster lists no
  voice for the owner).
- **A5. Integers, ±0:** binding 1,897 (swapped 544, duplicated 201, rotated 1,152: third_name 346,
  owner_unnamed 652, owner_no_insight 154); spelling 52 (val_v1 only); other_person 998
  (shares_label_token 101). By set: dev binding 200, other 180; val_v1 binding 1,397, spelling 52,
  other 679; val_v2 binding 300, other 139.

## Corpus: hidden insights

- **A6. Population:** exactly the committed scoreboard's S1 Insight nodes (`surfaceable is False`,
  `speaker_voice_type == "unknown"`): 14,841. The voice by S7's join (first `SUPPORTED_BY` quote →
  `_voice_of_quote`); principal = at least 0.30 of `_talk_share`.
- **A7. Classes, in order:** population gaps apart, never "minor": `voice_not_found`,
  `no_diagnostics` (6). `nobody_named` (no roster entry has a name) 3,557. `unplaced_principal`
  5,407, of which `one_to_one` (exactly one unplaced named entry, exactly one unnamed principal
  voice, and the insight is on it) 3,247 in 85 episodes. `unplaced_minor` 1,269.
  `placed_not_recorded_principal` 3,293 and `_minor` 638 (no named entry carries a `placed` key, so
  the record cannot say who is unplaced). `none_unplaced_principal` 541 and `_minor` 130 (`placed`
  recorded on every named entry). All ±0. No episode mixes present and absent `placed` keys (0).
- **A8. Seat check on the gold,** served cases only, A7's one-to-one rule: 26 scored, 16 right
  (`same_person` or case-insensitive equal), 9 another person, 1 the label supports no name; an
  unnamed principal voice with no label entry counted apart, never wrong; insights on the seated
  voice reported per verdict.

## Conservation, rows, outputs

- **A9. Conservation asserted in code (non-zero exit on failure):** the classed gold insight ids
  equal S7's `S7_wrong` ids exactly, sub-classes sum to binding, no id twice; A2 equals S7's rows
  by id; corpus classes plus gaps equal the scoreboard's S1 by the instrument's own enumeration,
  checked against the scoreboard; `one_to_one` ⊆ `unplaced_principal`; nothing written under the
  corpus or gold (the test hashes them before and after).
- **A10. Row fields.** Gold: set, case, episode_id, insight, voice, hidden, label, stored,
  segment_label, class, sub-class, owners (voice ids), each owner's published names,
  owner_label_weak, shares_label_token, stored_on_another_voice_unnamed_label,
  later_quote_on_stored, the definitions' commit sha. Corpus: episode_id, metadata relpath,
  insight, voice (or gap reason), talk share, principal, placed_recorded (all / none / no entries),
  n_unplaced, unplaced roles, n_unnamed_principal, one_to_one, class. Seat check: set, case,
  served, candidate, label name, role, confidence, verdict, insights on the voice.
- **A11. Outputs:** rows (JSONL) and a summary JSON in a required `--out` directory; the script
  refuses an `--out` inside the git toplevel; the record carries counts and case or episode ids
  only, no label names.

## Held-out, reading, tests, gates

- **A12. Held-out:** no gold set is untouched (the exploration ran over all three). Before step 8
  the builder lists in the record every pair and episode it read while defining the rules (the top
  8 pairs per class and 4 examples per sub-class it printed). The step-8 reviewer draws a seeded
  sample of the rest (seed recorded) and reads it in the transcripts: gold 20 binding pairs, at
  least 3 per sub-class, including every pair with two or more owners; corpus 20 episodes stratified
  over A7, none in a gold set. Mechanical class errors 0; semantic findings counted and reported,
  rules unchanged. The definitions are frozen at the commit and re-applied unchanged to the 2.1
  gold; changing a rule after 2.1 needs a new record.
- **A13. The step-8 reviewer reads in full:** every distinct (label, stored) pair in spelling (2)
  and other_person (67) with one quote and its segment; all 108 binding pairs at the pair level;
  all 81 `name_without_support` rows by distinct pair; all 26 seat-check cases plus the 1 not-served
  case (confirm it is excluded); the 6 gap insights.
- **A14. Scenario tests** on real-shaped temp corpora (invented names): every class and sub-class and
  the precedence; `SPEAKER_NN` on an owner is unnamed; two owners order-independent; spelling
  boundaries (a first-name variant in; a surname the matcher rejects out; a greeting fused to a first
  name out but flagged; a titled name); a missing `placed` key → placed_not_recorded; voice not
  found and missing diagnostics → gaps; share exactly 0.30 is principal; one-to-one with the insight
  on a placed voice does not count; a not-served gold case excluded; a dropped insight fails
  conservation; an `--out` inside the repo refused; read-only hashes. The step-8 reviewer
  mutation-tests each definition; a surviving mutant fails.
- **A15. Gates on the committed tree:** `make ci-fast; echo EXIT=$?` and `make type`, log attached;
  then the committed command re-run on gateworkB6c reproduces A5, A7 and A8 exactly.

## The reviewer's findings on steps 3 and 4

1. Reproduced exactly: 2,947 (380 / 2,128 / 439), all shown; binding 1,897 = 1,152 + 544 + 201; S1
   14,841 (equal to the committed scoreboard); 5,407 / 1,275 / 3,834 / 768 / 3,557; one-to-one
   3,247 in 85 episodes.
2. "Misspellings and fragments are about 102" is not supported: 1 fragment plus 101 sharing a token,
   65 of which are a greeting fused to a first name; the exploration's thresholds were ad hoc.
3. "16 of 27" counts a case not served on the clone (no served check in the scratch script): 16 of 26.
4. "None unplaced" 3,834 / 768: 3,931 of those 4,602 sit where no roster entry carries `placed`;
   only 671 truly have nobody unplaced.
5. Six insights counted "unplaced + minor" have no voice (5 without diagnostics).
6. The scratch third-name / none split counted raw `SPEAKER_NN` as a name: 346 / 806, not 932 / 220.
7. The scratch sub-class used the first owner only; the top-level split did not move.
8. The 51 first-name forms are spelling under A3 (52 with one more), not 153.
9. Not verified by the reviewer: the two negative text signals; "the decision trace exists only
   since #2276"; the reading list against the tracker.
10. The first commit of the record held gold-derived names; removed before it was pushed.
11. Not traced: which 0031 build the clone carries (its S1n is 7; the method's S1n row quotes 2,871
    for a different 0031 run).
