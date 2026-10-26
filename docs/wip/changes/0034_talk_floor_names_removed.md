---
id: 0034_talk_floor_names_removed
class: full
status: built, not applied
issue: none (the operator opens issues; naming arc, 2026-10-23)
touches:
  - src/podcast_scraper/upgrade/migrations/m0034_talk_floor_names_removed.py
  - pyproject.toml (package data upgrade/lists/*.json)
  - src/podcast_scraper/upgrade/reviewer_verdicts.py   # 4d: basis / segment_idx / line_sha256
  - src/podcast_scraper/upgrade/migrations/m0025_one_person_one_entry.py   # 4d: commercial guard
  - src/podcast_scraper/upgrade/rewrite_bridges_m0007.py   # D1: apply_to_payload returns a copy
---

# 0034 — the talk floor on served episodes: a name off a voice that barely speaks

Method: [CORPUS_CHANGE_METHOD.md](../../guides/CORPUS_CHANGE_METHOD.md). Plan:
podcast-scraper-eval-data `docs/wip/NAMING-GATES-2026-10-22.md`, the coding plan, item C7. A wrong
name is worse than a hidden insight (operator, 2026-10-19).

## 1. Intake

- **The pipeline change.** New episodes get the talk floor (`speaker_name_min_talk_share` 0.05 on
  `prod_dgx_full`, naming ladder doc v26).
- **Why served episodes need a migration.** They were written before the floor, and 88% of
  two-voice episodes were named by older code that the pipeline will not re-run. Text only: a
  voice's talk share comes from its segments on disk.
- **T3 is not part of this change.** The introduction-cue rule was measured beside the floor, but
  the operator decided on 2026-10-23 to ship the floor only. On the runs the corpus serves, T3's
  name extraction put phrases such as "York Times" on voices: 23 fires on the measured splits, 12
  right.

### 1b. Reading list (added 2026-10-26; the step-5 acceptance found section 1 had none)

- #2094 (open): D4 "tiny-cluster name theft" in 29 episodes: the cause behind this change. 0034
  takes the name off the tiny cluster; it does NOT give it back to the right voice.
- Naming ladder v40 / v53: the pipeline floor's host exemption. New episodes keep a short host
  voice by that rule; served episodes keep it by a `CORRECT` verdict here. The two differ.
- Merge plan `NAMING-BRANCH-MERGE-PLAN-2026-10-24.md`, step 5: the floor's measured costs (5 right
  host names off; self-introduced guests off; a guest exemption measured and not built: 3 right /
  6 wrong over the stored runs).
- #2364 (Patreon shout-outs published as guests), #2367 (rosters wrong beyond a missing name),
  #2105 (a guest published as the host), #2333 (cross-episode voice identity for merged / split
  clusters), #2294 (transcript text still names removed speakers), #2359 (clip drift with dynamic
  ads), #2334 (naming coverage repair and a nightly guardrail), #2200 (insights hidden because a
  voice went unnamed), #1897 (talk left unattributed), #2096 (the attribution epic).
- Unsourced in section 2 until re-derived: the "88%" and the 10-17 split (92 / 84 / 1 / 6 / 1)
  name no script or snapshot. The 10-25 baseline below does.

## 2. Outcome and baseline

- **Baseline, snapshot-prod-20261025** (`scripts/measure/corpus_scoreboard.py --corpus
  <snap1025> --label snapshot-prod-20261025`, 2,966 episodes, 130,401 insights): S1 16,810;
  S1n 7; S1c 0; S2 30,563 (unknown 16,810 / unidentified 13,659 / commercial 94); S3 9,359;
  S10a 190; S10b 2; S12 24; A1 382 hits in 290 episodes.

- **The outcome:** no name is served on a voice holding under 5% of its episode's talk, unless a
  review reads the name as that voice's own.
- **Baseline, 10-17 snapshot copy (`lost/gateworkB6c`, read only).**
  - `find_changes` lists 510 voices in 357 served episodes.
  - On the four labelled segments, 92 names are under the floor:
    - 84 bad: 81 ad / promo / clip, including 4 host-read ads, and 3 spurious;
    - 1 wrong;
    - 6 right and 1 role error. These are short host intros / outros and fragments; none is an
      ad read.

  Source: eval repo NAMING-RETHINK, "Talk floor 5%: host-read ads".
- **Prod:** not counted yet. The list is made on a fresh prod snapshot (section 5).

## 3. Diagnosis

- **The detector reproduces the measured rule on the labelled episodes.** Of the 92 measured
  names, 87 are listed.
- **The 5 not listed:** their labelled run is no longer the run the corpus serves; the detector
  reads the served one.
- **The 21 listed beyond the measurement:** voices the labels do not score (low-confidence labels,
  mostly clips, promos and unknowns).

## 4. Design

- **A frozen, reviewed list.**
  - `find_changes` runs the floor over a corpus copy.
  - `write_list` freezes it in `upgrade/lists/0034_talk_floor_names_removed.json`.
  - The migration applies that list and nothing else.
- **The review.** The reviewer reads each listed voice against the name the floor would remove,
  using the verdicts' own meaning:
  - `WRONG_PERSON` (the name is not this voice's) takes it off;
  - `CORRECT` / `CORRECT_MISSPELLED` keeps it, for example a host's short intro;
  - a listed voice without a verdict fails `verify`, and without a verdicts file nothing is
    written.
- **The rewrite.** It goes through 0029's voice-by-voice rewrite with 0031's derived surfaces:
  transcript prefixes and offsets, turns, diagnostics, context, ad maps, metadata counts, GI, KG
  and bridge. It is all or none per episode, and an episode whose stored name or role moved on is
  refused untouched.
- **The list is bound.** The verdicts header's `list_sha256` is the hash of the WHOLE list
  (every field of every entry), the list file is part of the migration's digest (`list_file`),
  and the receipts header records the list's hash. An entry edited after the review refuses the
  load.
- **Placeholder labels are no names.** A served label "SPEAKER_01", "Speaker", "Speaker 2" or
  "Unknown Speaker" (a transcription provider's shape for an unnamed voice) counts as unnamed, so
  the floor never lists one for review. On the 10-17 copy none was served: the list is unchanged
  (510 voices in 357 episodes).
- **No verdicts file:** `plan` prints one refusal line; `apply` raises and the upgrade run stops
  there with nothing written (the runner's contract, as 0031's).
- **Undo:** from `.podcast_scraper/upgrade-backups/0034/`, while each file is still as written.
- **Not in the registry yet.** An empty list with no verdicts file would fail `verify` on every
  upgrade run.
- **Tests** (`tests/unit/podcast_scraper/upgrade/test_m0034_talk_floor_names_removed.py`):
  - the floor rule;
  - list freeze and load;
  - the detector on a corpus;
  - apply, verify and undo, with the name off every surface the fixture writes (metadata,
    segments, transcript, turns, diagnostics, GI persons and quotes, KG);
  - an entry edited after the review refuses the load;
  - plan without a verdicts file is a refusal line;
  - the list file is part of the digest;
  - a placeholder label is no name to take off;
  - a `CORRECT` verdict keeping the name;
  - no verdicts file;
  - a moved-on voice refused;
  - not registered.

## 4b. Prediction on snapshot-prod-20261025 (dated 2026-10-26, before any verdict)

`MISSED:` the method puts the prediction (step 4) and the reviewer's acceptance (step 5) before
the build; 0034 was built first (2026-10-23). They are written now, before the review and the
clone measurement, for the operator to accept or reject.

- **Population:** `--find` on snapshot-prod-20261025 lists **556 voices in 397 episodes**
  (10-17 copy: 510 in 357). The 14 episodes night 1 ingested after the snapshot are written by
  the pipeline's own floor and are not expected on prod's list.
- **Own numbers, extrapolated from the labelled measurement** (92 under-floor names on the four
  labelled segments: 11 are the voice's own person: 6 right, 1 role error, 4 host-read ads, the
  rest not):
  - **kept by a `CORRECT` / `CORRECT_MISSPELLED` verdict: 66**;
  - **taken off by `WRONG_PERSON`: 490**.
- **Gating numbers on the clone after apply:** S1c stays 0; S12 does not rise (24); S1 / S2 rise
  only by the visible insights of the 490 voices whose names come off (the trade the operator's
  rule accepts: a wrong name is worse than a hidden insight), each listed by identity in the gate
  report; nothing outside the listed voices moves.

## 4c. The verdicts against the prediction: a miss, restated (2026-10-26)

- **Verdicts:** 556 of 556, four fresh reviewers (Opus), `scratchpad/m0034/verdicts_[1-4].tsv`:
  CORRECT 337, CORRECT_MISSPELLED 6, WRONG_PERSON 213. **Kept 343, taken off 213**, against the
  predicted 66 / 490.
- **Why it missed:** the prediction scaled the labelled segments' 11 / 92 to prod. Prod's
  under-floor voices are mostly hosts' own short intros, outros and self-introduced fragments:
  of the 343 kept, 318 have the served name in their deciding quote (63 of them ad-like), 24 only
  elsewhere in the voice's turns (8 ad-like), 1 nowhere in its text.
- **Restated before the clone measurement** (method D15): taken off 213, kept 343, on this
  snapshot, subject to the operator's two rule decisions below; S1 / S2 rise only by the visible
  insights of the 213, each listed by id; every other §3 number as in 4b.
- **For the operator:** (a) a host reading an ad keeps the host's name (71 kept voices are
  ad-like); (b) a correctly named clip of a public figure is kept by "is this voice that
  person". Either rule, reversed, flips those voices to taken off.
- **The population moved (D15), restated 2026-10-26 after the first clone gate:** 8 episodes (10
  entries: 8 `WRONG_PERSON`, 2 `CORRECT`) were refused by the shared rewrite, "transcripts: not a
  render, and a rewritten label is shared" (the #2294 shape: rewriting the label would change
  another voice's lines). They are taken off the list, so 0034 does not reach them: **546
  entries, 205 taken off, 341 kept**; the 8 wrong names there stay served (not reached, counted).
  The dropped entries are in the eval repo beside the full-evidence verdicts.
- **Evidence in the public repo:** the shipped verdicts file carries every row, verdict and name
  but no transcript text; the reviewers' deciding quotes are in podcast-scraper-eval-data
  `data/eval/migrations/0034_talk_floor_names_removed.verdicts-with-evidence.tsv` (private).
- **First clone gate** (before the drop, on snapshot-prod-20261025): rewrote 180 episodes,
  refused 8; S1 +87, S2 +87 (the insights of the taken-off voices, `insights_reattributed=87`),
  S10a +2. Re-measured on the committed list below.
- **Not yet done:** the step-8 reviewer reading every kept voice and a sample of 120 removals
  (acceptance F); the clone measurement on the committed list and the undo (step 7).

**Operator decisions (2026-10-26):**
- Operator accepted: the MISSED order (prediction and acceptance written after the build).
- A host reading an ad keeps the host's name ("keep").
- A correctly named clip of a public figure keeps the name ("keep").

## 4d. Revision after the step-8 NO-GO: the undecidable default reversed

Dated: 2026-10-26T08:28

- **Why:** the step-8 reviewer (section 8) found 8 right names in the 120-removal sample (bar 3),
  15 of 205 overall. 14 of the 15 are among the 89 removals whose only evidence was the default
  "no evidence it is them". **Operator, 2026-10-26: keep the name unless the text shows someone
  else.** A voice nothing decides is now kept (`CORRECT`, basis `undecidable`), not removed.
- **The change:** the 89 are re-read blind by two fresh readers under the new rule
  (`scratchpad/m0034/reread/`); the 116 removals that cite a deciding line are not re-read here.
  9 of those 116 the step-8 reviewer read as right (1) or undecidable (8): they go to the next
  step-8 reviewer, listed, not re-judged by the builder.
- **Prediction, written before either reader reports** (basis: the step-8 reviewer's own reading
  of the 89: 50 W, 14 C/Cw, 25 U):
  - population unchanged: 546 listed, 556 found, 10 dropped as before;
  - **taken off 166, kept 380**; episodes written **146** (was 180);
  - S1 and S2 rise by **+50 to +57** (57 = the visible insights on the 166 voices; on the 205,
    94 sat on the voices and 87 moved, the rest staying shown under a name another voice keeps);
  - S10a **+0 to +2** (both episodes of the +2 keep at least one removal);
  - must not move: S1c 0, S1n 7, S3 9,359, S10b 2, S12 24 (same ids), commercial 94, A1 382 / 290;
  - if the readers' W count differs from 166, the population is restated (D15) before the clone
    is measured again, not explained after.

Dated: 2026-10-26 (written before 09:03Z, the time of commit 275e76cc6 that carries it; corrected: this line first read "T09:25", a stamp I wrote without reading the clock)

- **The re-read:** readers A and B, blind to every earlier reading and to each other, agree on
  keep / remove for **85 of 89** (bar 80). The 4 splits (A keep, B remove) went to a fresh third
  reader, blind to both: all 4 removed (a house promo, a vox-pop shopper, a clip voice, a promo
  voice; in each the named person is shown speaking as another voice). Majority of three: **30
  removed, 59 kept**. Files: `scratchpad/m0034/reread/reader_{A,B}.tsv`, `third/reader_C.tsv`,
  `final_89.json`.
- **MISSED: the 4d prediction (166 removed).** The step-8 reviewer's reading of the 89 had 50
  removals; the three readers keep 20 of them. Restated before the clone is measured (D15):
  - **taken off 146, kept 400** (546 listed); episodes written **131**;
  - visible insights on the 146 voices: **59**; S1 and S2 rise by **+52 to +59** (on the 205, 7
    of the 94 stayed shown under a name another voice keeps);
  - S10a **+0 to +2** (both episodes still have a removal); every must-not-move number as above.
- **Bases for every row** (acceptance-B criterion 6): two fresh classifiers gave each of the 546
  rows a basis from the closed set without changing a verdict. On the 116 quoted removals one of
  them flagged **17 removals as absence-only** (no line shows someone else) and 1 keep as
  showing someone else. Those 18 were re-read blind by two more fresh readers (agreement **16 of
  18**; 101 of 107 over both re-reads), the 2 splits by a fresh third reader: **11 removed, 7
  kept**. Files: `scratchpad/m0034/basis/`, `reread2/`.

Dated: 2026-10-26 (written before 09:03Z, commit 275e76cc6; corrected: this line first read "T10:05", a stamp I wrote without reading the clock)

- **Restated again before the clone (D15):** **taken off 140, kept 406**; episodes written
  **126**; visible insights on the 140 voices **55**, so S1 and S2 rise by **+48 to +55**; S10a
  **+0 to +2**; must-not-move as above. Every one of the 140 lies inside the first 205, so no voice
  loses a name that round one kept.
- **Found while reading, not 0034's to fix:** in 2 of the 4 splits the named person's real voice
  is at or above the floor under another or no name (one carries a different person's name).
  0034 does not reach voices at or above the floor; these are listed as NOT COVERED in step 8.

**Clone measurement on db37f8b1c** (clean tree, `0034_talk_floor_names_removed.gate.json`,
measured 2026-10-26T09:27): 126 episodes, 140 voices, verify OK; S1 / S2 **+56** against the
restated +48 to +55: **MISSED by 1**, explained by insight id (`scripts/measure/m0034_evidence.py`,
criterion 12): 58 insights sit on the 140 voices, 56 go shown -> hidden `unknown`, 2 stay shown
under a name another voice keeps; the restatement used the round-1 reviewer's per-voice join,
which could not place 3 insights whose first quote starts inside a "Name: " prefix. S10a +1.
Must-not-move all equal.

Dated: 2026-10-26T09:50

- **A defect found by the census of that clone, in the shared rewrite (m0025):** the bridge step
  drops a `person:` identity whose id is in neither GI nor KG, without reading its aliases. One
  identity on the clone had an older unresolved id with the person's GI / KG id only as an alias;
  it was dropped although GI and KG keep the person (apply message `bridge_identities_removed=99`,
  98 of them the removed voices' persons). Repro test first:
  `test_a_bridge_identity_kept_by_an_alias_survives_the_rewrite` (fails on db37f8b1c).
- **The fix:** an identity is kept when its id OR any alias is a GI / KG person id; its sources
  are set from whichever matches.
- **Prediction for the re-measure on the fix** (before it runs): every number of the 09:27
  measurement unchanged (126 episodes, 140 voices, S1 / S2 +56, S10a +1, 1,499 files written, the
  same files) except `bridge_identities_removed` **98** (was 99) and that one episode's bridge,
  which keeps the identity. The same code serves 0031, 0033 and 0035: their exposure is measured
  from the snapshot's backups and reported in step 8.

**Re-measure on 9a7118eb4** (`gate_0034_r4`, clean tree): every gating number, both scoreboards,
the 1,499 files, the per-episode deltas and verify equal the 09:27 measurement, but
`bridge_identities_removed` is **112**, not the predicted 98: **MISSED**. Cause (the builder's):
the fix removes an identity whose id is a person the rewrite took off KG even when that id is
still a GI person (a person the KG rule drops while GI keeps it, e.g. through mentions), in 14
episodes; the old code kept those, rightly.

Dated: 2026-10-26T10:08

- **Correction:** an identity is kept when its OWN id is still a GI / KG person; when only an
  alias matches, it is kept unless its own id is a person the rewrite took off every voice.
- **Also in this revision** (for 0035, no change to 0034's behaviour): the verdicts loader takes
  an optional per-migration basis set; 0034 reads with the default set as now.
- **Prediction for the re-measure** (before it runs): every number of the 09:27 measurement
  unchanged, the same 1,499 files; `bridge_identities_removed` **98**; against 9a7118eb4 exactly
  14 episodes' bridges differ (they keep the identities the old code kept) and 1 differs against
  db37f8b1c (it keeps the alias-matched identity).

Dated: 2026-10-26T11:29

- **A defect in shared code, found by 0036's tests (D1):** `rewrite_bridges_m0007.apply_to_payload`
  returns the SAME dict it was given when the id map is empty; m0017's `_rewrite_bridge` (run by
  m0025's rewrite, so by 0034 and 0035) then does `bridge.clear(); bridge.update(out)`, which
  empties the bridge to `{}`. This is how 0025 left the 15 empty bridges prod serves. Repro test
  first (`test_the_shared_rewrite_never_writes_a_bridge_without_its_identities[renamed]`, red).
- **The fix:** `apply_to_payload` returns a copy, as `rewrite_ids` does (its contract: "in a copy").
- **Prediction for the re-measure** (before it runs): on the clones 0034 and 0035 wrote no new empty
  bridge (empty bridge files: snapshot 17 incl. superseded runs, after 0034 16, after 0035 16), so
  every number and every file of the last measurement is unchanged; only the code digest moves.

Dated: 2026-10-26T11:57

- **Round 3 (after the round-2 NO-GO and the operator's two rulings):** 43 voices re-read blind by
  two fresh readers, splits by a third (`scratchpad/m0034/reread3/`): the 7 wrong keeps the
  reviewer confirmed, the 10 moderate rows, all 14 voices reading the a16z disclaimer, and the 16
  mixed-voice rows the reviewer listed. Agreement bar: 39 of 43 (80 of 89 scaled).
- **Prediction, before either reader reports:** of the 43, 8 are removals today; after the re-read
  **14 to 29** (the 7 confirmed, the 1 other undecidable disclaimer reader, 0 to 3 of the 3
  `host_read` disclaimer readers, 0 to 10 of the moderate rows, less 0 to 2 mixed voices flipping
  to keep); removals in all **146 to 161**. S1 / S2 / S10a are restated as integers from the
  readers' counts BEFORE the clone is measured (the step the 0035 round skipped).

Dated: 2026-10-26T12:08

- **Round 3 result:** readers A and B agree on 40 of 43 (bar 39); the 3 splits to a fresh third
  reader, all removed. Final in the set: 17 removals (was 8): 10 keeps become removals, 1 removal
  becomes a keep (a mixed voice whose majority is the named person). Files:
  `scratchpad/m0034/reread3/` (`final_43.json`, `flips.json`).
- **Restated before the clone (the integers the 11:57 block promised):** removals **149**, kept
  **397** (546); episodes written **126 to 131**; S1 and S2 rise by **+49 to +69** (+56 measured
  for the 140; +3 from the 3 flipped voices whose insights the round-1 count covers; 0 to +10 from
  the 5 never counted; 0 to -10 for the voice turning back to a keep); S10a **+0 to +3**; S1c 0,
  S1n 7, S3 9,359, S10b 2, S12 24, commercial 94, A1 382 / 290 unchanged.

Dated: 2026-10-26T14:17

- **Operator rulings after the round-3 NO-GO (2026-10-26):**
  Operator accepted: 2026-10-26: a voice whose lines are only a sponsor tag or a house promo is an
  announcer (the name comes off) unless the text ties it to the named person; a host voice that
  also reads a disclaimer or synopsis keeps the name when most of its talk is the host's; and the
  MISSED episode count (135 against 126 to 131) is accepted.
- **Round 4:** the 20 KEPT voices the round-3 reviewer named or a sponsor / promo text scan flags
  (`scratchpad/m0034/reread4/`), re-read blind by two fresh readers under all three rulings,
  splits by a third; bar 18 of 20.
- **Prediction, before either reader reports:** of the 20, **3 to 12** become removals; removals
  in all **152 to 161**. Integers for episodes, S1 / S2 / S10a restated from the readers' counts
  before the clone.

Dated: 2026-10-26T14:23

- **Round 4 result:** readers A and B agree on 19 of 20 (bar 18); the 1 split (a mixed voice, 3 of
  27 segments the named person's, 16 someone else's) removed by a fresh third reader. 6 keeps
  become removals (4 sponsor-tag voices, 1 house-promo voice, the mixed voice); 14 stay kept
  (self-introductions inside a promo, an addressed caller, the round-3 reviewer's arguable caller).
- **Restated before the clone:** removals **155**, kept **391**; episodes written **138** (the 6
  sit in 5 episodes, 3 of them not yet written; computed from the r6 plan); S1 and S2 **+59 to
  +65**; S10a **+1 to +2**; every must-not-move number unchanged; empty bridges written 0.

Dated: 2026-10-26T15:29

- **Round 5, after the round-4 NO-GO (8d):** the cause is the builder's: the three operator rulings
  were applied to rows a scan or a reviewer flagged, not to every kept row, and one contested row
  (`2ede0e96` 01) was never re-read because the builder's list named `2ede0e96` 00. Now ALL 391
  kept voices are re-read blind under the three rulings, in 4 batches of 97-98, each by two fresh
  readers (8 readers), splits by a third; each reader states the named / other / backchannel split
  for every voice. Bar per batch: 88 of 98 (80 of 89 scaled).
- **Prediction, before any reader reports:** removals **162 to 185** (155 now; the round-4 reviewer
  confirmed 7 and found 2 probable in about 160 rows it read and 2 of 40 in a fresh sample, so
  about 5% of the 231 unread rows); kept **361 to 384**. Episodes, S1 / S2 / S10a restated from the
  readers' counts before the clone.

Dated: 2026-10-26T15:47

- **Round 5 result:** all 391 kept voices re-read blind. Keep / remove agreement per batch: 95 of
  98, 96 of 98, 95 of 98, 92 of 97 (bars 88, 88, 88, 87); 13 splits to a fresh third reader, who
  measured each mixed voice in seconds of talk (the builder's instruction, since 0034 measures talk
  seconds; segment counts and seconds disagree on 4 of the 13): 7 removed, 6 kept. 9 removals both
  readers agreed on. Files: `scratchpad/m0034/reread5/` (`final_r5.json`, `third/`).
- **One row decided against the readers' majority (operator to confirm):** `e7d4a370` 02. The voice is only a sponsor read and a house promo; all three readers found no
  line tying it to the name (reader C: "Name not tied by a line"), yet two kept it as a host read.
  The third ruling removes such a voice unless a line ties it; the deciding line is another voice's
  (seg 36; private evidence file). Written as `WRONG_PERSON` `ad_or_promo_not_own`.
- **Spellings:** readers marked 10 to 17 rows per reader `CORRECT_MISSPELLED` from their own
  knowledge of the person, not from the text; written as `CORRECT` (keep / remove unchanged).
- **Deciding segments:** every non-undecidable row's segment resolved to its list position in the
  served segments and checked against the reader's quoted line (`audit_idx.py`); readers wrote
  timestamps (2), 1-based positions (94) or list positions; 2 anchors set by hand (`0ec506f0`
  08: seg 253, his voice finishing a student's sentence; `60217010` 06: seg 1211, the segment that names the voice).
- **Restated before the clone:** removals **172** (155 + 17), kept **374**: inside the 15:29 range
  (162 to 185, 361 to 384). Episodes written **150** (the 17 sit in 17 episodes, 12 not written
  before; from the verdicts file, not a plan). S1 and S2 **+59 to +72** (+59 measured for the
  155; the r7 rate is 0.38 insights per removed voice, about +6 for 17, with a margin both ways
  because these are short promo and clip voices); S10a **+1 to +3**; every must-not-move number
  unchanged; empty bridges written 0.

Dated: 2026-10-26T17:21

- **Round 6, after the round-5 NO-GO (8e):** the cause is the builder's: the readers saw one
  episode at a time, and two of the reviewer's findings rest on a second episode of the same show.
  A deterministic scan (`scratchpad/m0034/recur.py`, `recur2.py`) found, for each kept voice, its
  lines of 40+ characters that recur word for word in another served episode of the feed: 151 of
  374 kept voices; 45 with the lines under a different served name elsewhere, 38 of them
  self-introductions inside recurring promos (tied by the text, kept by the third ruling). The 7
  without a self-introduction tie that the scan or the reviewer named were re-read by a fresh reader
  with the sibling episodes (`reread6/`): `iono.fm/e/1728203` 01 removed (the sibling `1728106`
  carries the same line as the interviewer's, who introduces the analyst by name);
  `CTf886...` 03 removed and its sibling `Qv7FB...` 00 confirmed removed (the show's produced
  presenter intro / outro, read by other named people elsewhere in the feed; no line ties either to
  the name); the three a16z host voices kept (first-person host talk at least half);
  `856dd2e4` 01 removed, confirmed by a second fresh reader (`reread6e/`): a cold-open news clip
  is 20.9 of its 32.5 s, the sign-off is read by different presenters in sibling episodes.
- **Not covered by the scan:** a produced spot whose wording differs from episode to episode (the
  `CTf886` presenter recurs by role, not by text: 8 of 83 s recur verbatim), and recurrences across
  feeds. The 38 promo self-introductions were not re-read this round.
- **Restated before the clone:** removals **175** (172 + 3), kept **371**; episodes written
  **152** (2 of the 3 in episodes not written before); S1 and S2 **+67 to +72**; S10a **+1 to
  +2**; every must-not-move number unchanged; bridges written as `{}` 0, as `{"identities": []}` 2
  (unchanged, 0036 rebuilds them). 0035 on the new base: every integer equal to its r5 run except
  possibly the shared episode `iono.fm/e/1728203` (0034 now takes the name off voice 01 there,
  0035 names voice 05); 0036 equal.

Dated: 2026-10-26T18:36

- **Round 6's conditions (8f), the builder's answers:**
  1. `038e11c4` 02: **kept**, basis `undecidable`. The voice only asks audience questions; nothing in
     the text shows it is someone else, and the operator's first rule keeps a name the text does not
     refute ("probably not the named person" is not text showing someone else).
     `d59a87e0` 09: **kept**, basis `addressed`. Two readers read the voice as the person introduced
     just before it performs; the deciding line names the person's role in the band, which does not
     show that another voice sings. Neither row flips.
  2. Ids only: `acceptance.md` lines 152 and 275 and this record's 15:47 block now carry no person
     name, no quoted and no paraphrased transcript line; a scan of every line this branch adds under
     `docs/` against every served name in the 0034 and 0035 verdicts files (`scratchpad/m0034/
     namescan.py`) finds 0 in the `.md` files (it also found one in the 0035 record, fixed). The gate
     reports' file paths carry public episode titles, as earlier migrations' do.
  3. The verdicts header (both files) now names round 6: `reviewed_commit` c453fb215, the reviewer
     line with the counts 175 / 371. The file's sha moves, so the chain is re-measured.
- **Prediction for the re-measure** (before it runs): every integer of 7e and of 0035's and 0036's
  last runs unchanged, plans with 0 differences; only the verdicts sha (and so the code digest)
  moves.

## 5. Still to do before apply

Dated: 2026-10-26T08:32

Revised acceptance for the 4d revision, by a fresh reviewer (not the builder):
[0034_talk_floor_names_removed.acceptance-B.md](0034_talk_floor_names_removed.acceptance-B.md).
It replaces criteria 6, 24 and 25 of the first acceptance, adds 32 (two blind readers) and 33 (the
nine carried removals), and names the evidence that closes 2, 3, 7, 12, 13, 17, 20, 21 and 23.
Its integer bars (80 of 89, 2 of 120, 5 outside the draw, 5 of 9) are the reviewer's choice, not
measured: for the operator to accept.

Operator accepted: 2026-10-26 (acceptance-B's integer bars; the two code changes it needs: the
verdicts file's `basis` / `segment_idx` / `line_sha256` columns, and the guard that keeps a
`commercial` insight or quote `commercial` when its voice loses a name).

Dated: 2026-10-26T09:50

Acceptance addendum for the bridge-alias fix (4d, 09:50), by a fresh reviewer:
[0034_talk_floor_names_removed.acceptance-C.md](0034_talk_floor_names_removed.acceptance-C.md):
the re-measure equals the 09:27 one on every integer except `bridge_identities_removed` 99 -> 98
and one episode's bridge; 5 of 5 fix mutants killed; the repro fails before and passes after. Its
bars are the reviewer's choice: for the operator to accept.

Dated: 2026-10-26T10:09

Acceptance addendum for the 10:08 correction (the bridge rule and the loader's per-migration
basis set), by a fresh reviewer:
[0034_talk_floor_names_removed.acceptance-D.md](0034_talk_floor_names_removed.acceptance-D.md).

Operator accepted: 2026-10-26 (acceptance-C's and acceptance-D's bars; the recorded misses: the
166 prediction, S1 +56 against +48 to +55, bridge_identities_removed 112 against 98 on 9a7118eb4).

One deviation from acceptance-B criterion 6, by the builder, for the next step-8 reviewer to
judge: a row with basis `undecidable` has no deciding line by definition, so the loader allows an
empty `segment_idx` / `line_sha256` on that basis only and refuses it on every other.

1. Make the list on a fresh prod snapshot copy (`--find <copy>`).
2. Do the full read audit: read every listed voice's stretch, and judge the downstream LLM impact
   by identity.
3. Have a fresh reviewer write the verdicts file.
4. Produce the clone outcome table: before and after on the snapshot copy, every surface.
5. Register the migration, with its version.
6. Get the operator's recorded go.
7. Apply with the corpus-migration skill.

Dated: 2026-10-26T11:31

Fresh-reviewer acceptance addendum for the 11:29 defect fix(es), integer bars before the code fix: [0034_talk_floor_names_removed.acceptance-E.md](0034_talk_floor_names_removed.acceptance-E.md).

## 6. Build

- Commits (branch `feat/m0034-list`, local): f9f228038 (list, verdicts, registration),
  275e76cc6 (the 4d revision: verdicts re-read under the operator's rule with basis columns; the
  commercial-type guard; talk seconds in the list entries; 0034's 42 step-8 mutants all caught),
  db37f8b1c (dated stamps corrected), 9a7118eb4 (bridge-alias fix; evidence script), a1698c214
  (the fix corrected; per-load basis sets), 2080d1cb2 (0035; 0034's final gate report).
- Tests: `tests/unit/podcast_scraper/upgrade/`, 603 passed on 2080d1cb2; the 9 acceptance-C/D
  mutants of the bridge rule and the loader all caught (`scratchpad/m0034/mut_fix2.py`).
- Evidence script (committed): `scripts/measure/m0034_evidence.py` (criteria 2, 7, 12, 13 by id;
  its output holds names and stays in the private eval repo).

## 7. Measure

Final clone measurement, a1698c214, clean tree, `0034_talk_floor_names_removed.gate.json`
(measured 2026-10-26, snapshot-prod-20261025):

- **Apply:** rewrote **126** episodes, refused 0; **140** voices unnamed; verify OK;
  `bridge_identities_removed` **98** (prediction 10:08: 98, met); against db37f8b1c exactly one
  bridge differs, against 9a7118eb4 the 14 the correction restores.
- **Scoreboard, before -> after:** S1 +56, S2 +56 (all `unknown`), S10a +1; S1c 0, S1n 7, S3
  9,359, S10b 2, S12 24, commercial 94, A1 382 / 290, insights 130,401, quotes 252,651 unchanged.
- **Insights by id** (criterion 12, `m0034_evidence.py`): 58 on the 140 voices; 56 shown -> hidden
  `unknown`; 2 stay shown under a name another voice keeps; 0 moved outside the removed voices;
  Iw_unid, Iw_unk, Iw_com 0.
- **People who lose their last voice** (criterion 13): 275 person entries; bridge 98 removed /
  17 kept, GI 30 / 15, KG 112 / 3; kept by a mention (8), by another voice (7), by KG edges (3).
- **Not reached** (criterion 2): 5,461 named voices at or above the floor; 5,565 under-floor voices
  already unnamed; 5,496 placeholder labels; (d) 72 of the 140 removals in 67 episodes where a
  person's real voice at or above the floor is served unnamed (a heuristic, over-counting; listed
  by key in the evidence file); (e) 17 removals whose name a voice at or above the floor also holds.
- **Labels (criterion 7):** 84 of the 92 labelled names join; 61 disagree (59 labelled ad / promo /
  clip and kept, 48 of them on a self-introduction; 2 labelled host-read ads and removed). All 61
  are listed for the step-8 reviewer; none is settled by the builder.
- **Files written: 1,499 changed** in the 126 episodes (receipts 1,518: 19 rewritten with identical
  bytes, 14 KG, 3 GI, 2 bridge).
- **Derived surfaces** (`scripts/measure/derived_surface_census.py`, read with the checks below):
  `insight_density` / `insight_sentiment` stale in 126 episodes (refreshed on prod by
  `enrich --only insight_density,insight_sentiment` for those episodes); search-index rows of the
  126 (incremental `index`); context consistent (126 of 126 equal a rebuild from the after-state;
  the census's 20 are rebuilt-unchanged files); `.anon.txt` is the pre-naming render by design
  (D-40); the per-episode manifest is processing provenance with no product reader; ad-map
  offsets off: 0.
- **Undo cycle** (criterion 21, `scratchpad/m0034/undo_cycle.py` on a copy of the after-clone):
  undo restored 1,518, refused 0; corpus content equal to the snapshot (0 of 143,530 files differ;
  only the backups, the receipt log and `upgrade_ledger.json`); apply again 126 episodes; undo again
  the same; no other migration's backups changed. UNDO-CYCLE PASS. On prod an undo also removes
  0034's ledger entry (`undo` does not).

### 7b. Final measurement, round 3 (ffaae4ae0, clean tree)

Supersedes the a1698c214 numbers above (`0034_talk_floor_names_removed.gate.json` is this run):
- **Apply:** rewrote **135** episodes, refused 0; **149** voices unnamed; verify OK;
  `bridge_identities_removed` 103.
- **Scoreboard:** S1 +59, S2 +59, S10a +1; S1c 0, S1n 7, S3 9,359, S10b 2, S12 24, commercial 94,
  A1 382, insights 130,401, quotes 252,651 unchanged.
- **Against the 12:08 restatement:** removals 149 (met), S1 / S2 +59 (in +49 to +69), S10a +1 (in
  +0 to +3); episodes **135 against 126 to 131: MISSED by 4** (more of the 10 new removals fell in
  episodes not already written than the range allowed).
- **Files:** 1,603 changed (receipts 1,624: 21 rewritten with identical bytes, 15 KG, 3 GI, 3
  bridge). Empty bridge files after: 15 (snapshot 17 incl. superseded runs): none written.
- **Undo cycle:** restored 1,624, refused 0; corpus content equal to the snapshot (0 of 143,636
  files); apply again 135 episodes; undo again the same; no other migration's backups changed.
  UNDO-CYCLE PASS.

### 7c. Final measurement, round 4 (a859902ab, clean tree)

Supersedes 7b (`0034_talk_floor_names_removed.gate.json` is this run): rewrote **138** episodes,
**155** voices unnamed, refused 0, verify OK; S1 +59, S2 +59, S10a +1; S1c 0, S1n 7, S3 9,359,
S10b 2, S12 24, commercial 94, A1 382, insights 130,401, quotes 252,651 unchanged; 1,636 files
changed (receipts 1,658: 22 identical-byte rewrites, 16 KG, 3 GI, 3 bridge); empty bridges written
0. Against the 14:23 restatement: removals 155 (met), episodes 138 (met), S1 / S2 +59 (in +59 to
+65), S10a +1 (in +1 to +2). Undo cycle: restored 1,658, refused 0, corpus content equal to the
snapshot (0 of 143,670 files), apply again 138, undo again the same: UNDO-CYCLE PASS.

### 7d. Final measurement, round 5 (a5a9a6909, clean tree)

Supersedes 7c (`0034_talk_floor_names_removed.gate.json` is this run, measured 2026-10-26T15:57:54):
rewrote **150** episodes, **172** voices unnamed, refused 0, verify OK; S1 +67, S2 +67, S10a +1;
S1c 0, S1n 7, S3 9,359, S10b 2, S12 24, A1 382 / 290, insights 130,401, quotes 252,651 unchanged;
1,779 files changed (receipts 1,801: 22 identical-byte rewrites). Against the 15:47 restatement:
removals 172 (met), episodes 150 (met), S1 / S2 +67 (in +59 to +72), S10a +1 (in +1 to +3).
Bridges: 150 written; 0 written as `{}`; 2 written as `{"identities": []}` over the snapshot's `{}`
(feed `aef1e4c0` item 0002, feed `e739a158` item 0001, the same 2 as rounds 3 and 4; 7c's "empty
bridges written 0" counted only `{}`); 0036 rebuilds both (they are 2 of its 16 damaged bridges).
Undo cycle (`undo_cycle_r8.log`): restored 1,801, refused 0, corpus content equal to the snapshot
(0 of 143,813 files), apply again 150, undo again the same, other migrations' backups changed 0:
UNDO-CYCLE PASS.

### 7e. Final measurement, round 6 (eea65f01f, clean tree)

Supersedes 7d (`0034_talk_floor_names_removed.gate.json` is this run, measured 2026-10-26T17:34:54):
rewrote **152** episodes, **175** voices unnamed, refused 0, verify OK; S1 +67, S2 +67, S10a +1;
S1c 0, S1n 7, S3 9,359, S10b 2, S12 24, A1 382 / 290, insights 130,401, quotes 252,651 unchanged;
1,805 files changed (receipts 1,827). Against the 17:21 restatement: removals 175 (met), episodes
152 (met), S1 / S2 +67 (in +67 to +72), S10a +1 (in +1 to +2); the 3 round-6 voices carry no
insight that S1 counts. Bridges: 0 written as `{}`, 2 as `{"identities": []}` (the same 2 as 7d;
0036 rebuilds them). Undo cycle (`undo_cycle_r9.log`): restored 1,827, refused 0, corpus content
equal to the snapshot (0 of 143,839 files), apply again 152, undo again the same, other
migrations' backups changed 0: UNDO-CYCLE PASS.

### 7f. Re-measure after round 6's conditions (d94e4d13a, clean tree)

`0034_talk_floor_names_removed.gate.json` is this run: plan, files (1,805), gating moves, both
scoreboards, episode deltas and verify equal to 7e, plan differences 0; only the verdicts sha moved
(header to round 6). The 18:36 prediction met. Undo cycle (`undo_cycle_r10.log`): UNDO-CYCLE PASS.

## 8. Verdict

Step-8 reviewer (a fresh agent, not the builder), 2026-10-26. Commit under review `f9f228038` (the committed list
`upgrade/lists/0034_talk_floor_names_removed.json`, sha256 `093b1956...`). Data: before = snapshot-prod-20261025 scratch copy,
after = the gate's clone `gate34`. Nothing was written into the repo except this section; no `make ci-fast`
was run (the builder's step). Voices are named by episode id + voice id only; the per-voice reading is in the
reviewer's scratch file `step8_reading.tsv` (556 rows). Earlier reviews and the builder's tables were inputs, not evidence.

### 8.1 Numbers re-derived with the reviewer's own code

All scripts are in the reviewer's scratchpad (`.../scratchpad/m0034/step8/`); none is the builder's. The shared helpers
used are the product's `select_served_artifacts` (served-copy rule) and `_segment_char_spans` (the S7 join).

| What | Command (reviewer's code) | Result |
| --- | --- | --- |
| Population | `pop.py <before>`: floor over each served episode's `segments.json` (per-voice sum of `end - start` over attributed segments, own placeholder test, strict `< 0.05`) | **556 voices in 397 episodes**; equal to the builder's `--find` key for key (set difference 0 both ways); names and roles equal |
| List = population | `cmp.py` | shipped list 546 (389 episodes) + the 10 dropped entries (8 episodes) = the 556, disjoint; the drop is the documented D15 restatement |
| Verdict coverage | `outcome.py` | 546 rows = the 546 list keys; CORRECT 335, CORRECT_MISSPELLED 6, WRONG_PERSON 205; K 341, W 205; unreviewed 0, duplicates 0, outside list 0 |
| Names off (O2) | `outcome2.py`: segment labels before vs after, every served episode | **205** W voices lost the label (= \|W\|); **341** K voices byte-unchanged; **0** other label changes; segment text, times and voice ids identical in all 2,964 readable episodes |
| Episodes written | sha256 census of both trees (`census.py`, 142,011 / 144,177 files) | 2,138 changed files outside `.podcast_scraper` = 2,137 content files + `upgrade_ledger.json`; all in the **180** episodes holding a W key (= the set of W episodes); 0 changes in the 209 K-only episodes; 0 files removed; 1 new file (the receipts) |
| Receipts | `rec_chk5.py` | 2,165 rows, every `sha_before` / `sha_after` equal to the census; 28 of them are no-op rewrites; every changed file is in the receipts except the runner's ledger. 299 receipt paths are un-normalised (`run_x/../run_x/...`), cosmetic but a path compare needs `normpath` |
| O1 | `pop.py <after>` + `after_cmp.py` | the floor detector on the after-clone lists **351 keys = K (341) + the 10 dropped**; W keys 0; new keys 0 |
| Gating board | committed `corpus_scoreboard.py` on both trees, plus own GI diff (`outcome3.py`) | S1 16,810 → 16,897 (+87); S2 30,563 → 30,650 (+87); S2 unidentified 13,659 and commercial 94 unchanged; S10a 190 → 192 (+2); S1c 0; S1n 7; S3 9,359; S10b 2; S12 24; unresolved refs 244; A1 382 / 290 / 2; insights 130,401; quotes 252,651. Equal to the builder's gate |
| Insights moved (own diff, by insight id) | `outcome3.py` | exactly 87 insights changed, all shown -> hidden `unknown`; 84 sit on W voices by the S7 join, 3 have no placeable quote and moved by their `speaker` label; none was `commercial` or `unidentified` |
| Offsets | `offsets.py` | S12 24 -> 24, **same 24 quote ids**; ad-map mismatches (`source_length` / excised text) 0 -> 0; turns `segments_sha256` mismatches 0 -> 0; ad-free segment out-of-range 1 -> 1 (same) |
| Gold | `gold_insights.py` on both trees, per-insight diff (`golddiff.py`) | S7-wrong unchanged (380 / 2,128 / 439). **5 insights left S7-correct, on 3 gold-named voices** (section 8.4) |
| Derived surfaces | committed `derived_surface_census.py` on the after-clone | stale after the apply: `.insight_density` 180, `.insight_sentiment` 180, `.manifest` 180, `.anon.txt` 37, `.context.json` 30 (the removed names are absent from them; only an episode title carries one), `.adfree.admap` 6 (offsets match, mismatches 0); search-index rows of the 180 episodes |
| Spot checks | `spot.py` on 5 episodes | W: 3 episodes, the removed `<name>:` prefix lines go to the raw voice id, line counts conserved; K: the self-introduced voice's lines kept (3 of 3) and a K-only episode is byte-identical in both states |

Not reached, as integers on this snapshot (acceptance 2): named voices at or above the floor 5,456; under-floor voices already
unnamed (raw id or empty label) 5,565; placeholder-labelled under-floor voices 0; under-floor named voices whose name another
voice in the episode also carries 42 (25 of them in W). 2 served episodes name a transcript file that is absent, so the
detector cannot read them. The 14 post-snapshot episodes and the D4 count were not measured.

### 8.2 Mutations (each against the repo tests, in a copy; the repo files were never edited)

42 mutants of the floor rule, the verdict mapping, the list binding, the shared-name guard, the offset shift and the verdicts
loader were run against the m0034 tests and then against the whole `tests/unit/podcast_scraper/upgrade/` directory (555
tests, which pass unmutated). **26 caught, 16 survived.**

Caught: floor 0.04; denominator = largest voice; each placeholder alternative; unnamed voices listed; verdict mapping keep-on-W,
remove-on-CORRECT, remove-always; verdicts load ignoring the whole list; receipts `list_sha256` dropped; `list_file()` pointing
elsewhere; verify ignoring a missing verdict; apply ignoring the verdicts; plan raising; receipts file name; shared-name insight
guard (`_shared_labels`); unnaming leaves an insight shown; offset shift in a segment and in a diff run; ad-map length check;
verdicts-loader `list_sha256`, key-outside-list, duplicate key, stale name.

**Survivors (unguarded paths):**

| Mutation | Why it matters |
| --- | --- |
| `<` -> `<=` in the floor | the boundary is untested; 60 named voices sit in [4.5%, 5.5%] on this snapshot |
| `FLOOR` 0.05 -> 0.06 | only 0.04 is caught |
| denominator counts only named voices; unattributed segments counted; negative span not clamped (equivalent on this data) | the share is the whole rule |
| served name taken from the last segment instead of the first | |
| `CORRECT_MISSPELLED` also removes the name | 6 production rows carry that verdict and no test uses it |
| receipts header `verdicts_sha256` dropped | the audit trail of which verdicts ran |
| `verify` ignores refused episodes | the check that found the 8 refusals could be removed silently |
| detector over every copy instead of the served copies | superseded runs would enter the list |
| no `.txt` guard; backup tag `0034` -> `0035` | |
| shared-name refusal ("transcripts: not a render, and a rewritten label is shared") removed | this is the guard that refused 8 episodes at the first gate; nothing in the repo tests it |
| unnaming does not set `speaker_voice_type = "unknown"` | |
| `write_list` unsorted (equivalent for loading) | |
| commercial-type guard | there is none in the code: a W voice typed `commercial` would be retyped `unknown` (acceptance 12); 0 instances here, 0 tests |

Acceptance 23 asks for 0 survivors or an equivalent mutant argued. Equivalent: the unsorted `write_list` and the negative-span clamp
on this data. The others are not argued, so criterion 23 fails.

### 8.3 Reading

All 556 voices were read, not the 120 the acceptance asks for: the 341 K, all 205 W, the 10 dropped entries. The order was
shuffled (seed 77; the sample draw used seed 20261026), the builder's verdict was not shown while reading, and an identity needed the voice's own words
(self-introduction, an address answered, a host reading in the host's own voice, a diarization fragment shown by continuity) or
an independent label. The packets show only a window around each voice's first lines, so where the packet could not decide, the
whole transcript was grepped, and where the builder's cited line contradicted the reading it was taken. A voice that cannot be
decided is counted as remove ("U"), as the operator's rule says.

| Stratum | Read | Agree with the builder | Disagree |
| --- | --- | --- | --- |
| K (kept) | 341 | 339 | 2 (would remove) |
| W: host-role, shared-name and random sample (acceptance 25) | 120 | 112 | **8 (would keep)** |
| W: the rest (guest-role) | 85 | 78 | 7 (would keep) |
| Dropped entries (not applied) | 10 | n/a | read 6 keep / 4 remove; the record counts 8 W + 2 K, the per-entry verdicts are not in either file |

Of the 205 W rows, 157 are clear removals (an ad, a promo, a clip, a news or archival voice, another person's lines, a label that
is not a person), 33 are undecidable (counted as agree), 15 are voices whose name I judge right. 89 of the 205 W rows carry the
deciding line "no evidence it is them"; 12 of the 15 disagreements carry exactly that line. The default does not separate "not
this person" from "no self-introduction".

The 15 disagreements (episode id, voice; basis from the whole transcript):

- Seven I rate firm, 3 of them proved by the gold labels (8.4): `715ff48a` SPEAKER_01 (gold), `730396fd` SPEAKER_02 (gold),
  `e32aa079` SPEAKER_06 (gold, and the host thanks him by name at the end), `3cab0976` SPEAKER_02 (the show's host: opens, welcomes
  the guest, asks the questions, closes), `a1395046` SPEAKER_00 (the interviewer in a two-person interview; the host's intro is on
  an unnamed voice), `47e1118c` SPEAKER_00 (introduces the guest and asks the opening questions), `ee5028a8` SPEAKER_03 (the
  interviewee the host introduced, speaking at length).
- Five more I rate right on the content or an introduction: `e5b7caa6` SPEAKER_03 (addressed by the host, then answers),
  `a0e15df4` SPEAKER_01 (the guest; 56 lines), `774dd005` SPEAKER_03 and `201de292` SPEAKER_03 (guests answering after the host's
  welcome), `cfff7e08` SPEAKER_03 (an investor introduced by his firm in the host's lead-in).
- Weak: `2050514a` SPEAKER_02 (a host fragment), `urn:bbc:podcast:m002c71f` and `urn:bbc:podcast:m002myr1` SPEAKER_08 (expert clips
  introduced by the host).

The 2 K voices I would remove: `1_2766128` SPEAKER_12 (a radio-drama character: the name is the character's, not a guest) and
`519b7338` SPEAKER_03 (a garbled first name kept under `CORRECT_MISSPELLED`). K has no firm wrong-person keep in 341. Where a first
reading was wrong, the builder's line or the gold label was right: I first called `730396fd` SPEAKER_02 wrong and `715ff48a` SPEAKER_01 undecidable, and four K rows
wrong or undecidable, and corrected them from the gold labels and the builder's quoted lines.

Where the builder was right: a voice that merges two people (two hosts' self-introductions, two reporters, two voicemail callers)
was removed, and 4 of those rows would have been kept by a reading of the opening lines alone (`2329358a` SPEAKER_07, `2e6155cd` SPEAKER_07,
`63b8eeb5` SPEAKER_09, `10fe0293` SPEAKER_05); I agree with the removals.

### 8.4 Independent check: the gold labels

33 of the listed voices appear in the labelled sets (dev_v1, val_v1, val_v2). Of the 13 with a gold name that the
builder kept, all 13 are right (S7-correct stays). Of the voices the builder removed that have a gold name, **3 of 3 are right
persons**: `730396fd` SPEAKER_02 (val_v1, 2 insights), `715ff48a` SPEAKER_01 (val_v2, 1 insight), `e32aa079` SPEAKER_06 (val_v2, 2
insights). 5 insights leave S7-correct (2 in val_v1, 3 in val_v2); one insight with no placeable quote enters it in val_v2, so the net is 11,498 -> 11,496 (val_v1) and 3,706 -> 3,704 (val_v2); dev_v1 unchanged. S6 (labelled voices whose
published name became worse) is **3, not 0**. This is evidence that does not depend on my reading.

### 8.5 Against the prediction (4b, 4c) and the restatement

| Item | Predicted / restated | Measured | Result |
| --- | --- | --- | --- |
| Population | 556 voices, 397 episodes | 556 / 397, same keys | pass |
| Kept / removed (4b) | 66 / 490 | 341 / 205 (546 applied) | the 4b prediction missed by a wide margin; 4c restated it |
| Kept / removed (4c, after the drop) | 341 / 205, 546 entries | 341 / 205 | pass |
| Episodes written | 180 (first gate 180, refused 8) | 180; refused 0 on the clone | pass |
| S1c, S12 | 0, 24 | 0, 24 (same ids) | pass |
| S1 / S2 | rise "only by the visible insights of the removed" | +87 / +87 (the upper bound by the S7 join is 94) | pass with the explanation in 8.1 |
| S10a | +2 | +2 | pass |
| Nothing else moves | S3, S10b, commercial, A1, counts | unchanged | pass |

Order: the 4b prediction and the acceptance list were written after the build (the `MISSED:` line; accepted by the operator in
4c). The count of removals changed after the first clone gate (213 -> 205), which section 4c states.

### 8.6 The acceptance list, criterion by criterion

| # | Result | Evidence |
| --- | --- | --- |
| 1 | pass | own code: 556 / 397, equal both ways; the 60 named voices in [4.5%, 5.5%] were listed by the reviewer, not the builder |
| 2 | fail (not done by the builder) | the record holds none of (a) to (e); the reviewer's integers are in 8.1 |
| 3 | fail | list and hash binding exist and are tested (`list_sha256` in the verdicts header, the digest and the receipts); the entries carry the share only as a string, not the talk seconds and the episode total as numbers |
| 4 | partial | registered before 0035 (test asserts the index); 0034 verify true in the gate; the other migrations' verify lines not compared |
| 5 | pass, restated | 546 rows for the 546 applied keys (the 10 dropped are documented); closed set, no duplicates, header complete |
| 6 | fail | the shipped file has no segment index or text per row (the evidence is in the private repo); "no evidence it is them" stands for 89 rows and no row is marked `undecidable:` |
| 7 | fail | the 92 labelled under-floor names were not compared by the builder; the reviewer's join finds 3 unexplained disagreements (8.4) |
| 8 | pass | the clip rule and the host-reads-an-ad rule are recorded in 4c |
| 9 | pass | 351 keys = K + the 10 dropped; W 0; new 0 |
| 10 | pass | 205 removed, 341 kept unchanged, 180 episodes written, refused 0 |
| 11 | partial | the removed name is gone from segments, transcripts, turns, diagnostics names, the context and the ad-free files; it remains in the GI Person nodes (25: 12 kept by mentions, 12 spoken by other voices' quotes, 1 other), 31 bridge identities, 9 KG labels, 1 unplaced roster entry, and the decision trace (history); the builder lists none of them |
| 12 | fail | no insight-id lists in the gate report; Iw_vis = 94 by the join, 10 stay shown (4 under a shared name, 6 with no `speaker` field), 3 more moved without a placeable quote; no commercial guard in code |
| 13 | fail (not done) | no list of the people who lost their last voice |
| 14 | pass | every "must not move" number identical, same ids |
| 15 | partial | S1n 7 and S7-wrong unchanged; S11 not measured by the scoreboard |
| 16 | partial | +87 / +87 / unidentified 0 / +2 equal the id-level diff, not "exactly \|Iw_vis\|" (94); S4, S5, S8, S9 not measured |
| 17 | **fail** | S7-correct falls on 3 gold-named voices; S6 = 3 |
| 18 | pass | see 8.1 (the gate's suffix tally mis-bins some `.txt` names; the count 2,137 is right) |
| 19 | pass for what was checked | quotes, ad maps, turns hash; the metadata GI / KG count recount was not redone |
| 20 | fail | no census in the record; the committed census shows the enricher sidecars, manifests and anon files stale |
| 21 | fail (not done) | no full-clone undo in the record or the gate report |
| 22 | partial | removed (205), name kept by another voice (25), Person kept by mentions (12), CORRECT (335) and CORRECT_MISSPELLED (6) kept, moved-on refusal and no-verdicts refusal (tests); removed from a `commercial` voice: 0 clone instances and no test |
| 23 | **fail** | 16 survivors (8.2) |
| 24 | pass with 2 flags | 341 read; no firm wrong-person keep; 2 questionable (8.3) |
| 25 | **fail** | 8 of the 120 (the bar is 3), 15 of all 205 |
| 26 | pass | 3 W and 2 K episodes in both states |
| 27, 28, 29 | not verifiable | prod only; the search-index rows of the 180 episodes are a step-10 item |
| 30 | not verified | `make ci-fast` is the builder's step |
| 31 | pass | the operator's line follows the `MISSED:` in 4c |

### 8.7 What is not covered (as long as what is covered)

- Prod: the plan diff, the search-index rows, the apply counts, the 14 post-snapshot episodes.
- The 8 episodes of the 10 dropped entries: their names stay served. By the reading, 4 of the 10 are wrong persons (`3e537123` SPEAKER_07,
  `6e2afd94` SPEAKER_02, `7e247ae4` SPEAKER_01, `994123ec` SPEAKER_02) and keep a wrong name on disk. The refusal ("a rewritten label
  is shared", the #2294 shape) has no repo test (8.2).
- 4 insights stay shown under a name on a W voice because the name is shared with another voice: `730396fd` SPEAKER_08 holds 2 of
  them, on a clip voice. The shared-name guard leaves them.
- The D4 shape of #2094: removing the name does not give it back to the right voice; the main voice of the right person stays
  unnamed in many of these episodes (for example `a1395046`, `3cab0976`).
- 58 K voices carry a misspelled or garbled form of the person's name (the verdicts file marks 6 `CORRECT_MISSPELLED`); they are served as
  they are. One is a wrong first name (`519b7338` SPEAKER_03).
- `make ci-fast`, the undo, `upgrade verify` of the other migrations, S4 / S5 / S8 / S9 / S11, the metadata count recount, and any
  reading beyond the first lines of a voice for the 341 K rows other than by grep and the builder's quoted line.
- Two served episodes with no readable transcript file (the detector cannot reach them).

### 8.8 Reasons for the verdict

The mechanics are sound: the population, the counts, the files, the offsets and the gating board all match the record, and the
removed names are gone from the transcript surfaces. The verdicts are not. Acceptance 25 sends the whole verdicts file back for a
full re-read at more than 3 wrong removals in 120; it has 8, 3 further proved by the gold labels. The cause is one default ("no
evidence it is them") that removes the right name from fragments, interviewers and guests who never introduce themselves. Criterion
17 fails independently (S6 = 3), criterion 23 fails (16 survivors, including the guard that refused the 8 episodes and the
`CORRECT_MISSPELLED` mapping), and criteria 2, 3, 6, 7, 12, 13, 20 and 21 are not met.

What a repair needs: re-read the W rows without the self-introduction default (the reviewer's 15 and the per-row table in
`step8_reading.tsv` are a starting point, not a verdict), flip the confirmed ones to CORRECT, re-bind the list and verdicts hashes,
re-run the clone gate (and the gold diff, the undo and the census), add the tests of 8.2, restate the prediction, and send the
result to a new step-8 reviewer.

Verdict: NO-GO

## 8b. Verdict (round 2)

Dated: 2026-10-26. Step-8 reviewer, round 2: a fresh agent, not the builder and not the round-1 reviewer. Commit under review
`7fd0eff36` (0034's files last changed in `a1698c214`); data: before = `snap1025/corpus` (snapshot-prod-20261025 scratch copy),
after = `gate34e` (0034 applied by the gate; its ledger records 0034, the snapshot's does not). Nothing was written into the
repo except this section. Voices are named by episode id prefix and voice id only. Earlier reviews and the builder's tables
were inputs, not evidence. My scripts are in the reviewer scratchpad `r8b34/` (none is the builder's).

### 8b.1 How the reading was done (what I read, what I did not)

- I read myself, from the whole transcript and before opening any verdict: the 9 carried removals (criterion 33), the 2
  flagged keeps, v181 and x032, and about 40 voices where a reader and the builder differed.
- I did NOT personally read every voice. 12 fresh Sonnet readers read 254 voice-items blind to every verdict and to each other
  (all 140 W rows, all 65 rows flipped to keep, 49 criterion-7 rows not already in those, the 9 carried). Each of their
  disagreements with the builder (40) I read and settled myself. I did not re-read the roughly 214 agreements; a reader agreeing
  with the builder is weaker evidence than my own reading, and I say so.
- The removal draw (criterion 25): seed 918273 (not 77, not 20261026), W = 140, the 7 carried W rows excluded, 133 left,
  29 feeds, 120 drawn with at least 1 per feed; the 13 not drawn were read as well (every W row was read).

### 8b.2 Numbers re-derived with my own code

| What | Result |
| --- | --- |
| Population | own floor code over `select_served_artifacts`: **556 voices in 397 episodes**; the shipped list's 546 are all in it; the other **10** are the documented D15 drop |
| Verdicts file | 546 rows = 546 list keys; CORRECT 395, CORRECT_MISSPELLED 11, WRONG_PERSON **140**; sha256 of the file equals the receipts header `e0df37e3...`; list sha equals the header `4285e56b...` |
| Criterion 6, integers | public `verdict` / `basis` / `segment_idx` / `line_sha256` equal the private evidence file for **546 of 546**; rows with no basis 0; W rows whose evidence text begins "no evidence" or states an absence 0; W rows without a line 0; `line_sha256` equals sha256 of the segment text in the before corpus for **521 of 521** rows that carry one (0 misses) |
| Undecidable rows | **26** (25 flipped from W, 1 other); 25 carry no segment_idx / line_sha256, 1 carries both |
| Revision flips | 65 rows W (round 1: 205) to K; K to W 0; bases of the 65: addressed 28, undecidable 25, continuity 4, host_read 4, self_intro 2, clip_own_person 2 |
| Names off | own diff of every `segments.json`: **140** (file, voice) pairs lost their label = the 140 W keys, set difference 0 both ways; no other label changed; text, times, voice ids identical |
| Insights by id | own diff of every `.gi.json`: 118 GI files changed; exactly **56** insights changed, all shown -> hidden `unknown`, named-before; 0 removed; 0 other property moves |
| Scoreboard | committed command, both trees: S1 16,810 -> 16,866 (+56); S2 30,563 -> 30,619 (+56), unknown = S1, unidentified 13,659, commercial 94; S10a 190 -> 191 (+1); S1n 7; S1c 0; S3 9,359; S10b 2; S12 24; insights 130,401; quotes 252,651; A1 382 / 290 / 2. Equal to the gate report and to 4d's restated prediction except S1/S2 +56 against the recorded +48..+55 (the recorded miss, accepted) |
| Tests at HEAD (in a `git archive` copy) | `tests/unit/podcast_scraper/upgrade`: **609 passed** |

### 8b.3 Criterion 33: the carried nine, read first and blind

Read from the whole transcript before any verdict was opened. Mine / the shipped file:

| # | Key (episode, voice) | My reading | Shipped |
| --- | --- | --- | --- |
| 1 | `tag:soundcloud,2010:tracks/2135080263` 02 | keep: backchannels and a "thank you for having us" after the guests are welcomed; nothing shows someone else | CORRECT_MISSPELLED (addressed) |
| 2 | `f004446b` 03 | stays W: the same voice says "thanks for having us" and "our next paper", guest lines | W |
| 3 | `d867d354` 00 | stays W: another voice says the named host "has said before" of these lines' speaker | W |
| 4 | `0e910a81` 03 | stays W: the voice answers "Joe" and asks the co-host a question; the guest is on another voice | W |
| 5 | `urn:bbc:podcast:m002c70n` 07 | stays W: a listener couple's exchange before the experts are introduced | W |
| 6 | `0308f04c` 02 | stays W: an acted poem that says the speaker "has been dead since" a past year | W |
| 7 | `eae61d33` 06 | stays W: a voicemail caller ("I'm 35 years old", shares a parent's subscription) | W |
| 8 | `730396fd` 02 | keep, with a flaw: the self-introduced candidate's ad and interview clips, but the cold open's first lines are another speaker's | CORRECT (clip_own_person) |
| 9 | `iono.fm/e/1728203` 08 | stays W: it asks the question; the named analyst answers on 09 | W |

Result: 9 of 9 read and settled in writing; additional flips needed 0 (the shipped file already keeps 1 and 8). The 5-of-9 trigger
does not fire; the 116 stay. Row 8 shows the pattern of 8b.5: a named voice holding a stretch of someone else.

### 8b.4 Criterion 25: removals

All 140 W rows were read (120 drawn plus 13 undrawn plus the 7 carried). 137 read as a removal. Three differed:

- `2329358a` 07 (in the draw): the voice says "I'm ..." and then "And I'm ..." of the other host: two people on one voice. W stands.
- `655e460d` 00 (in the draw, host role): the reader found it undecidable; it reads a third-person synopsis and the legal
  disclaimer, the same class as the other 5 a16z narrator voices the builder removed. W stands.
- `38156bae` 04 (in the draw): a voicemail caller whose segment also holds a second caller's one-line introduction. By the
  majority test used for the kept ad voices (8b.5) this is a right name removed: **1 wrong removal** (arguable).

Bar: at most 2 in the 120 draw; found 1 (arguable). Outside the draw (72 host-role rows read; 2(e); 7 rows): 0 wrong. **Criterion 25
passes.** The 137 agreements are one blind Sonnet reading each, not mine.

### 8b.5 Criterion 24: the rows flipped to keep, and the two flagged rows

Bar: 0 wrong keeps. The readers read the 65 flipped rows blind: OWN 29, UNDECIDABLE 10, OTHER 26. I settled the 26 myself.
Of the 25 flipped rows with basis `undecidable`, the readers said OTHER on **20**, UNDECIDABLE on 4, OWN on 1: the default
the operator ordered ("keep unless the text shows someone else") keeps rows where the text does show someone else.

**Wrong keeps I confirm (text shows someone else): 7.**

- 5 a16z announcer voices named after a person who is not in the episode or not the speaker: `277d8505` 01, `3346a500` 00,
  `34e47cf8` 00, `382b7a08` 01, `25f3d896` 00. Each reads a third-person synopsis of the episode and/or the legal disclaimer. My own
  scan of the list found 14 voices that read that disclaimer: the builder removed 6 as `another_person` (`1b259b80` 01, `2e282338` 01,
  `360e9305` 02, `655e460d` 00, `6be2f208` 02, `6eb845ef` 02), kept 3 as `host_read` and kept these 5 as `undecidable`. One class, two
  treatments.
- `ee665e22` 05: fragments that complete another voice's sentences (`ee665e22` 04); the named person introduces herself on 06.
- `1_2766128` 12 (flagged by round 1): a character in an audio drama; the same basis the builder used to remove `0308f04c` 02.
  Flip to W.

**Moderate, for the next reader to settle (text points elsewhere but less sharply): 10.** `75146f27` 02 (a one-line show tagline
before the guest is introduced), `iono.fm/e/1728203` 01 (asks the questions while 05 answers; the same shape as carried row 9, which
the builder removed), `50f2f680` 11 (a promo read and a politician clip), `2ede0e96` 01 (promo reads and an interviewer in a field
clip), the sponsor-lead-in-only voices `c1b9decd` 09, `00fc117b` 13, `6af9750d` 09 and `6af9750d` 08 (two different names on adjacent
sponsor reads of one episode), `a4c1c410` 02 and `2ede0e96` 00 (one fragment each that other speech shows is not theirs).

Settled as kept (reader OTHER, not shown): `62fdb4dc` 03, `b27552ed` 04, `a0e15df4` 01, `a1395046` 00 (contamination by one or two
lines of the other person, the majority being the named person's), `10fe0293` 05, `856dd2e4` 01, `64c72df8` 00,
`brilliantexperience.podbean.com/602a0e4a` 01, `db8ca4cd` 03, `730396fd` 02 (gold-labelled right person, with a speech clip on the voice).

Flagged row 2, `519b7338` 03: the same co-host's other voice, self-introduced under the transcript's own garble of the first name. It
is the right person under a wrong first name: kept as `CORRECT_MISSPELLED`, not a wrong-person keep. The published roster still
holds a nonexistent first name; 0034 does not fix that.

**Criterion 24 fails: 7 confirmed wrong keeps (bar 0), 10 more to settle.** A consistency finding the builder must resolve: voices
with a minority of another speaker's lines are removed in 2 rows (`2329358a` 07 and `38156bae` 04) and kept in
10 K rows (`c85ca0b8` 07, `b7d968f5` 02, 03 and 08, `6baab6b1` 03, `c196b474` 04, `1294ea90` 03, `2a3e72ef` 01, `29b52580` 00 and 04).
It needs one written rule from the operator.

### 8b.6 Criterion 7: the 61 label disagreements

I used the builder's list of 61 (`criterion_7.disagreements`) and did not re-derive the 92 labelled names from the 10-17
measurement: **MISSED** by me. All 61 were read (readers blind, then me on every non-agreement).

- 2 label-keep / verdict-remove, `fb846750` 01 and `2a588a0a` 04 (the show host's name): both voices say lines of an acted
  two-person shoe skit while the host introduces the show on another voice. The labels' "host-read ad" is a label error; the
  removals stand. Both earlier readers were right.
- 59 label-remove / verdict-keep: 41 read OWN and 1 undecidable (the label scores participation; the verdict scores identity;
  settled by the operator ruling of 4c, 48 of the 59 on a self-introduction); 10 are minority-contamination keeps (8b.5); 1 is
  the gold-labelled `730396fd` 02; 1 is `62fdb4dc` 03 (kept, undecidable); **5 are verdict errors**: `1_2766128` 12 and the four
  sponsor/promo voices `c1b9decd` 09, `6af9750d` 08, `6af9750d` 09, `2ede0e96` 01.

Bar 0 unexplained: met in form, but 5 settle as verdict errors, so the criterion fails on its second clause (a disagreement settled
as a verdict error is corrected, and step 7 re-runs).

### 8b.7 Criterion 6, the builder's deviation

Judged: **acceptable in principle, not as built.** An `undecidable` row has no deciding line by definition, and requiring one would
force a made-up line. But (a) 25 of the 26 such rows then carry nothing an auditor can check; (b) the readers called 20 of those 25
OTHER and 6 of the 7 confirmed wrong keeps are in it, so the bucket is not a harmless default; (c) the exception is not
pinned: the mutant that lets ANY basis pass with an empty `segment_idx` / `line_sha256` survives 609 tests (acceptance-D 12
requires that refusal to be tested). Accept the exception only with that test and with the 7 above corrected.

### 8b.8 Mutations (in a `git archive 7fd0eff36` copy; the repo was not touched)

53 mutants run against the m0034, verdict-loader, m0025, m0031, m0035 and m0036 tests; survivors re-run against the whole
`tests/unit/podcast_scraper/upgrade` directory (609 tests). **47 killed, 6 survived on both.**

Survivors: the receipts file name changed (equivalent if the constant is shared by write and undo; no test pins it); `require_basis`
off for 0034 (a basis-less file would load); the commercial-type guard on QUOTES removed (the insight guard is killed); quotes not
set `unknown` when unnamed; **bridge `sources` not set from the match** (acceptance-C 14 asks for a test; absent); the loader
letting any basis through with no line (above). Acceptance-B 23 bar: 0 survivors, at most 2 argued equivalent: **fails** (6, 1 arguable).
Acceptance-C's five mutants: 4 of 5 killed (sources survives); acceptance-D's sixth and the loader's three: killed. The round-1
survivors now killed include the floor boundary, 0.06, the denominator, the served-name choice, the shared-name refusal,
`CORRECT_MISSPELLED`, `verify` on refused episodes and the detector over every copy.

### 8b.9 Undo and derived surfaces

- Undo: not re-run by me. The builder's cycle log (`undo_cycle_r5.log`) shows undo restored 1,518, refused 0; apply again 126
  episodes; undo again 1,518; corpus content differing from the snapshot 0 (1,519 bookkeeping files); other migrations' backups
  changed 0. The receipts file on the after-clone holds 1,519 lines (header plus 1,518), consistent. Criterion 21 rests on the
  builder's log, not on my run.
- Derived surfaces: I ran the committed census on the after-clone: `insight_density` 126, `insight_sentiment` 126, `manifest` 126
  stale; `anon.txt` 26; `adfree.admap` 4 (offsets match, mismatches 0); `context` 20 (the record says these are rebuilt-unchanged);
  bridge and turns 0. The ladder (re-enrich the 126, incremental index of the 126, anon by design) is sound and matches the record.
  The census also prints "episodes written: 162" against 126 applied: unexplained, not chased.
- Acceptance-C 20 to 23 (exposure of 0031, 0033, 0035 to the alias defect, by backups): the record reports no integers for them
  (4d says "reported in step 8"; section 7 has none): **not evidenced.**

### 8b.10 What is NOT covered

- Prod: criteria 27 to 29 (the plan diff against the clone's, the 14 post-snapshot episodes, the search-index rows of the 126
  episodes by targeted edit, apply counts equal to the restated prediction minus refusals, backup ownership, daytime outside
  00:00-04:30Z) cannot be done on the clone and are untouched. The outcome hook's accept file is for them.
- The 92 labels were not re-derived (8b.6); `make ci-fast` (builder's step, not run); the sample of 214 agreements not re-read by me;
  derived-surface census count 162; the 10 dropped entries and their 8 episodes (their wrong names stay served); D4 of #2094; the
  58 misspelled K names; reprocess durability; S4, S5, S8, S9, S11.
- The next reader must read the 10 moderate rows and the 7 corrected rows afresh; I did not give those 10 a final verdict.

### 8b.11 Criteria summary

| Criterion | Result |
| --- | --- |
| 1, 5, 9, 10, 14, 18 (population, coverage, names off, scoreboard, files) | pass, own code (8b.2) |
| 6 | verdicts/columns/shas pass 546 of 546; the deviation as built fails (8b.7) |
| 7 | fail: 5 verdict errors; 92 labels not re-derived |
| 12 | pass for ids: 56 moved ids = S1 delta; 0 commercial moved in this data; the quote-side commercial guard is untested (survivor) |
| 21 | rests on the builder's log |
| 23, acceptance-C 14 | fail: 6 survivors, sources mutant alive |
| 24 | **fail: 7 confirmed wrong keeps, 10 to settle** |
| 25 | pass: 1 arguable wrong removal in 120 |
| 33 | pass: 9 of 9, 0 flips needed |
| acceptance-C 20 to 23 | not evidenced |
| 27 to 29 | prod, open |

### 8b.12 Reasons for the verdict

The mechanics are sound and re-derived: the population, 140 names off, 406 byte-unchanged, 56 insights by id, the scoreboard, the
hashes, the 609 tests. The verdicts still fail the operator's own rule. The "keep unless the text shows someone else" default
kept 20 of 25 undecidable rows that an independent reader reads as someone else, and I confirm 7 wrong names kept (5 are an
announcer class the builder removed 6 times elsewhere), against a bar of 0, plus 10 more to settle. The shipped test suite leaves
the quote-side commercial guard, the bridge `sources` and the loader's basis exception unguarded, and the exposure of 0031, 0033
and 0035 is not measured.

What a repair needs: flip the 7 to `WRONG_PERSON` with a basis and a deciding line each (`segment_idx`, `line_sha256`); have a
fresh reader settle the 10; get the operator's one rule for minority contamination; add the 6 tests and kill the 6 mutants; report
the C 20 to 23 integers; restate the prediction (W rises from 140; S1 and S2 rise with it; D15); re-bind the list and verdict
hashes; re-run the clone gate, the undo cycle and the census; then a new step-8 reviewer reads the changed rows.

Verdict: NO-GO the 7 confirmed wrong-name keeps (criterion 24, bar 0; 5 are the a16z announcer class the builder removed 6 times elsewhere), 5 verdict errors among the label disagreements (criterion 7), 6 surviving mutants incl. bridge sources and the unpinned basis exception (criteria 23, C14, D12), acceptance-C 20-23 not evidenced

**Operator rulings after the round-2 step-8 NO-GO (2026-10-26):**
- Operator accepted: 2026-10-26: a voice that mixes two people keeps the name it is served under
  when MOST of its talk is the named person, and loses it when most is someone else ("majority
  decides"); a minority of another person's lines does not remove it.
- Operator accepted: 2026-10-26: a voice that only reads a show's intro synopsis in the third
  person, or its legal disclaimer, is shown by the text to be someone else (an announcer): the
  name comes off, for every such voice alike.

## 8c. Verdict (round 3)

Dated: 2026-10-26. Step-8 reviewer, round 3: a fresh agent, not the builder and not the round-1 or round-2 reviewer. Commit under
review `9e39accc5`. 0034's code last changed in `ffaae4ae0` and the gate ran on it clean; `git diff ffaae4ae0 HEAD --stat` shows
only record and gate files since. Data: before = `snap1025/corpus`, after = `gate34f`. Nothing was written into the repo except this
section; every mutation ran in a `git archive 9e39accc5` copy (`scratchpad/r8c34/code`). Voices are named by episode-id prefix and
voice id only. My scripts are in `scratchpad/r8c34/mine/` (none is the builder's or an earlier reviewer's). The `secrets-scan` script
is blocked in this shell, so I ran my own regex pass over `git diff 7fd0eff36 HEAD` (AWS key, private key, gh / sk / slack token,
bearer, key=value): 0 hits in 3,420 diff lines; gitleaks was not run.

### 8c.1 Numbers re-derived with my own code

| What | Result |
| --- | --- |
| Population | floor over `select_served_artifacts` and each served `segments.json`: **556 voices in 397 episodes**; the 546 shipped keys are all in it; the other 10 are the documented D15 drop |
| Verdicts | 546 rows; CORRECT 386, CORRECT_MISSPELLED 11, WRONG_PERSON **149** => kept **397**. Public `verdict` / `basis` / `segment_idx` / `line_sha256` equal the private evidence file for 546 of 546 (0 diffs). The 16 rows with an empty `segment_idx` / `line_sha256` are all CORRECT + `undecidable` (17 undecidable rows; one carries a line) |
| Hash chain | sha256 of the shipped verdicts file = the receipts header `verdicts_sha256` (`13537382...`); `rv.list_sha256(load_list())` = the verdicts header = the receipts header (`4285e56b...`); all 546 list entries carry parseable `talk_s` / `total_s` with share < 5% |
| Names off | label diff of every `segments.json` before vs after: **149** voices lost their label = the 149 W keys (set difference 0 both ways); **0** K labels changed; text, times, voice ids identical (0 diffs) |
| Insights by id | own diff of every `.gi.json`: exactly **59** insights changed, all shown -> hidden `unknown` (`surfaceable` None -> False, `speaker_voice_type` None -> unknown); 0 removed |
| Scoreboard | committed command on both trees (52 flattened fields): S1 16,810 -> 16,869 (**+59**), S2 30,563 -> 30,622 (**+59**), S2 unknown +59, S10a 190 -> 191 (**+1**); every other field equal (S1c 0, S1n 7, S3 9,359, S10b 2, S12 24, commercial 94, unidentified 13,659, A1 382 / 290, insights 130,401, quotes 252,651) |
| Episodes | **135** episodes hold a W key (126 earlier + 10 new W voices in 10 previously unwritten episodes - 1 for the voice that went back to a keep); gate: rewrote 135, refused 0; receipts 1,625 lines (header + 1,624) |
| Empty bridges | 17 files with no identities before and after (16 served; 15 are literally `{}`); **0** new, **0** healed |
| Tests at HEAD | `tests/unit/podcast_scraper/upgrade` in the archive copy: **668 passed** |
| Undo | not re-run by me; the builder's `undo_cycle_r6.log` (13:27, after the 12:21 gate): restored 1,624, refused 0, corpus content differing 0; apply again 135; undo again the same; other migrations' backups changed 0 |

### 8c.2 The round-2 NO-GO items, one by one

| Round-2 item | Closed? | Evidence |
| --- | --- | --- |
| The 7 confirmed wrong keeps | **No: 5 of 7** | The 5 a16z announcer voices (`277d8505` 01, `3346a500` 00, `34e47cf8` 00, `382b7a08` 01, `25f3d896` 00) are `WRONG_PERSON` now (read in 8c.4). `ee665e22` 05 and `1_2766128` 12 are still `CORRECT`; I read both myself and still find them wrong keeps (8c.4) |
| The 10 to settle | Settled by readers, **not on evidence for 5 of the 7 kept** | 3 became removals (`75146f27` 02, `a4c1c410` 02, `2ede0e96` 00), 7 stay kept; read in 8c.4. The 5 sponsor / promo voices among the kept 7 are kept by the default; independent labels contradict 4 of them |
| Announcer class, all 14 disclaimer readers | **Yes in outcome, not in the record** | My own regex over the 546 for the disclaimer text finds 12 voices (3 kept, 9 removed); with the 3 it misses (`382b7a08` 01, `34e47cf8` 00, `655e460d` 00) the 43 hold 15 a16z intro / outro / disclaimer voices: 12 removed, 3 kept (`38112b09` 00, `a3717c93` 01, `ad8ee5b3` 02). The 3 kept read the intro in the first person ("I sit down with ...") and the outro, and are served as the host; the removed read it in the third person under a guest's name. That is a defensible reading of "shown to be someone else", but the operator's text says a voice that "only reads ... its legal disclaimer" is an announcer, and 2 of the 3 (`a3717c93` 01, `ad8ee5b3` 02) never speak inside the conversation. The first-person distinction is written nowhere in the record |
| Mixed voices under "majority decides" | **Yes, with 1 row the text cannot settle** | 11 `merged_voices` removals; the ones I read have the other person's talk as most of the voice (`38156bae` 04: the caller's own half-sentence, then a second caller for the rest; `6be2f208` 02: a third-person synopsis and the outro, 98 lines, the named guest is in the synopsis). In my 40-row sample 7 kept voices carry a minority of another speaker's lines (`7ee00723` 04, `b0e586d9` 04, `6ebbe026` 00, `f28385d3` 01, `c1b9decd` 03, among others) and each is mostly the named person. `2329358a` 07 flipped to keep: the promo is read by two hosts on one voice and the text does not give a majority either way, so the default keeps it |
| The 5 criterion-7 verdict errors | **No** | `1_2766128` 12, `c1b9decd` 09, `6af9750d` 08, `6af9750d` 09, `2ede0e96` 01 are all still `CORRECT`; the re-read said `undecidable` (4) or `self_intro` (1). The labels (`non_participant`, no name) are independent of the readers and still disagree; "undecidable, so kept" is the default rule, not an explanation of the label |
| The 6 surviving mutants | **Yes, 6 of 6 killed** | 8c.6 |
| Acceptance-C 20 to 23 (exposure of 0031 / 0033 / 0035) | **No** | The record still holds no integer for them (line 251 promises them "in step 8"; grep for exposure finds only that promise and my round-2 note). The builder's scratch script run on the snapshot's backups prints, per migration backup, identities dropped while an id or alias is still a served GI / KG person: 0015 11, 0017 4, 0020 10, 0025 89, and none for 0031 or 0033 (0035 has no backup in the snapshot: NOT MEASURED). Those are my run of their script, not a record entry, with no denominator, no (a) / (b) / (c) class and no control |
| Criterion 6's deviation (empty line only for `undecidable`) | **Yes** | `test_only_an_undecidable_keep_may_leave_its_deciding_line_empty` (3 bases + a `WRONG_PERSON` case) kills the any-basis mutant (8c.6). The cost stands: 16 of 17 `undecidable` rows carry nothing an auditor can check, and 6 of the 7 kept "moderate" rows are in that bucket |

### 8c.3 The fresh sample: 40 kept rows outside the 43

Draw: `random.Random(1026131).sample` over the 371 K rows that are not among the 43 (not 77, 20261026 or 918273); 40 rows in 38
episodes (bases in the draw: self_intro 32, addressed 6, host_read 1, clip_own_person 1; the draw holds no `undecidable` row, 7 of
the 371 are). I read each from the transcript (the voice's lines with one line of context, every line naming the person, the
episode's first and last lines, the voice's self-introducing lines) and opened the builder's basis only afterwards.

- **Result: 40 of 40 are right keeps.** 0 wrong. 33 are self-introductions or addressed-and-answering; 7 are a person introduced by
  the host just before speaking, a listener or caller who gives their own name, or a clip of the named person.
- 7 of the 40 carry a minority of another speaker's lines (a clip line or a credit line inside a promo or a vox pop) and each is mostly
  the named person (the majority ruling keeps them).
- Notes that are not defects: one voice (`urn:bbc:podcast:p0p4mkxp` 06) reads a presenter intro that names the series' usual host in the third
  person, never names itself, and is `host_read`; nothing in the text shows someone else. One served name is garbled (`https://iono.fm/e/...` 02
  carries a stray surname word), the right person under a wrong surname, already `CORRECT_MISSPELLED`.
- Beyond the draw, over all 397 K rows: (a) a scan for kept voices whose own lines say the served surname in the third person
  without a self-introduction pattern: 38 flagged; I read the matching line of 21 of them (the rest are sample rows or the host
  voices below): 19 are first-person introductions, one (`urn:bbc:podcast:m002m8zz` 05) is a presenter line "This one came in from"
  followed by "my name is" on the same voice, and one is `1_2766128` 12 (8c.4); (b) a scan for the legal disclaimer over the
  K rows: only the 3 host voices of 8c.2 and one host voice with 113 lines (`29e0b258` 00).
- A sample of 40 with 0 wrong gives no rate; the wrong keeps below were found in the 43 re-read rows, which is where the defects
  concentrate (the `undecidable` bucket and the sponsor / promo class).

### 8c.4 What I read among the 43, and the wrong keeps I confirm

I read 21 of the 43 from the transcript myself: the contested rows (`ee665e22` 05,
`1_2766128` 12, `2329358a` 07, `38156bae` 04, `75146f27` 02, `5114c6d7` 02, `a4c1c410` 02, `62fdb4dc` 03, `2ede0e96` 00, `6be2f208` 02,
`38112b09` 00, `a3717c93` 01, `ad8ee5b3` 02, `c1b9decd` 09, `6af9750d` 08 / 09, `2ede0e96` 01, `00fc117b` 13, `50f2f680` 11, `b27552ed` 04,
`iono.fm/e/1728203` 01) and the first 14 voice lines of the 5 a16z flips (`277d8505` 01, `3346a500` 00, `34e47cf8` 00, `382b7a08` 01,
`25f3d896` 00: each reads a third-person synopsis naming other people as host and guest, then the outro). I did NOT re-read the other
17 of the 43: `1b259b80` 01, `360e9305` 02, `655e460d` 00, `6eb845ef` 02, `2e282338` 01 (removals) and the 12 self-introduction / continuity
keeps (`b7d968f5` 02 / 03 / 08, `6baab6b1` 03, `c196b474` 04, `1294ea90` 03, `2a3e72ef` 01, `29b52580` 00 / 04, `c85ca0b8` 07, `a0e15df4` 01,
`a1395046` 00); for them I rely on the two blind readers' agreement (all 17 agreed A = B) and, for the 5 removals, the disclaimer-text
match. Among the 7 removals I read in full (`38156bae` 04, `75146f27` 02, `5114c6d7` 02, `a4c1c410` 02, `62fdb4dc` 03, `2ede0e96` 00,
`6be2f208` 02) and the 5 flips: **0 wrong removals found** (`a4c1c410` 02 is two lowercase news-clip lines, the weakest). Keeps
I read and accept: `38112b09` 00, `a3717c93` 01, `ad8ee5b3` 02 (the host's own first-person intro, 8c.2), `2329358a` 07 (8c.2),
`b27552ed` 04 (a guest, 27 short lines), `iono.fm/e/1728203` 01.

**Wrong keeps I confirm (bar 0):**

1. **`ee665e22` 05** (27 lines, 28.6 s). 13 of its lines (21.8 s, 76% of the voice's talk) complete a sentence that voice 04 (served
   as another guest) has just begun; 3 lines (2.8 s) answer a question and are followed by voice 06; 11 lines (4.0 s) are
   backchannels. By the majority ruling the name comes off. Both blind readers wrote `undecidable` (the packet did not say to read for
   sentence continuation). Round 2 found the same.
2. **`1_2766128` 12**. A dramatised caller inside a story told by voice 08 (asked for a name, it answers with a first name). Both readers took the
   in-story name as `self_intro`. The readers' own instruction lists "the name is a fictional / acted character not the speaker" as
   `OTHER`, and the builder removed `0308f04c` 02 (an acted poem) and `eae61d33` 06 (a voicemail caller) on that ground. Arguable,
   because the served name equals the character's name; it is a wrong keep under the instruction and inconsistent with those two.
3. **`6af9750d` 08** (31 lines). It reads one produced sponsor spot three times in one episode, word for word. My scan of every
   segments file for the spot's text finds 6 files: the spot's body voice is served as a different show host in `00fc117b`, as this
   voice here, and unnamed in 2 other episodes. One spot, one recorded voice: at most one of those names is right. The label is
   `non_participant` with no name. Probable, not certain.

**Contested, kept by the default, not settled by evidence:** `c1b9decd` 09, `6af9750d` 09, `00fc117b` 13 (the one-line sponsor
lead-in, served as one person in 3 episodes and unnamed in 3), `2ede0e96` 01 (a promo line repeated 6 times, a gift line and two
field-interview questions), `50f2f680` 11 (two lines of a clip, then a promo; lines naming The Wirecutter Show are served under 8 different person names across
the corpus, not all of them this promo). The text neither shows nor excludes the named person; the labels for `c1b9decd` 09, `6af9750d` 08 / 09 and `2ede0e96` 01 say `non_participant`, no name (`00fc117b` 13 has no label). I
do not count these as confirmed wrong keeps. I do count them as the reason criterion 7 is not closed.

### 8c.5 The 12:08 restatement against 7b: episodes 135 against 126 to 131

Judged: **a real miss, honestly recorded in 7b, avoidable, and not a defect of the migration.** The ten added removals (`25f3d896` 00,
`277d8505` 01, `2ede0e96` 00, `3346a500` 00, `34e47cf8` 00, `382b7a08` 01, `5114c6d7` 02, `62fdb4dc` 03, `75146f27` 02, `a4c1c410` 02) sit in
**10 different episodes, none among the 126 already written**; the one voice that went back to a keep (`2329358a` 07) was the only
removal of its episode: 126 + 10 - 1 = **135**, a lookup over the keys the 12:08 block already held. "126 to 131" was written without that
lookup. The other 12:08 integers hold against my own numbers: removals 149 (met), S1 / S2 +59 (inside +49 to +69), S10a +1 (inside
+0 to +3), every must-not-move number equal. Acceptance-B 10 (written episodes = exactly the episodes holding a W key; refused 0)
holds at 135. What is missing is the operator's line: the method counts a `MISSED:` only after one, and 7b has none (the two earlier misses
have theirs).

### 8c.6 D1 (acceptance-E) and the mutations

D1 mutants, in the archive copy:

| Mutant | Result |
| --- | --- |
| `apply_to_payload` returns the input again | **killed** (`test_the_shared_rewrite_never_writes_a_bridge_without_its_identities[renamed]`); the repro is red on the unfixed code |
| `apply_to_payload` returns `dict(payload)` (shallow) | **survived** 668 tests. Acceptance-E 4 asks "killed or argued"; the record argues nothing. It is probably equivalent for the one caller (m0017's clear-then-update keeps the same `identities` list object alive in the copy), but that argument is not written |

Acceptance-E, criterion by criterion: **1** (re-measure equals the last, only the digest differs): superseded, the W set changed, and
7b states its own numbers, which I re-derived (8c.1). **2** (repro 3 of 3 green after, red before): the mutant above is the unfixed
code; 668 pass on the fixed code. **3** (empty bridge files, 0 new): 0 new and 0 healed (8c.1); the record says 15 and I count 17 files with no identities
(16 served, 15 literally `{}`). **4**: 1 of 2 mutants killed, the second not argued. **5** (every user of m0017's bridge rewrite has the
repro or a test of its own, or an argument): m0017 and m0020 have neither (`git diff 7fd0eff36 HEAD` touches neither test file; the record
has no sentence on them). **D1 is not closed**: criteria 4 (second half) and 5 fail on documentation and two small tests, not on behaviour.

The 16 mutants against the whole `tests/unit/podcast_scraper/upgrade` directory (668 tests unmutated), in a
`git archive 9e39accc5` copy:

| # | Mutant | Result |
| --- | --- | --- |
| M1 | receipts file name changed | killed |
| M2 | `require_basis` off for 0034 | killed |
| M3 | quote commercial-type guard removed | killed |
| M4 | quote not typed `unknown` when unnamed | killed |
| M5 | bridge `sources` not set from the match | killed |
| M6 | loader lets any basis leave `segment_idx` / `line_sha256` empty | killed (`...[self_intro]`) |
| M9 | acceptance-C 11: aliases ignored | killed |
| M10 | acceptance-C 12: alias match keeps a removed voice's person | killed |
| M11 | acceptance-D 6: `taken_off` applied to an identity whose own id is still a person | killed |
| M12 | floor `<` -> `<=` | killed |
| M13 | floor 0.05 -> 0.06 | killed |
| M14 | `CORRECT_MISSPELLED` also removes | killed |
| M15 | shared-name refusal removed | killed |
| M16 | insight commercial-type guard removed | killed |
| M7 / M8 | D1, above | killed / **survived** |

**15 of 16 killed.** The 6 round-2 survivors (M1 to M6) are all killed. Acceptance-B 23 (0 survivors, at most 2 argued equivalent):
1 survivor, M8, probably equivalent and unargued.

### 8c.7 What is NOT covered

- Prod: criteria 27 to 29 (plan diff, the 14 post-snapshot episodes, search-index rows, apply counts, ownership, daytime), untouched
  and untouchable on the clone. The derived-surface ladder of 7 (re-enrich the 135, incremental index) is not run.
- I read 54 of the 397 kept rows (40 sampled + 14 of the 43) and 12 of the 149 removals; the other 343 kept rows and the other 137
  removals rest on round 2's reading, the blind readers and my regex scans. A 40-row sample with 0 wrong gives no rate.
- The undo cycle was not re-run by me (the builder's r6 log). Acceptance-B 17 (S7 / S6 on the gold sets, `gold_insights.py`) was not re-run;
  I only checked that the 3 gold-named voices are `CORRECT`. S4, S5, S8, S9, S11 were not measured. `make ci-fast` is the builder's step.
- The 92 labelled names were not re-derived from the 10-17 measurement; I used the builder's `criterion_7` rows.
- The 10 dropped entries and their 8 episodes (their wrong names stay served), D4 of #2094 (the right voice stays unnamed), the 58
  misspelled K names, reprocess durability, and the garbled served surname above.
- Whether the first-person host intro under the announcer ruling is what the operator meant. That is the operator's call, not mine.

### 8c.8 Reasons for the verdict

The mechanics are right and re-derived: 556 / 397, 149 off, 397 byte-unchanged, 59 insights by id, S1 / S2 +59, S10a +1, 135 episodes,
the hash chain, 668 tests, and 15 of 16 mutants killed with all six round-2 survivors dead. The announcer class is applied the same way
to all 15 voices that read the disclaimer or the synopsis, and the mixed-voice ruling is applied by majority where the text allows it.
A fresh 40-row sample of kept rows is clean. But criterion 24's bar is 0 wrong keeps and I confirm one with a measured majority
(`ee665e22` 05, 76% of its talk is another voice's sentence completions) and two more on the readers' own instruction and on the
spot's cross-episode names (`1_2766128` 12, `6af9750d` 08), while the criterion-7 label disagreements are not explained beyond the
default; acceptance-C 20 to 23 are still not in the record; D1 lacks its two tests (or arguments) and one argued mutant; the
135-against-126-to-131 miss has no operator line. None of this makes any name worse than the snapshot serves today (a keep is the
status quo); the repair is small, but it changes the measured outcome (removals 149 -> 151 or more), so it needs a clone re-measure
and a reader of the changed rows, not a condition on this verdict.

What a repair needs, in this order:

1. Flip `ee665e22` 05 to `WRONG_PERSON` (basis `another_person`, the deciding line, `segment_idx`, `line_sha256`); decide `1_2766128` 12 and
   `6af9750d` 08 with the operator's one rule on acted characters and produced sponsor spots, and the five contested sponsor / promo
   voices of 8c.4 with it; re-bind the verdicts and list hashes.
2. Restate removals, S1 / S2 / S10a **and episodes** as integers from the keys (a lookup, as in 8c.5), then re-run the clone gate, the
   undo cycle and the census, and have a fresh reader read the changed rows only.
3. Put the acceptance-C 20 to 23 integers in the record with denominators, classes (a) / (b) / (c) and the control (my run of the
   builder's script: 0015 11, 0017 4, 0020 10, 0025 89, 0031 none, 0033 none, 0035 no backup), and note which of them 0036 repairs.
4. D1: one test each for m0017 and m0020 (an empty id map, the bridge keeps its identities) or one sentence each, and kill or argue
   the shallow-copy mutant.
5. The operator's lines: the 135 miss; the first-person host intro / disclaimer distinction (or its reversal for `a3717c93` 01 and
   `ad8ee5b3` 02).

### 8c.9 Summary

1. Mechanics re-derived with my own code: 556 voices / 397 episodes; 149 names off (= the 149 W keys, 0 other label changes); 397 kept byte-unchanged; 59 insights moved by id; S1 / S2 +59; S10a +1; 135 episodes; every other scoreboard field equal; 668 tests pass at HEAD.
2. Hash chain holds in three places (verdicts file sha = receipts header; list sha in verdicts header = receipts header = my recomputation); public columns equal the private evidence for 546 of 546.
3. Round-2 wrong keeps: 5 of 7 closed (the a16z announcers); `ee665e22` 05 is still a wrong keep (76% of its talk is another voice's sentence completions), `1_2766128` 12 arguable.
4. The 10 moderate rows: 3 became removals; 7 stay kept, 5 of them sponsor / promo voices kept by the default with independent labels against 4.
5. Announcer class: 12 removed, 3 host voices kept on a first-person reading that the record never states; majority rule applied; 2329358a 07 is the one the text cannot settle.
6. Fresh seeded sample (seed 1026131) of 40 kept rows outside the 43: 40 of 40 right; 0 wrong.
7. Mutants: 15 of 16 killed; all 6 round-2 survivors killed; the shallow-copy D1 mutant survives, unargued. D1 criteria 4 and 5 are not met (no m0017 / m0020 test or argument).
8. 12:08 against 7b: episodes 135 against 126 to 131 is a real, avoidable miss (126 + 10 - 1); no operator line. Acceptance-C 20 to 23 still not in the record.
9. Criterion 6's deviation is now pinned by a test and closed; criterion 7 is not closed (4 label disagreements explained only by the default).
10. NO-GO, narrowly: one confirmed and two probable wrong keeps plus three documentation gaps; the repair changes the measured outcome, so it needs a re-measure and a reader of the changed rows.

Verdict: NO-GO the wrong keeps under criterion 24 (bar 0: `ee665e22` 05 confirmed, `1_2766128` 12 and `6af9750d` 08 probable; the criterion-7 sponsor / promo label disagreements explained only by the default), acceptance-C 20-23 still not in the record, D1 (acceptance-E 4-5) without its m0017 / m0020 tests or arguments and one unargued surviving mutant, and no operator line for the 135-against-126-to-131 miss

**Acceptance-C 20-23, exposure of other migrations to the bridge defects** (read only on
snapshot-prod-20261025; `scratchpad/bridge_exposure.py`, `bridge_consistency.py`, `m0036/gaps.py`):
identities dropped from a bridge while their id or an alias was still a GI / KG person, by the
migration backup that last held them: 0015 11, 0017 4, 0020 10, 0025 89; 0029, 0031 and 0033: 0;
0035 has no backup in the snapshot (not applied there) and was measured on its clone instead (its
gate writes no empty bridge; publishable gaps are 0036's population). Corpus-wide, 115 GI / KG
persons in 45 episodes had no bridge identity (74 of them in 15 bridges 0025 left `{}`, D1): the
repair is 0036. None of these are 0034's to change.


## 8d. Verdict (round 4)

Dated: 2026-10-26. Step-8 reviewer, round 4: a fresh agent, not the builder and not the round-1, round-2 or round-3 reviewer. Commit
under review `0e9f4454d`. 0034's code and list are unchanged since `ffaae4ae0`; since round 3 the only source change under
`src/` or `tests/` that touches 0034 is the verdicts file (14 lines) and the D1 test file (`git diff 9e39accc5 HEAD --stat -- src
tests scripts`). The gate ran on `a859902ab` clean and `git diff a859902ab HEAD --stat -- src tests scripts pyproject.toml` is empty.
Data: before = `snap1025/corpus`, after = `gate34g`. Nothing was written into the repo except this section; every mutation ran in a
`git archive 0e9f4454d` copy (`scratchpad/r8d34/code`). Voices are named by episode-id prefix and voice id only; no name and no
transcript text is quoted here. My scripts are in `scratchpad/r8d34/mine/` (none is the builder's or an earlier reviewer's).
Secrets: `secrets-scan --range 9e39accc5..HEAD` (gitleaks 8.30.1) PASS; my own regex pass over `git diff 7fd0eff36 HEAD` found 0
hits.

### 8d.1 Numbers re-derived with my own code

| What | Result |
| --- | --- |
| Population | floor over `select_served_artifacts` and each served `segments.json`: **556 voices in 397 episodes**; the 546 shipped keys are all in it (10 not shipped: the documented D15 drop); every list name equals the derived served label (0 differ) |
| Verdicts | 546 rows; CORRECT 380, CORRECT_MISSPELLED 11, WRONG_PERSON **155** => kept **391**. W episodes: **138**. Public `verdict` / `basis` / `segment_idx` / `line_sha256` equal the private evidence file for 546 of 546 (0 diffs). 10 rows have an empty `segment_idx` / `line_sha256`, all CORRECT + `undecidable` (11 undecidable rows, one carries a line) |
| Hash chain | `sha256` of the shipped verdicts file `4d451ccb...` = the receipts header `verdicts_sha256`; `rv.list_sha256(load_list())` = `4285e56b...` = the verdicts header = the receipts header; all 546 entries carry parseable `talk_s` / `total_s` with share < 5% (0 exceptions); receipts file 1,659 lines (header + 1,658) |
| Names off | label diff of every served `segments.json`, before vs after: **155** voices lost their label = the 155 W keys (set difference 0 both ways); **0** K labels changed; text, times and voice ids identical (0 diffs) |
| Insights by id | own diff of every `.gi.json`: exactly **59** insights changed, all shown -> hidden `unknown`, in 33 episodes; 0 added, 0 removed |
| Scoreboard | committed command on both trees (52 flattened fields): S1 16,810 -> 16,869 (**+59**), S2 30,563 -> 30,622 (**+59**), S2 unknown +59, S10a 190 -> 191 (**+1**); every other field equal (S1c 0, S1n 7, S3 9,359, S10b 2, S12 24, commercial 94, A1 382 / 290, insights 130,401, quotes 252,651) |
| Episodes | **138** episodes hold a W key = the gate's "rewrote 138", refused 0; the restated 14:23 integers (155 / 391 / 138 / +59 in +59 to +65 / +1 in +1 to +2) all hold against my numbers |
| Tests at HEAD | `tests/unit/podcast_scraper/upgrade` in the archive copy: **674 passed** |
| Undo | not re-run by me; the builder's `undo_cycle_r7.log`: restored 1,658, refused 0, corpus content differing 0 (of 143,670), apply again 138, undo again the same, other migrations' backups changed 0: UNDO-CYCLE PASS |

### 8d.2 The round-3 NO-GO items, one by one

| Round-3 item | Closed? | Evidence |
| --- | --- | --- |
| `ee665e22` 05 (confirmed wrong keep) | **Yes** | now `WRONG_PERSON`, basis `merged_voices` |
| `6af9750d` 08, `6af9750d` 09, `c1b9decd` 09, `00fc117b` 13, `50f2f680` 11 (sponsor / promo voices kept by the default) | **Yes** | all five are `WRONG_PERSON` / `ad_or_promo_not_own` now |
| `1_2766128` 12 (acted caller, probable) | **Not changed, not counted** | both blind readers wrote `self_intro` and it stays `CORRECT`; I did not re-read it; arguable, as in round 3 |
| `2ede0e96` 01 (a round-3 contested row) | **No, never re-read** | see 8d.3 item 1: the builder's `reread4/build.py` names `"2ede0e96 00"` (a row already removed, filtered out by the `ship[k] != "WRONG_PERSON"` line), not `01`; the row is not among the 20 keys in `reread4/keys.json` |
| Operator's third ruling applied consistently across ALL kept rows | **No** | 8d.3: 7 kept voices, found by my scans and by a fresh sample, are wrong keeps under the operator's three rulings (1 sponsor / promo-tag voice, 1 interviewer voice carrying the guest's name, 5 mixed voices whose majority is someone else); 2 more are probable |
| Acceptance-C 20-23 | **Partly** | the record now carries integers (0015 11, 0017 4, 0020 10, 0025 89, 0029 / 0031 / 0033 0, 0035 no backup), which I reproduced with the builder's script. Still missing: the denominators (criterion 20), the control "1 hit before the fix, 0 after" (21), the (a) / (b) / (c) classes (23). My own alias-only signature (acceptance 20's definition), per migration backup: bridge files in backup / with a served counterpart / episodes with a hit / identities hit: 0015 93 / 93 / 0 / 0, 0017 60 / 60 / 3 / 3, 0020 24 / 24 / 0 / 0, 0025 40 / 40 / 0 / 0, 0029 76 / 76 / 0 / 0, 0031 53 / 53 / 0 / 0, 0033 1 / 1 / 0 / 0; the builder's broader "id or alias" definition gives the record's 11 / 4 / 10 / 89, so the record's numbers and acceptance 20's signature disagree and the record does not say which it reports. 0036 exists as the repair record (criterion 22). Control (21) not measured by me (no 09:27 clone) |
| D1 tests for m0017 / m0020 and the deep-copy contract | **Yes** | `test_bridge_rewrite_empty_id_map.py`: m0017 bridge rewrite with an empty id map keeps the identities (2 params), m0020 reaches it only through m0017's `_Episode` (identity asserted), `apply_to_payload({}...)` returns a deep copy (nested mutation of the result leaves the input unchanged). My 4 mutants in the copy: returns the input (4 tests fail), `dict(payload)` (1 fails), `copy.copy(payload)` (1 fails), list-copy only (1 fails): **4 of 4 killed**. The record still carries no sentence on it (documentation only) |
| Operator line on the 135 miss | **Yes** | 14:17 block: "the MISSED episode count (135 against 126 to 131) is accepted" |
| Criterion 7 label disagreements | **Not re-examined** | not re-run; the five sponsor / promo rows it named are decided above except `2ede0e96` 01 |

### 8d.3 Wrong keeps I find (bar 0): 7 confirmed, 2 probable

Rule applied: the operator's three rulings of 2026-10-26 (mixed voice: majority decides; announcer; sponsor tag / house promo unless
the text ties the voice to the person, by self-introduction, address, or being introduced just before speaking). Seconds are the
printed segment times of the voice's own lines. A host-read ad kept under ruling 1 and a tie by introduction (`60217010` 06: the
host says the colleague's name, then the voice reads a first-person testimonial) are not counted.

Confirmed:

1. **`2ede0e96` 01.** 12 lines in 19.3 s: 9 are sponsor / house-promo tags (a gift-subscription tag at segment 8, and seven
   one-line tags for a house show at 700 to 707), 3 are interview questions (59, 73, 473). No self-introduction, no address, nothing
   introduces it. Mostly a tag voice with nothing tying it to the named person. One of the five rows round 3 named; dropped from
   round 4 by an id typo (above). Basis `undecidable`, kept.
2. **`https://iono.fm/e/1728203` 01** (basis `continuity`, deciding segment 197). 6 lines, 22 s, all questions or interjections put
   TO the unnamed voice 05, which gives the long answers (123 s). Segment 197 only announces the clip. The same two questions
   (segments 199 and 200) appear in the served episode `https://iono.fm/e/1728106`, segments 43 and 44, on a voice labelled with
   another person's name, and the answers to them (45 onward) are on a voice labelled with a near spelling of the name 1728203 gives to voice 01. At most one of the two
   labellings is right; the in-episode dialogue and the second episode say it is not this one. Round 3 "read and accepted" this row.
3. **`f6be2d45` 12.** 54 s: segments 278 to 280 (21 s) are another show's promo, 281 (15 s) is the named person's promo, 372 (18 s)
   is a story voice in a print shop. The named person's share is 15 of 54 s (28%); even without the first promo, 372 alone is more.
4. **`23620594` 13.** 39 s: segment 141 (23 s) is a two-host promo for another show (it addresses a named co-host and carries a
   second speaker's "Yeah"), 142 / 144 / 145 (16 s) are the named person's promo. 41% named.
5. **`13fcb6ee` 07.** 60 s: segments 259 to 267 (about 53 s) are a produced listener-voicemail spot that recurs word for word in 14
   other episodes under other labels; 406 to 409 (about 9 s) are a live audience member introducing herself. The two are not shown
   to be one person and the spot is 87% of the voice.
6. **`cc425097` 00.** 17 s: segments 256 and 290 (10 s) each complete a sentence that voice 04 (served as the guest) has just begun
   (the previous segment ends mid-sentence); 421 and 422 (the sponsor-supporter line, about 7 s) are the named person's. 59% is the
   guest's own speech, the same shape as `ee665e22` 05 (76%) that round 3 called a confirmed wrong keep.
7. **`fc255834` 04.** 33 s of a fan vox pop: only segments 472, 475, 479, 484 and 490 (about 6 s) are the interviewer's; the rest
   (477 to 478, 480 to 483, 485 to 487, 491 to 495, 498) are fans answering. The named person is about 18% of the voice.

Probable (margins thin, not counted in the bar-0 statement):

8. **`7a7bc90f` 05.** 12 s: segment 0 (5 s) names the person as interviewer; 628 / 629 / 631 (7 s) are listener clips from the same
   recurring spot as item 5.
9. **`f1dfec6a` 13.** 46 s: segment 164 (24 s) is the same two-host promo as item 4, 165 to 168 (22 s) the named person's.

Arguable, not counted: `e7d4a370` 02 (a sponsor read and a membership promo by a voice labelled as a co-host of the show; the
text says "our show" and "our listeners" but names nobody, and the same recording carries the same label in 7 other episodes, one
under another co-host's name; kept under ruling 1 as a host-read ad, which sits uneasily with ruling 3's "tied by such a line"),
`0ec506f0` 08 (a student line that continues another voice's sentence, 5 of 14 s), `0165ea4c` 02 (an interviewer voice that also
completes the guest's sentences at 100 and 115; I did not read all 14 lines against the guest's).

How I found them (so the set can be re-run): (a) every K voice with 6 or fewer lines (103 of 391) read in full; (b) 70 K voices
flagged by four scans (promo wording share, promo without a first-person tie, sentence-continuation share, short without a tie);
(c) a cross-episode scan for K voices whose lines recur verbatim in other episodes under a different label (items 2, 5);
(d) a fresh sample (8d.4). The builder's round-4 set of 20 came from the round-3 reviewer's list and one promo-wording scan
(`reread4/sponsor_candidates.json`, 14 voices); it holds none of items 1 to 7.

### 8d.4 The fresh sample: 40 kept rows

Draw: `random.Random(4034526).sample` over the sorted 391 K keys (not 77, 20261026, 918273 or 1026131). I read each voice's first
lines, its lines naming the person, and one neighbouring line from the other side.

- **Result: 38 right, 2 wrong** (items 6 and 7 above). The one voice that looked wrong at first (`60217010` 06, an ad read) is
  tied by the host introducing the person immediately before the read, so it counts as right. 0 undeterminable.
- Wrong keeps in the draw 2 of 40 (5%). That is not a rate for the 391 (the rows I scanned are not the sample), but it shows the
  defect is not confined to the 20 rows round 4 re-read.

### 8d.5 What is NOT covered

- 231 of the 391 K rows were in none of my sets (not 6 or fewer lines, not flagged, not sampled); I read none of them. The 70
  flagged rows I read in part: the ones named in this section and about 45 others; the rest of the flagged rows I judged from the
  scan output and the self-introduction line only.
- The 155 removals: I read none this round (round 3 read 12). The six new removals (`ee665e22` 05, `6af9750d` 08 / 09, `c1b9decd`
  09, `00fc117b` 13, `50f2f680` 11) I did not re-read; I rely on both blind readers agreeing.
- The undo cycle (builder's log only), acceptance-B 17 (S6 / S7 on the gold sets), S4, S5, S8, S9, S11, `make ci-fast` (the
  builder's step), prod criteria 27 to 29, the derived-surface ladder of 7.
- 16 mutants of round 3: not re-run (0034's code is unchanged since `ffaae4ae0`); only the 4 D1 mutants were run, all killed.
- The 8 episodes of the 10 dropped entries (their wrong names stay served), D4 of #2094, the 58 misspelled K names, reprocess
  durability, the criterion-7 label disagreements as a class.
- Whether the operator's ruling 3 ("tied by such a line") also covers a host-read ad with no tie (the `e7d4a370` 02 question).

### 8d.6 Reasons for the verdict

Everything mechanical holds and is re-derived: 556 / 397, 155 off = the 155 W keys, 391 byte-unchanged, 59 insights by id, S1 / S2
+59, S10a +1, 138 episodes, the hash chain in three places, 674 tests, D1 closed (4 of 4 mutants). Round 3's confirmed and
probable wrong keeps are closed except `2ede0e96` 01, which was never re-read. But criterion 24's bar is 0 wrong keeps and the
operator's third ruling was applied to the rows the round-3 reviewer happened to name and to one promo-wording scan, not to all
391 kept rows. I find 7 wrong keeps under the operator's own rulings (2 of them in a sample of 40), 2 more probable, and
the round-4 build lost a named row to a typo. Each is a name left on a voice whose text shows it is someone else's; none
makes anything worse than the snapshot serves today, but the bar is 0. The repair is small (7 to 9 rows to `WRONG_PERSON`, basis,
deciding segment and hash, re-bind the verdicts and list, restate removals 162 to 164 and the episodes by lookup, re-run the gate,
the undo cycle and a reader of the changed rows) and it needs a second scan of all 391 under the three rulings, not another list
of names.

What a repair needs, in this order:

1. Flip items 1 to 7 (and decide 8 and 9) with deciding segment and line hash; re-bind the verdicts and list hashes.
2. Scan the 231 K rows I did not touch with the scans in 8d.3 (promo wording, tie by self-introduction / address / introduction,
   sentence continuation, verbatim recurrence across episodes) and have two blind readers read what they flag, with the 70 flagged
   rows.
3. Restate removals, episodes, S1 / S2 / S10a as integers from the keys before the clone is re-measured; re-run the gate, the undo
   cycle and the census.
4. Acceptance-C 20 to 23: add the denominators, the control and the classes, and say which definition the record's 11 / 4 / 10 / 89
   follow.

### 8d.7 Summary

1. Mechanics re-derived with my own code: 556 voices / 397 episodes; 155 names off = the 155 W keys, 0 other label changes; 391 kept byte-unchanged; 59 insights by id; S1 / S2 +59; S10a +1; 138 episodes; every other scoreboard field equal; 674 tests pass at HEAD.
2. Hash chain holds in three places; public columns equal the private evidence for 546 of 546.
3. Round-3 wrong keeps: `ee665e22` 05, `6af9750d` 08, and the four contested sponsor / promo rows are closed (now `WRONG_PERSON`); `1_2766128` 12 unchanged and arguable; `2ede0e96` 01 never re-read (id typo in the round-4 build).
4. The operator's third ruling is not applied across all kept rows: I find 7 wrong keeps (sponsor / promo or mixed voices whose majority is someone else) and 2 probable; `https://iono.fm/e/1728203` 01 is confirmed against a second served episode.
5. Fresh seeded sample (seed 4034526) of 40 kept rows: 38 right, 2 wrong (`cc425097` 00, `fc255834` 04).
6. D1 closed: m0017 / m0020 tests plus the deep-copy test; 4 of 4 mutants (returns input, shallow `dict`, `copy.copy`, list-only) killed.
7. Operator line on the 135 miss: present (14:17).
8. Acceptance-C 20 to 23: integers now in the record, reproduced; denominators, control and classes still missing; my alias-only signature gives 0017 3 hits in 3 of 60 bridges and 0 elsewhere, the record's broader definition gives 11 / 4 / 10 / 89.
9. Not covered: 231 of 391 kept rows unread; the 155 removals not re-read this round; undo cycle, gold sets, ci-fast, prod.
10. NO-GO on criterion 24 (bar 0): 7 confirmed and 2 probable wrong keeps, one lost to a build typo; the repair is small but changes the measured outcome, so it needs a re-measure and a reader of the changed rows.

Verdict: NO-GO criterion 24 bar 0 not met: 7 confirmed wrong keeps under the operator's three rulings (`2ede0e96` 01, `https://iono.fm/e/1728203` 01, `f6be2d45` 12, `23620594` 13, `13fcb6ee` 07, `cc425097` 00, `fc255834` 04) and 2 probable (`7a7bc90f` 05, `f1dfec6a` 13), found by scans of the 391 kept rows and a fresh sample of 40 (2 wrong), with `2ede0e96` 01 lost from the round-4 re-read by an id typo; acceptance-C 20-23 still lacks denominators, control and classes

**Acceptance-C 20-23, with denominators, classes and control** (2026-10-26, round 4's condition;
`scratchpad/bridge_exposure2.py` on snapshot-prod-20261025, read only; per migration backup: bridges
the backup holds, then dropped person identities by class: (a) alias-only, the own id no longer a
GI / KG person while an alias still is and no served identity carries it (the alias defect fixed in
a1698c214); (b) the own id still a GI / KG person, yet no served identity carries it):
- 0012 187 bridges, 0 / 0; 0014 39, 0 / 0; 0015 93, (b) 11; 0016 45, 0 / 0; 0017 60, (b) 4;
  0019 1, 0; 0020 24, (b) 10; 0025 40, (b) 89; 0029 76, 0; 0030 120, 0; 0031 53, 0; 0033 1, 0;
  (0018, 0021-0024, 0027 hold no bridge backup). Class (a): 0 in every migration by this signature.
  The round-4 reviewer's own alias-only signature gives 3 hits in 3 of 60 bridges for 0017 and 0
  elsewhere: the two signatures differ in whether an alias already carried by another served
  identity counts; not settled here, and both are listed.
- **Control:** the same check on 0034's clone at 9a7118eb4 (before the alias fix) gave 1 class-(a)
  identity in 1 episode; after the fix, 0.
- Class (b) is the bridge-gap population 0036 repairs (with the empty bridges 0025 left); 0034's
  own exposure is 0 in both classes after the fix.


## 8e. Verdict (round 5)

Dated: 2026-10-26. Step-8 reviewer, round 5: a fresh agent, not the builder and not the round-1 to round-4 reviewer. Commit under
review `a8d4eba2b` (clean tree). Data: before = `snap1025/corpus`, after = `gate34h`. Nothing was written into the repo except this
section; mutants ran in a `git archive a8d4eba2b` copy. Voices are named by episode-id prefix and voice id only; no name and no
transcript text is quoted. My scripts are in `scratchpad/r8e34/` (none is the builder's or an earlier reviewer's). I opened the
round-5 reader files only after my own blind sample and scans were written down.

### 8e.1 Numbers re-derived with my own code

| What | Result |
| --- | --- |
| Verdicts | 546 shipped keys, all in the frozen list, every name equal to the list's; CORRECT **374**, WRONG_PERSON **172**, 0 CORRECT_MISSPELLED; W episodes **150**. Against the round-4 file: the 155 earlier W rows are all still W (0 flipped back), 17 rows are new |
| Public vs private | `verdict`, `basis`, `segment_idx`, `line_sha256`, `name` equal for 546 of 546 (0 diffs), same key set |
| Hash chain | 528 rows carry a segment and a hash: `sha256(private evidence line)` = public `line_sha256` = `sha256(strip(served segment text at segment_idx))` for **528 of 528**; 468 point at the voice's own line, 60 at another voice's line (an introduction or address). The other 18 rows carry neither: all CORRECT / `undecidable` (the rule). Shipped verdicts sha `8332a450...` = gate = receipts header; list sha `4285e56b...` = verdicts header = receipts header (`rv.list_sha256(load_list())` in the archive copy) |
| Names off | per-row comparison of 389 served `segments.json` (285,771 rows): the **2,392** rows of the 172 W voices lose their label (`voice_type` none -> `unknown`); **0** label changes on any other row, **0** changes to text, times, voice id or role; the 172 W keys all lose the label, 0 K key changes |
| Tree | 1,780 files differ = 1,779 corpus files + the ledger; 0 missing; 150 episodes. The 374 kept voices sit in files whose rows I compared one by one (above) |
| Insights by id | own diff of the 146 changed `.gi.json`: **67** insights changed, all shown -> `connect` with the speaker moved off the name, in 37 episodes; 0 insight nodes added or removed; 38 Person nodes and 116 SPOKEN_BY edges removed; quote nodes re-anchored (8,096 offset-only changes, 17 with the `Name: ` prefix dropped from the text). Equal to the apply message (`insights_reattributed=67`) |
| Scoreboard | committed command (`scripts/measure/corpus_scoreboard.py`) on both trees, 52 flattened fields: S1 16,810 -> 16,877 (**+67**), S2 30,563 -> 30,630 (**+67**), S2 unknown +67, S10a 190 -> 191 (**+1**); every other field equal (diff script `sbd.py`) |
| Restated integers | 172 removals, 374 kept, 150 episodes, S1 / S2 +67, S10a +1: all inside the 15:29 and 15:47 ranges (162 to 185 / 361 to 384; +59 to +72; +1 to +3) |
| Tests | archive copy: `test_m0034_talk_floor_names_removed.py` + `test_reviewer_verdicts.py` **79 passed**; the full upgrade directory (674 at round 4) not re-run |
| Undo | not re-run by me; read `undo_cycle_r8.log`: two cycles, restored 1,801, refused 0, corpus content differing 0 of 143,813, other migrations' backups changed 0, UNDO-CYCLE PASS |
| Bridges | 150 written, the 2 `{"identities": []}` over `{}` (same two as rounds 3 and 4) are named in 7d; 0036 rebuilds them |

Mutants (archive copy only): floor `<` -> `<=` (killed: `test_a_voice_at_exactly_the_floor_keeps_its_name`); one W row flipped to
CORRECT in the shipped verdicts (killed: `test_registered_with_its_list_and_verdicts_shipped`); one basis set to a made-up value (killed,
same test); **one `line_sha256` altered: SURVIVED** (79 pass). No committed test checks the hash against a transcript (the
transcript and the evidence file are not in the repo), so the hash chain is guarded only by a reviewer's own check. Mutants of my own
checks: a changed K label and a W label left on a row, each injected into my per-row comparison: both flagged (1 each).

### 8e.2 The round-4 NO-GO items, one by one

| Round-4 item | Closed? | Evidence |
| --- | --- | --- |
| `2ede0e96` 01 | **Yes** | W, basis `ad_or_promo_not_own`, segment 8 |
| `f6be2d45` 12, `23620594` 13, `13fcb6ee` 07, `cc425097` 00, `fc255834` 04 | **Yes** | all W, basis `merged_voices`, each read by me (8e.4) |
| `7a7bc90f` 05, `f1dfec6a` 13 (probable) | **Yes** | W, `merged_voices` |
| `https://iono.fm/e/1728203` 01 (confirmed) | **No** | still CORRECT; the basis was changed from `continuity` to `addressed`, segment 197. See 8e.3 item 1 |
| `1_2766128` 12, `0165ea4c` 02, `e7d4a370` 02, `0ec506f0` 08 (arguable) | decided | `1_2766128` 12 and `0165ea4c` 02 stay CORRECT (I read both: self-introduction in a call, and an interviewer voice; I count neither wrong); `e7d4a370` 02 and `0ec506f0` 08 are W now (8e.5, 8e.4) |
| Scan of ALL kept rows under the three rulings | **Done by the builder** (4 batches, 8 readers, 13 splits to a third reader); my check of the result in 8e.3 | the builder's "5% of the unread 231" prediction (162 to 185) held: 17 new removals |
| Acceptance-C 20 to 23 | **Yes, except the control** | 8e.6 |
| D1 tests, operator line on the 135 miss | carried (closed in round 4) | not re-opened |

### 8e.3 Criterion 24 (bar 0 wrong keeps): what I did and found

Method, in the order I did it:

1. **Fresh seeded draws, read blind.** `random.Random(55130426).sample` of **45** of the 374 K keys; then `random.Random(8851027).sample` of 30 more among the K rows with basis `self_intro` not in the first draw. Seeds not used by any earlier round (77, 20261026, 918273, 1026131, 4034526). For each voice: its first lines, a middle and last line, its lines naming the person, and the neighbouring lines of other voices, plus the full voice where a doubt arose. **75 read: 75 right, 0 wrong.** Arguable, counted right: `tag:soundclo` 02 (about 37 of 48 s are theme-song lyrics, the singer is not shown, the name comes from an intro of "the creators"; nothing shows someone else), `substack:pos` 02 (a sponsor clip introduced by the host just before it plays).
2. **Targeted scans over all 374 K rows** (own code, `scan.py`, `recur.py`, `intro2.py`, `runs.py`): promo-wording share of the voice's seconds >= 0.5 (24 voices: every one carries a self-introduction inside the promo, so ruling 3 keeps them); share of seconds that complete another voice's unfinished sentence >= 0.25 (18 voices, all read); lines recurring verbatim in other episodes under a different name (14 voices, 1 real: item 1 below; the rest are the same boilerplate lines read by a host); self-introductions inside a voice naming someone else (6 voices, all misspellings or a two-host promo merged into one voice); voices in several distant runs with the name tied in only a small share (89, 14 read). I also opened **all 87 K rows whose basis is not `self_intro`** (`addressed` 35, `continuity` 22, `undecidable` 18, `clip_own_person` 9, `host_read` 3): first lines and context for each, the full voice for about 25.
3. Then the round-5 reader files for the rows I had flagged.

Found:

1. **`https://iono.fm/e/1728203` SPEAKER_01 (CORRECT, `addressed`, segment 197): a CONFIRMED wrong keep, the round-4 item the builder did not close.** In the same file: voice 06 announces the analyst (segments 196 to 198), then voice 01 only asks and prompts (6 lines, 22 s; segments 199, 200, 212, 220 to 222) and voice 05 gives every long answer (202 to 225). The deciding segment 197 is voice 06's announcement, not a line of voice 01. In the served episode `https://iono.fm/e/1728106` the same two question lines are segments 43 and 44 under the anchor's label, and the answer (45 onward) under the analyst's label. The builder's own carried row 9 (`iono.fm/e/1728203` 08, W: "it asks the question; the named analyst answers") has this shape. Readers: A WRONG_PERSON on exactly this structure, B CORRECT, third reader C CORRECT while writing that "which of the two is [the analyst] is not shown". The readers' instructions describe one episode's transcript per packet and bar the change records, where round 4's cross-episode finding is written; two of three readers outvoted the one who read the structure. The text shows someone else: a keep fails the bar.
2. **`gid://art19-episode-locator/V0/CTf886...` SPEAKER_03 (CORRECT, `undecidable`): an unresolved inconsistency, not counted as confirmed.** The voice is the show's intro and outro spot (third-person synopsis of the debate, a promo code, "like and subscribe", 21 of 22 lines), named by the feed's host list (`llm_resolution`, `known_hosts`). The same show's spot in `gid://art19-episode-locator/V0/Qv7FB...` SPEAKER_00, same label, was removed as an announcer by both readers. Here reader A removed, B kept, third reader C kept on one first-person hand-over line. The same produced spot is treated two ways; the readers' own instruction says a recurring produced spot is an announcer, which makes this a removal.
3. Not wrong, checked because a scan flagged them: `1_2721875` 02 (the co-host named in the opening, asks every question), `9214c273` 04 (the guest's other voice per the episode intro), `ec8ef41e` 01 (tail fragments of the guest's own sentences), `brilliantexp` 01, `10fe0293` 05, `ee5028a8` 03, `6211e32b` 02 (each continues the same person's sentence), `d9759ebd` 03 (named 39 s of speech vs 18 s shown to be other actors, indeterminate clips 26 s: no majority other).

Result: **1 confirmed wrong keep in 374, found by a targeted scan and by reading the round-4 item; 0 in 75 randomly drawn rows.** The bar is 0.

### 8e.4 The 17 new removals, read

All 17 read against the transcripts: **17 right under the rulings, 0 wrong removals.** Per row (segment times are the printed ones, hand-added):
`0ec506f0` 08 (a chatbot persona 4 s; a student's statement continuing another voice's sentence 11 s); `1199806b` 05 (a fictional fruit character); `13fcb6ee` 07 (a recurring produced voicemail spot, 53 of 60 s, and a live audience member); `2ede0e96` 01; `4e16892d` 02 (a call recording of the reporter and the people she calls); `7a7bc90f` 05; `cc425097` 00; `e60b15c2` 03 (an owner, a cook and a reporter in one voice); `f6be2d45` 12; `fc255834` 04 (fan vox pop); `feefff28` 00 (15 of 21 s a town-hall clip); `gid://art19...Qv7FB...` 00 (announcer). Five rest on the builder's choices: `23620594` 13 (named 15 of 39 s, 3 of 4 segments), `f1dfec6a` 13 (15 of 44 s, 3 of 5 segments), `6ebbe026` 00 (23 of 49 s, 6 of 8 segments), `730396fd` 02 (22 of 49 s, 9 of 12 segments) decide on seconds against segments (8e.5 a); `e7d4a370` 02 decides on ruling 3 (8e.5 b).

### 8e.5 The builder's two choices

**(a) Majority of a mixed voice in seconds of talk, not segment count: follows the rulings as written.** The ruling says "majority decides" and the readers were told "MOST of the voice's talk"; talk is time, and the segment lengths in these voices run from 0 s to 24 s, so a count lets one-second fragments outvote a 24 s spot. It was applied only to the 13 split rows, so I tested the other direction: the seconds-based scans of 8e.3 (promo share, continuation share, recurrence share) find no other K voice where seconds would flip an agreed keep. Two of the four rows it decides are thin (45% and 47% named by my hand count); the other two are 38% and 34%. The rulings do not name the unit; the record says seconds, the operator should confirm it once.

**(b) `e7d4a370` SPEAKER_02 written as a removal against the readers' 2-of-3 keep: follows ruling 3 as written.** The voice is only a sponsor read (segments 12 to 23) and the house promo (473 to 485), 18 of 18 lines; no line ties it to the name (no self-introduction, no address); the episode's other voice says the co-host is away (segment 36). Ruling 3 removes such a voice "unless the text ties it"; ruling 1 (a host reading an ad keeps the name) applies only if the voice is shown to be the host, and the text does not show it. The two other rows with this label are W as well (`8536937e` 02, `c2fe4ed8` 03), so the removal is consistent. The rulings pull against each other here; the builder marked it "operator to confirm" and I agree it needs that yes. I do not count it as a wrong removal. `CORRECT_MISSPELLED` written as `CORRECT`: keep and remove are unchanged (the migration treats the two alike); the cost is that the 58 misspelled kept names are not visible in the verdicts file, which is not a defect of this change.

### 8e.6 Acceptance-C 20 to 23

Own code (`expo.py`, `expo2.py`, `expo3.py`) on `snap1025/corpus/.podcast_scraper/upgrade-backups/`: bridges in the backup (denominator): 0012 187, 0014 39, 0015 93, 0016 45, 0017 60, 0019 1, 0020 24, 0025 40, 0029 76, 0030 120, 0031 53, 0033 1 (0018, 0021 to 0024, 0027 hold none): equal to the record. Criterion 20's literal signature (dropped by id, alias still a GI / KG person) gives 4 hits: 0012 one, 0017 three. I read all four: each dropped id is a title-prefixed or plural form whose alias is the id of a served person identity, so the person is carried; with that exclusion (the record's class (a)) the count is **0 in every migration**. Class (b) (own id still a GI / KG person, no served identity carries it): 0015 11, 0017 4, 0020 10, 0025 89, **equal** to the record. 0034's own exposure: 148 changed bridges, 0 hits by either signature. The record's block now states the class definitions; the "not settled" sentence is settled by this reading. Control (criterion 21, 1 identity in 1 episode before the fix, 0 after): **not re-run by me**, the 09:27 clone is not among my inputs; the builder's statement stands unchecked.

### 8e.7 Other findings

- Record lines 352 to 362 (the 15:47 block) write a person's name three times and quote a transcript sentence. The no-names rule (ids only) applies to the record; rewrite those lines with ids before landing.
- The altered-hash mutant survives in CI (8e.1): no committed check protects `line_sha256`.
- The readers' packets are one episode's transcript each (per their instructions); a cross-episode check (same lines under another label) was never part of the re-read, which is why item 1 survived. My `recur.py` is that check: run on all 374 K it names one real case.

### 8e.8 What is NOT covered

- I opened 152 of the 374 K rows (75 drawn + 87 by basis, 10 in both); the scans also made me read some of the other 222, which I did not enumerate. At least the part of those 222 (all `self_intro`) that no scan flagged was never opened by me. A wrong keep among them that is not a promo, a continuation, a recurrence, a mismatched self-introduction or a multi-run voice passes my scans. The `self_intro` rows with a clip inside a larger interview voice are the shape I can least exclude.
- Of the 87 non-`self_intro` rows I read first lines and context for all and the full voice for about 25; the rest I judged from the printed lines.
- The 155 removals of round 4: not re-read by me; I rely on round 4's reading and the readers' agreement. Of the 17 new ones I read all, but with hand-added segment times for the five seconds-dependent rows.
- Seconds shares for the other 378 voices: only scans, not a read of every voice's turn lengths.
- Control 21; the undo cycle (log only); acceptance-B 17 (S6 / S7 on the gold sets), S4, S5, S8, S9, S11; `make ci-fast` on the committed tree; prod criteria 27 to 29; the derived-surface ladder of criterion 7 (search-index rows, enrichment outputs) for the 172 / 174 voices.
- The full upgrade test directory (674 at round 4) and the 16 round-3 mutants: not re-run; only the 79 0034 tests and 4 mutants.
- The 10 dropped entries (D15), the 58 misspelled kept names, D4 of #2094, reprocess durability.
- Whether the operator's seconds reading (a) and the ruling-3 reading (b) are what he means: the operator's yes is still open.
- Whether the two iono.fm episodes' labelling is a wrong label in the snapshot only for 1728203 or also 1728106 (I read 1728106's labelling as right; I did not check the analyst's voice there end to end).

### 8e.9 Reasons for the verdict

Every mechanical check holds and is re-derived: 546 keys, 172 names off = the 172 W keys, 374 kept rows byte-unchanged, 67 insights by id, S1 / S2 +67, S10a +1, 150 episodes, the hash chain in three places for 528 of 528 rows, the 17 new removals right. Acceptance-C 20 to 23 are closed except the control. But criterion 24's bar is 0, and one wrong keep that round 4 confirmed with a second served episode is still kept: the builder re-read it, two of three readers outvoted the one who saw the question-and-answer structure, and the cross-episode evidence never reached them. A second voice is the same produced spot as one the readers removed in a sibling episode. The repair is small: set `iono.fm/e/1728203` 01 to `WRONG_PERSON` (`another_person`, deciding segment 199 or 212, with its line hash) and settle `CTf886...` 03 beside its sibling; re-bind verdicts and list shas; restate removals 172 -> 173 or 174, episodes 150 -> 150 or 151 (1728203 already holds a W key; the other episode holds none), S1 / S2 stay +67 (neither name carries an insight in the served GI), S10a +1; re-run the gate, the undo cycle and a reader of the two changed rows; add a cross-episode recurrence pass (`recur.py` shape) to the re-read so the same evidence reaches the reader; rewrite record lines 352 to 362 with ids.

### 8e.10 Summary

1. Mechanics re-derived with my own code: 546 keys, CORRECT 374 / W 172, 150 episodes; 2,392 rows of the 172 W voices lose their label, 0 other label, text, time, voice-id or role changes in 285,771 rows.
2. Hash chain holds for 528 of 528 rows (evidence line = public hash = served segment text), the 18 rows without a line are `undecidable`; verdict and list shas equal in the gate, the receipts and the verdicts header.
3. 67 insights changed by id (shown -> hidden), 0 added or removed; committed scoreboard command: S1 / S2 +67, S10a +1, the other 42 of 52 fields equal (the 10 that differ are the corpus path, the label, and S1 / S2 / S2-unknown / S10a in the per-language and total blocks).
4. Round-4 items: 6 of 7 confirmed and both probable wrong keeps closed; `https://iono.fm/e/1728203` 01 is NOT closed (second-episode evidence).
5. Criterion 24: 75 randomly drawn K rows (seeds 55130426, 8851027) 0 wrong; all 87 non-`self_intro` rows opened and five scans over all 374 K: 1 confirmed wrong keep and 1 unresolved inconsistency (`CTf886...` 03 against its removed sibling).
6. The 17 new removals: 17 right, 0 wrong removals; five of them depend on choices (a) and (b).
7. Choice (a) seconds of talk and choice (b) the `e7d4a370` 02 removal both follow the rulings as written; the operator's yes on both is still open.
8. Acceptance-C 20 to 23: denominators and class (b) 11 / 4 / 10 / 89 reproduced; class (a) 0 once an alias carried by a served identity is excluded; control 21 not re-run.
9. Mutants: 3 of 4 killed in the archive copy; the altered `line_sha256` survives (no committed check). Record lines 352 to 362 carry a name and a transcript sentence.
10. NO-GO on criterion 24 (bar 0).

Verdict: NO-GO criterion 24 bar 0 not met: one confirmed wrong keep (`https://iono.fm/e/1728203` SPEAKER_01, round-4 item 2, unclosed: the voice only asks questions that the same file's voice 05 answers and that the served episode `https://iono.fm/e/1728106` gives to the anchor's label) and one unresolved inconsistency (`gid://art19-episode-locator/V0/CTf886...` SPEAKER_03, the same produced announcer spot that its sibling `gid://art19-episode-locator/V0/Qv7FB...` SPEAKER_00 had removed); repair is two rows plus a re-measure, a cross-episode recurrence pass for the reader and ids-only rewrite of record lines 352 to 362

## 8f. Verdict (round 6)

Dated: 2026-10-26. Step-8 reviewer, round 6: a fresh agent, not the builder and not the round-1 to round-5 reviewer. Commit under
review `c453fb215` (clean tree; the gate ran on `eea65f01f`, and `git diff --stat eea65f01f c453fb215 -- src scripts` is empty).
Data: before = `snap1025/corpus`, after = `gate34i`. Nothing was written into the repo except this section; mutants ran in a
`git archive c453fb215` copy, deleted afterwards. Voices are named by episode-id suffix and voice id only; no name and no
transcript text is quoted. My scripts are in `scratchpad/r8f34/` (none is the builder's or an earlier reviewer's). I read the
round-6 reader files and `recur.json` only after my blind sample and my own scans were written down.

### 8f.1 Numbers re-derived with my own code

| What | Result |
| --- | --- |
| Verdicts | 546 shipped keys, 0 not in the frozen list, 0 name differences; CORRECT **371**, WRONG_PERSON **175**; W episodes **152**. Against round 5 (`git diff 708776dc6 c453fb215`): exactly 3 rows moved CORRECT to WRONG_PERSON (`856dd2e4` 01, `CTf886...` 03, `https://iono.fm/e/1728203` 01), nothing moved back |
| Public vs private | `verdict`, `basis`, `segment_idx`, `line_sha256`, `name` equal for 546 of 546, same key set |
| Hash chain | 529 rows carry an index and a hash: `sha256(private deciding line)` = public `line_sha256` = `sha256(strip(served segment text at segment_idx))` for **529 of 529**, 0 bad. The 17 rows with neither are all CORRECT / `undecidable` |
| Labels | per-row comparison of all 2,966 served `segments.json` (2,125,790 rows): the only changed fields are `speaker_label` and `voice_type`, on **2,428** rows; all 2,428 sit in the 175 W voices (each of the 175 loses its label on at least one row; after: label none, `voice_type` unknown); **0** changed rows outside W voices; 0 changes to text, times, voice id, role. The 371 kept voices are byte-unchanged in this surface |
| Scoreboard | committed `scripts/measure/corpus_scoreboard.py` on both trees, 52 flattened fields (`sbd.py`): S1 16,810 -> 16,877 (**+67**), S2 30,563 -> 30,630 (**+67**), S2 unknown +67, S10a 190 -> 191 (**+1**), in the total and the `en` block; every other field equal |
| Restated integers | 175 removals, 371 kept, 152 episodes, S1 / S2 +67, S10a +1: inside the 17:21 ranges (175, 152, +67 to +72, +1 to +2) |
| Gate / hash | `verdicts_sha256` in the gate = `rv.load(...).sha256` of the shipped file (`142ce3ae...`); list sha `4285e56b...` = verdicts header |
| Undo | `undo_cycle_r9.log` read: restored 1,827, refused 0, corpus content differing 0, other migrations' backups changed 0, twice, UNDO-CYCLE PASS. Not re-run by me |
| Tests | archive copy: `test_m0034_talk_floor_names_removed.py` + `test_reviewer_verdicts.py` 79 passed; 156 passed in the wider upgrade + outcome selection |

### 8f.2 The round-5 NO-GO items, one by one

| Round-5 item | Closed? | Evidence |
| --- | --- | --- |
| `https://iono.fm/e/1728203` 01 (confirmed wrong keep) | **Yes** | now W, `another_person`, segment 199, hash chain intact. I re-read it: the voice's 6 lines (segments 199, 200, 212, 220 to 222) are the interviewer's questions; the same two question lines are segments 43 and 44 of `https://iono.fm/e/1728106` under the interviewer's label, who introduces the analyst at segment 7 there; the analyst's answers in 1728203 are the unnamed voice 05 |
| `CTf886...` 03 vs sibling `Qv7FB...` 00 | **Yes** | both W, `another_person`. 03 is 84 s in 22 segments: 21 segments are the show's intro synopsis, the hand-over to the host and the like-and-subscribe outro; one 0.8 s fragment. The ruling on an announcer applies as written |
| Cross-episode recurrence pass for the reader | **Yes, as a scan** | the builder's `recur.py` (151 of 374 kept voices have a 40+ character line recurring in another episode of the feed; 7 without a self-introduction tie re-read with sibling episodes). I ran my own (8f.3). Limits in 8f.3 and 8f.7 |
| Ids-only rewrite of the 15:47 block | **Yes, with two leftovers** | no person name in any line added this round (38 added lines, scanned against every name in the verdicts file and for capitalised pairs); the 15:47 block has no name and no quoted transcript sentence now. Leftover 1: it still paraphrases one transcript line ("the co-host's line at seg 36 ...") for `e7d4a370` 02. Leftover 2, not in the block: `0034_..acceptance.md` line 275 carries a person's name and a quoted transcript sentence (it dates from `f9f228038`; round 5 did not list it) |
| Altered-`line_sha256` mutant survives | **No** | still survives (8f.6); not a round-5 NO-GO item but still true |

### 8f.3 Criterion 24 (bar 0 wrong keeps): what I did and found

Method, in the order I did it:

1. **Fresh seeded draw, read blind.** `random.Random(61803398).sample` of **45** of the 371 CORRECT keys (a seed in no earlier round or
   file I could grep). For each voice: its first lines, a middle and last line, the lines that name the person, the neighbouring lines of
   other voices, the episode roster with seconds, and, for every line of 40+ characters, the label that line carries in every other
   served episode of the same feed (`view.py`). **45 read: 45 right, 0 wrong.** Arguable, counted right: `urn:bbc:podcast:p0p2dn1j` 11
   and `p0p4mkxp` 06 (an archive-pick presenter intro with third-person synopsis plus first-person lines; the same intro carries the same
   label in 40 lines elsewhere in the feed, the label is tied by a self-introduction in `p0p94prv`, and a stand-in presenter is
   named in `p0p1980m`, whose labelled voice is already W); `b7d968f5` 02 (13 s: 5 s of another voice's sentence, 8 s promo with a
   self-introduction; promo is the majority).
2. **My own scans over all 371 kept rows** (`scan1.py`, `scan2.py`, `dup.py`, `addr.py`, `promo.py`, `runs.py`, `short.py`):
   (a) lines recurring in other episodes, share of the voice's seconds: 93 voices at 0.5 or more, every one carries a
   self-introduction; (b) recurring lines carried by a different named label elsewhere: 45 voices, I opened ~25 and the rest are
   instances of the same fixed spots (one NYT promo family per name); (c) two voices with the same served name in one episode:
   15 episodes, I opened the small twin in 11; (d) a voice addressed by name at least twice with the next speaker never being it: 6,
   all opened; (e) promo-wording share 0.3 or more without a self-introduction: 1 voice (opened); (f) voices whose seconds sit mostly
   in runs not tied by a self-introduction or a name just before: 89, I opened ~30 of them; (g) **all 17 `undecidable` rows** and
   **all 45 non-`self_intro`, non-`undecidable` rows** not otherwise read (first, second and last lines, roster; about a dozen in full).
   I did not count the union of rows opened.
3. Then the builder's round-6 files (`reread6/`, `reread6e/`, `recur.json`).

Found:

- **No confirmed wrong keep.**
- **Unresolved, not counted as wrong keeps, each needs a written settlement (criterion 24, last bullet):**
  - **`038e11c4` SPEAKER_02 (`undecidable`)**: the episode is a live meetup recording with two hosts and an audience and no guest
    introduced. The voice only asks two audience questions (8 segments, 51 s). The only mention of the served person in the whole
    transcript is segment 504, said by a host in the third person ("our colleague ... co-teaches"). The served metadata calls the
    person the episode's guest, source `introduced`, which no line in the text supports. I read it as probably an audience member
    who is not the named person; the text does not exclude the named person being in the audience. Same genus as the round-5
    iono item, but without a second episode that decides it.
  - **`d59a87e0` SPEAKER_09 (`addressed`, segment 2164)**: the deciding line names the person as the band's bass player; it does not say he
    sings or speaks. The voice is sung song lines and the audience's hallelujah chorus after "here's your turn" (segments 2170 to
    2179, 2206 to 2212). The same episode has a second voice (04) named by a first-name self-introduction that may be the same
    person. Nothing in the text shows who sings.
- Checked because a scan flagged them, right: the 3 a16z host voices (kept as first-person host talk at least half), the 38 promo
  self-introductions (same fixed spots, majority of each voice is the promo), `0ec506f0` 00, `3cab0976` 02 (the interviewer, not the
  guest), `a0e15df4` 01 (backchannel and answers of the guest, which is not decided by the label on voice 03 either), `6211e32b`
  02, `77d27e49` 01, `e142348b` 02, `8f93902a` 04 (34 of 36 s the interview clip), `10fe0293` 05 (continues the guest's sentence).

**Scan coverage, what the recurrence scan (the builder's and mine) can miss:** a produced spot whose wording changes from
episode to episode (the `CTf886...` presenter recurs by role: 8 of 83 s verbatim); lines under 40 characters (backchannel, one-line
credits); the same wrong label recurring in every episode (nothing differs, so no signal); a voice that appears once; recurrence
across feeds (my scan includes it for fixed lines, the builder's does not; both find only verbatim text); ASR variants of the
same line; and the interview shape where the only evidence is who answers whom (round 5's iono case needed a sibling that
happened to carry the same news item). The scan names candidates; it does not read them.

Result: **0 confirmed wrong keeps in 371; 0 in 45 randomly drawn rows; 2 unresolved rows (above).** The bar is 0 confirmed.

### 8f.4 The 3 round-6 removals, read

All 3 read against the served transcripts and, for the first two, the sibling episodes: **3 right under the rulings, 0 wrong removals.**
`https://iono.fm/e/1728203` 01 and `CTf886...` 03: 8f.2. `856dd2e4` 01 (`merged_voices`, segment 10): 33 s in 8 segments; the
cold-open news reading is 20.9 s (a third-person bulletin, segments 8 to 10), the sign-off credits 11.5 s (segments 158 to 162).
Majority in seconds is the other speaker by the builder's unit; the credits are read by different presenters in sibling
episodes. This is the least certain of the three: the cold-open reading could be the named co-host, in which case the voice
would be mixed with the credit read and the majority would flip. A removal is the safe direction under the operator's ranking.

### 8f.5 The builder's choices awaiting the operator

- Majority in seconds: follows the rulings as written (the rulings do not name the unit; the operator's yes is still open). Round 6 changes
  no row on this unit.
- `e7d4a370` 02 removed against a 2-of-3 keep: follows ruling 3 as written (sponsor read and house promo only, 18 of 18 lines, no line
  tying it); the other two voices with that shape are W (`8536937e` 02, `c2fe4ed8` 03).
- Sibling episodes as evidence and a produced presenter role as an announcer: follows the rulings as written for `CTf886...` 03 and
  `Qv7FB...` 00 (third-person synopsis, hand-over, outro); I did not find the role rule applied to a voice that also speaks as the person
  (the `p0p...` presenter voices above are kept, rightly: their label is tied elsewhere in the feed).

### 8f.6 Mutants (archive copy only)

| Mutant | Result |
| --- | --- |
| first W row set to CORRECT (verdict column only) | killed (`test_registered_with_its_list_and_verdicts_shipped`) |
| same, evidence column rewritten to match | killed |
| one basis set to a made-up value | killed |
| **one `line_sha256` altered** | **SURVIVED** (79 passed; 156 passed in the wider selection) |
| **one `segment_idx` changed** | **SURVIVED** |
| **one CORRECT row deleted from the shipped file** | **SURVIVED** (the run-time guard "a listed voice without a verdict fails verify" is separate, and not hit by these tests on the shipped file) |
| my own checks | a changed kept label and a W label left on a row, each injected into the per-row comparison: both flagged |

The hash chain and the kept rows are protected in CI only by the future outcome file bound to the verdicts sha; there is no outcome
for 0034 yet (`ls src/podcast_scraper/upgrade/outcomes`: 0031, 0033).

### 8f.7 What is NOT covered

- **Rows opened.** 45 read in depth; all 17 `undecidable`; all 45 other non-`self_intro` rows by first, second and last lines and roster;
  about 55 more flagged by scans. The union is not counted and is below 371. The `self_intro` rows (287) that no scan flagged were
  not opened except by the draw. A wrong keep among them that is not a recurrence, a mismatched self-introduction, an
  addressed-and-unanswered voice, a multi-run voice or a promo passes my scans. The shape I can least exclude is a short clip
  inside a larger interview voice whose self-introduction belongs to another run.
- **Depth.** For the non-`self_intro` rows I read first/last lines and context, the full voice for about a dozen; I did not
  read every voice's turn lengths to re-measure seconds; I did not re-measure the 13 split rows of round 5 or the 4 seconds-decided rows.
- **The 172 earlier removals** (of 175): not re-read by me; I rely on round 4 and round 5. Only the 3 new ones were read.
- **Fixture / other surfaces:** the byte-unchanged check covers the served `segments.json`; I did not diff the other 1,800 changed files
  class by class (transcript text, GI, KG, bridges, roster, search index) for the kept voices; the 67 insights I rely on through S1 / S2
  and the apply message (`insights_reattributed=67`), not by my own insight-id diff; bridge files (2 written as empty identities) not
  re-checked.
- **Not run:** `make ci-fast` on the committed tree; the full upgrade test directory (only 156 selected); the undo cycle (log only);
  acceptance-B 17 (S6 / S7 on the gold sets), S4, S5, S8, S9, S11; acceptance-C control 21; the prod criteria 27 to 29; the
  derived-surface ladder of criterion 7 (search-index rows, enrichment outputs) for the 175 voices; the 0035 and 0036 re-measures
  (read the record lines only).
- **Provenance in the shipped file:** the verdicts header still says `reviewed_commit: f9f228038` and "round 5 ... 391 kept voices; round 4 for
  the 155 removals"; the file now has 371 kept and 175 removals. Correcting it changes the file sha.
- **Not decided here:** the operator's yes on the three choices in 8f.5; the two unresolved rows in 8f.3; the 10 dropped entries, the
  58 misspelled kept names (written `CORRECT`), reprocess durability.
- **Whether `856dd2e4` 01 is the named co-host** (8f.4): not decidable from the transcript.

### 8f.8 Reasons for the verdict

Every mechanical check holds and is re-derived: 546 keys, 175 names off = the 175 W keys, 371 kept voices unchanged in the served
segments, S1 / S2 +67, S10a +1, 152 episodes, the hash chain for 529 of 529, the undo log. Both round-5 NO-GO rows are closed
by evidence I re-read (the iono interviewer lines match the sibling episode's; the art19 spot is announcer-only in both episodes),
and the 3 round-6 removals are right under the rulings. I found no confirmed wrong keep in 45 drawn rows and in the rows my scans
flagged. Two rows are unresolved (8f.3): one live-meetup audience questioner named by a metadata guess, one voice named by a line that
makes the person a bass player. Criterion 24 does not count a row the reader merely cannot confirm, so neither blocks; but each is the
kind of row the operator ranks worst, so they are settled in writing and, if flipped, re-gated. The conditions below are small and
checkable.

### 8f.9 Summary

1. Mechanics re-derived with my own code: 546 keys, CORRECT 371 / W 175, 152 episodes; 2,428 rows of the 175 W voices lose their label, 0 other label, text, time, voice-id or role changes in 2,125,790 rows.
2. Hash chain 529 of 529 (evidence line = public hash = served segment text); verdicts sha `142ce3ae...` = gate; list sha = header.
3. Scoreboard (committed command): S1 / S2 +67, S10a +1, the other fields equal; undo log PASS (read).
4. Round-5 items: iono 1728203 01 closed (interviewer lines match 1728106); `CTf886...` 03 and `Qv7FB...` 00 both W, closed; recurrence pass done as a scan; ids-only closed for this round's lines, two leftovers (acceptance.md line 275; one paraphrase in the 15:47 block).
5. Criterion 24: 45 drawn rows (seed 61803398) 0 wrong; 17 `undecidable` + 45 other non-`self_intro` rows + ~55 scan-flagged rows opened; 0 confirmed wrong keeps; 2 unresolved (`038e11c4` 02, `d59a87e0` 09).
6. The 3 round-6 removals: 3 right, `856dd2e4` 01 the least certain.
7. Builder choices: seconds, `e7d4a370` 02, sibling evidence / produced presenter role all follow the rulings as written; the operator's yes is open.
8. Mutants: hash, segment index and a deleted kept row all survive the committed tests; protected only by the future outcome binding.
9. Recurrence scan misses: varying produced spots, lines under 40 characters, a wrong label recurring identically, single-appearance voices, interview-structure cases.
10. GO WITH CONDITIONS (below).

Verdict: GO WITH CONDITIONS (1) settle `038e11c4` SPEAKER_02 and `d59a87e0` SPEAKER_09 in writing, each kept with its argued basis or flipped to WRONG_PERSON, and if either flips re-bind verdicts and list shas and re-run the gate, the undo cycle and the 0035 / 0036 re-measures; (2) remove the person name and the quoted sentence at `0034_talk_floor_names_removed.acceptance.md` line 275 and the transcript paraphrase for `e7d4a370` 02 in the 15:47 block; (3) bring the verdicts-file header (`reviewed_commit`, reviewer line, counts) to round 6 in the same change as (1), or state in the record that it is stale and that the outcome binds sha `142ce3ae...`; (4) the operator's recorded yes on the three builder choices in 8f.5 before any apply; (5) build the 0034 outcome bound to the final verdicts sha and run `make ci-fast` on the committed tree, since no committed test guards the hash, segment index or a deleted kept row

**Round 6's conditions, closed by the builder (2026-10-26, after 8f):**
- **C1** `038e11c4` 02 kept (`undecidable`) and `d59a87e0` 09 kept (`addressed`), each with its
  argued basis written in the 18:36 block; neither flips.
- **C2** ids only: `acceptance.md` lines 152 and 275 and the 15:47 block rewritten; the scan of
  every line the branch adds under `docs/` against every served name finds 0 in the `.md` files.
- **C3** the verdicts header names round 6 (both files); re-measured (7f), every integer equal.
- **C4** open: the operator's recorded yes on the three builder choices (majority in seconds;
  `e7d4a370` 02 removed under the third ruling; sibling episodes as evidence and a produced
  presenter role as an announcer), before any apply.
- **C5** the outcome, bound to the final verdicts sha: below; `make ci-fast` on the committed tree:
  below.

**Outcome (C5, 2026-10-26):** `scripts/measure/build_outcome.py --migration 0034_talk_floor_names_removed` on the gate
report measured at c2c0dcad2 (clean tree; the final gate after the format / lint / type /
docstring fixes, every number and plan equal to the d94e4d13a run): `upgrade/outcomes/0034_talk_floor_names_removed.json`, outcome 8fddf1dd075a2fb0f915f4e3605abc3645d4502c29336eee583d4264c2357876.
`make ci-fast` on the commit holding it: below.

**Operator, 2026-10-26, on the 12 step-9 points:** "I'm okay with all 12 points and decisions we took. Let's keep moving forward." C4 closed: the three builder choices (majority in seconds; `e7d4a370` 02 removed under the third ruling; sibling episodes as evidence and a produced presenter role as an announcer) are accepted.
