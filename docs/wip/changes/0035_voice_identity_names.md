---
id: 0035_voice_identity_names
class: full
status: landed
issue: none (the operator opens issues; naming arc, 2026-10-23)
touches:
  - src/podcast_scraper/upgrade/migrations/m0035_voice_identity_names.py
---

# 0035 — the voice identity rung on served episodes that carry fingerprints

Method: [CORPUS_CHANGE_METHOD.md](../../guides/CORPUS_CHANGE_METHOD.md). Plan:
podcast-scraper-eval-data `docs/wip/NAMING-GATES-2026-10-22.md`, the coding plan, item C8. A wrong
name is worse than a hidden insight (operator, 2026-10-19).

## 1. Intake

- **The pipeline change.** New episodes get the voice identity rung (`voice_identity_rung`, naming
  ladder doc v26 / v29). It compares an episode's voice fingerprints with the same feed's other
  episodes:
  - a voice that matches a fingerprint group named by vote takes that name;
  - a name leaves a voice that is clearly not that person.
- **Why served episodes need a migration.** Served episodes were named before the rung. They
  already carry what it needs, so it runs from text on disk with no DGX:
  - the diarizer's fingerprints (`voice_embeddings` in the speaker-diagnostics sidecar, captured
    since #2339);
  - the feed's published names.
- **Scope.** On prod, 413 of 2,939 served episodes (14%) carried fingerprints on 2026-10-23, all
  written since 2026-10-12. Episodes without fingerprints are not touched.

## 2. Outcome and baseline

- **The outcome:** a fingerprinted voice is served with its feed's voted name when it matches that
  group, and loses a name when it is clearly not that person, only where a review confirms it.
- **Measured** (eval repo `autoresearch/voice_embedding_match`, pre-registered val_v4, 10 feeds
  never used to shape it): wrong names 23 -> 17 (p = 0.031), correct names 70 -> 77 (p = 0.014).
- **Baseline, 10-17 snapshot copy (`lost/gateworkB6c`, read only), after the review fixes:
  10 voices (7 named, 3 unnamed).**
  - The first run, before the fixes, listed 15. The 5 it no longer lists all hold 1.1-4.3% of
    their episode's talk, under the floor (M2). They include the two that looked wrong on sight
    (a guest's name replaced by another person's; a sub-5% voice given a host's name).
  - The absolute-path fix (M1) added no voice on this copy: 121 of its 2,685 served metadata
    files store an absolute path, and none of them changes a group here. Prod's share is not
    measured.
  - None of the 10 has been read yet (section 5, step 2).
- **Prod:** not counted yet. The list is made on a fresh prod snapshot (section 5). The 10-17 copy
  predates most fingerprinted episodes, so the prod list will be larger.

## 3. Diagnosis

- **The detector is the pipeline's own rule.** `episode_changes` calls the rung's
  `voice_identity.feed_history`, `build_groups` and `decide` on the stored fingerprints, with the
  pipeline's defaults (threshold 0.7, 60 s minimum talk for a group member). Nothing is
  re-implemented.
- **Served names, not ladder names.** The voice's current name and role come from the served
  segments (0034's `served_by_voice`), so the list describes what the reader sees today.
- **The talk floor comes after the rung.** In the pipeline a voice under 5% of the talk ends
  unnamed whatever the rung says. A host-read ad carries the host's fingerprint, so the rung
  alone would put the host's name on it, and a review could only answer "it is the host's
  voice". Voices under the floor are not listed; they are 0034's.
- **Siblings with an absolute stored path.** Prod metadata often stores the transcript path as
  written at ingest (`/app/output/...`). The rung's feed history now maps it into the corpus
  being read (`voice_identity.sibling_diagnostics_path`), so a snapshot copy sees every sibling.

## 4. Design

- **A frozen, reviewed list**, as 0034's.
  - `find_changes` replays the rung over a corpus copy.
  - `write_list` freezes it in `upgrade/lists/0035_voice_identity_names.json`.
  - The migration applies that list and nothing else.
- **The review.** The reviewer reads each listed voice:
  - a RENAME is judged by its new name: `CORRECT` / `CORRECT_MISSPELLED` applies it;
    `WRONG_PERSON` leaves the voice as it is;
  - an UNNAMING is judged by the name it takes off: `WRONG_PERSON` applies it; `CORRECT` keeps
    the name;
  - a listed voice without a verdict fails `verify`, and without a verdicts file nothing is
    written.
- **The list is bound.** The verdicts header's `list_sha256` is the hash of the WHOLE list
  (every field of every entry: both names, both roles, the evidence), the list file is part of
  the migration's digest (`list_file`), and the receipts header records the list's hash. An
  entry edited after the review refuses the load.
- **A renamed voice's source is `voice_identity`**, as the pipeline records the rung's names, so
  a name this migration applies never votes in the feed bank (`voice_identity._NON_VOTING_SOURCES`).
  That includes a respelling of the same person ("Jon Smith" -> the group's "John Smith"), which
  the shared rewrite would otherwise leave on its old source (`RESPELLING_TAKES_RENAMED_SOURCE`).
- **Placeholder labels are no names.** A served label "SPEAKER_01", "Speaker", "Speaker 2" or
  "Unknown Speaker" counts as an unnamed voice (0034's `_LABEL`), so the review never judges one.
- **The rewrite.** Through 0029's voice-by-voice rewrite with 0031's derived surfaces (0034's
  episode class): transcript prefixes and offsets, turns, diagnostics, context, ad maps, metadata
  counts, GI, KG and bridge. All or none per episode; an episode whose stored name or role moved on
  is refused untouched.
- **Undo:** from `.podcast_scraper/upgrade-backups/0035/`, while each file is still as written.
- **Registered** (2026-10-26, `upgrade/registry.py`, after 0034, `to_version` 2.7.29), once the
  list and its verdicts were committed; until then it was kept out of the registry, as 0034 was.
- **Tests** (`tests/unit/podcast_scraper/upgrade/test_m0035_voice_identity_names.py`; each fix
  from the review mutation-checked: reverting it fails its test):
  - the detector names a host's unnamed voice from the feed's fingerprints;
  - an episode without fingerprints is not touched;
  - a reviewed rename is written on every surface, verified and undone;
  - a rename read as `WRONG_PERSON` is not written;
  - review names: a rename by its new name, an unnaming by its old one;
  - a listed voice without a verdict fails verify and nothing is written;
  - a voice under the floor is not listed, even when it is the host's voice;
  - siblings stored with an absolute path still vote;
  - a name this migration applied does not vote afterwards, a respelling included;
  - registered, with its list and verdicts shipped;
  - the dry-run rows carry counts and input sha256s (the outcome hook's plan diff).

## 4b. List and verdicts on snapshot-prod-20261025 (2026-10-26)

`MISSED:` as 0034's record: the prediction and the acceptance come after the build; for the
operator to accept.

- **List:** `--find` on snapshot-prod-20261025 lists **163 voices: 139 renames, 24 unnamings**,
  sharing no voice with 0034's list (10 episodes carry entries of both, on different voices).
- **Verdicts:** 163 of 163, two fresh reviewers (Opus), `scratchpad/m0035/verdicts_[12].tsv`.
  Renames: **61 applied** (CORRECT / CORRECT_MISSPELLED), **78 left as they are**
  (WRONG_PERSON: the text does not support the fingerprint's name). Unnamings: **24 of 24
  applied** (WRONG_PERSON on the name they take off).
- **Restated prediction on this snapshot:** 61 voices take the fingerprint group's name, 24 lose
  a name, 78 are untouched; nothing outside those 85 moves. The val_v4 measurement (wrong 23 ->
  17, correct 70 -> 77) scored the rung's answers, not text-confirmed ones; here more than half
  the renames are not confirmed by the text, so the reach is smaller than the rung's rate
  suggested.
- **Not yet done:** the step-8 reviewer's read and verdict; the clone measurement and undo.

Dated: 2026-10-26

- **Predicted before the clone measurement** (the restated counts above): 85 voices written (61
  renames, 24 unnamings), 78 listed voices untouched, 0 voices outside the 163 listed.
- `MISSED:` S1, S2 and S3 were not predicted in integers before the clone measurement. The values
  below were MEASURED (section 7) and are recorded as measurements, not as a prediction:
  S1 +196, S2 -93, S3 -226. The split, by insight (step-8 reviewer, section 8): 403 go from shown
  to hidden-`unknown`, all on the 22 voices that lose a name; 207 go from hidden-`unknown` to
  shown and 289 from hidden-`unidentified` to shown, on applied renames. S1 = 403 - 207 = +196;
  S2 = 403 - 207 - 289 = -93. For the operator to accept or reject, as with the build-order miss.
- **Must not move** (measured, section 7): S1c 0, S12 24, S10a 192, S10b 2, A1 382 hits in 290
  episodes, 130,401 insights, 252,651 quotes.

**Operator decisions (2026-10-26):**
- Operator accepted: the MISSED order (prediction and acceptance written after the build).
- A host reading an ad keeps the host's name ("keep").
- A correctly named clip of a public figure keeps the name ("keep").

## 4d. Revision: the operator's rule applied to 0035, and the left renames re-read

Dated: 2026-10-26T09:20

- **Operator, 2026-10-26** (after 0034's step-8 NO-GO; same rule for 0035): an UNNAMING applies
  only when the text shows the voice is someone else (no-evidence unnamings are kept); a RENAME
  applies only with positive evidence in the text (unchanged); all 78 left renames and the 4
  unnamings applied on "no evidence" are re-read blind by two fresh readers, splits by a third;
  `Buzzsprout-20005821` SPEAKER_00 is dropped from the list (right name, wrong role `guest` from
  the feed's group; the migration cannot correct a role from a review); every verdict states its
  basis and deciding line (the columns 0034 now requires; 0035's step-8 condition 4).
- **Prediction, before any reader reports** (basis: the step-8 reviewer's K6 sample, 11 of 48
  confirmable, about 16 of 78; its K5 reading of the unnamings, 3 undecidable):
  - list: 163 -> **162** (one dropped);
  - renames applied **71 to 76** (60 kept of the 61 applied, plus 11 to 16 of the 78);
  - unnamings applied **20 to 21** (24 minus the 3 or 4 without evidence);
  - S1 rises by at most **+184** (+196 minus the 12 insights on the 3 kept unnamings) and falls
    further with each newly applied rename of an unnamed voice: **+150 to +184**;
  - S2 **-105 to -93**; S3 **-260 to -226**; S1c 0, S12 24, S10b 2, commercial 94, A1 382 / 290
    unchanged; measured on snapshot-prod-20261025 with the revised 0034 (275e76cc6) applied first;
  - if the readers' counts fall outside these, the population is restated before the clone.

Dated: 2026-10-26T09:38

- **MISSED: the 4d prediction (11 to 16 of the 78 renames added).** The first two re-readers added
  49 and 44; their agreement was 72 of 82, under the 74 bar, so the work stopped and went to the
  operator. Cause (the builder's): the instructions accepted "the episode's host speaking as host"
  without saying how the host is known; both readers used the episode description, one more
  broadly. 15 of the renames both readers apply rest on an answered address, 2 on a self-intro.
- **Operator, 2026-10-26:** the description may identify the host only when it names exactly one
  host and the transcript shows one host voice; a co-hosted show needs the transcript itself;
  elimination never counts. Fresh readers re-read all 82 plus the 17 applied renames resting on
  `host_read` / `continuity` (99), the A/B readings of the 82 discarded; bar 90 of 99 (74 of 82
  scaled).
- **Restated prediction, before the fresh readers report:** of the 78 left renames, **17 to 32**
  added (the 17 both earlier readers applied on an address or a self-intro, plus up to 15 one-host
  `host_read`); of the 17 applied host / continuity renames, **8 to 17** stay; unnamings applied
  **21 to 22** (the 20 shown by the text, plus 1 to 2 of the 4); renames applied in
  all **60 - (0 to 9) + (17 to 32) = 68 to 92**; S1 / S2 / S3 restated as integers after the
  readers settle the counts and before the clone (D15).

- **Result** (`scratchpad/m0035/reread4/`): fresh readers C and D agree on 82 of 99 (bar 90);
  the 17 splits (7 a solo host split into two voices, 3 one guest name spelt alike for two
  guests, 7 single calls) went to a fresh third reader E under the operator's split-host ruling;
  majority of three. Final, 162 entries: **renames applied 92** (60 carried less those the host
  rule now refuses, plus those the 78 now support), **not added 46**; **unnamings applied 21**,
  **kept 3**. Within the restated 68 to 92 and 21 to 22. A fresh `--find` reproduces the list
  entry for entry, the operator's exclusion included (`EXCLUDED` in the migration, tested).

**Clone measurement on 2080d1cb2** (clean tree; base: snapshot-prod-20261025 with the final 0034,
a1698c214, applied and in the ledger; `scratchpad/m0035/gate_0035_r2.json`): rewrote **99**
episodes, refused 0; verify OK; S1 **+72**, S2 **-404**, S3 **-407**, S10a **-4**; S1c 0, S1n 7,
S10b 2, S12 24, commercial 94, A1 382, insights 130,401, quotes 252,651 unchanged.

- **MISSED, four numbers:** S1 +72 against +150 to +184, S2 -404 against -105 to -93, S3 -407
  against -260 to -226, S10a -4 against acceptance-B's -2 to +2. All on the side of more shown.
- **MISSED, a step:** the 09:38 restatement said S1 / S2 / S3 would be restated as integers once
  the readers settled the counts and before the clone; the builder did not restate them. The
  ranges predate the re-read, which applies 92 renames where 4d assumed 71 to 76.
- The moves are explained by insight id by the step-8 reviewer (they are not explained here).

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

## 5. Acceptance

Dated: 2026-10-26T09:21

Revised acceptance for the 4d revision, by a fresh reviewer (not the builder):
acceptance evidence (in git history). Its integer bars
(74 of 82 reader agreement, the sample sizes, the S10a range) are the reviewer's choice, not
measured: for the operator to accept.

Operator accepted: 2026-10-26 (0035's rule decisions in 4d: unnamings need text showing someone
else; renames need positive evidence; all 78 left renames re-read; Buzzsprout-20005821 SPEAKER_00
dropped).

Operator accepted: 2026-10-26 (acceptance-B's bars; the host rule: a description identifies the
host only when it names one host and the transcript shows one host voice, elimination never
counts; a solo host split into two voices takes the name on a mid-sentence continuation; the
recorded misses: the 11 to 16 prediction, and reader agreement 72 of 82 then 82 of 99 under the
bars, settled by a third reader on the 17 splits).

The first acceptance, below, stands for what acceptance-B does not replace.

MISSED: no acceptance criteria were written by a reviewer before the build (2026-10-23) or
before the clone measurement. The criteria 0035 was judged against are the step-8 reviewer's
(section 8, "Criteria"), written by the same reviewer who gave the verdict.

Operator accepted: 2026-10-26 (the MISSED order of steps 4 and 5, for 0034 and 0035).

Dated: 2026-10-26T11:31

Fresh-reviewer acceptance addendum for the 11:29 defect fix(es), integer bars before the code fix: acceptance evidence (in git history).

## 6. Build

- Code: f9f228038 on `feat/m0034-list` (2026-10-26T06:40Z): the migration, its frozen list
  (`upgrade/lists/0035_voice_identity_names.json`, 163 entries), the verdicts file
  (`upgrade/verdicts/0035_voice_identity_names.tsv`, list_sha256 `9b4d929ab138…`, verdicts
  sha256 `83e3fcd2e3fc…`), the registry entry and `dry_run_report` through 0034's `plan_rows`.
- The verdicts file carries every row, verdict and name and no transcript text. The reviewers'
  deciding quotes are in podcast-scraper-eval-data
  `data/eval/migrations/0035_voice_identity_names.verdicts-with-evidence.tsv` (private).
- Tests: `tests/unit/podcast_scraper/upgrade/`, 555 passed on f9f228038. The 23 mutants the
  step-8 reviewer found surviving are condition (2) of section 8: their tests are not written
  yet.

## 7. Measure

Final clone measurement on 2080d1cb2 (clean tree), base snapshot-prod-20261025 with the final
0034 (a1698c214) applied and in the ledger; report `0035_voice_identity_names.gate.json`. The
round-1 measurement (06:59, f9f228038) is superseded; its report is kept in the private eval repo
(`0035_voice_identity_names.round1.gate.json`).

- **Apply:** rewrote **99** episodes, refused 0; verify OK (its message is 0034's inherited text,
  "off the floor's name", which misdescribes a rename: a step-8 item).
- **Scoreboard moves:** S1 +72, S2 -404, S3 -407, S10a -4; S1c 0, S1n 7, S10b 2, S12 24,
  commercial 94, A1 382 / 290, insights 130,401, quotes 252,651 unchanged. Against the prediction:
  see 4d (MISSED, four numbers and a skipped restatement).
- **Files: 1,358 changed** (receipts 1,359: one KG rewritten with identical bytes).
- **Derived surfaces** (`scripts/measure/derived_surface_census.py`, read with the checks used for
  0034): `insight_density` / `insight_sentiment` stale in 99 episodes (prod: `enrich --only
  insight_density,insight_sentiment` for them); search-index rows of the 99 (incremental `index`);
  context consistent (99 of 99 equal a rebuild from the after-state; the census's 15 are
  rebuilt-unchanged files); `.anon.txt` pre-naming by design; the manifest is provenance; ad-map
  offsets off: 0.
- **Undo cycle** (`scratchpad/m0035/undo_cycle.py` on a copy of the after-clone, compared with the
  0034 after-clone): undo restored 1,359, refused 0; corpus content equal (0 of 144,890 files
  differ; only backups, the receipt log); apply again 99 episodes; undo again the same; no other
  migration's backups changed. UNDO-CYCLE PASS. On prod an undo also removes 0035's ledger entry,
  and 0035 is undone before 0034.
- **One episode is written by both 0034 and 0035** (the final plans share 1 metadata path; an
  earlier statement here that they share none came from the round-1 plans and was wrong). 0035 was
  measured on the 0034 after-clone, so for that episode it reads 0034's output: prod's plan diff
  for 0035 allows 0 differences provided prod's 0034 output equals the clone's, which 0034's own
  plan diff and receipt check first.

- **Re-measure on ffaae4ae0** (clean tree; base: snapshot-prod-20261025 with 0034 round 3 applied
  and in the ledger; acceptance-C): S1 +72, S2 -404, S3 -407, S10a -4, 1,358 files with the same
  suffixes, the same 99 episodes, per-episode deltas equal: every number equal to the 2080d1cb2
  measurement (met); verify OK with its own message; empty bridge files after: 15 (none written).
  The base's own scoreboard differs (0034 now removes 149), so before / after totals move by that.
  0034 and 0035 share 1 episode. Undo cycle on this base: restored 1,359, refused 0, corpus
  content equal (0 of 144,996 files), apply again 99, undo again the same: UNDO-CYCLE PASS.

- **Final re-measure on a859902ab** (base: 0034 round 4 applied and in the ledger): every number
  and file equal to the ffaae4ae0 measurement, both scoreboards included (the 6 removals 0034's
  round 4 added carry no insights); verify OK; 1 episode shared with 0034; empty bridges written 0.
  Undo cycle on this base: restored 1,359, refused 0, 0 of 145,030 files differ: UNDO-CYCLE PASS.

- **Re-measure on 0034 round 5** (a5a9a6909, clean tree; base `gate34h`: 0034 round 5 applied and
  in the ledger; `0035_voice_identity_names.gate.json` is this run, measured 2026-10-26T16:18:19):
  plan 99 episodes with 0 differences against the a859902ab run; files 1,358; episode deltas, verify
  and every gating move equal (S1 +72, S2 -404, S3 -407, S10a -4); only the base's scoreboard
  moved (+8 hidden-unknown from 0034's 17 new removals). Prediction ("unchanged except the base")
  met. Undo cycle (`undo_cycle_r5.log`): restored 1,359, refused 0, 0 of 145,173 files differ,
  apply again 99, undo again the same: UNDO-CYCLE PASS.
- **0025's verify after 0035 (round 4's finding, step-9 item 8):** on the snapshot 0025's verify
  already fails 1 episode (feed `8ee73e4a` item 0001, ingested 2026-10-22); after 0034 the same 1;
  after 0035, 5: 0035 adds 4. **Correction (2026-10-26, measured on a probe clone):** the builder's
  earlier sentence here ("the split solo host's name now sits on two voices ... one person listed
  twice") was not checked and is false. Each of the 4 episodes has one person id per person; 0025
  flags them only for spelling: 0035 gives the voice the spelling of its feed-wide fingerprint
  group, 0025 prefers the spelling the episode's own title or description states (3 people, 4
  episodes; the snapshot's 1 is the same kind). **Operator, 2026-10-26: accepted as known** (step-9
  item 11): no code change; the prod handover lists the 5 episodes as known 0025 verify flags,
  spelling only, and the 2.7 runbook's step 5 (`upgrade verify`) is read with that exception.

- **Re-measure on 0034 round 6** (eea65f01f, clean tree; base `gate34i`; this record's `.gate.json`
  is this run, measured 2026-10-26T17:49:38): every gating move, both scoreboards, episode deltas
  and verify equal to the r5 run; files 1,359 (r5 1,358). The plan differs in 1 episode only, the
  shared `iono.fm/e/1728203` (predicted at 0034's 17:21 block): 0034 now takes the name off that
  episode's voice 01, so 0035 no longer merges its voice 05 into a person already placed there
  (r5: placed_merged 1, quotes_moved 26, admaps_carried 1) and adds the person instead (r6:
  kg_persons_added 1, entities_added 1, context_rebuilt 1, metadata_counts_synced 1; one more KG
  file). Undo cycle (`undo_cycle_r6.log`): restored 1,359, refused 0, 0 of 145,199 files differ,
  apply again 99, undo again the same: UNDO-CYCLE PASS. Prod's plan diff for 0035 is taken
  against this run.

- **Re-measure on 0034 after its round-6 conditions** (d94e4d13a, clean tree; base `gate34j`; this
  record's `.gate.json` is this run): plan, files, gating moves, both scoreboards, episode deltas
  and verify equal to the r6 run, plan differences 0. Undo cycle (`undo_cycle_r7.log`): UNDO-CYCLE
  PASS.

## Not covered by the tests

- An unnaming on a corpus: the test checks its review semantics, not a rewrite. The rewrite
  path is 0034's, which its own tests cover with an unnaming on every surface.
- The surface check reads a quote's speaker through its `speaker_id` and the Person node, as the
  readers do (`gi/load.py`, `server/app_gi_view.py`): the rewrite clears the authored
  `speaker_name`. The fixture's Person node for the raw id `SPEAKER_00` stays; real GI makes no
  Person from a raw id (`gi/speakers.py`).
- A rename to a name already on another voice in the same episode. `decide` is the pipeline's; the
  migration does not add a guard of its own.
- The 10-17 detector output has not been read voice by voice yet; that is step 2.

## 8. Verdict

Step-8 reviewer: a fresh Opus advisor, 2026-10-26, not the builder and not a resumed reviewer. Code
under review: f9f228038 (clean tree; the two `.gate.json` files are untracked). Earlier reviews
(the two list reviewers, the m0034 acceptance) were read as inputs. Every number below comes from
the reviewer's own stdlib code in `scratchpad/r8/` (list at the end), run on the two clones; none
is taken from the builder's table. `make ci-fast` was not run, as instructed.

### Criteria

Written before judging, from the method, this record's goal and section 4b, and 0034's acceptance
file as the template.

- **K1 Population.** The frozen list is what the detector finds on the snapshot, key for key, with
  entries equal; 139 renames + 24 unnamings = 163; no voice shared with 0034's list; the verdicts
  header binds the whole list.
- **K2 Verdict coverage.** 163 of 163 rows, closed verdict set, header complete, author not the
  builder, each row carries the line that decides it (D13); a row the reviewer could not decide is
  marked as such.
- **K3 Own outcome numbers (4b restated prediction).** Voices renamed = the 61 `CORRECT` /
  `CORRECT_MISSPELLED` renames; voices unnamed = the 24 unnamings; the 78 other renames untouched;
  nothing outside those 85 voices moves; episodes written = the episodes holding an applied voice
  (72); refused 0.
- **K4 Gating numbers.** S1c = 0, S1n does not rise, S12 does not rise. S1, S2, S3 may move only
  under the trade rule (a wrong name is worse than a hidden insight), each move explained by
  insight id.
- **K5 Applied changes are right.** Every one of the 85 applied changes is read; a rename must
  have positive identity evidence for the new name (self-introduction, an address answered by
  this voice, a host read, continuity), an unnaming must remove a name the voice does not own.
- **K6 Left changes.** At least 30 of the 78 renames left as they are, seeded, are read; every
  voice where I would apply is counted as a disagreement.
- **K7 Mutation.** Each definition is mutated against its tests; a mutation that leaves every
  test green is an unguarded path.
- **K8 Safety of the apply.** Undo, plan diff, derived-surface census exist and were measured.
- **K9 Landing.** The record, the shipped outcome and `make ci-fast` satisfy the method, so the
  landing test is green.

### Findings

**K1 Population: PASS.**
- `m35.find_changes` run on the pre-0034 snapshot copy (`snap1025/corpus`) returns 163 keys; the
  list has 163; only-detector 0, only-list 0, entries different 0
  (`scratchpad/r8/refind.py`). This reuses the builder's detector, so it proves reproducibility,
  not that the detector is right.
- 139 renames and 24 unnamings by my own count of the list file (`r8/join.py`). Overlap with
  0034's list: 0 voices, 10 episodes (`r8/ov.py`).
- The same detector on the clone that has 0034 applied (gate34) returns 166: 3 extra, all in one
  feed's renames between two host/guest labels on three episodes (`iono.fm/e/1730679` SPEAKER_02,
  `iono.fm/e/1728106` SPEAKER_01, `iono.fm/e/1729809` SPEAKER_02). The cause is that 0034 changes
  which names vote in the feed's groups. The list is frozen, so these are not applied. Record it:
  they are 0036 material, and the list must stay the pre-0034 one.
- `rv.load` accepts the verdicts header's `list_sha256`
  (`test_registered_with_its_list_and_verdicts_shipped` passes, 29 tests green on the unmutated
  tree).

**K2 Verdict coverage: PARTIAL.**
- 163 rows, keys equal to the list's keys, verdicts only from the closed set (51 `CORRECT`, 10
  `CORRECT_MISSPELLED`, 78 + 24 = 102 `WRONG_PERSON`).
- The shipped TSV writes "reviewer's deciding quote kept in podcast-scraper-eval-data" in every
  evidence cell, a pointer to a sibling repo. D13 asks for the deciding evidence with the entry.
- In the full-evidence file, 74 of the 78 rename rows left as they are carry the identical text
  "no evidence it is them" (the other 4 state a refutation); 4 of the 24 unnamings carry it too
  (a name removed for lack of evidence). The label `WRONG_PERSON` therefore covers both "refuted"
  and "not confirmed". 0034's acceptance asked for `undecidable:` rows counted apart; here they
  are not, so the file cannot be audited row by row for those 78; I did it by reading (K6).

**K3 Own outcome numbers: PASS.** Commands and results:
- `r8/treediff.py`: sha256 over every file of gate34 and gate35 (`.podcast_scraper` excluded).
  130,385 files before; 130,386 after, the one new file being `voice_identity_names.jsonl`; 991
  files changed (equals the gate report's 991); 72 `.metadata.json` changed = 72 episodes written;
  0 changed files outside those 72 run directories.
- `r8/join.py`: applied = 61 renames (51 + 10) + 24 unnamings = 85; renames left 78; the episodes
  of the applied keys = 72, and they equal the 72 written episodes. The voices whose segment
  `speaker_label` / `speaker_role` differ between gate34 and gate35 are exactly the 85 applied
  keys: 0 extra, 0 missing. So the 78 left are byte-for-byte untouched in the segments, and
  nothing else in a written episode moved.
- `r8/diagchk.py`, gate35 diagnostics for the 85 keys: the 61 renames have `resolved_name` = the
  listed name, `named` true, `source` = `voice_identity` (37 were unnamed, 24 replaced a name; 8 of
  those are spelling variants by string similarity); the 24 unnamings have `named` false, `source`
  `raw`. The gate's `voices_named=37`, `roster_entries_unnamed=24`,
  `diagnostics_voices_renamed=85`, `rewrote 72 episode(s)` all equal these counts.
- Against the section 4b prediction (61 / 24 / 78 untouched / nothing else moves): met exactly.

**K4 Gating numbers: PASS under the trade rule; S1 +196 is explained, not a defect.**
- `r8/own_s.py` reads the 72 written episodes' GI before and after (all other files are identical
  by the tree diff): per insight id `surfaceable`, `speaker_voice_type`, `speaker`; the voice is
  the raw `speaker` of the segment holding the first `SUPPORTED_BY` quote's `char_start` (S7's
  join, by the quote's cited file name); talk share from the diagnostics `talk_time_s`, principal
  at 0.30. Result, after minus before: S1 +196, S2 -93 (unknown +196, unidentified -289,
  commercial 0), S3 -226, S1n 0. `r8/s12.py`: S1c 0 and S12 0 before and after on the 72. These
  equal the builder's gate figures (S1 +196, S2 -93, S3 -226); S1c (0), S1n (7 = 7), S12 (24 = 24),
  S10a (192 = 192) are as the gate reports for the whole corpus.
- By insight id the S1 move is three transitions, nothing else changed state:
  - 403 insights go from shown to hidden-`unknown`. All 403 sit on 22 voice keys, every one of
    them an applied unnaming (the 24 unnamings, 2 of which have no shown insight).
  - 207 go from hidden-`unknown` to shown, on applied renames of an unnamed voice.
  - 289 go from hidden-`unidentified` to shown, on applied renames. S1 = 403 - 207 = +196;
    S2 = 403 - 207 - 289 = -93.
- Is each of the 403 a trade the rule accepts? 391 sit on voices whose name I read as wrong (the
  name belongs to another person, so the insights were misattributed and are now hidden). 12 sit
  on three voices I cannot decide either way (`ce8e9a37` SPEAKER_04: 7, `substack:post:220234130`
  SPEAKER_01: 4, `youtube:MWX36ZYnsm0` SPEAKER_00: 1); a hidden insight there replaces a name that
  may or may not be right, which is the rule's side of the trade. The largest are `substack:post:177641179`
  SPEAKER_00 (56), `4e6324ca` SPEAKER_01 (55), `33269f09` SPEAKER_00 (46), `substack:post:221372257`
  SPEAKER_00 (38), `93208e83` SPEAKER_00 (35), `Buzzsprout-20005821` SPEAKER_01 (34), `a31bf15e`
  SPEAKER_03 (34), `29d987a7` SPEAKER_03 (26). In all eight the long
  voice is a person other than the one named (mostly the guest, served under the host's name),
  so the insights were shown under the wrong person.
- Cost the operator should see first: `4e6324ca` ends with zero visible insights (S10a +1 there,
  offset by -1 in two other episodes: net 0, 192 = 192).
- Whole-corpus S7 / S6 / S8 / S11 were not measured by the gate and not by me: NOT VERIFIED.

**K5 Applied changes: PASS, 0 wrong of 85 by my reading, with the following named.**
- Reading: `step8_reading.tsv` rows 0-84 (all 85). Renames (61): 51 with positive identity
  evidence (an answered address, a self-introduction, the host's introduction of the guest, a
  continuation of a two-host introduction); 9 `c`, thin: fingerprint at 0.91-0.97 plus a
  solo-monologue or host-read context, with no name in the text (`05598a71` SPEAKER_01,
  `1_2766234` SPEAKER_02, `20ebeb5b` SPEAKER_01, `41ca1334` SPEAKER_01, `4ae8ea8d` SPEAKER_00,
  `4e743da6` SPEAKER_01, `75b67139` SPEAKER_01, `a353187e` SPEAKER_01, `iono.fm/e/1729071`
  SPEAKER_05); 1 with a wrong new role.
- Role defect: `Buzzsprout-20005821` SPEAKER_00 gets role `guest` (from the feed's group) while
  the voice introduces itself as the host standing in for the usual host. Its partner SPEAKER_01
  is correctly unnamed. 12 of the 61 renames change the role (11 guest to host, 1 host to guest);
  this is the one host to guest.
- Unnamings (24): 21 where the text shows the name is another person's (the speaker is the guest
  or a different host; the pair is a swap whose other half the rung fingerprint-matches); 3
  undecidable, hidden under the rule: `ce8e9a37` SPEAKER_04, `substack:post:220234130` SPEAKER_01,
  `youtube:MWX36ZYnsm0` SPEAKER_00.
- Inconsistency between applied and left: the same initial-plus-surname spelling variant is applied on
  `Buzzsprout-19147714` SPEAKER_01 and left as `WRONG_PERSON` on `Buzzsprout-19066916` SPEAKER_01
  (same person, same pattern, 0.92 and 0.87). The applied one changes no identity.
- Not a defect of 0035, but it limits the benefit: in 11 swap pairs the unnaming is applied and
  its partner rename (the voice the fingerprint says owns the name) is left. 12 shown insights
  stay on those partner voices under a name I read as wrong (`93208e83` SPEAKER_02: 7;
  `substack:post:221372257` SPEAKER_01: 4; `33269f09` SPEAKER_01: 1). Before 0035 they were
  shown under the same wrong names.

**K6 Left changes: 11 disagreements in 48 reads.**
- Sample: `random.Random(8035).sample(range(78), 40)` over the 78 left renames sorted by
  (episode id, voice), plus the 8 partner voices of applied unnamings that the sample did not
  contain (the other 3 of the 11 partners were in the sample). 48 voices read, whole transcript
  searched for the new name, not only the packet.
- I agree with 37 (31 + 1 + 5 where nothing in the text identifies the voice: promo and ad voices,
  panel members whose names the transcription spells alike, mixed voices that hold two speakers'
  lines, spelling variants of what may be two people). I would apply 11:
  - strong (self-introduction, or a host introduction plus a two-voice episode): `1de6bacf`
    SPEAKER_00 (the voice says it hosts the show under the listed name), `97da41be` SPEAKER_04
    (the voice's line names itself, and the guest addresses it by that name),
    `Buzzsprout-19280635` SPEAKER_01, `Buzzsprout-19545344` SPEAKER_01, `01a0aee8` SPEAKER_00,
    `93208e83` SPEAKER_02;
  - moderate (elimination or the host's address): `4e6324ca` SPEAKER_02, `substack:post:221372257`
    SPEAKER_01, `33269f09` SPEAKER_01, `66ed7536` SPEAKER_03;
  - cosmetic: `Buzzsprout-19066916` SPEAKER_01.
- Sample proportion 8 of 40 (20%). The other 38 of the 78 were not read; at that proportion about
  16 of the 78 are confirmable. All 11 are on the conservative side: nothing wrong is applied, a
  hidden or wrongly named voice stays as it is.
- Weak spot of the review itself: the same pattern (a host welcomes a guest by name, two voices,
  the other answers) is applied on `019fece8` SPEAKER_00 and left on `01a0aee8` SPEAKER_00,
  `Buzzsprout-19280635` SPEAKER_01 and `Buzzsprout-19545344` SPEAKER_01.

**K7 Mutation: 43 mutants, 20 caught, 23 survived; every file restored** (`r8/mut.py`; both
sources' sha256 equal to the pre-run values, `git status` shows only the two untracked gate
files). Each mutant ran `test_m0035_voice_identity_names.py` and `test_m0034_talk_floor_names_removed.py`
(29 tests) with `-x`.
- Caught (20): MIN_TALK_S to 100000; floor skip removed; floor skip inverted; rename role forced
  to the old role; rename branch dropped; talk seconds zeroed; `_reviewed` rename on
  `WRONG_PERSON`; rename applied on any verdict; `review_names` old-first; `review_names`
  new-only; `RENAMED_SOURCE` to `roster`; diagnostics path; feed directory one level up; 0034's
  FLOOR to 0.0 and to 0.5; 0034's `_LABEL`; 0034's `talk_by_voice` sign; 0034's list field order;
  0034's list hash in the receipt; 0034's verify ignoring missing verdicts.
- Survived (23): the whole unnaming path is unguarded.
  - Detector, unnaming branch (4): branch dropped; old name kept; `same[-1]` to `same[0]`; and
    served names ignored by `decide`.
  - `_reviewed`, unnaming branch (3): applies on `CORRECT`; applies on any verdict; never
    applies. The 24 unnamings in the shipped list depend on this branch; the gate clone exercises
    it (24 of 24 written), the unit tests do not.
  - `_reviewed` drops `CORRECT_MISSPELLED`: the 10 misspelling renames depend on it.
  - `RESPELLING_TAKES_RENAMED_SOURCE` to False: the respelling test uses a name pair that
    `same_person` rejects (`r8/sp.py`: False), so it never reaches the branch; the 8 respellings
    in the list rely on it (on the clone all 61 have source `voice_identity`).
  - `BACKUP_TAG` set to 0034's and `RECEIPTS_FILE` set to 0034's: undo and receipts of the two
    migrations would collide, and no test notices.
  - Constants of the rung: THRESHOLD to 0.99 and to 0.2; MIN_TALK_S to 0; model name ignored; the
    no-groups early return; `new_role or old_role`.
  - Others: `<=` at the floor (equivalent in practice), empty episode-id guard, `to_version`,
    `max` to `min` in the similarity text, 0034's `served_by_voice` first-vs-last, 0034's verify
    ignoring a refused episode.

**K8 Safety of the apply: NOT VERIFIED by me.**
- The undo test, the per-episode plan file and the derived-surface census are not in the record or
  in the gate report (the record's 4b says they are not yet done). The gate report holds the plan
  hash (`plan_sha256`) and per-episode inputs, and gate35 has `.podcast_scraper/upgrade-backups/0035`.
  I did not run the undo (it would write into the clones).
- Verify on the clone is reported ok in the gate report ("every voice the review confirmed is off
  the floor's name": 0034's message string, still printed by 0035, a wording defect).

**K9 Landing: FAIL at f9f228038.**
- `test_every_registered_migration_under_the_regime_is_allowed_to_land` is red (run by me:
  `1 failed, 29 passed` in `test_change_gate.py`). For 0035: step 4 has no `Dated:` line, steps 6
  and 7 are missing, step 8 was missing until this section; there is no shipped
  `upgrade/outcomes/0035_voice_identity_names.json`, so step 8 cannot name its sha256 yet. Same
  for 0034.
- Sections 1 to 5 of this record are the pre-build text: it still says "Not in the registry yet"
  and lists a "not registered" test, while f9f228038 registers it and that test was replaced.
  Doc and code diverge.
- The prediction (4b) gives counts only. It states no integer for S1, S2, S3; the measured +196,
  -93, -226 were never predicted (the operator accepted the order MISSED, not the missing
  integers).
- Secrets scan over `origin/main..HEAD`: gitleaks 8.30.1, PASS.

### Per-criterion result

- K1 PASS. K2 PARTIAL (evidence pointer; no refuted / unconfirmed split). K3 PASS. K4 PASS (S1
  +196 explained by insight id, 391 + 12 of 403 are hidden-for-wrong-or-undecidable-name). K5
  PASS (0 wrong applied; 9 thin, 3 undecidable, 1 role, 1 inconsistency named). K6 11 of 48
  disagreements, all on the conservative side. K7 23 unguarded mutants. K8 NOT VERIFIED. K9 FAIL.

### Not covered / not verified

- Whole-corpus S7 / S6 / S8 / S11, and gold sets: not measured by the gate or by me.
- Undo, derived-surface census, prod plan diff (step 10): not seen.
- 38 of the 78 left renames: not read. The 163 list is only the rung's candidates; a wrong name
  the rung does not flag is not in scope.
- The packets the list reviewers saw: I searched the whole transcript, which the reviewers may not
  have done; a reviewer who saw only the packet had less evidence than I did.
- The detector re-run is the builder's code. My independent code covers the apply, the numbers
  and the reading, not the detector's rule.
- `make ci-fast`: not run (instruction), so K9 is open.

### Reviewer's code (not committed; `scratchpad/r8/`)

`treediff.py`, `own_s.py`, `join.py`, `s12.py`, `diagchk.py`, `pairs.py`, `s1voices.py`,
`resid.py`, `refind.py`, `refind2.py`, `ov.py`, `sp.py`, `mut.py` (+ `mut.log`,
`mut_results.json`), `show2.py`, `autog.py`, `grep.py`. Per-voice reading:
`scratchpad/m0035/step8_reading.tsv` (133 rows: 85 applied, 40 seeded left, 8 partner voices).

Verdict: GO WITH CONDITIONS Conditions: (1) make the record landable before any hand-off: section 4 gets a `Dated:` line with integers for S1, S2, S3 (+196, -93, -226 as measured, with the 403 / 207 / 289 split) and the 85 / 78 counts, sections 6 and 7 are written (the undo result, plan file and derived-surface census attached), the stale "not in the registry" text is fixed, the outcome is built with `build_outcome.py` and its sha256 named in section 8, and `make ci-fast` passes on the committed tree with the log attached (today the landing test is red for 0034 and 0035); (2) add the unit tests the 23 surviving mutants name: the detector's unnaming branch, `_reviewed` for an unnaming (applies on `WRONG_PERSON`, keeps on `CORRECT`), `CORRECT_MISSPELLED`, a respelling pair `same_person` accepts, `BACKUP_TAG` and `RECEIPTS_FILE` distinct from 0034's, and the role fallback; (3) the operator sees the table first (S1 +196 as the hidden side of the trade, 12 of the 403 on undecidable voices, `4e6324ca` left with zero visible insights) and rules on the 11 voices in K6 and the 12 shown insights left under a wrong name on partner voices, and on the role of `Buzzsprout-20005821` SPEAKER_00; none of the 11 is applied without a fresh verdict row and a rebuilt list hash; (4) the shipped verdict rows carry the deciding line for the 85 applied entries and mark the 74 "no evidence" rows and the 4 unnamings that carry it `undecidable:` and the others `refuted:`, so the file can be audited without the sibling repo.

## 8b. Verdict (round 2)

Step-8 reviewer, round 2: a fresh agent, 2026-10-26, not the builder, not the round-1 reviewer, not a
reader (C, D, E) and not a resumed session. Code under review: HEAD `7fd0eff36` on
`feat/m0034-list` (migration, list, verdicts and tests identical to the measured commit `2080d1cb2`;
`git diff 2080d1cb2 7fd0eff36 -- src tests` touches only m0036 and the registry line). Judged
against acceptance evidence (in git history). Data read only: base `gate34e` (snapshot
`snapshot-prod-20261025` with the final 0034), after `gate35b`. Every integer below is from a command
I ran, with my own code in `scratchpad/r8b35/` (`c1.py` to `c23.py`, `treediff.py`, `mut.py`, `refind.py`,
`undo2.py`, `view*.py`); ids only, no names, no transcript text. Earlier reviews were inputs, not evidence.

Seed: I recorded it AFTER I had opened the shipped verdicts file for counts (I needed the counts to
build the populations), so it is not blind in the sense of 37; the draws are over keys sorted by
(episode id, voice) and the seed is **350826** (differs from 8035, 77 and 20261026).
The worktree held uncommitted edits made by another session while I worked (the D1 bridge-aliasing
block of 11:29, `acceptance-C`, tests); I reviewed HEAD, `git diff` showed no change under `src/` at
12:45, and I did not review those edits.

### Outcome first (criterion 41)

Measured on `gate34e` -> `gate35b`; prediction = 4d (09:20, restated 09:38).

| Number | 4d prediction | Measured by me | Pass? |
| --- | --- | --- | --- |
| list | 162 (138 / 24) | 162 (138 renames, 24 unnamings) | yes |
| renames applied | 71-76, restated 68-92 | **92** (82 `CORRECT` + 10 `CORRECT_MISSPELLED`) | in the restated total; its parts are not (below) |
| renames not added | - | **46** | - |
| unnamings applied / kept | 20-21, restated 21-22 | **21** / **3** | yes |
| voices rewritten | 91-97, restated 89-114 | **113** (= 92 + 21) | yes (restated) |
| episodes written | never stated | **99** (= episodes of the 113 applied keys) | miss: no integer was stated |
| S1 | +150 to +184 | **+72** | miss |
| S2 | -105 to -93 | **-404** | miss |
| S3 | -260 to -226 | **-407** | miss |
| S10a | -2 to +2 (acceptance-B) | **-4** | miss |
| S1c / S12 / S10b / commercial / A1 / insights / quotes / S1n | 0 / 24 / 2 / 94 / 382 in 290 / unchanged / unchanged / no rise | 0 / 24 / 2 / 94 / 382 in 290 / 130,401 / 252,651 / 7 -> 7 | yes |

How I got them: the committed `score_episode` run on the 99 written episodes, before and after, differs
only on S3 -407, S2 -404, S2-unidentified -476, S1 +72 (= S2-unknown +72) and S10a -4; every other counter
(S1c, S1n, S12, S10b, commercial, A1, insights, quotes) is unchanged. My own join (insight id -> first
SUPPORTED_BY quote -> the segment holding its `char_start` -> raw voice -> diagnostics `talk_time_s`
share >= 0.30; `c5.py`) gives the same S1 +72, S2 -404, S3 -407. The other 2,867 episodes are
byte-identical (below), so the whole-corpus deltas equal these. The gate report agrees
(`S1 16866 -> 16938`, `S2 30619 -> 30215`, `S3 9359 -> 8952`, `S10a 191 -> 187`).

### The S1 / S2 / S3 / S10a miss, by insight id

Three transitions, nothing else changed state (`c5.py`, `c6.py`, `c7.py`), and **0** moved insights sit on
a voice that is not an applied key of the right kind:

- **386** insights shown -> hidden-`unknown`, on **19** voices, every one an applied unnaming (21 applied;
  the other 2 have no shown insight). Per voice (episode id prefix, voice: insights): `substack:post:177641179`
  SP00: 56; `4e6324ca` SP01: 55; `33269f09` SP00: 46; `substack:post:221372257` SP00: 38; `93208e83` SP00: 35;
  `a31bf15e` SP03: 34; `Buzzsprout-20005821` SP01: 34; `29d987a7` SP03: 26; `9ee5a39d` SP01: 12; `66ed7536`
  SP02: 12; `5cdca311` SP02: 10; `gid..RGR5` SP00: 7; `ce8e9a37` SP04: 7; `d86f541d` SP02: 5; `9616ab52`
  SP04: 4; `iono.fm/e/1727005` SP03: 2; `gid..YjYu` SP04: 1; `ba73ffec` SP07: 1; `7c41b808` SP01: 1.
- **314** hidden-`unknown` -> shown, on **28** voices, applied renames. **476** hidden-`unidentified` ->
  shown, on **30** voices, applied renames. S1 = 386 - 314 = +72; S2 = 386 - 314 - 476 = -404.
- By origin of the rename (round-1 verdicts file in the private eval repo): the 56 carried renames give
  192 + 289 (round 1 had 207 + 289: the 4 renames now not added carried 15 unknown insights); the **36**
  renames newly added from the 78 give **122 + 187 = 309** insights shown, which the 4d range (built on 11 to
  16 added) could not contain. The 3 kept unnamings hold 12 + 4 + 1 = **17** shown insights
  (`1_2766140` SP00: 12, `substack:post:220234130` SP01: 4, `youtube:MWX36ZYnsm0` SP00: 1), which is
  exactly round 1's 403 minus today's 386.
- S10a: 4 episodes go from no visible insight to some (`a353187e-ea29-4e21-940d-7129aad784fa`,
  `b22b4fbe-1f25-11f1-990b-1718674eb070`, `b28f69e0-1f25-11f1-990b-37773ffde874`,
  `bd1441d7-cb92-44bc-b86d-5fc0eb97af6b`), all by renames; **0** go the other way. `4e6324ca` ends with 6 of
  61 insights shown (55 before); it is not zero-visible.

Judgement of the MISSED step. The 09:38 text promised S1 / S2 / S3 as integers "after the readers settle
the counts and before the clone"; they were not restated, so four numbers landed outside every stated
range and the episodes integer was never stated. It is a process miss and 4d records it honestly, but
that restatement is the D15 gate and it is not met. The 09:38 text also said "within the restated 68 to
92": true of the total (60 - 4 + 36 = 92), false of its parts (36 added against 17 to 32 stated). The
direction is the favourable one (fewer hidden insights than predicted, more shown) and every moved
insight is on a voice I read (below), so I do not send it back to step 3; the operator must accept it
at step 9, and the integers above become the prediction prod's step 11 is held to.

### Per criterion (acceptance-B)

- **31 Population: PASS.** `find_changes` on the pre-0034 snapshot and on `gate34e`: 162 found, 0 only
  in the detector, 0 only in the list, 0 entries different (48 s, 64 s). The sole exclusion is a named
  rule (`EXCLUDED`) with a test; its key is absent from the list, the shipped and private verdicts, the
  receipts and the plan, and its 109 segments, its roster entry and its diagnostics entry are unchanged (`c14.py`, `c24.py`).
  Voices shared with 0034's list: **0**; shared episodes 10; episodes written by both: **1**, `https://iono.fm/e/1728203`
  (13 files in both receipts; 13 of 13 have 0035's `sha_before` equal to 0034's `sha_after`; 0035
  rewrote its KG with identical bytes, the one receipt whose file did not change). My list hash
  `cbf1a4ab...` equals the verdicts header; the receipts header carries the same list hash and the
  verdicts sha `84ac945a...` equals the shipped file's.
- **32 Basis: PARTIAL.** Rows with no basis 0 of 162; basis outside its set 0; unnamings applied without a
  line 0 (21 of 21 `another_person` with `segment_idx`); evidence cells beginning "no evidence" 0; shipped
  vs private file 0 differing fields over 162 rows; the sha256 of the segment text at `segment_idx` equals
  `line_sha256` for **117 of 117** rows (`c4.py`). Not met: **45** `undecidable` rows ship with an empty
  `segment_idx` and `line_sha256` (acceptance asks 162 of 162); and the loader accepts a rename
  `CORRECT` / `CORRECT_MISSPELLED` with basis `undecidable` and APPLIES it (`hole.py`: "loaded; applies
  [('e', 'SPEAKER_00')]"). The shipped file has 0 such rows, so nothing wrong is applied today; the
  guard the operator's rule (b) needs does not exist, and no test could fail for it.
- **33 Kept stays kept: PASS on the apply path, short on the report.** The 49 kept or left voices (46 renames
  and 3 unnamings): 0 changed segment label or role, 0 changed diagnostics entry, 0 changed roster
  (name, role) (`c21.py`). The plan and receipts do not name them by key (`MWX36ZYnsm0` occurs 0 times in the gate
  report); only the verdicts file does.
- **34 Reread: operator-accepted below the bar.** C and D agreed on **82 of 99** (bar 90, scaled from 74 of 82);
  the 17 splits (all renames, none an unnaming) went to E, majority of three: 14 applied, 3 not added
  (`c13.py`). The operator accepted this in the record.
- **35 Numbers:** table above. Voices whose label or role moved = **113** = the applied keys, **0** extra,
  **0** missing; segment text unchanged (`c3.py`). Tree diff over every file outside `.podcast_scraper`
  (`treediff.py`): 130,649 files before, 130,650 after (the new one is the receipts file), **1,358**
  changed, 99 `.metadata.json`, no change outside those 99 episodes; receipts 1,359 = 1,358 + 1 identical.
- **36 Mutation: PARTIAL.** 57 mutants (the 43 of round 1, re-based, plus 14 for this revision) against the
  whole `tests/unit/podcast_scraper/upgrade/` directory in a `git archive` copy: **54 caught, 3 survived**,
  files restored (`mut.log`). 20 of round 1's 23 survivors are now caught; the 3 left are the survivors listed here. Survivors: (1) `<` ->
  `<=` at the talk floor, equivalent (a share exactly at the floor); (2) dropping the `if not groups: return`
  in `episode_changes`, equivalent (I read `voice_identity.decide`: with no groups every voice is skipped
  and it returns `{}`); (3) `to_version` 2.7.29 -> 2.7.28, not equivalent, no test pins it. Over the bar
  by one real survivor, plus the missing guard above, which no mutant can show because the check is absent.
  New-revision mutants caught: exclusion ignored; unnaming on `CORRECT`; unnaming on `undecidable`; 0035
  basis set not passed; basis columns not required; empty `segment_idx` / `line_sha256`; no basis; basis
  outside the set; non-digit index; sha format; `require_basis` not enforced. (The mutant that drops `undecidable` from `CORRECT` is caught only through 0034's registered file, not by any test of 0035's own guard.)
- **37 Reading: 0 wrong applied.** I read all **92** applied renames (36 new + 8 thin + 19 of the draw + 29
  others; `view.py`, `view2.py`, `view6.py`), all 24 unnamings, 30 of 46 left renames, the dropped entry and
  its partner, and found for each applied one a line or a chain in the text, the thinnest listed below:
  **0 wrong applied renames, 0 wrong applied unnamings**. The 9 thin renames: 8 are applied, each with a text
  tie I found beyond the stored line (`1_2766234` SP02: the voice addresses the other guest by name; `20ebeb5b`
  SP01, `4ae8ea8d` SP00, `4e743da6` SP01, `75b67139` SP01: mid-sentence hand-offs with the named host voice;
  `41ca1334` SP01: the other host later addresses it by first name; `a353187e` SP01: solo host self-intro;
  `iono.fm/e/1729071` SP05: introduced by a headline, then asks); `05598a71` SP01 is flipped to not added. The
  20-draw (seed 350826): 19 applied right, 1 (`substack:post:195859978` SP00) not added, no first name in the text,
  agreed. Three applied renames rest on a thinner chain than their stored line: `iono.fm/e/1728203` SP05
  (the line introduces the voice before it, not it; the same clip in `iono.fm/e/1728106` shows SP05's words
  are the guest's answer, so the name is right); `urn:bbc:podcast:p0p2vv0y` SP06 (a hand-over plus a
  third-person credit; the basis label `addressed` does not match its line); `1_2766234` SP02 (closed set of
  three). Role changes: 13 guest -> host, **0** host -> guest. Unnamings: all 21 are shown by the text (the voice speaks of the name as another person, is addressed as
  someone else, or states its own identity); the two weakest are `66ed7536` SP02 (closed set: the host voice is SP03, which addresses the three
  guests) and `ba73ffec` SP07 (a mixed voice that carries both hosts' intro lines; the text shows part of
  it is the other host). Kept: `youtube:MWX36ZYnsm0` SP00 and `substack:post:220234130` SP01 are right
  keeps (nothing shows another person). `1_2766140` SP00 is a **grey keep**: the host addresses it as a
  different spelling of the name, the sibling voice is tied to the name by content (`1_2766140` SP01 is applied
  on that), so the 12 insights stay under a name the text may well give to the other guest; the two similarly
  transcribed guests now share one roster entry. Round 1's `ce8e9a37` SP04 is now applied with a stronger line
  than the stored one (the voice says "take it away" to the person whose name it carried).
- **38 Left renames: 1 of 30 missed (bar 3).** Population is 46, not 62-67, because the readers added 36.
  Seeded 30 of 46: I would add one with a positive line: `flightcast:01M4BDW19ZFPED9TQ4S5DP60PY` SP00 (the
  episode description names one host and the guest; the readers used the feed description's two co-hosts).
  Two more would be added by closure within a two-person show (`9ee5a39d` SP00, `93208e83` SP02); the
  operator's rule says elimination never counts, so I count them as left correctly.
- **39 Safety of the apply: PARTIAL.** Undo: on a copy-on-write copy of the after-clone, `m0035.undo` restored
  **1,359**, refused 0, and the tree (ledger included, receipts log aside) equals the 0034-only clone, **0** differing files
  (`undo2.py`). Then `m0034.undo`: restored 1,518, refused 0, and the tree equals `snapshot-prod-20261025`
  except `upgrade_ledger.json`, whose only difference is `"version": "2.7.28"` against `"2.7.27"`.
  The gate applies the migration directly, so 0035 never has a ledger entry on the clone; the record's
  sentence "on prod an undo also removes 0035's ledger entry" was not exercised. Derived surfaces
  (`derived_surface_census.py` on `gate35b`): stale `insight_density` / `insight_sentiment` / manifest in 99,
  `.anon.txt` 93, ad map 10, context 15, bridge 0, turns 0, ad-map offsets off 0; its header says "episodes
  written: 217", a count I did not reconcile (the tree diff says 99). The outcome file does not exist yet.
- **40 Gold: no effect, vacuously.** `gold_insights.py` on both clones: the reports are equal apart from the
  snapshot name (dev_v1 2452 / 380 / 1313, val_v1 11498 / 2128 / 4252, val_v2 3707 / 439 / 1109); 0 of the 751
  gold cases name any of the 99 written episodes, so S7 / S6 cannot see this change.
- **41 Order:** the outcome table is not first in section 7 (mechanics lead); the table above is.

### Other findings

- **Inherited verify message.** `verify` returns ok = True with "every voice the review confirmed is off the
  floor's name" (gate report). The ok value is right; the sentence is 0034's and false for a migration that
  mostly adds names. It is the line the operator reads at the end of a prod run.
- **13 episodes end with one name on two voices** (0 before; `c11.py`, `c12.py`): 10 are a solo host split into
  two voices (the operator's ruling), 3 are not: `1_2766140` (two guests), `db4aff8e` (the second holder
  already carried the name; the text shows it is the other host), `iono.fm/e/1728203` (the second holder is
  the interviewer). Roster entries merge the voices; the GI of `1_2766140` carries two Person nodes with one name.
- **Residual wrong names stay by design** on partner voices of applied unnamings (the round-1 note); the
  excluded entry's partner is correctly unnamed and the excluded voice keeps its present name.
- Bridges: the 99 bridges 0035 rewrote have identities before and after (0 empty); the pending D1 fix is
  not triggered on this clone.
- Secrets: `gitleaks git --log-opts="origin/main..HEAD"`: 8 commits, no leaks.
- `make ci-fast` not run by me and no log is attached to the record; the landing test is red for 0035 on
  exactly one item (no shipped outcome), plus 0034 (NO-GO line) and 0036 (no gate report).

### Not covered / not verified (round 2)

- Prod: the plan diff, the counts on the 14-odd post-snapshot episodes, the search-index rows of the 99
  (incremental index) and the `enrich --only` refresh: not evidenced on a clone.
- Whole-corpus S7 / S6 / S8 / S9 / S11: not measured by the gate or by me; S7-wrong (gating) is unmeasured
  here because no gold case touches these episodes. My protection against a wrong shown name is the reading.
- The runner path (apply, ledger entry, undo, `upgrade verify` reads pending, the scoreboard JSON equal
  before and after): not run; the outcome hook blocks a regime migration until its outcome is built.
- 16 of the 46 left renames were not read; 29 of the 92 applied renames were read quickly (one line plus
  context), not with the whole transcript; the 80 carried rows' bases were assigned by a classifier and
  I re-read only the lines I list above.
- The packets the readers saw: I searched whole transcripts, which they may not have; I did not check
  their diligence.
- Persons I called right are right by text and by the fingerprint agreeing; if both are wrong together
  (a name said about someone else) I would not see it.
- Reprocess durability; per-language rows (all `en`); decision-trace names that stay in history; the
  pipeline rung itself (mutants outside `upgrade/`).
- The D1 fix and `acceptance-C` (uncommitted, 11:29 and 11:31): not reviewed; its prediction (every number
  unchanged, only the code digest moves) is consistent with what I measured (0 empty bridges written).
- Whether 82 of 99 is high enough: a judgement the operator already took, not a measurement.

Verdict: GO WITH CONDITIONS (1) close the loader hole before any hand-off: a rename `CORRECT` / `CORRECT_MISSPELLED` with basis `undecidable` is refused by 0035's loader, with a test that fails without it (acceptance-B 32 and 36), and give `to_version` a pin or an argument, argue in writing the 2 equivalent mutants (the floor `<=`, the no-groups early return), and make 0035's verify message say what 0035 did; (2) put the D1 fix (acceptance-C) in the same code batch, then ONE re-measure on a clean tree at the final commit on the same base: every integer in the table above unchanged (162; 92 / 46 / 21 / 3; 113 voices; 99 episodes; 1,358 files; S1 +72, S2 -404, S3 -407, S10a -4; 0 empty bridges new), the sorted path set equal, refused 0, verify OK; then build the outcome with `build_outcome.py` AFTER the final code, name its sha256 here, and attach a `make ci-fast` log on the committed tree with the landing test green (0 failing, 1 run); (3) the operator accepts, at step 9, the missed restatement (S1 / S2 / S3 / S10a outside every range, the episodes integer never stated, 36 added against 17 to 32) and the restated integers become the prediction for prod's step 11; (4) before any prod apply: run the runner path on a clone once the outcome exists (ledger entry written and removed by undo, `upgrade verify` reads pending), record that an undo leaves the ledger `version` at its higher value unless reset, and add the kept and left voices by key to the plan or receipts or have the operator accept the verdicts file as that list; (5) the operator rules on the grey cases: `1_2766140` SP00 (12 insights kept shown under a name that may be the other guest's) and the 45 `undecidable` rows shipping without a closest line; and prod's 0035 plan is taken only after prod's 0034 is applied and its output equals the clone's, with `https://iono.fm/e/1728203` (written by both) the only difference allowed in the plan diff.

## 8c. Verdict (round 3)

Step-8 reviewer, round 3: a fresh agent, 2026-10-26, not the builder, not the round-1 or round-2
reviewer, not a reader (C, D, E), not a resumed session. Code under review: HEAD `9e39accc5` on
`feat/m0034-list`. `git diff ffaae4ae0 9e39accc5 --stat` touches only `docs/wip/changes/` (no `src`,
no `tests`), so the re-measure commit `ffaae4ae0` and HEAD run the same code; `migration_digest` of
0035 at HEAD, computed by me from a `git archive` copy, is `45de67894956...`, equal to the gate
report's `code_digest`, and the report says `dirty: false`. Judged against round 2's conditions
(section 8b), `acceptance-B` and `acceptance-C`. Data read only: base `gate34f` (snapshot
`snapshot-prod-20261025` with 0034 round 3), after `gate35c`. Mutations and the runner path ran only in
copies (`scratchpad/r8c35m`, `r8c35`, `r8c35_run`); nothing in the repo was edited but this section.
My code: `scratchpad/r8c35_*.py` (find, verd, tree, score, voices, shared, empty, empty2, mut, d1,
undo, digest, cmp, gcmp). Ids only. I read no voice this round (no seed): the reading of round 2
(0 wrong applied) is an input, not re-done. `make ci-fast` was not run by me.

### Outcome first

Every integer below is from a command I ran. "Record" is the claim in section 7 / 4d / 8b.

| Number | Record | Mine | Same? |
| --- | --- | --- | --- |
| list; renames / unnamings | 162; 138 / 24 | 162; 138 / 24 (list file) | yes |
| detector vs list, on the pre-0034 snapshot and on `gate34f` | 162, 0 differences | 162 and 162 found; only-detector 0, only-list 0, entries different 0 (both runs) | yes |
| renames applied / not added | 92 / 46 | 92 (82 `CORRECT`, 10 `CORRECT_MISSPELLED`) / 46 | yes |
| unnamings applied / kept | 21 / 3 | 21 / 3 | yes |
| voices whose label or role moved | 113 | 113 = the applied keys; 0 extra, 0 missing; 0 of the 49 left or kept moved | yes |
| episodes written | 99 | 99 = the episodes of the 113 applied keys | yes |
| files changed | 1,358 | 1,359 differ between `gate34f` and `gate35c` outside `.podcast_scraper`: 1,358 content files + `upgrade_ledger.json` (the after-clone's ledger holds a 0035 entry stamped 12:33:00; I did not establish who wrote it); 99 `.metadata.json`; the new file is the receipts log | yes |
| path set vs round 2's | equal | 1,358 vs 1,358; 0 added, 0 removed against round 2's `treediff.json` | yes |
| S1 / S2 / S3 / S10a, committed `score_episode` on the 99, before / after | +72 / -404 / -407 / -4 | +72 (663 -> 735) / -404 (1,537 -> 1,133) / -407 (650 -> 243) / -4 (4 -> 0); S2-unidentified -476; A1 hits 12 -> 12, S1c 0, S12 0 in these episodes | yes |
| other 2,867 episodes | byte-identical | no file outside the 99 episodes differs | yes |
| gate report vs the 2080d1cb2 report | "every number equal" | `gating_moves` equal, `files_changed_by_suffix` equal, 70 `episode_deltas` equal, `plan_sha256` equal (`334dadf49e94...`); only `code`, `code_digest`, `verify.message`, `measured_at` differ | yes |
| empty bridge files | 15 after, 0 written | snapshot 17, `gate34f` 15, `gate35c` 15; 0 of the 1,358 changed bridges is empty | yes (see below) |
| verdicts file | 84ac945a... | sha256 `84ac945a...`, list header `cbf1a4ab...`; 45 `undecidable` rows, all with empty `segment_idx` / `line_sha256`; 0 rename `CORRECT` and 0 unnaming `WRONG_PERSON` on `undecidable` | yes |

Empty bridges. The 11:29 prediction and `acceptance-C` bar 3 said 16 (snapshot 17, after 0034 16, after
0035 16). Measured: 15 and 15. Cause, from my diff of the two lists: two of the snapshot's 17 empty
bridges (feeds `rss_feeds.simplecast.com_aef1e4c0`, `rss_feeds.simplecast.com_e739a158`) are
non-empty on `gate34f` (0034 round 3 writes an `identities` key into them); `gate35c` adds none. The
bar's intent (0 new from 0035) is met; the integer 16 is not, and the record states 15 without saying
the prediction was 16.

Tests: `tests/unit/podcast_scraper/upgrade` in the copy of HEAD: **668 passed** (50 s). The landing
test `test_every_registered_migration_under_the_regime_is_allowed_to_land` is **red** at HEAD (4
items; one is 0035's missing shipped outcome, one is 0034's NO-GO); expected before the outcome exists.

### Round-2 conditions, one by one

**(1a) Loader hole: closed for `CORRECT`, open for `CORRECT_MISSPELLED`.** Mutants run against the
whole upgrade test directory in `r8c35m`:
- rename check removed: caught (`test_a_rename_on_undecidable_is_refused`);
- rename check on `CORRECT_MISSPELLED` only: caught;
- unnaming check removed: caught; skip condition inverted: caught;
- **rename check on `CORRECT` only (the `CORRECT_MISSPELLED` half dropped): SURVIVED** (668 passed).
  A `CORRECT_MISSPELLED` rename on `undecidable` would load and apply, and no test says otherwise.
  The shipped file has 0 such rows, so nothing wrong is applied today; the test needs a second
  parameter.

**(1b) `to_version`: closed.** 2.7.28 and 2.7.30 both caught by
`test_the_version_and_the_verify_message_are_0035s_own`.

**(1c) The two equivalent mutants "argued in writing": NOT done by the builder.** The record has no
new argument; the only one is round 2's sentence. My own check:
- No-groups early return in `episode_changes`: dropping it survives, and it is equivalent in output.
  `voice_identity.decide` skips every voice when `groups` is empty (`or not groups: continue`), so
  `owner` stays empty and it returns `{}`; the function then returns `(eid, {})` as before.
- `<` to `<=` at the talk floor: survives, and it is not strictly equivalent. It differs for a voice
  whose share is exactly 0.05; 0034 uses `<` (`share < floor`), so 0035's `<` is the consistent
  reading. No voice at exactly the floor is in the list that I know of (I did not scan for one); a
  boundary test (5 s of 100 s) kills it, so an argument or that test must be in the record.

**(1d) Verify message: closed at the gate, open in the tests.** The gate report's `verify.message` is
now "every rename the review confirmed carries the fingerprint group's name, and every unnaming the
review confirmed is off its voice". The test reads the attribute `_verified`, not what `verify()`
returns: the mutant that makes 0034's `verify` return its own constant text for every subclass
**SURVIVED** (668 passed).

**(2a) D1 in the same code batch: closed.** `ffaae4ae0` carries the fix and the migrations' changes.
Repro `test_the_shared_rewrite_never_writes_a_bridge_without_its_identities`: **3 passed** on HEAD;
with the fix reverted in the copy, **1 failed (`[renamed]`), 2 passed**. Reverting the fix is caught.

**(2b) ONE re-measure at the final commit, every integer unchanged: closed.** Table above. The report
was measured on a clean `ffaae4ae0`; HEAD has the same code digest.

**(2c) Outcome built with `build_outcome.py`, its sha256 named here, `make ci-fast` log attached with
the landing test green: NOT done.** The outcome cannot exist before a GO verdict (the script reads the
record's last `Verdict:`); this verdict is that gate. In a COPY with a synthetic verdict line the
script wrote an outcome and the digest in it equalled the migration's digest at HEAD, so the build
works on this code; that outcome is not the shipped one.

**(3) The operator accepts the missed restatement at step 9:** open, listed below.

**(4a) The runner path on a clone: closed, by me.** In `r8c35_run` (a copy-on-write copy of `gate34f`,
ledger at 2.7.28, 0034 in it), with the outcome built in my copy of the repo:
- `upgrade status`: pending 0035, 0036; `upgrade run --dry-run --to 2.7.29`: "outcome hook: plan
  diff 0 episode(s)", would rewrite 99 episodes;
- `upgrade run --to 2.7.29 --yes --no-snapshot`: exit 0, "rewrote 99 episode(s) ... diagnostics_voices_renamed=113
  ... roster_entries_unnamed=21 ... voices_named=61", ledger gains `0035_voice_identity_names`
  `to_version 2.7.29`, version 2.7.29; the hook's outcome record has `verdict: PASS`, `failures: []`,
  `plan_diff: []`;
- the tree after the runner equals `gate35c` except `upgrade_ledger.json` (1 file of 130,387);
- `m0035.undo`: restored **1,359**, refused **0**; the ledger entry is **removed**; `upgrade status`
  reads 0035 pending again; the ledger `version` stays **2.7.29** (not rewound; `record_reverted`
  says so on purpose); the tree equals `gate34f` except the receipts log and that version;
  `upgrade verify` on the undone copy: exit 0, every applied migration OK.
  The record's sentence "on prod an undo also removes 0035's ledger entry" is true; the version
  remark is not in the record.

**(4b) The kept and left voices by key in the plan or receipts: NOT done.** The record is unchanged on
this. The 49 (46 renames not added, 3 unnamings kept) are named only in the verdicts file; the
operator can accept that file as the list (step 9 below).

**(5) Grey cases and the prod sequence:** open for the operator; the sequence is judged below.

### The episode both 0034 and 0035 write, and prod's plan-diff rule

It is the single run dir of `https://iono.fm/e/1728203`: the receipts share **13** files in **1** run dir;
for **13 of 13** 0035's `sha_before` equals 0034's `sha_after`; 0035 rewrote one KG with identical bytes
(the 1 of its 1,359 receipts with `sha_before == sha_after`).

The rule in section 7 ("0 differences, provided prod's 0034 output equals the clone's") is right for how
prod runs it, and round 2's wording ("this episode the only difference allowed") is right only for a
dry run taken before 0034 is applied. Evidence: `_prepare_regime` refuses a regime run with more than one
migration pending ("runs alone ... apply the earlier ones first, then this one with --to its version"),
so 0035's plan is taken after 0034 is applied; the clone's plan for 0035 was taken on the clone with 0034
applied; the plan rows carry input sha256s, and 13 of 13 inputs of this episode are 0034's outputs. So
equal 0034 output gives equal rows. A dry run on prod before 0034 is applied would show this episode
as a difference by that chain; I did not run that dry run.

### Acceptance-C

1. Re-measure equals the last on every number and the same file set, verify OK, refused 0: **met** (above;
   refused 0 and verify ok are in the gate report; path set compared).
2. Repro: 3 of 3 pass after the fix, at least 1 fails without it, on both trees: **met** (3 passed; 1 failed,
   2 passed).
3. Empty bridge files after 0035 equal after 0034, 0 new: **met in intent** (15 and 15); the stated 16 was
   not (cause above).
4. Mutation, fix reverted: **met** (1 of 1 killed).
5. The users of m0017's bridge rewrite: m0025 (so 0034, 0035) runs the repro. m0017 and m0020 need a test
   that feeds an empty id map and asserts the bridge keeps its `identities`, or one written sentence
   each: **NOT met**. `git diff` shows no change to the m0017 or m0020 tests and the records contain
   no such sentence. Their tests pass with the fix (668 passed), so no regression is shown; the bar is
   a bar.

### Other findings

- `acceptance-B` 36 survivors now: 1 real (`CORRECT_MISSPELLED`, above) and 1 on `verify()` text, plus
  the 2 argued above (one of them needs a boundary test or a sentence). Round 2's three survivors are
  all addressed in code (`to_version` caught); the bar of 0 survivors except 2 argued equivalents is
  met only once those two tests and the two written arguments exist.
- The secrets scan `gitleaks git --log-opts="origin/main..HEAD"`: 11 commits, no leaks.
- The wording of verify: "carries the fingerprint group's name" is a promise `verify()` keeps only as
  far as 0034's check does (no episode still to write, none refused). Same strength as 0034's message.

### For the operator at step 9 (items, not decisions)

1. Accept the missed restatement: S1 +72 against +150 to +184, S2 -404 against -105 to -93, S3 -407
   against -260 to -226, S10a -4 against -2 to +2, the episodes integer never stated (99), 36 renames
   added against 17 to 32 stated. The direction is more insights shown. The integers for prod's step 11
   become: list 162 (138 / 24); applied 92 + 21 = 113 voices; not added 46, kept 3; 99 episodes;
   1,358 files; S1 +72, S2 -404, S3 -407, S10a -4; empty bridges new 0.
2. Rule on the grey keep `1_2766140` SPEAKER_00 (12 insights stay shown under a name that may be the other
   guest's) and on the 45 `undecidable` rows shipping with an empty `segment_idx` and `line_sha256`
   (acceptance-B 32 asked for 162 of 162; the loader allows empty only on `undecidable`).
3. Accept the verdicts file as the key-level list of the 49 kept / left voices, or require them in the
   plan.
4. Round 2's open reading findings: three applied renames on a thinner chain than their stored line
   (`iono.fm/e/1728203` SPEAKER_05, `urn:bbc:podcast:p0p2vv0y` SPEAKER_06 whose basis label does not match its
   line, `1_2766140` SPEAKER_02), and 13 episodes ending with one name on two voices, 3 of them not a solo host
   split (`1_2766140`, `db4aff8e`, `iono.fm/e/1728203`). The record has no answer to either.
5. Prod order: 0034 first and its output for `iono.fm/e/1728203` checked equal to the clone's; then 0035
   alone with `--to 2.7.29`; on undo 0035 before 0034; an undo leaves the ledger `version` at 2.7.29
   unless the operator resets it. After the apply: `enrich --only insight_density,insight_sentiment` for
   the 99 episodes and an incremental `index` for their search rows (round 2's census: stale in 99).

### Not covered / not verified (round 3)

- No voice was read this round; the "0 wrong applied" rests on round 2's reading of 92 renames and 24
  unnamings, and 30 of 46 left renames; 16 of the 46 left were never read by a reviewer.
- `make ci-fast`: not run by me; no log in the record. Landing test red at HEAD (4 items).
- The shipped outcome: not built (needs this verdict); its sha256 is not known.
- A prod dry run (before and after 0034): not run; prod's plan diff, the post-snapshot episodes, the
  search-index rows and the `enrich` refresh are not evidenced on a clone.
- Whole-corpus S6 / S7 / S8 / S9 / S11 and gold: not measured by me; round 2 found 0 of 751 gold cases touch
  the 99 episodes.
- Whether any listed voice sits at exactly 5.0% of its episode's talk: not scanned.
- The detector reruns are the builder's code (`find_changes`), so they show reproducibility, not that the
  rule is right. My independent code covers the tree, the scoreboard on the 99, the voices that moved, the
  verdict counts and the receipts chain.
- Reprocess durability; per-language rows; decision-trace names that remain in history; the pipeline rung
  itself.

Verdict: GO WITH CONDITIONS Conditions: (1) before the outcome is built, add tests that need no `src` change and so keep the code digest: a `CORRECT_MISSPELLED` rename on `undecidable` is refused (the survivor), what `verify()` returns for 0035 is the 0035 text (the survivor), a voice at exactly the floor share is listed (kills `<=`) or a written argument in this record, and a written argument that the no-groups early return changes no output; (2) acceptance-C bar 5: a test for m0017 and for m0020 (empty id map, the bridge keeps its `identities`) or one sentence each in the record; (3) the record states the 15 against the stated 16 empty bridges and its cause, and that an undo leaves the ledger `version` at 2.7.29; (4) after (1) to (3), build the outcome with `build_outcome.py` on the committed tree, name its sha256 in this section, attach a `make ci-fast` log with the landing test green (0035 and its record only; 0034's NO-GO line is 0034's), and if any `src` file changed re-measure once more with every integer unchanged; (5) the operator takes the step-9 items above (the missed restatement and the integers for step 11, the grey keep `1_2766140` SPEAKER_00 and the 45 `undecidable` rows without a closest line, the verdicts file as the key-level list of kept and left voices, round 2's thin-chain and two-voices-one-name findings), and prod runs 0034 first, checks its output for `https://iono.fm/e/1728203` equals the clone's, then 0035 alone with `--to 2.7.29`, with 0 differences allowed in the plan diff.

**Builder's notes on the round-3 conditions (2026-10-26):**
- **Equivalent mutants, argued:** (a) dropping the no-groups early return in `episode_changes`:
  `voice_identity.decide` returns `{}` when `groups` is empty, so nothing downstream differs; (b)
  `<` to `<=` at the talk floor: differs only for a voice at a share of exactly 0.05; not strictly
  equivalent, so a boundary test pins it (0034's `test_a_voice_at_exactly_the_floor_keeps_its_name`
  covers the shared floor). **Correction (2026-10-26T16:36, round 4's condition 1):** that sentence
  was false: 0034's test cannot reach 0035's own comparison, and the `<=` mutant survived every
  test. Now `test_the_talk_floor_boundary_is_strict` (5.0 s of 100 listed, 4.99 not) pins 0035's
  comparison: with `<=` it fails (1 failed, 2 passed), restored it passes. Tests only; the code
  digest does not move.
- **Empty bridge files:** 15 before and 15 after 0035 on the round-3 base, not the 16 the 11:29
  block and acceptance-C stated (0034 round 3 turned 2 of the snapshot's 17 into non-empty
  `{"identities": []}`, which 0036 now treats as damaged); 0035 writes none, which is the bar's
  intent. MISSED integer, recorded.
- **Undo and the ledger:** an undo removes 0035's ledger entry, but the ledger `version` stays at
  2.7.29 (the reviewer's runner-path run); a prod undo of 0035 resets `version` to 2.7.28 by hand,
  a step written into the prod handover for 0035 (the 2.7 runbook has no such step; round 4's
  condition 2).


## 8d. Verdict (round 4)

Step-8 reviewer, round 4: a fresh agent, 2026-10-26, not the builder, not the round-1, -2 or -3
reviewer, not a reader (C, D, E), not a resumed session. Code under review: HEAD `0e9f4454d` on
`feat/m0034-list`. `git diff a859902ab 0e9f4454d -- src tests config scripts` is empty (the last two
commits are records only), so the measured commit `a859902ab` and HEAD run the same code.
`migration_digest` of 0035, computed by me from a `git archive` copy of HEAD, is `7b45818107ef...`,
equal to the gate report's `code_digest`; the report says `dirty: false`. Judged against round 3's
conditions (section 8c), `acceptance-B` and `acceptance-C`. Data read only: base `gate34g` (snapshot
`snapshot-prod-20261025` with 0034 round 4 applied), after `gate35d`. Mutations ran only in
`scratchpad/r8d35` (a `git archive` copy); the runner path ran in `scratchpad/r8d35b` (a second
archive copy, with a synthetic verdict) against `scratchpad/r8d35_run` (a clonefile copy of
`gate34g`). Nothing in the repo was edited but this section. My code: `scratchpad/r8d35_*.py`
(tree, find, verd, score, s12, s3, s10, voices, stems, rcpt, ed, floorscan, mut, undo, digest, land).
Ids only. I read no voice this round (no seed): round 2's reading (0 wrong applied) is an input,
not re-done. `make ci-fast` was not run by me.

### Outcome first

Every integer below is from a command I ran. "Record" is the claim in sections 7 / 4d / 8b / 8c.

| Number | Record | Mine | Same? |
| --- | --- | --- | --- |
| list; renames / unnamings | 162; 138 / 24 | 162; 138 / 24 (list file) | yes |
| detector vs list, on `gate34g` | 162, 0 differences | `find_changes`: 162 found, only-detector 0, only-list 0, entries different 0 | yes |
| renames applied / not added | 92 / 46 | 92 (82 `CORRECT`, 10 `CORRECT_MISSPELLED`) / 46 (42 `undecidable`, 2 `another_person`, 1 `clip_other_person`, 1 `merged_voices`) | yes |
| unnamings applied / kept | 21 / 3 | 21 (`another_person`) / 3 (`CORRECT`, `undecidable`) | yes |
| voices whose label or role moved | 113 | 113 = the applied keys; 0 extra, 0 missing; 0 of the 49 left or kept moved; 0 segment texts changed | yes |
| episodes written | 99 | 99 = the episodes of the 113 applied keys | yes |
| files changed | 1,358 | 1,359 differ between `gate34g` and `gate35d` outside `.podcast_scraper`: 1,358 content files + `upgrade_ledger.json`; the one new file is the receipts log; 99 `.metadata.json`; per-suffix counts equal the gate report's table | yes |
| files outside the 99 episodes | 0 | 0 of 1,358 (each name checked against the 99 metadata and transcript stems) | yes |
| receipts | 1,359 | 1,359 = 1,358 + 1 identical bytes; `sha_before` equals the base file 1,359 of 1,359; `sha_after` equals the after file 1,359 of 1,359; receipt set equals the changed set once `/../` path segments are normalised | yes |
| S1 / S2 / S3 / S10a on the 99, before / after | +72 / -404 / -407 / -4 | committed `score_episode`: +72 (663 -> 735) / -404 (1,537 -> 1,133) / -407 (650 -> 243) / -4 (4 -> 0); S2-unidentified -476 | yes |
| same, with my own counting | - | S1 and S2 from the GI `surfaceable` / `speaker_voice_type` properties: 663 -> 735 and 1,537 -> 1,133; S3 from my own join (first `SUPPORTED_BY` quote, `char_start`, the pipeline's `_segment_char_spans`, share >= 0.30): 650 -> 243; S10a: episodes whose insights are all hidden 4 -> 0 | yes |
| per-episode deltas | 70 episodes | 70 episodes with a delta, keys equal, 0 differ from the gate report's `episode_deltas` | yes |
| must not move | A1 12, insights, quotes, S2 commercial | on the 99: A1 12 -> 12, insights 4,396 -> 4,396, quotes 8,695 -> 8,695, S2 commercial 1 -> 1 | yes |
| gate report vs the round-3 report | "every number equal" | `applied_before`, `apply_message`, `episode_deltas`, `files_changed`, `files_changed_by_suffix`, `gating_moves`, `plan`, `plan_sha256` (`334dadf49e94...`), `scoreboard_before`, `scoreboard_after`, `snapshot`, `verdicts_sha256`, `verify` equal; `clone`, `code`, `code_digest`, `measured_at` differ | yes |
| empty bridge files | 15 after, 0 written | `gate34g` 15, `gate35d` 15; 0 of the 1,358 changed bridges is empty | yes |
| verdicts file | `84ac945a...` | sha256 `84ac945a29ad...`, list header `cbf1a4ab...`; 45 `undecidable` rows, all with empty `segment_idx` and `line_sha256`; 0 rename `CORRECT` / `CORRECT_MISSPELLED` and 0 unnaming `WRONG_PERSON` on `undecidable` | yes |
| tests | - | `tests/unit/podcast_scraper/upgrade` in the copy of HEAD: 674 passed (52 s) | - |
| secrets, history | - | `gitleaks git --log-opts="origin/main..HEAD"`: 14 commits, no leaks; 0 merge commits; `origin/main` is an ancestor of HEAD | - |

Nothing the record states about the numbers is false. The list is shipped as 162 with every one of
its 162 voices at a served talk share of at least 0.0527 (my scan, below).

### Round-3 conditions, one by one

**(1) Tests that keep the digest.** Ten mutants, each run against the whole `upgrade` test
directory in `r8d35` (`r8d35_mut.log`): 8 killed, 2 survived.

| Mutant | Result | Caught by |
| --- | --- | --- |
| a1: rename check on `CORRECT` only (the `CORRECT_MISSPELLED` half dropped) | killed | `test_a_misspelling_rename_on_undecidable_is_refused` (new) |
| a2: rename check on `CORRECT_MISSPELLED` only | killed | `test_a_rename_on_undecidable_is_refused` |
| a3: rename check removed | killed | `test_a_rename_on_undecidable_is_refused` |
| b1: 0034's `verify` returns its own constant text for every subclass | killed | `test_verify_after_a_reviewed_apply_returns_0035s_own_message` (new, runs `verify()`) |
| b2: 0035's `_verified` set to 0034's text | killed | `test_the_version_and_the_verify_message_are_0035s_own` |
| e1 / e2: `to_version` 2.7.28 / 2.7.30 | killed | the same test |
| f1: the D1 fix reverted (`apply_to_payload` returns the same dict) | killed | `test_bridge_rewrite_empty_id_map.py` (3 of its 4 tests fail) |
| c1: `<` to `<=` at the talk floor in `episode_changes` | **survived** (674 passed) | nothing |
| d1: the no-groups early return dropped | survived, equivalent | - |

- **(1a) `CORRECT_MISSPELLED` on `undecidable`: closed.** a1 was round 3's surviving mutant; the new
  test kills it, and a2 and a3 are still killed.
- **(1d) What `verify()` returns: closed.** b1 was round 3's surviving mutant; the new test applies,
  calls `verify()` and reads the returned message.
- **(1c, `d1`) The no-groups early return: closed, the builder's argument is right.** I read
  `voice_identity.decide`: `if v not in vecs or talk < min_voice_talk or not groups: continue`, so
  with no groups `owner` stays empty and it returns `{}`; the loop in `episode_changes` then adds
  nothing.
- **(1c, `c1`) The floor `<=`: NOT closed as written.** The builder's note says a boundary test
  "pins it (0034's `test_a_voice_at_exactly_the_floor_keeps_its_name` covers the shared floor)".
  That test calls `m34.floor_changes`; 0035's comparison is its own line in `episode_changes`
  (`seconds.get(voice, 0.0) / total < m34.FLOOR`), which that test cannot reach, and the mutant
  survived all 674 tests. The harm is nil for what ships: the list is frozen and `episode_changes`
  only runs for `--find`; I computed the served talk share of all 162 listed voices (the served
  episode of each, segments `start` / `end`): smallest 0.0527, 0 under 0.05, 0 within 1e-6 of 0.05,
  so `<=` cannot change the list. The sentence is false; the survivor is harmless. A condition below
  asks for the test or the corrected sentence.

**(2) D1 in the same code batch and the other users of m0017's bridge rewrite: closed.**
`test_bridge_rewrite_empty_id_map.py`: with the fix reverted in `r8d35`, 3 failed (the deep-copy
test and both `test_m0017_bridge_rewrite_with_an_empty_id_map_keeps_the_identities` cases), 1
passed; with the fix, 4 passed. `grep` over `src` finds one importer of `apply_to_payload` outside
its module: m0017 line 65; m0020 reuses m0017's `_Episode`, and the new test asserts
`m20._Episode is m17._Episode` and `m20.plan_maps is m17.plan_maps`. That is an identity test, not
a behavioural one, and it holds by construction; it is acceptable as the "one written sentence".

**(3) The 15 against 16 and the ledger version: recorded; one pointer is false.** The builder's
notes state 15 and 15 and the cause (0034 round 3 turned two of the snapshot's 17 into non-empty
`{"identities": []}`); I measured 15 and 15 again on the round-4 base, 0 written. The ledger
`version` after an undo I reproduced (below). The note says an undo "resets `version` to 2.7.28
by hand (the runbook step)"; the 2.7 upgrade runbook (since removed) had 0 hits for `2.7.28`,
`0035`, `0034` or `undo`, so there is no such runbook step to point at.

**(4) Outcome, sha256, `make ci-fast`: not done, and cannot be before this verdict.** In
`r8d35b`, with my section ending in a synthetic verdict line, `build_outcome.py` wrote an outcome
(sha256 `447417caa2fe...`, synthetic, not the shipped one) whose `code_digest` is `7b458181...`.
`check_landing` on that tree reports for 0035 exactly one item: "step 8 does not name the outcome's
sha256"; on the unmodified tree it reports "no shipped outcome". The other two items in the landing
test are 0034's and 0036's NO-GO lines, not 0035's.

**(5) Grey cases and the prod sequence:** for the operator (items below).

### The runner path on a copy

`r8d35_run` is a clonefile copy of `gate34g` (ledger at 2.7.28, 0034 round 4 in it), run with the
code of `r8d35b` (the built outcome in place):

- `upgrade status`: pending 0035 and 0036. `upgrade run --dry-run --to 2.7.29`: "outcome hook: plan
  diff 0 episode(s)", would rewrite 99 episodes.
- `upgrade run --to 2.7.29 --yes --no-snapshot`: exit 0, "rewrote 99 episode(s) ...
  diagnostics_voices_renamed=113 ... roster_entries_unnamed=21 ... voices_named=61"; the hook's
  outcome record has `verdict: PASS`, `failures: []`, `plan_diff: []`, `code_digest` `7b458181...`;
  ledger gains `0035_voice_identity_names` at 2.7.29. The tree after the runner equals `gate35d`
  except `upgrade_ledger.json` (1 of 130,387 files).
- `m0035.undo`: restored 1,359, refused 0; the ledger entry is removed; `upgrade status` reads 0035
  pending again; the ledger `version` stays 2.7.29. The tree equals `gate34g` except the ledger
  and the receipts log (1 changed, 1 added).
- `upgrade verify` on the undone copy: **exit 1**, 32 migrations OK and one FAIL:
  `0025_one_person_one_entry: 1 episode(s) still list one person twice`, naming
  the metadata of episode id `998e8fc4-c6fc-11f0-b555-4bf8880724ca`.
  This is not 0035's: 0035 wrote 0 files of that episode; its metadata file has the same sha256
  (`afc208f875e8`) in the snapshot copy, `gate34g` and `gate35d`; and m0025's own per-episode
  `plan()` returns the same counters on all three. The undone copy equals `gate34g`, so the failure
  is in the base. Round 3 recorded exit 0 for this check on its base; I did not reconcile that.
  0034's verdicts list this episode, so it is reported to the caller for 0034's reviewer; it is not a 0035 defect.

### Other findings

- Receipts: 179 of 1,359 relpaths in 0035's receipts carry a `/../run_<id>/` segment (the same form
  in 0034's receipts: 197 lines). Undo restored all 1,359, so it works; the form is cosmetic.
- The 4d prediction remains missed and accepted in the record; nothing here changes that.
- Round 3 wrote "`1_2766140` SPEAKER_02" as the third thin-chain rename. The list has
  `1_2766234` SPEAKER_02 (a rename, `addressed`); `1_2766140` has SPEAKER_00 (the kept unnaming) and
  SPEAKER_01. The reviewer's id was a typo for `1_2766234`.

### For the operator at step 9 (each answerable yes or no)

1. Accept the missed restatement: S1 +72 against +150 to +184, S2 -404 against -105 to -93, S3 -407
   against -260 to -226, S10a -4 against -2 to +2, the episodes integer (99) never stated, 36
   renames added against 17 to 32 stated, and hold prod's step 11 to: list 162 (138 / 24); 113
   voices (92 renames, 21 unnamings); 46 not added, 3 kept; 99 episodes; 1,358 files; S1 +72, S2
   -404, S3 -407, S10a -4; new empty bridges 0?
2. Keep `1_2766140` SPEAKER_00 as it is (12 insights stay shown under a name the text may give to
   the other guest)?
3. Ship the 45 `undecidable` rows with an empty `segment_idx` and `line_sha256` (acceptance-B 32
   asked for 162 of 162; the loader allows empty only on `undecidable`)?
4. Accept the verdicts file as the key-level list of the 49 kept or left voices (46 renames not
   added, 3 unnamings kept), since the receipts and the plan name only the 113 applied?
5. Accept the three applied renames resting on a thinner chain than their stored line, from round
   2's reading (`iono.fm/e/1728203` SPEAKER_05, `urn:bbc:podcast:p0p2vv0y` SPEAKER_06,
   `1_2766234` SPEAKER_02)?
6. Accept the 13 episodes that end with one name on two voices, 3 of them not a solo-host split
   (`1_2766140`, `db4aff8e`, `iono.fm/e/1728203`; round 2's finding, not re-derived by me)?
7. Run in this order: 0034 first, its output for `iono.fm/e/1728203` (the one episode both write;
   13 files) checked equal to the clone's; then 0035 alone with `--to 2.7.29`, 0 differences
   allowed in the plan diff; undo 0035 before 0034, and reset the ledger `version` by hand after an
   undo; after the apply, `enrich --only insight_density,insight_sentiment` for the 99 episodes and
   an incremental `index` for their search rows?
8. Is `upgrade verify` failing at 0025 on `998e8fc4-c6fc-11f0-b555-4bf8880724ca` (in the snapshot,
   not written by 0035) something to fix before 0034 and 0035 run on prod, or to leave?

### Not covered / not verified (round 4)

- No voice was read this round; "0 wrong applied" rests on round 2's reading of 92 renames and 24
  unnamings and 30 of 46 left renames; 16 of the 46 left were never read by a reviewer.
- Mutation: 10 mutants this round (the 8 killed and 2 survivors above), not round 2's 57; the other
  47 were not re-run. The mutants outside `upgrade/` (the pipeline rung itself) were not run.
- `make ci-fast`: not run by me; no log in the record. The shipped outcome does not exist; its
  sha256 is unknown until after this verdict.
- A prod dry run (before and after 0034), prod's plan diff, the post-snapshot episodes, the
  search-index rows of the 99 and the `enrich` refresh: not evidenced on a clone.
- Whole-corpus S6 / S7 / S8 / S9 / S11 and gold: not measured by me; round 2 found 0 of 751 gold
  cases touch the 99 episodes. S10a, S1, S2, S3 were measured on the 99 only; the other 2,867
  episodes are byte-identical (no changed file outside the 99), so whole-corpus moves equal these.
- The detector reruns (`find_changes`) are the builder's code: they show reproducibility, not that
  the rule is right. My independent code covers the tree, the receipts chain, the scoreboard on the
  99 (S1 / S2 by property, S3 by my own join, S10a by property), the voices that moved and the
  verdict counts. S10a's `insights_from_gi` was replaced by a `surfaceable` rule and agreed (4 -> 0).
- Why `upgrade verify` read exit 0 in round 3 and exit 1 now: not reconciled.
- The 13 two-voices-one-name episodes and the thin-chain renames: round 2's findings, not re-derived.
- Reprocess durability; per-language rows; decision-trace names that remain in history.

Verdict: GO WITH CONDITIONS Conditions: (1) the record's sentence that 0034's floor test covers 0035's own `<` is false (mutant `<=` survived 674 tests): add a test of `episode_changes` at exactly 5.0% talk (tests only, the digest `7b458181...` does not move) or replace the sentence with the survivor and the fact that none of the 162 listed voices is under 0.05 or within 1e-6 of it (smallest 0.0527); (2) delete "(the runbook step)" from the ledger-version note or write that step into the prod handover for 0035, since the repo's runbook has none; (3) after (1) and (2), build the outcome with `build_outcome.py` on the committed tree, name its sha256 in this section, and attach a `make ci-fast` log with the landing test showing no 0035 item (0034's and 0036's NO-GO lines are theirs); a change under `src` needs one more re-measure with every integer unchanged, a tests-or-record change does not; (4) the operator answers the step-9 items 1 to 8 above, and prod runs 0034 first, checks its output for `https://iono.fm/e/1728203` equals the clone's, then 0035 alone with `--to 2.7.29`, with 0 differences allowed in the plan diff.

**Outcome (condition 3, 2026-10-26):** `scripts/measure/build_outcome.py --migration 0035_voice_identity_names` on the gate
report measured at c2c0dcad2 (clean tree; the final gate after the format / lint / type /
docstring fixes, every number and plan equal to the d94e4d13a run): `upgrade/outcomes/0035_voice_identity_names.json`, outcome decb3449bdfc6c6d3be9eef21a2662b100615454a2bd361d7616f8d8d7f17918.
`make ci-fast` on the commit holding it: below.

**Operator, 2026-10-26, on the 12 step-9 points:** "I'm okay with all 12 points and decisions we took. Let's keep moving forward." Round 4's step-9 items 1 to 8 accepted as listed, and the prod order (0034, check `iono.fm/e/1728203` against the clone, 0035 alone with `--to 2.7.29` and 0 plan differences, 0036; undo in reverse with the ledger version reset by hand; then the sidecar refresh and an incremental index for the written episodes); 0025's verify flags accepted as known (see the correction in section 7).

## 10. Before apply / deploy

In the deployment, on approval (2026-10-27), after
0034, on the deployed build. Dry run 09:27Z: `would rewrite 99 episode(s)`, 27 counters equal to the
clone's (e.g. segments_relabelled 56747, insights_reattributed 1590, voices_named 61);
**outcome hook: plan diff 0 episode(s)**.

## 11. After apply / deploy

- **Apply** 2026-10-27 09:32:42-09:43:01Z, `--to 2.7.29`, EXIT=0; receipt counters equal to the
  dry run and the clone (27 counters). Undo recorded: `m0035_voice_identity_names.undo`, then the
  ledger version reset to 2.7.28 by hand.
- **`upgrade verify`** 10:08Z: 0035 [OK]; FAILs: 0025 on the 5 known episodes (accepted, 8d) and
  0030 on 2 episodes whose host-read sponsor blocks 0035's renames exposed. 0030 was re-run on those
  2 (guarded dry run = 2 episodes, 8 counters; 14 files written), after which 0030 is [OK].
- **Ownership:** 5 root-owned paths (0035, 0036, the 0030 re-run) chowned to the corpus owner; OWNERSHIP
  PASS.
- **Derived surfaces** (once, after 0034, 0035, 0036 and the 0030 re-run): `enrich --only
  insight_density,insight_sentiment` 248 ok, 0 failed; incremental `index` scanned 3004, skipped
  2756, reindexed 248, EXIT=0. No rebuild.
- **Snapshots:** the `upgrade-backups/` snapshots were deleted on approval, after they were
  validated (2026-10-27). The undo above therefore no longer has its
  backup: forward-only from here.
- **Not verified:** the corpus-scope enrichers after the next nightly (open
  item 7, 2026-10-28); the search-index rows of the written episodes, read one by one; the app in
  a browser on a written episode.
