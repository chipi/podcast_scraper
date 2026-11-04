---
id: 0044_title_spelling_voices_renamed
class: full
status: open
issue: "#2363 (B5's stored episodes; arc item L6)"
touches:
  - src/podcast_scraper/upgrade/migrations/m0044_title_spelling_voices_renamed.py
  - src/podcast_scraper/upgrade/lists/0044_title_spelling_voices_renamed.json
  - src/podcast_scraper/upgrade/verdicts/0044_title_spelling_voices_renamed.tsv
  - src/podcast_scraper/upgrade/registry.py
  - tests/unit/podcast_scraper/upgrade/test_m0044_title_spelling_voices_renamed.py
---

# 0044 — stored voices take the spelling their own episode title states (B5's stored episodes)

Method: [CORPUS_CHANGE_METHOD.md](../../guides/CORPUS_CHANGE_METHOD.md) §4, full class. Arc list
`docs/wip/INGESTION-ARC-2026-11-03.md`, item L6. A migration, so its apply is a deploy step that
waits for the operator's go, after 0043's. Ids in this record; names only as the frozen list (a
published list, the precedent of 0034–0043) carries them.

## 1. Intake

- **B5** (`fe231664`, 2026-10-15, #2363): the people an episode title names are spelling
  references, so new episodes put the title's spelling on the voice. Episodes stored before it
  keep the heard (ASR) spelling.
- **The replay** (eval-data `ingestion-arc-l6`, `data/eval/ingestion-arc/l6/`):
  `scripts/measure/roster_replay.py` over snapshot-prod-20261103, roster at `fe231664^` against
  `fe231664`, all 3,130 member episodes: 27 voices renamed, each the heard spelling moved to the
  title's spelling of the same person (read by the builder, 2026-11-04).
- **Why not a ledger rule** (the arc first filed L6 under tier L): of the 27, 22 are one-episode
  ids in no ledger group, and the title's spelling is an id nowhere; a join cannot rename them
  (scratchpad `l6_vs_pv.py`: today's mode and `publisher_verified` each join 3 of 27).
- **Served state on the snapshot, per voice** (scratchpad `l6_surfaces.py`, `l6_list.py`), matched
  the way 0029's rewrite matches (the voice's segments, `speaker` -> `speaker_label`):
  18 voices still carry the heard spelling; 4 already carry the title's (D2 or 0041); 3 are unnamed
  now (`SPEAKER_xx`); 1 carries a third spelling of the title's person; 1 carries the show's host's
  name although the replay reads the voice as the titled guest.
- **Machinery:** 0037 (`m0037_named_voices_corrected`): a frozen list of renames, each naming its
  deciding segment; 0035's reviewed verdicts with basis columns bound by the list's sha256; 0029's
  voice-by-voice rewrite with 0031's derived surfaces; a stored state that is neither the listed
  old nor the new name refuses the episode.

## 2. Outcome and baseline

- **Outcome:** each listed voice the review confirms carries its episode title's spelling of its
  person on every surface 0029 and 0031 rewrite (segments, transcript markers, metadata roster,
  GI and KG persons, bridge, diagnostics), as B5 gives a new episode.
- **Baseline:** snapshot-prod-20261103 with 0043 applied through `upgrade run --to 2.7.37` (the
  clone this gate runs on): the 20 listed voices carry their stored names.

## 3. Diagnosis by loss

One cause: the stored episodes predate B5. Ranked population, section 1. The 3 unnamed voices are
not a spelling loss: naming an unnamed voice is a different claim, not in this list.

## 4. Prediction

Dated: 2026-11-04

- **List:** 20 entries in 20 episodes (18 heard spellings, 1 third spelling, 1 host name on the
  titled guest's voice).
- **Applied:** the entries the blind review marks `CORRECT` or `CORRECT_MISSPELLED`; predicted 18
  to 20 (the two odd title spellings and the host-name case are the ones a reader may refuse).
- **Episodes written:** the applied entries' episodes, 1 each; no other episode.
- **Gating scoreboard:** no S1/S1c/S10 rise; S2 may move by the applied voices (a renamed voice
  stays named).
- **Must not move:** every episode not listed (gate's file list); 0029–0043's verify on the copy;
  the identity ledger's joins except the renamed ids (the ledger refresh after `upgrade run`
  respells to group spellings; predicted 0 files written for these one-episode ids).
- **Derived:** the incremental index re-embeds the written episodes (their segment text changes
  only in `Name:` markers); counted in step 7.

### 4b. Rows added by the acceptance read, with the blind verdicts (2026-11-04)

The verdicts (a fresh blind reader, Sonnet, 6 minutes, scratchpad packet `l6_packet.json`; ids,
basis and deciding segment only): **18 `CORRECT`, 2 `WRONG_PERSON`**. Entry 19 (episode
`a02c151a-dde6-4804-9827-a3df80b8352c`, stored a third spelling of the titled guest) is the show's
host's voice: the stored name is another person's, and the listed rename would not fix it. Entry 20
(episode `145e5a55-510d-449e-bfce-3304561dc680`, stored the host's name) introduces itself as the
host: the stored name is right, the replay's rename was wrong. Neither applies.

1. **Integers:** renames 18, unnamings 0, refused 0, episodes written 18, swap groups 0; files
   written 150 to 260 (0037 wrote about 12 per episode). No scoreboard total is predicted to move
   (0037's lesson); the gold's coverage of these 18 voices is not checked, so no S7 move is
   predicted.
2. **Population, 27 = 18 + 4 + 3 + 2:** reached 18 voices in 18 episodes; not reached: 4 already
   on the title's spelling, 3 unnamed (`SPEAKER_xx`), 2 refused by the review; episodes outside the
   replay's 3,130 members (stored after the snapshot) are not on the list. The 9 not reached are
   byte-identical after the apply.
3. **Must not move, per listed episode:** every other voice of the 18 episodes (name, role) moves
   0; `upgrade verify` 0 new failures; a second apply writes 0 files; the undo restores every file
   byte-identical with 0 refused.
4. **Role changes:** 0 (the one host-to-guest entry, 20, is not applied).
5. **Duplicates inside an episode:** entries whose new name already sits on another voice of the
   same episode: predicted 0; whose old name also sits on another voice: predicted 0. Measured on
   the clone before the gate.
6. **Ledger refresh after `upgrade run`:** of the 18, the ids in a ledger group (3 of the 27 were,
   `l6_vs_pv.py`) are respelled to their group's spelling; predicted: no applied voice goes back to
   its heard spelling, and the refresh writes files in 0 to 3 of the 18 episodes.
7. **Derived surfaces per applied voice:** segments, transcript markers (`.txt`, `.adfree.txt`,
   `.cleaned.txt`), roster, diagnostics, GI and KG persons, bridge identity, quotes and insights
   follow the voice; insight counts per voice unchanged. Index: the incremental index re-embeds the
   18 written episodes.
8. **Review:** verdicts bound to the list's sha256; a listed voice without a verdict fails
   `verify`; an entry without its deciding segment is refused at load (0037's rule); the verdicts'
   deciding segment and its sha256 carry the evidence, the list's evidence text is the same for each.

### 4c. Revision: 0045 is registered after 0044 (supersedes nothing above)

Dated: 2026-11-04T17:47

`upgrade/registry.py` gains one entry after 0044's (`0045_episode_voices_corrected`, record
`0045_episode_voices_corrected.md`); this migration's entry, list, verdicts and module do not
change. As in 0043's revision 4e: no measured number of this migration moves; its code digest
moves (the registry is in its import closure), so it is re-gated (plan sha256 equal to the last
gate's) and its outcome rebuilt; on prod it runs in its own run (`--to` its version) before 0045,
which a run without `--to` with both pending refuses.

## 5. Acceptance

Prediction read 2026-11-04 by a fresh agent (reviewer, Sonnet; not the builder, not the verdict
reader), 4 minutes, before any code. Verdict: **rows added** (4b, items 1 to 8; its item 9 is in
Not covered). Step-8 spot checks it added: entry 20's roster unchanged (not applied); entry 19 and
the ledger-joined entries read again after the refresh; the odd title spellings (entries 6, 8, 14,
15, 16) against the verdicts; nickname-to-formal entries (3, 7, 14) leave no short form on a voice;
entry 15's non-ASCII markers moved in every transcript; untouched voices in 2 or 3 listed episodes
unchanged; the 9 not reached byte-identical by sha256; files and episodes written against 4b.

### 5b. Acceptance of revision 4c

Dated: 2026-11-04T17:52

Fresh agent (reviewer, Sonnet), `minutes 3`: rows added, each taken:

1. Prod order: `--to 2.7.38` (0044), then `--to 2.7.39` (0045), each in its own run after its own go; a run without `--to` with more than
   one pending is refused ("runs alone", `cli_handlers` R1), not partly applied.
2. `--to` leaves the later migrations out of `pending`, so `check_static`'s first-pending rule
   sees the migration being applied.
3. `applied_before` is unchanged; the only check the edit moves is the code digest.
4. `upgrade/registry` is in this migration's import closure (scratchpad `c2_closure.py`); its list
   and verdicts sha256 are unchanged.
5. The re-gate's `verify_all` is reproduced, 0 regressions; an unapplied 0045 does not fail it;
   verify and undo read only this migration's own receipts and tag.
6. Registry-order tests: 0044's test asserts it follows 0043; 0045's test asserts it follows 0044.
7. 0045's gate and outcome are separate and later; its clone has 0043 and 0044 applied and
   recorded.
8. The gate script applies the migration directly, so `--to` does not apply to the re-gate.
9. The re-gate reproduces plan sha256 `9ae9d480…` and the measured rows exactly.

## 6. Build

- `upgrade/migrations/m0044_title_spelling_voices_renamed.py`: 0037's `NamedVoicesCorrectedMigration`
  with its own id (to 2.7.38), list, verdicts, receipts (`title_spelling_voices_renamed.jsonl`)
  and backup tag (`0044`); `load_list` holds 0037's rule (renames only, a deciding segment).
- `upgrade/lists/0044_…json` (20 entries), `upgrade/verdicts/0044_…tsv` (the blind reader's 20
  rows: verdict, basis, segment index and the segment text's sha256, no text; bound to the list's
  sha256), `upgrade/registry.py` (after 0043; 0043's revision 4e and acceptance 5d).
- Tests (`test_m0044_title_spelling_voices_renamed.py`, 6, fictional names, written before the
  module was placed): a heard spelling takes the title's on segments, roster, GI and transcript,
  and the undo puts it back; the host's name on the titled guest's voice takes the guest and the
  role; a `WRONG_PERSON` verdict writes nothing; a stored name that moved on is refused untouched;
  the shipped list (20 in 20 episodes); registered after 0043, own backups and receipts, verdicts
  load against the list. Commit `5d37558e`.
- `MISSED:` the tests ran against the drafted module before acceptance in the scratchpad, not
  first against a failing build.

## 7. Measure

Clone: snapshot-prod-20261103 with 0043 applied through `upgrade run --to 2.7.37` (hook receipt
PASS; recorded in record 0043, step 8). Gate: `corpus_change_gate.py` on `5d37558e`, dirty false
(`0044_title_spelling_voices_renamed.gate.json`). Checks on the gate copy: scratchpad
`m44_checks.py`, `m44_marker2.py`, `m44_cleaned.py`, `l6_dups.py`.

| Row (4 / 4b) | Predicted | Measured |
| --- | --- | --- |
| renames applied | 18 | **18** (`diagnostics_voices_renamed` 18; targets after: 0 episodes) |
| episodes written | 18 | **18** |
| files written | 150 to 260 | **206** |
| refused, unnamings, voice role changes | 0, 0, 0 | **0, 0, 0** (segment rows whose `speaker_role` changed in the 18 episodes: 0, `m44_roles.py`). `kg_roles_set` 1: the KG node `person:adam-mosseri` goes from `mentioned` to `guest`, since the renamed voice is that person |
| gating scoreboard | no total moves | **no gating number moved** |
| `upgrade verify`, all applied | 0 new failures | **43 ok, 0 failing, 0 regressions** |
| the 20 listed voices after | 18 on the title's spelling, 2 as stored | **20 of 20** |
| other voices of the listed episodes | 0 moved | **0** |
| the 2 refused entries' episodes | byte-identical | **0 files moved** |
| new / old name on another voice of the episode | 0 / 0 | **0 / 0** |
| second apply | 0 files | **0 episodes, 0 files** |
| undo | byte-identical, 0 refused | **209 restored, 0 refused, 209 of 209 equal to the clone** |
| ledger refresh after the apply | 0 to 3 episodes respelled, none back to the heard spelling | **3 episodes (9 files), 0 refused; 0 applied voices back on the heard spelling**; joins 649 -> 652: `person:andrea-tosoto`, `person:jacob-liberman` and `person:nikhyl-singhal` join their groups |
| old-name `Name:` markers left in the 18 episodes' transcripts | 0 | **2 lines in 1 file**: `6410d1c5-6b80-4908-a7b3-ab62b46c7576` `.cleaned.txt` (below) |

- **The 2 lines:** the stored cleaned transcript of `6410d1c5` already labels lines of another
  voice with this guest's heard name (by the segments, lines 256 and 276 are another voice's
  text). 0029's cleaned-line rewrite relabels only the lines whose text is this voice's, so those
  2 keep the old label. A misattribution in that file before 0044, left as it was; 45 of the
  voice's own lines carry the title's spelling.
- **Spot checks:** entries 19 and 20 (refused) unchanged; the nickname-to-formal entries carry no
  short form on any voice (the new/old-on-another-voice counts above); entry 15's non-ASCII
  heard spelling has no marker left in `.txt`, `.adfree.txt` or `.cleaned.txt` (`m44_marker2.py`
  lists only `6410d1c5`).
- Not measured here: the incremental index's re-embed count (predicted 18), the enrichers of the
  written episodes, and the gold's coverage of the 18 voices.

**Builder: GO.** Deciding numbers: 18 renames in 18 episodes and nothing else, every listed voice as
the review decided, undo whole, no gating number moved.

## 8. Verdict

Verdict read 2026-11-04 by a fresh agent (reviewer, Sonnet; not the builder, not the step-5 reader,
not the verdict reader), inputs as file paths, ran only the unit tests.

Review budget: minutes 5, tokens 60000

```text
REVIEW 0044_title_spelling_voices_renamed 2026-11-04 commit 3d6f6119 reviewer Sonnet (fresh) minutes 5
PREDICTION vs GATE: rows equal (18, 18, 206 in 150-260, 0/0/0, verify 0 new failures, 20 of 20, other voices 0, undo 209/209, second apply 0, refresh 3 in 0-3); outside the rows: 2 cleaned-transcript lines in 6410d1c5 (named), the KG node role change (not predicted)
REQUIRED ITEMS: verdicts bound to the list's canonical sha256; 20 rows for 20 entries; the 2 WRONG_PERSON not applied (0 files moved, test); spot checks done except the 7 unreached by sha256 and the odd spellings beyond the by-elimination note; verdicts header reviewed_commit is not the gate commit (list unchanged since)
AFTER-STATE READ: counts only; no rendered excerpt; index re-embed, enrichers, gold coverage not shown
OPEN CALLS: what voices_named 14, entities_dropped 6, kg_persons_removed 11 count; kg_roles_set 1; the 2 cleaned lines; entry 19's wrong-person voice; by-elimination verdicts
DOUBTED NUMBER: voices_named=14
VERDICT: GO WITH CONDITIONS (1) state what voices_named, entities_dropped and kg_persons_removed count; (2) the 7 unreached voices' episodes byte-identical, or named as not checked; (3) the verdicts header's reviewed_commit stated or updated; (4) the index re-embed count measured
```

Conditions, each answered (2026-11-04):

1. **The counters** (0025's rewrite, `m0025_one_person_one_entry.py`): `voices_named` counts a
   listed voice whose new name had no roster entry carrying that voice, so the entry is attached
   (the unplaced entry for that person takes the voice, or one is added): 14 of the 18;
   `unplaced_dropped` 2 is that unplaced entry dropped where the placed voice already holds the
   person. `kg_persons_removed` / `kg_persons_added` 11 / 11 are the heard-spelling speaking
   Person nodes replaced by the title's; `entities_dropped` / `entities_added` 6 / 23 are the
   roster's transcript-provenance entity aliases recomputed from the new names. Each counts the
   listed voices' own surfaces: `voice_names` holds only listed voices.
2. **The 7 unreached** (4 on the title's spelling, 3 unnamed), on the gate copy after the apply
   and the ledger refresh against the clone (scratchpad `m44_cond.py`): **0 of 7 episodes with a
   differing file**.
3. **`reviewed_commit` `f766a0b63`** is the branch head when the reader worked; the list and the
   verdicts were first committed in `5d37558e` and have not changed since; the binding is the
   list's sha256 in the header.
4. **Index:** the indexer's own fingerprint (`m44_cond.py`, as 0043's `m43_fp.py`): **18 of 18
   written episodes change it**, so prod's incremental `index` re-embeds 18; rows 11 removed, 11
   added (the KG Person rows); 0 episodes with moved metadata and an unchanged fingerprint.

Verdict: GO WITH CONDITIONS

- **Condition 1:** applied on prod only after 0043, in its own run (`--to 2.7.38`), on the
  operator's go, with prod's plan diffed against this gate's.
- **Condition 2:** after the apply, the incremental index re-embeds the written episodes (18 on the
  clone) and the ledger refresh respells at most the 3 measured; prod's counts recorded in step 11.
- **Condition 3:** the wrong-person voice the review found (episode
  `a02c151a-dde6-4804-9827-a3df80b8352c`) is not fixed by this migration; it needs its own list.

**Measured after the verdict** (2026-11-04, a copy of clone-0043, `upgrade run --to 2.7.38 --yes
--no-snapshot` on `a35cbd04`):

- **Prod's apply path:** the outcome hook applied 0044, receipt verdict PASS, 0 failures, 0 plan
  differences, `S1c_inconsistent` 0; 18 episodes with the gate's counters; "Now at version 2.7.38;
  0 pending"; the ledger refresh after it: 652 joins, 3 added, 3 episodes (9 files) respelled, as
  on the gate copy.
- **Gold:** `gold_insights.py` on clone-0043 and on this copy, totals only: dev_v1 S7_correct 2,440,
  S7_wrong 392, S7_unnamed 1,313, S8_undecidable 77 on both; val_v7 2,334, 299, 799, 5 on both.
  No case's served name moves: the gold covers none of the 18 voices' insights.
- **Enrichers:** `enrich --only insight_density,insight_sentiment` (no `--force`): 123 runs ok, 0
  failed each, EXIT=0; 123 is what the staleness check picked after 0043 and 0044 together, not
  broken down by migration.

Outcome: `src/podcast_scraper/upgrade/outcomes/0044_title_spelling_voices_renamed.json`, sha256
`816bc81ec1d21936b5ee155750ec8d20c3a5394026f8a02b693c7aaff64bb857` (gate on `5d37558e`, dirty
false, plan sha256 `9ae9d480…`).

## 9. Operator review

_TODO_

## 10. Before apply / deploy

_TODO_

## 11. After apply / deploy

_TODO_

## 12. Retrospective

_TODO_

## Not covered

- **Found by the review, not fixed here:** episode `a02c151a-dde6-4804-9827-a3df80b8352c`'s voice
  stored with a third spelling of the titled guest is the show's host's voice: a wrong-person name
  on a served voice. A rename to the host is a 0037-class correction with its own list and review.
- Episode `6410d1c5-6b80-4908-a7b3-ab62b46c7576`: 2 cleaned-transcript lines of another voice
  labelled with this guest's heard spelling (section 7).
- The 3 unnamed voices the replay names (naming an unnamed voice is a different claim) and the 4
  already on the title's spelling.
- Only the speaker name moves: the heard spelling inside spoken text, insight and quote text,
  summaries and KG mentions stays; the episode title is untouched.
- Other episodes of the same people stored with the heard spelling that the replay did not reach,
  and episodes stored after the snapshot.
- The search-index rows are refreshed by the incremental index after the apply, not by the
  migration; the enrichers of the written episodes run after it.
- The verdict reader's evidence for 3 entries (6, 12, 14) is by elimination: the only other voices
  are the host and a co-guest named elsewhere.
