---
id: 0045_episode_voices_corrected
class: full
status: open
issue: "none (found by 0044's blind review; ingestion arc)"
touches:
  - src/podcast_scraper/upgrade/migrations/m0045_episode_voices_corrected.py
  - src/podcast_scraper/upgrade/lists/0045_episode_voices_corrected.json
  - src/podcast_scraper/upgrade/verdicts/0045_episode_voices_corrected.tsv
  - src/podcast_scraper/upgrade/registry.py
  - tests/unit/podcast_scraper/upgrade/test_m0045_episode_voices_corrected.py
---

# 0045 — one episode's two guest-named voices: the host's voice and the guest's spelling

Method: [CORPUS_CHANGE_METHOD.md](../../guides/CORPUS_CHANGE_METHOD.md) §4, full class. Found by
0044's blind review (record `0044_title_spelling_voices_renamed`, section 4b, entry 19). A
migration: its apply is a deploy step after 0044's, on the operator's go. Ids in this record;
names only in the frozen list (published, the precedent of 0034–0044).

## 1. Intake

- **Episode `a02c151a-dde6-4804-9827-a3df80b8352c`** (feed `rss_feeds.simplecast.com_2e104dc7`,
  run `e1b90677`), on snapshot-prod-20261103 with 0043 and 0044 applied (`clone-0044`): the
  title names one guest. Voices: SPEAKER_00 the first host (19.4% of talk), SPEAKER_01 a heard
  spelling of the titled guest (32.2%), SPEAKER_02 a second heard spelling of the same guest
  (45.0%), five unnamed voices (3.6% together). The roster lists the first host and both
  spellings as guests; the show's second host is on no voice.
- **0044's blind reader** judged SPEAKER_02 (its entry 19) to be the show's second host, by its
  segment 23 and the voice addressing the first host. 0044 does not apply that entry (a rename to
  the title's spelling would have been wrong); its Not covered names this episode.
- **SPEAKER_01** was not on 0044's list: the replay's rename for this episode pointed at
  SPEAKER_02.
- **Machinery:** 0037's (renames with a deciding segment, blind verdicts with basis columns bound
  by the list's sha256, swaps applied whole, 0029's rewrite with 0031's derived surfaces), as 0044.

## 2. Outcome and baseline

- **Outcome:** SPEAKER_02 carries the show's second host's name and the host role; SPEAKER_01
  carries the title's spelling of the guest; one guest entry on the roster, two hosts.
  **As delivered** (after the verdicts, 4b): SPEAKER_01's rename was refused, so SPEAKER_01 keeps
  its stored name; only SPEAKER_02's half of this outcome is met.
- **Baseline:** clone-0044 (snapshot-prod-20261103, 0043 and 0044 applied through `upgrade run`).

## 3. Diagnosis by loss

One episode. Cause not traced: the episode predates B5 (title spellings), and the second host's
voice took a guest name at naming time (two spellings of one guest on two voices).

## 4. Prediction

Dated: 2026-11-04T17:50

- **Applied:** the entries a fresh blind reader marks `CORRECT` / `CORRECT_MISSPELLED`; predicted 2.
- **Episodes written:** 1; **files written** 8 to 16; refused 0; unnamings 0; swap groups 0.
- **Roles:** SPEAKER_02 guest -> host (1 role change); SPEAKER_01 stays guest.
- **Roster after:** the first host, the second host (on SPEAKER_02), the guest (on SPEAKER_01);
  no entry for either heard spelling.
- **Must not move:** every other episode; SPEAKER_00 and the unnamed voices; `upgrade verify` 0 new
  failures; a second apply writes 0; the undo restores byte-identical with 0 refused; no gating
  scoreboard total moves (S10/S1 unchanged; the episode keeps its named voices).
- **Ledger refresh after `upgrade run`:** the second host's id joins its existing group; files
  respelled: 0 or 1 episode.
- **Derived:** the incremental index re-embeds 1 episode.
- **Duplicates:** the new host name is on no other voice of the episode (checked on clone-0044).

### 4b. Rows added by the acceptance read, with the blind verdicts (2026-11-04)

The verdicts (a fresh blind reader, Sonnet, 3 minutes, scratchpad packet `m45_packet.json`):
**SPEAKER_02 `CORRECT`** (basis `self_intro`, segment 23); **SPEAKER_01 `WRONG_PERSON`** (basis
`another_person`, segment 14: the voice's segments the reader saw are host talk). The packet held
the first 20 of SPEAKER_01's 109 segments, so the guest's later segments were not in it; the
verdict stands as given and SPEAKER_01 keeps its stored name (Not covered).

1. **Integers:** renames applied 1 (with a role change), unnamings 0, refused 0, swap groups 0;
   the two entries are independent (different old names, neither new name equals an old one), so
   a verdict on one does not gate the other; verdicts bound to the list's sha256; a listed voice
   without a verdict fails `verify`. Episodes written 1; files 6 to 14.
2. **Roles by segment:** SPEAKER_02's 271 segment rows go guest -> host; SPEAKER_00, SPEAKER_01 and
   the 5 unnamed voices move 0 in name and role.
3. **Roster after:** the first host unchanged; the second host on SPEAKER_02; SPEAKER_01's entry
   unchanged; the SPEAKER_02 heard-spelling guest entry gone; no host entry duplicated. Whether the
   second host is attached to an existing roster entry or added (`voices_named` /
   `unplaced_dropped`) is measured at step 7.
4. **KG and GI:** the heard-spelling speaking Person node of SPEAKER_02 replaced by the second host's
   (about 1 removed, 1 added); the second host's node role host; the same host's nodes in other
   episodes unchanged; `kg_roles_set` 0 or 1, measured.
5. **Served and scored:** SPEAKER_02's insights and quotes move from a guest's attribution to the
   host's; per-voice insight count unchanged; the gold's coverage of this episode is read at step 7
   by the scorer's totals (no move predicted).
6. **Duplicates:** the second host's name on no other voice of the episode; neither heard spelling
   left on SPEAKER_02.
7. **Must not move:** every other episode, including 0044's 7 unreached and 2 refused; no
   `Name:` marker of SPEAKER_02's old spelling left in `.txt`, `.adfree.txt`, `.cleaned.txt`.
8. **Ledger refresh:** neither voice goes back to a heard spelling; the second host's id is in its
   existing group (joins unchanged or +1); files respelled in 0 or 1 episode.
9. **Derived:** the written episode's index fingerprint changes (1 of 1), 0 episodes with moved
   metadata and an unchanged fingerprint; the enrichers re-run it.
10. **Machinery:** registered after 0044, to 2.7.39, own backup tag `0045`, receipts
    `episode_voices_corrected.jsonl`; applied after 0044 in its own run; undo byte-identical with 0
    refused; a second apply writes 0; `upgrade verify` 0 new failures; 0037's rename takes the
    listed new role (unit test, 0044's).

## 5. Acceptance

Prediction read 2026-11-04 by a fresh agent (reviewer, Sonnet; not the builder, not the verdict
reader), 4 minutes, before any code. Verdict: **rows added** (4b). Spot checks for step 8: the
roster (2 hosts, the guest entry of SPEAKER_01 as stored, no duplicate); segment roles; KG nodes and
roles; markers in the three transcripts; both entries against their verdicts; insights and quotes
following the voice; after the ledger refresh; 0044's 9 untouched episodes byte-identical; index
fingerprint; undo, second apply, a gate on a clean commit.

## 6. Build

- `upgrade/migrations/m0045_episode_voices_corrected.py`: 0037's migration with its own id (to
  2.7.39), list, verdicts, receipts (`episode_voices_corrected.jsonl`) and backup tag (`0045`), as
  0044. The list (2 entries), the verdicts (2 rows bound to the list's sha256, deciding segment and
  its text's sha256), the registry entry after 0044 (0043's revision 4f / acceptance 5e, 0044's
  4c / 5b).
- Tests (`test_m0045_episode_voices_corrected.py`, 7, fictional names): 0044's scenario tests on
  0045's class (a heard spelling renamed on every surface and undone; a refused rename writes
  nothing; a moved name refused), plus the 0045 shape (a guest's heard name on the second host's
  voice takes the host's name and role on the segments and the roster), the shipped list (2 in 1
  episode), the shipped review (the host rename `CORRECT`, the other `WRONG_PERSON`), registered
  after 0044 with its own backups and receipts.
- `MISSED:` the module and tests were drafted from 0044's in the scratchpad before acceptance; the
  tests ran against the draft, not first against a failing build.

## 7. Measure

Clone: `clone-0044` (snapshot-prod-20261103, 0043 and 0044 applied through `upgrade run`, hook
receipts PASS). Gate: `corpus_change_gate.py` on `3a4adaf47` (the branch rebased onto PR B),
dirty false (`0045_episode_voices_corrected.gate.json`). Checks: scratchpad `m45_checks.py` (gate
copy against the clone), `gold_insights.py` totals.

| Row (4 / 4b) | Predicted | Measured |
| --- | --- | --- |
| renames applied | 1 | **1** (targets: 1 episode before, 0 after) |
| episodes / files written | 1 / 6 to 14 | **1 / 11** |
| gating scoreboard | no total moves | **no gating number moved** |
| `upgrade verify`, all applied | 0 new failures | **44 ok, 0 failing** |
| files outside the episode | 0 | **0** (23 changed: 11 in the episode, 12 the gate's bookkeeping) |
| segment rows | SPEAKER_02's 271 guest -> host; others 0 | **271 SPEAKER_02 rows guest -> host with the new label; 0 other rows** |
| roster | 2 hosts, SPEAKER_01's entry as stored, the heard entry gone, no duplicate | **2 hosts, 1 guest; the heard name gone; 1 host-name entry; SPEAKER_01's stored name kept** |
| KG persons | about 1 removed, 1 added; the host's node role host | **miss: 0 removed, 0 added**; the host's node `person:kevin-roose` existed as `mentioned` and goes to `host` |
| old-name markers in `.txt`, `.adfree.txt`, `.cleaned.txt` | 0 | **0, 0, 0** (37, 37, 34 lines carry the host's name) |
| index fingerprint | changes (1 of 1) | **changes** |
| second apply / undo / re-apply | 0 / byte-identical, 0 refused / 1 | **0 / 11 of 11 equal, 0 refused / 1** |
| ledger refresh | no voice back on a heard spelling; 0 or 1 episode respelled | **0 files written, joins 652 unchanged; SPEAKER_02 keeps the host's name** |
| gold totals | no move | **dev_v1 2,440 / 392 / 1,313 / 77 and val_v7 2,334 / 299 / 799 / 5, as before** |

- **The KG miss:** the episode's KG held no speaking node under the heard spelling, and the host
  was already a `mentioned` node; the rename moves that node's role. Same shape as 0044's
  `person:adam-mosseri`.

**Builder: GO.** Deciding numbers: 1 voice renamed in 1 episode, nothing outside it, undo whole.

## 8. Verdict

Verdict read 2026-11-04 by a fresh agent (reviewer, Sonnet; not the builder, not the step-5 reader,
not the verdict reader), inputs as file paths, ran only the unit tests.

Review budget: minutes 5, tokens 60000

```text
REVIEW 0045_episode_voices_corrected 2026-11-04 commit 43891f75 (gated code 3a4adaf47 = HEAD's src/tests) reviewer Sonnet (fresh) minutes 5
PREDICTION vs GATE: rows equal except the named KG miss; unpredicted counts bridge_identities_removed 1, entities_added 1, context_rebuilt 1
REQUIRED ITEMS: verdicts bound to the list's sha256 (recomputed); 2 rows; WRONG_PERSON not applied (plan holds SPEAKER_02 only); 7 tests pass; module is 0044's with id, version and labels changed; gate on a clean commit; spot checks in section 7; per-voice insight count not shown
AFTER-STATE READ: the gate JSON and section 7's figures; roster, rows, markers and undo are the builder's scratchpad checks, not re-run
OPEN CALLS: section 2's outcome names SPEAKER_01, which is refused; SPEAKER_01 left open; the deciding index 23 against a 20-segment packet (a global index); one reader on SPEAKER_02, corroborated by 0044's reader
DOUBTED NUMBER: the ledger refresh's 0 files / 652 joins, not in the gate JSON
VERDICT: GO WITH CONDITIONS (1) section 2 and 4 say SPEAKER_01 keeps its stored name; (2) the unpredicted counts explained, per-voice insight count checked or in Not covered; (3) SPEAKER_01 tracked as follow-up work
```

Conditions, each answered (2026-11-04):

1. Section 2 carries "as delivered: SPEAKER_01 keeps its stored name"; 4b row 3 already said it.
2. The counts: `bridge_identities_removed` 1 is the heard-spelling identity of SPEAKER_02 in the
   episode's bridge; `entities_added` 1 is the roster's transcript-provenance alias for the host's
   name; `context_rebuilt` 1 is the episode's context file rebuilt from the new roster (0025's
   rewrite, as 0044's counters in its step 8). Per-voice insight counts were not checked (Not
   covered); `insights_reattributed` 9 and `quotes_reattributed` 16 follow the voice to the host.
3. Tracked in the arc list (`INGESTION-ARC-2026-11-03.md`, "Mine, scheduled"): SPEAKER_01, a
   whole-voice reading as its own reviewed entry. The deciding segment index is the segment's
   index in the episode's segments, as every verdicts file's (`segment_idx`).

Verdict: GO WITH CONDITIONS

- **Condition 1:** applied on prod only after 0044, in its own run (`--to 2.7.39`), on the
  operator's go, with prod's plan diffed against this gate's.
- **Condition 2:** SPEAKER_01 of the episode stays as stored until its own reviewed entry.

Outcome: `src/podcast_scraper/upgrade/outcomes/0045_episode_voices_corrected.json`, sha256
`a840b1862383606971d68b9375c9a9e6d32f8242959ef56b4a7919a028433c50` (gate on `3a4adaf47`, dirty
false, plan sha256 `3500641d…`).

## 9. Operator review

_TODO_

## 10. Before apply / deploy

_TODO_

## 11. After apply / deploy

_TODO_

## 12. Retrospective

_TODO_

## Not covered

- **SPEAKER_01 is still open.** The reader refused its rename from the voice's first 20 of 109
  segments (host talk); the packet did not show its later segments, so whether SPEAKER_01 is the
  guest, the first host's split voice or a merge is not decided. It keeps its stored guest name.
  Settling it needs a reading of the whole voice, as its own reviewed entry.
- Per-voice insight counts before and after were not checked.
- The first host's voice (SPEAKER_00) and the 5 unnamed voices are untouched; whether the
  diarization split a host across voices is not traced.
- Only the speaker name and role move: the heard spelling inside spoken text, insight and quote
  text stays; the episode title is untouched.
- The search-index rows are refreshed by the incremental index after the apply; the enrichers
  re-run the written episode after it.
