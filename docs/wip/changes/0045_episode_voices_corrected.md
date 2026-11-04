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

_TODO_

## 8. Verdict

_TODO_

## 9. Operator review

_TODO_

## 10. Before apply / deploy

_TODO_

## 11. After apply / deploy

_TODO_

## 12. Retrospective

_TODO_

## Not covered

_TODO_
