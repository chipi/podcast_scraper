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

## 5. Acceptance

Prediction read 2026-11-04 by a fresh agent (reviewer, Sonnet; not the builder, not the verdict
reader), 4 minutes, before any code. Verdict: **rows added** (4b, items 1 to 8; its item 9 is in
Not covered). Step-8 spot checks it added: entry 20's roster unchanged (not applied); entry 19 and
the ledger-joined entries read again after the refresh; the odd title spellings (entries 6, 8, 14,
15, 16) against the verdicts; nickname-to-formal entries (3, 7, 14) leave no short form on a voice;
entry 15's non-ASCII markers moved in every transcript; untouched voices in 2 or 3 listed episodes
unchanged; the 9 not reached byte-identical by sha256; files and episodes written against 4b.

## 6. Build

_TODO_

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
