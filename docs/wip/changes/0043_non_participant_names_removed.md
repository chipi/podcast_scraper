---
id: 0043_non_participant_names_removed
class: full
status: open
issue: none (ingestion arc 2026-11-03, items B6 residual and B8 residual; #2364, #2123 shipped)
touches:
  - src/podcast_scraper/upgrade/migrations/m0043_non_participant_names_removed.py
  - src/podcast_scraper/upgrade/lists/0043_non_participant_names_removed.json
  - src/podcast_scraper/upgrade/verdicts/0043_non_participant_names_removed.tsv
  - tests/unit/podcast_scraper/upgrade/test_m0043_non_participant_names_removed.py
  - src/podcast_scraper/upgrade/registry.py
---

# 0043 — a name on a voice that only reads an ad or a supporter shout-out comes off (text only)

Method: [CORPUS_CHANGE_METHOD.md](../../guides/CORPUS_CHANGE_METHOD.md) §4. Arc list:
`docs/wip/INGESTION-ARC-2026-11-03.md`, B6 / B8 residuals. Names are kept out of this record; the
frozen list (package data, as 0034's and 0037's) carries the published names.

## 1. Intake

- **#2123 (B8, shipped):** ad readers named as people. The pipeline's ad signatures
  (`AdSignatures.is_ad_voice` in `roster.classify_voices`) type such a voice `commercial`, no stated
  name exempting it. m0016 removed what was on disk then, but m0016 skips a name any source STATED,
  and one of its sources is the metadata's `content.detected_guests`. In artifacts before schema
  1.2.0 (#2075, `workflow/metadata_generation.py` schema note) that field is a projection of the
  roster by role, so the old roster's own guest seat for a self-introduced ad reader counted as a
  statement of it. All 33 class-A roster entries are pre-1.2.0 (no `placed` key), role `guest`.
- **#2364 (B6, shipped fc4e0dbd):** supporter shout-outs and house promos published as guests. The
  roster's `_self_introduced_promo_voices` now types them `commercial`. Stored episodes keep the
  old roster.
- **0034 (landed): every voice here was on its list and was KEPT.** 0034's reviewers judged
  whether the name is the voice's own person, and the operator's third ruling (0034 §4c,
  2026-10-26) keeps a sponsor-tag / house-promo voice named when "the text ties it to the named
  person". A self-introduction inside the promo is such a tie: all 59 voices below (39 at intake,
  20 added at step 5) carry `CORRECT` / basis `self_intro` in
  `upgrade/verdicts/0034_talk_floor_names_removed.tsv`.
- **The conflict this change must not paper over:** today's pipeline publishes these 59 voices
  `commercial` with no name (section 3); 0034's ruling keeps them named on stored episodes. Both
  cannot stand. 0043 sides with the pipeline. That reverses 59 rows of a recorded operator ruling,
  so the operator decides it before the build (section 9, open calls 1 and 2).
- **m0016** (`upgrade/migrations/m0016_ad_reader_names_removed.py`): the commercial typing
  (`_VoiceEpisode._commercial_after_whole`) this change reuses for its type.
- **0034 / 0037 machinery** (frozen reviewed list, verdicts bound by list sha256, 0029's
  voice-by-voice rewrite with 0031's derived surfaces): the rewrite this change reuses.
- **Readers of these surfaces:** the player's roster and transcript speaker labels, KG Person nodes
  (guest), the bridge identity rows, the search index's transcript chunks (speaker prefixes).

## 2. Outcome and baseline

- **Outcome:** no stored episode serves a name on a voice that today's roster code, run on the
  episode's own stored inputs with the corpus's ad signatures, types `commercial` and leaves
  unnamed, in the three classes below.
- **Baseline, snapshot-prod-20261103** (`scripts/measure/corpus_scoreboard.py --corpus
  <snap1103> --label snapshot-prod-20261103`, 3,130 episodes, 138,469 insights): S1 18,589;
  S1n 7; S1c 0; S2 32,488 (unknown 18,589 / unidentified 13,804 / commercial 95); S3 9,373;
  S10a 197; S10b 2; S12 6 (244 outside: transcript file not found); A1 400 hits in 301 episodes.
- **The change's own number:** served names on such voices: **59 voices in 52 episodes** (39 in 39 before step 5 added class C).

## 3. Diagnosis by loss

Three classes, each found by a committed or stated rule over the snapshot, then every voice read.

- **Class A, 33 voices (m0016's stated-name exemption).** `scripts/measure/m0043_candidates.py
  --corpus <snap1103>` (after `ad_signatures.write_for_corpus`, 32 s) takes m0016's `find_hits`
  excluded hits (36) and keeps those stated ONLY by `content.detected_guests` and absent from the
  publisher's text (feed and episode titles and descriptions): 33 rule_a, 3 kept (2 named by the
  publisher's text, 1 a stated host). Four readers: R1 22 voices, R2 7, R3 2, R4 2; two feeds.
  Every voice's text read: each is a house ad or a subscription promo read by the named person.
  Talk share 0.002 to 0.038; signatures' recurring fraction 0.30 to 1.00.
- **Class B, 6 voices (#2364's rule on old rosters).** The B6 replay (`roster_replay.py` old vs
  new over the stored inputs, 12 voices lost) resolved on the snapshot: 7 still served (roster
  entry `self_intro` guest, segment labels), 5 already gone. Of the 7, 6 are supporter shout-outs
  (3) or another show's / the publisher's subscription promo (3); talk share 0.002 to 0.040. The
  7th is the show's own host reading the show's premium promo: not a target (the host is a
  participant; the name is right and the role error is another item).
- **Today's code agrees on all 39 of classes A and B** (class C below: all 20). Replaying `resolve_speaker_roster` (current tree) on each
  episode's stored inputs with signatures built in memory: `{A unnamed commercial: 33, B unnamed
  commercial: 6}`; 0 named. So new episodes are already right; only stored ones carry the names.
- **What the names reach today (per voice, the snapshot):** other voices carrying the same name in
  the episode 0; GI insights 0, GI quotes 0, GI Person nodes 0 (the ad voices produced no insight);
  KG Person nodes 11 (one per episode, in 11 of the 39 episodes; KG is per episode, nothing
  shared); `content.detected_guests` entries 33 (class A); roster entries 39; segment rows of the
  39 voices 446 (A 295, B 151), ad-free segment rows 446; transcript prefixes on all 39.
- **Class C, 20 voices in 13 episodes (added at step 5, 2026-11-04).** m0016's own RECURRENCE hits
  on the snapshot's rebuilt signatures (`find_hits`: 20; language-only 0). m0016's hit set was
  frozen at its apply, and these voices arrived or turned recurrent since. Each is another show's
  host or writer promoting their own show inside this one: five readers C1 to C5 (7, 4, 3, 3 and 2
  voices) and class A's reader R4 once more (1). Roster entries `placed: true`, `self_intro`, guest; 0034 kept all 20
  (`CORRECT` / `self_intro`). Today's code: `{recurrence unnamed commercial: 20}`
  (`m43_rec.py`, the replay as above). Reach: GI 0 / 0 / 0; KG Person nodes 20 (one per voice);
  `detected_guests` 0; roster entries 20; segment rows 228, ad-free 228; no episode shared with
  classes A / B. Talk share 0.004 to 0.024.
- **The census (added 2026-11-04, after Review 2).** The three rules above are partial: the B6
  replay compared old roster code with new, so a voice old code did not name either, yet stored
  named by older code still, never showed. `scripts/measure/m0043_census.py` replays the current
  tree's roster on every served episode's stored inputs with the corpus's signatures built in
  memory (8 shards, about 160 s each, 3,130 episodes, 0 errors) and lists every stored-named voice
  it unnames and types `commercial`: **120**. All 59 above are in it. Of the other 61:
  - **group D, 52 voices**: non-hosts reading another show's or the publisher's promo, the same
    kind as class B (four journalists' subscription promos on one daily show 34, one network's
    promo for another show 11, the rest single voices);
  - **group E, 8 voices on the show's own stated host name**: 2 host names on German-language ad
    voices, 6 hosts reading their own show's promo or premium spot (4 of them are the ids section 4
    first left out);
  - 1 more stated host voice, found only by the committed census (its episode stores the
    transcript path written at ingest, which the scratch census did not resolve).
  The operator's call 3 (section 9): D joins class B (58), the 2 German-ad voices are class E, the
  hosts' own-show promos keep their names. `scripts/measure/m0043_list.py` classifies the census
  in that order and freezes the list: **113 listed** (A 33, B 58, C 20, E 2), 7 not listed (6 stated
  hosts and the BizNews host by id). Checked against the per-class sets above: same 113 keys, same
  classes. 0034 kept 108 of the 113 (`CORRECT`; basis `self_intro` 107, `undecidable` 1); 5 were
  not on 0034's list.
- **What the 113 names reach (`m43_impact113.py`, the snapshot):** every listed name leaves its
  episode (no other voice carries it). Segment rows 1,172 (A 295, B 593, C 228, E 56), ad-free
  1,155; KG Person nodes 78 in 68 episodes; `detected_guests` 44, `detected_hosts` 2 (class E: the
  host's name leaves those 2 episodes' roster and KG too, since no other voice carries it there);
  GI Person nodes 2, quotes 2, insights 1 (visible), all class B.

## 4. Prediction

Dated: 2026-11-04 (restated the same day at step 5, before any code: class C added, rows below)

On snapshot-prod-20261103, after apply on a clone:

- **Listed voices 59 in 52 episodes** (A 33 in 33, B 6 in 6, C 20 in 13); **applied 59** if every
  verdict confirms, per class A 33 / B 6 / C 20; episodes written 52.
- **Own numbers:** served names on the listed voices 59 -> 0; roster entries removed 59; KG Person
  nodes removed 31 (11 in 11 episodes + 20 in 13 episodes: 24 episodes); `content.detected_guests` entries removed 33; segment
  rows typed `commercial` 674 (A 295, B 151, C 228) and ad-free segment rows 674; unnamed voices on
  the corpus +59.
- **Must not move:** S1 18,589; S1n 7; S1c 0; S2 32,488 with each bucket (unknown 18,589,
  unidentified 13,804, **commercial 95**: commercial counts insights, and no GI node carries these
  voices, so typing them commercial adds 0); S3 9,373; S10a 197; S10b 2; S12 6; A1 400 / 301; the
  hidden-insight totals of #2200 / #2198 (S1, S10a) unchanged; every file outside the 52 episodes
  byte-identical; no voice outside the 59 changes name, role or type, inside the 52 episodes too.
- **By identity:** the 59 are the frozen list's entries (episode id, voice), class in the evidence
  field; ids in the list file, not repeated here. Not listed, by id: class-A kept
  `substack:post:196570345` SPEAKER_01, `47f783de-0450-4f3b-9a6c-251b63ddff6e` SPEAKER_07,
  `743e06dc-a88d-11f1-a4dd-878b8bea9556` SPEAKER_00; class-B host `https://iono.fm/e/1726296`
  SPEAKER_00. The 59 reversed 0034 keeps are the 59 listed ids.
- **Population not reached:** the 4 ids above; voices typed commercial by none of the three rules;
  episodes ingested after the snapshot (the pipeline's own rules).

### 4b. Restated after the census (supersedes the rows above)

Dated: 2026-11-04T10:51

`MISSED:` the migration module was written (uncommitted) before this restatement, against the
59-voice prediction; the population changed, the code did not.

- **Listed 113 voices in 103 episodes** (A 33 in 33, B 58 in 58, C 20 in 13, E 2 in 2; 3
  episodes hold voices of two classes); **applied 113** if every verdict confirms; episodes
  written 103.
- **Own numbers:** served names on the listed voices 113 -> 0; roster entries removed 113 (the
  names leave their episodes); KG Person nodes removed 78; `detected_guests` entries removed 44,
  `detected_hosts` 2; segment rows typed `commercial` 1,172 and ad-free rows 1,155; GI Person nodes
  removed 2, quotes unattributed 2 and typed `commercial`, insights hidden 1 (typed `commercial`).
- **Scoreboard:** S2 32,488 -> **32,489** (commercial 95 -> **96**: the one insight; unknown and
  unidentified unchanged); S1 18,589 unchanged (the insight is typed commercial, not unknown);
  S10a 197 -> 197 or 198 (198 only if that insight was its episode's last visible one).
- **Must not move:** S1 18,589; S1n 7; S1c 0; unknown 18,589; unidentified 13,804; S3 9,373; S10b
  2; S12 6; A1 400 / 301; every file outside the 103 episodes byte-identical; no voice outside the
  113 changes name, role or type.
- **By identity:** the 113 frozen entries (episode id, voice; class in the evidence field) in
  `upgrade/lists/0043_non_participant_names_removed.json`. Not listed, by id: stated hosts
  `substack:post:196570345` SPEAKER_01, `61f3a12e-bcd4-4285-9f36-1e6ddbb3cdf6` SPEAKER_03,
  `Buzzsprout-17024331` SPEAKER_00, `Buzzsprout-17503238` SPEAKER_00,
  `ef347198-77cb-4bf9-8e08-118d52d75132` SPEAKER_00, `ea283f31-8b41-42a8-9147-b49f01293e32`
  SPEAKER_00; the BizNews host `https://iono.fm/e/1726296` SPEAKER_00.
- **Trade by identity:** the 2 class-E episodes (`743e06dc-a88d-11f1-a4dd-878b8bea9556`,
  `47f783de-0450-4f3b-9a6c-251b63ddff6e`) lose their host's name: it sat only on the ad voice.

### 4c. Revision after the first clone gate (supersedes 4b where it differs)

Dated: 2026-11-04T11:26

The first gate (section 7, measurement 1) left `speaker_role` (guest 1,071 + 1,054 ad-free rows,
host 56 + 56) on the segments of the unnamed commercial voices and their diagnostics role (guest
109, host 2). The pipeline writes no segment role and diagnostics role `unknown` for a commercial
voice, and a reader counting hosts from segments would still count the 2 class-E ad voices as
hosts. Revision: the unnamed listed voices' segment rows lose `speaker_role`; their diagnostics
voice gets role `unknown`.

- **Applied 111** (the review kept 2: section 8 input); episodes written 101.
- **Own numbers after the revision:** segment rows typed `commercial` 2,237 (raw 1,127, ad-free
  1,110), each without `speaker_role`; diagnostics voices typed `commercial` 111, role `unknown`
  111; roster entries removed 110 (one listed voice has no roster entry); KG Person nodes removed
  76; `detected_guests` 43, `detected_hosts` 2; GI Person nodes 1, quotes 1, insights 1 (each typed
  `commercial`).
- **Scoreboard:** as measured at gate 1 and unchanged by the revision: S2 32,489 (commercial 96),
  every other number as at the baseline.
- **Must not move:** S1 18,589; S1n 7; S1c 0; unknown 18,589; unidentified 13,804; S3 9,373;
  S10a 197; S10b 2; S12 6; A1 400 / 301; files written as at gate 1 (the role edit lands in files
  gate 1 already writes: 0 episodes change only by role); `verify_all` with 0 regressions; every
  voice outside the 111 keeps its segment role and diagnostics role, the 2 kept voices
  (`0d5b552f-016f-47a7-8325-e993f48e9ea2` SPEAKER_08, `tag:soundcloud,2010:tracks/2178767775`
  SPEAKER_02) included; a second apply writes 0 files.
- **By identity:** the key is deleted (`speaker_role` absent, as the pipeline writes), not set to
  null. The host-role rows (56 raw + 56 ad-free, diagnostics host 2) are the 2 class-E ad voices,
  `743e06dc-a88d-11f1-a4dd-878b8bea9556` SPEAKER_00 and `47f783de-0450-4f3b-9a6c-251b63ddff6e`
  SPEAKER_07. Host voices by segment role: `743e06dc…` 1 -> **0** (the ad voice was its only host
  seat; the host's name, on no other voice, already leaves the episode at 4b), `47f783de…` 3 ->
  **2** (SPEAKER_04 and SPEAKER_14 keep their host role). The other 109 voices' rows go from
  guest to no role.
- **Wording:** "GI Person nodes 1" is the Person node removed; `gi_nodes_typed_commercial` 2 is
  the quote and the insight.

### 4d. Revision after Review 2 (supersedes nothing above)

Dated: 2026-11-04T14:21

Review 2's census test found `_move_census` computing the talk shares from the total before the
move: right when the voice leaves another type (the total is unchanged), wrong when the voice had
no type before (its seconds join the total but the shares do not see them; the shares then sum
above 1). Revision: the shares are computed from the total after the move.

- **On snapshot-prod-20261103:** 0 of the 111 applied voices have a diagnostics voice without a
  `voice_type` (scratchpad `m43_census_chk.py`), so **no measured number moves**: every 4c row and
  the gate's counters stay as measured; the re-gate (step 7, measurement 3) must reproduce them
  exactly, plan sha256 aside if the code digest moves it.
- **Must not move:** everything in 4c; the case is reached only by a voice with no type.
- **Rows added at its acceptance** (fresh agent, `minutes 3, tokens ~14k`): the sibling case, a
  previous type the census has no bucket for (or one holding fewer seconds than the voice), also
  moves the total; computing from the total after the move covers both. **Census row:** at
  measurement 3, every written diagnostics file with a `voice_census` has shares summing to 1
  within 0.001 (count of such files stated there). **By identity:** 0 of the 111 applied voices hit
  either case on the snapshot (`m43_census_chk.py`); for a typed voice with a bucket the total is
  unchanged, so its shares are the same values as at measurement 2; the 2 kept voices and voices
  outside the 111 are not touched. **Repro first:** the census test, parametrised none / a counted
  type / a type without a bucket, fails on the 4c code for none and no-bucket, and passes after.
  Not verified: whether the pipeline's own census counts an untyped voice's seconds; the fix assumes
  it counts every voice it lists.

### 4e. Revision: 0044 is registered after 0043 (supersedes nothing above)

Dated: 2026-11-04T16:48

`upgrade/registry.py` gains one entry after 0043's (`0044_title_spelling_voices_renamed`, record
`0044_title_spelling_voices_renamed.md`); 0043's own entry, list, verdicts and module do not change.

- **No measured number of 0043 moves:** the gate applies 0043 alone; its plan reads 0043's own
  list. The re-gate after the edit must give plan sha256 `d167d230…` and every 4c row exactly.
- **The code digest moves** (the registry is in 0043's import closure), so 0043's outcome is
  rebuilt after the re-gate, as each time code in its digest moved.
- **On prod:** 0043 runs first (`--to 2.7.37`, alone, as the hook requires the first pending
  migration), then 0044 in its own run after its own go.

## 5. Acceptance

Fresh agent, 2026-11-04: `minutes 4, tokens ~45k`. Rows added above: the reading-list items below,
the segment-row integer (674), commercial 95 stated, the #2200 / #2198 totals, the ids, the class
split, the KG population. Its reading of #2365 led to class C (m0016's recurrence hits). Added to
section 1's reading list: #2365 (closed; another show's host's promo published as a guest, "4 served records
still carry the old name until a relabel": they are in class C), #2362 (closed; host seat never
typed commercial), #2331 (closed; sponsor reads take voices), #2200 and #2198 (open; insights hidden
by unnamed voices: not moved here, GI 0), #1227 (closed; pre-roll ad narrators as guests),
`docs/guides/NAMING_KNOWN_ISSUES.md` K14, `docs/guides/NAMING_LADDER.md`.

Spot checks for step 8:

- (a) the 3 class-B promo voices: the episode's real participants keep their names;
- (b) the not-listed ids unchanged (4 at step 5; 7 after the census, section 4b);
- (c) the class-B host keeps name and role;
- (d) one voice per reader (R1-R4, C1-C5): roster, segment labels, transcript prefix, bridge
  identity row unnamed and typed `commercial`;
- (e) the KG Person nodes gone (31 at step 5; 76 after the census and the review), no GI node touched beyond the predicted ones;
- (f) one search-index transcript chunk per class: the prefix gone, rows outside the written episodes (52 at step 5; 101 after the census and the review)
  untouched (step 10, prod).

### 5b. Acceptance of revision 4c

Dated: 2026-11-04T11:28

Fresh agent (reviewer), `minutes 3, tokens ~14k`: 4c's integers are consistent with measurement 1
(arithmetic). Rows added to 4c on its reading: the must-not-move integers restated, the 2 kept
voices by id, the host-role rows tied to the 2 class-E voices with the host-seat count per
episode (1 -> 0, 3 -> 2), key deleted not nulled, 0 episodes changing only by role, the GI
counter wording. Spot checks for step 8 added:

- (g) both class-E episodes: no segment row of the ad voice carries `speaker_role`; host seats by
  segment role 0 and 2;
- (h) one class A or B voice: segment rows without `speaker_role`, diagnostics role `unknown`,
  neighbouring voices unchanged;
- (i) one kept voice: segments and diagnostics untouched, name and role;
- (j) idempotence: a second apply writes 0 files.

### 5c. Acceptance of revision 4d

Dated: 2026-11-04T14:22

Fresh agent (reviewer), `minutes 3, tokens ~14k`: the fix is right for the stated case and, with
the total taken after the move, covers the sibling case (a previous type without a census
bucket). Rows it asked for are in 4d (census row, identity, repro first, the pipeline-census
assumption named). Spot check for step 8: measurement 3's census row.

### 5d. Acceptance of revision 4e

Dated: 2026-11-04T16:49

Fresh agent (reviewer, Sonnet), `minutes 3`: rows added, each taken:

1. `upgrade run` without `--to` on prod, with 0043 and 0044 pending, is refused ("runs alone",
   `cli_handlers` R1), not a partial apply; the order is `--to 2.7.37` (0043), then `--to 2.7.38`
   (0044) after its own go.
2. `--to 2.7.37` leaves 0044 out of `pending`, so `check_static`'s first-pending rule sees 0043.
3. 0043's `applied_before` is unchanged; the only check the edit moves is the code digest.
4. `upgrade/registry` is in 0043's closure (measured 2026-11-04, scratchpad `c2_closure.py`);
   0043's list and verdicts sha256 are unchanged.
5. 0043's verify and undo read only its own receipts and tag; the re-gate's `verify_all` (42 ok,
   0 regressions) is reproduced, and 0044 registered but unapplied must not fail it.
6. The registry-order tests: 0043's test pins its predecessor only; 0044's test asserts it follows
   0043.
7. 0044's gate and outcome are separate and later; its clone has 0043 applied and recorded.
8. Not taken as written: the reader asked for `--to` on the 0043 re-gate; the gate script applies
   the migration directly (not through `upgrade run`) and checks only that the clone's ledger holds
   every earlier migration, so `--to` does not apply to it.

## 6. Build

- **Migration** `upgrade/migrations/m0043_non_participant_names_removed.py`: 0034's reviewed-list
  machinery (verdicts bound by the list's sha256, a listed voice without a verdict fails
  `verify`, no verdicts file writes nothing) and 0029's voice-by-voice rewrite with 0031's derived
  surfaces; on top, the listed voices' segment rows (raw and ad-free) and the GI nodes the
  rewrite hides are typed `commercial`, the diagnostics voice and its summary census move to
  `commercial`, and a name that leaves the episode leaves `detected_hosts` / `detected_guests`.
  Its own verdict bases: `WRONG_PERSON` / `non_participant`, `CORRECT` / `participant`. The list
  refuses an entry that renames, or names no class (A, B, C, E) and deciding segment. Not
  registered until the verdicts exist.
- **List** `upgrade/lists/0043_non_participant_names_removed.json` (113 entries) from
  `scripts/measure/m0043_census.py` and `scripts/measure/m0043_list.py` (section 3).
- **Tests:** `tests/unit/podcast_scraper/upgrade/test_m0043_non_participant_names_removed.py`
  (every surface unnamed and typed commercial, census moved, `detected_guests` trimmed, verify and
  undo; a `participant` verdict keeps the name; another question's basis is unusable; no verdicts
  writes nothing; an unplaced statement keeps the name in `detected_guests`; the list's refusals;
  its own backups; not registered). Mutation: each of the four added steps made a no-op fails a
  test (4 of 4 killed). `tests/unit/scripts/measure/test_m0043_list.py` (class order, the host
  and excluded ids, every entry loads) and `test_m0043_candidates.py`.
- `MISSED:` the module tests were written after the module, not before; the mutation run is the
  evidence that they test it.

## 7. Measure

**Measurement 1** (2026-11-04T11:22, `corpus_change_gate.py` on snapshot-prod-20261103, work copy
`gate-0043`; report `0043_non_participant_names_removed.gate.json` of that run): computed class
full; gating move S2 +1 only; files changed 1,110; `verify` ok; `verify_regressions` 0. Apply:
101 episodes, 0 refused; `roster_entries_unnamed` 110, `segments_typed_commercial` 2,237,
`diagnostics_voices_typed_commercial` 111, `kg_persons_removed` 76, `detected_guests_removed` 43,
`detected_hosts_removed` 2, `gi_nodes_typed_commercial` 2, `quotes_reattributed` 1,
`insights_reattributed` 1, `quotes_retexted` 12 (m0023's offset carry; S12 stays 6),
`entities_added` 44 / `entities_dropped` 99 (m0025's `normalized_entities` resync to the placed
roster). Against 4b with the 2 kept voices taken out: every number equal except roster entries
(110, not 111: one listed voice has no roster entry) and GI quotes (1, not 2: the other is a kept
voice's). Undo on the full work copy: restored 1,155, refused 0, all 1,155 byte-identical to the
snapshot, `verify` pending again (101 episodes). The excerpts (10 voices, both kept ones among
them) showed the unnamed voices keep `speaker_role`: revision 4c.

**Measurement 2** (after revision 4c; gate measured 2026-11-04T11:43:44, report
`0043_non_participant_names_removed.gate.json`, plan sha256 `d167d230…`): computed class full;
gating move S2_hidden +1 only (32,488 -> 32,489; commercial 95 -> 96); files changed 1,110;
`verify` ok; `upgrade verify` over all 42 applied migrations ok, 0 failing, `verify_regressions`
0. Apply counters identical to measurement 1. Checks on the work copy
(`scratchpad m43_m2.py`):

| 4c / 5b row | Predicted | Measured |
| --- | --- | --- |
| applied voices / episodes | 111 / 101 | 111 / 101 |
| segment rows typed commercial, no `speaker_role`, no label | 2,237 | 2,237, 0 problems |
| diagnostics voices role `unknown`, commercial, unnamed | 111 | 111 |
| other voices' segment role unchanged (rows) | all | 119,618 rows, 0 moved |
| other voices' diagnostics role unchanged | all | 0 moved |
| the 2 kept voices untouched | yes | yes |
| host seats by segment role, class E | 0 and 2 | 0 and 2 |
| roster / KG / `detected_guests` / `detected_hosts` | 110 / 76 / 43 / 2 | 110 / 76 / 43 / 2 |
| GI Person, quote, insight | 1 / 1 / 1 | 1 / 1 / 1 (2 nodes typed commercial) |
| S2 (commercial) | 32,489 (96) | 32,489 (96) |
| S1, S1n, S1c, S3, S10a, S10b, S12, A1 | unchanged | unchanged |
| second apply | 0 files | 0 episodes |
| full-clone undo | restored all, refused 0, byte-identical, verify pending | 1,155 restored, 0 refused, 1,155 / 1,155 identical, verify pending (101) |

Per-voice excerpts (10 voices: one per class A, C, E, both kept voices, five class B; ids,
labels, roles and types before and after, the deciding segment's text kept private in the
scratchpad `m43_excerpts.json`): every unnamed voice goes from its name to no label, type
`commercial`, and from measurement 2 no role; the kept voices are byte-unchanged; the other voices
of each episode keep their labels (an episode's label count drops only by its own listed voices).

Derived-surface census: the files written per suffix are in the gate report
(`files_changed_by_suffix`: metadata 101, segments 202, transcripts and ad-free text 201 (`transcripts_rerendered`; 21 of them under odd suffix keys, file names with dots), turns
30, diagnostics 101, GI 90, KG 77, bridge 91, context 100, ad maps 83, cleaned text 34). Not
written on the clone and left for step 10 on prod: the search-index rows of the 101 episodes
(targeted reindex) and the enrichment outputs built from the changed files.

**Builder: GO.** Deciding number: 111 of 111 reviewed non-participant voices unnamed and typed
commercial with 0 problems in 121,855 rows, S2 the only moving gating number (+1, as predicted),
0 verify regressions, full undo byte-identical.

**Measurement 3** (after revision 4d and the tier-L code; gate 2026-11-04T14:37:04 on clean
`804826ade`; re-run 14:52:16 on `26ed7d59a` after a variable rename the spelling check asked for, equal again): equal to measurement 2 in everything but its time and code stamp (plan sha256
`d167d230…`, files 1,110, S2 +1 only, `verify_all` 42 ok, 0 regressions); outcome rebuilt
(sha256 `fe2692aa…` after the re-run). Revision 4d's census row: 101 diagnostics files written, 101 with a
`voice_census`, 101 summing to 1 within 0.001 (`scratchpad m43_census_row.py`).

**Ledger after 0043** (added 2026-11-04 after step 8, a derived surface the gate does not run: on
prod `upgrade run` ends with the ledger refresh and, since 0041, its respell). On a fresh copy of
the snapshot (`scratchpad m43_ledger.py`): a baseline refresh first wrote 0 files and changed 0 of
652 joins; after the 0043 apply, the refresh `upgrade run` does wrote **0 files** (respell written
0, refused 0) and the ledger went from 652 to **649 joins**, 0 display spellings changed. The 3
joins removed are spelling variants whose group 0043 empties: `person:brian-chen`,
`person:catherine-benholt` and `person:leila-faldil`. Residue: two promo voices today's roster
still names stay outside 0043 (B10's recurrence blind spots): one Daily episode
(`c89d8533-9397-49b1-9849-0bc560b5a443` SPEAKER_06, the variant spelling) and one Planet Money
episode (`2ffbf42c-503f-4b51-9462-af3c5626c900` SPEAKER_05). After 0043 the first no longer joins
the group and shows its own (variant) spelling at read time: 1 voice's displayed spelling moves.

**Re-embed check** (`scratchpad m43_fp.py`, the indexer's own row builder and fingerprint, before
vs after on the same copy): of the 101 written episodes, 100 change fingerprint, so prod's
incremental `index` re-embeds them; 1 changes no embeddable row's text or metadata. Rows: 76
removed (the KG entity rows of the removed Person nodes), 0 added, 4,751 with moved metadata, all
inside re-embedded episodes; **0 episodes** with moved metadata and an unchanged fingerprint, so no
search row stays stale and no in-place row edit is needed. Step 10 expects `reindexed` 100.

## 8. Verdict

Verdict read 2026-11-04 by a fresh agent (reviewer, Sonnet; not the builder, not a step-5
reader), inputs as file paths, ran nothing over the corpus.

Review budget: minutes 5, tokens 40000

```text
REVIEW 0043_non_participant_names_removed 2026-11-04 commit 5c410dcb6 reviewer Sonnet (fresh) minutes 4
PREDICTION vs GATE: 4c compared row by row, 18 rows equal, 0 missed (applied 111 / 101 episodes; segments 2,237; diagnostics 111; roster 110; KG 76; detected_guests 43 / hosts 2; GI 2 typed, quote 1, insight 1; S2 +1, commercial 95 -> 96; every must-not-move number equal; verify ok, 0 regressions; host seats 0 and 2; second apply 0; 119,618 other rows unmoved)
REQUIRED ITEMS: gate, undo (1,155 restored, 0 refused, byte-identical, verify pending), second apply, census, verify_all, excerpts, builder GO with deciding number present; ci-fast log missing
AFTER-STATE READ: excerpts show labels gone, type commercial, other voices kept, kept voices unchanged, but they are the measurement-1 state (role still present); spot checks (c), (d), (e) not evidenced by id; section 5 wording stale (31, 52, "4 ids")
OPEN CALLS: calls 1-3 recorded as operator-accepted; the 2 kept voices consistent with call 3; none new
DOUBTED NUMBER: none by arithmetic
VERDICT: GO WITH CONDITIONS  every 4c integer and must-not-move number holds, undo and second apply clean; ci-fast log, 4c-state excerpts and checks (c), (d), (e) by id missing
```

Conditions, each answered:

1. `make ci-fast` on the committed tree, log attached: **open** (running at the time of writing;
   result below).
2. Excerpts on the 4c state and checks (c), (d), (e) by id: **done** (`scratchpad
   m43_checks2.out`, on the work copy after measurement 2). One voice per reader of classes A, C
   and E (12 voices) and both kept voices: every unnamed voice goes from its transcript prefix
   lines (1 to 5) to 0, its bridge identity (1, or 0 where it had none) to 0, its roster entry to
   0, its segment role (guest, or host for E) to none and its diagnostics role to `unknown`; the 2
   kept voices keep prefixes, bridge identity, roster entry and roles (guest -> guest). (c) the
   BizNews host `https://iono.fm/e/1726296`: episode not written (`m43_spot.json`, with the other
   6 not-listed ids). (e) GI attribution moved on exactly 3 nodes, all in
   `29b52580-0f56-41a6-8b8a-288bd604bc01`: 1 Person removed, 1 quote (`quote:d0ebd498559efb4e`)
   and 1 insight (`insight:7d18f7a8ab112598`) unattributed and typed commercial.
3. Section 5's stale numbers: **done** (the step-5 values kept, the current ones beside them).

Verdict: GO WITH CONDITIONS

- **Condition 1:** `make ci-fast` green on the committed tree, log attached (the first run, while
  this section was still empty, failed only the landing check that requires this line).
  **Done** 2026-11-04 on `0758c88cf` (clean tree): `MAKE_EXIT=0`; unit 17,475 passed, 58 skipped;
  then 2,263 passed, 46 skipped; 244 passed; 40 passed (log `scratchpad ci-fast-0043b.log`, sha256
  `fb6526c518a8c3ed…`, kept local: it holds machine paths).
- **Condition 2:** at the apply, prod's dry-run plan diffed against the clone's (method step 10:
  prod holds episodes the snapshot lacks), and the search-index rows of the written episodes
  reindexed (targeted, no rebuild).

Gate on committed code: re-run 2026-11-04T12:08:57 on a clean tree (`91794771c`, dirty false),
equal to measurement 2 in everything but its time and code stamp (plan sha256 `d167d230…`).
Outcome: `src/podcast_scraper/upgrade/outcomes/0043_non_participant_names_removed.json`, sha256
`632a047f34630b0347f6bd74d00064062a6932d0498620c8935d2a1c61b2a920` (rebuilt 2026-11-04 after
0044's registry entry, revision 4e; gate re-run on `5d37558e`, dirty false, plan sha256
`d167d230…` equal, 42 verify ok, 0 regressions; earlier builds `584ff3c0…` (after C2's fix),
`5866ae29…`, `fe2692aa…`, `27bca008…` and `f625eb63…`, each stale once code in its digest moved).

**Rehearsal of prod's apply path** (2026-11-04, a copy of snapshot-prod-20261103, `upgrade run
--to 2.7.37 --yes --no-snapshot`, code `bc03d093`): the outcome hook applied 0043, receipt verdict
PASS, 0 failures, 0 plan differences, `S1c_inconsistent` 0; 101 episodes with the gate's counters;
"Now at version 2.7.37; 0 pending"; the ledger refresh after it: 649 joins, 3 removed, as measured.

## 9. Operator review

- **Open call 1:** 0043 reverses 0034's third-ruling keeps for these 39 self-introduced promo and
  shout-out voices, aligning stored episodes with the shipped pipeline (#2123, #2364).
  Operator accepted: 2026-11-04 0043 reverses the 0034 third-ruling keeps for the 39 listed voices.
- **Open call 2:** class C (20 voices, 13 episodes, section 3) is the same ruling's keeps, found
  at step 5. Reverse those too (59 in all)?
  Operator accepted: 2026-11-04 0043 also reverses the 0034 keeps for the 20 class-C voices (59 in all).
- **Open call 3:** the whole-corpus census (section 3) finds 52 more non-host promo voices
  (group D, joined to class B) and 8 voices on the show's own host name (group E). Add D; of E,
  only the 2 host names on German-language ad voices?
  Operator accepted: 2026-11-04 0043 adds group D (52) and the 2 German-ad host-name voices of
  group E (113 in all); the 6 hosts' own-show promo voices keep their names.
- **Open call 4** (Review 2, minor 4): the 2 class-E host names, on no other voice in their
  episodes, also leave `detected_hosts` (the migration's own rule for a name that leaves its
  episode). Keep that?
  Operator accepted: 2026-11-04 class E stays as built (the name leaves `detected_hosts` too).

## 10. Before apply / deploy

_TODO_

## 11. After apply / deploy

_TODO_

## 12. Retrospective

_TODO_

## Not covered

- Class B is the B6 replay's output, a replay over the whole corpus with the old and new roster
  code; this record does not re-run it. Voices #2364's rule would type commercial and the replay
  did not list (an episode whose replay errored) are not counted.
- The ad signatures are rebuilt every 6 h on prod; class A is computed from the snapshot's rebuild
  (2026-11-04). Prod's list can differ; the list is frozen from the snapshot and verified per entry
  against prod's stored name and role (refused, untouched, when moved on).
- The search index's chunk rows for the 52 episodes are a step-10 item (targeted reindex of those
  episodes), not measured on the clone.
- The class-B host voice's wrong role (guest) is not fixed here.
- m0016's exemption still counts a pre-1.2.0 `content.detected_guests` as a statement; 0043 does
  not change m0016. The pipeline no longer writes the field (schema 1.2.0), so only stored
  artifacts are affected, and 0043 removes the name from it where the name leaves the episode.
