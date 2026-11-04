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
  person". A self-introduction inside the promo is such a tie: all 39 voices below carry
  `CORRECT` / basis `self_intro` in `upgrade/verdicts/0034_talk_floor_names_removed.tsv`.
- **The conflict this change must not paper over:** today's pipeline publishes these 39 voices
  `commercial` with no name (section 3); 0034's ruling keeps them named on stored episodes. Both
  cannot stand. 0043 sides with the pipeline. That reverses 39 rows of a recorded operator ruling,
  so the operator decides it before the build (section 9, open call 1).
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
- **Today's code agrees on all 39.** Replaying `resolve_speaker_roster` (current tree) on each
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
  host or writer promoting their own show inside this one: C1 a Pivot co-host 7 voices, C2 the other
  co-host 4, C3 Criminal's host 3, C4 Rest is Entertainment's host 3, C5 an Athletic writer 2, and
  class A's reader R4 once more (1). Roster entries `placed: true`, `self_intro`, guest; 0034 kept all 20
  (`CORRECT` / `self_intro`). Today's code: `{recurrence unnamed commercial: 20}`
  (`m43_rec.py`, the replay as above). Reach: GI 0 / 0 / 0; KG Person nodes 20 (one per voice);
  `detected_guests` 0; roster entries 20; segment rows 228, ad-free 228; no episode shared with
  classes A / B. Talk share 0.004 to 0.024.

## 4. Prediction

Dated: 2026-11-04 (restated the same day at step 5, before any code: class C added, rows below)

On snapshot-prod-20261103, after apply on a clone:

- **Listed voices 59 in 52 episodes** (A 33 in 33, B 6 in 6, C 20 in 13); **applied 59** if every
  verdict confirms, per class A 33 / B 6 / C 20; episodes written 52.
- **Own numbers:** served names on the listed voices 59 -> 0; roster entries removed 59; KG Person
  nodes removed 31 (11 + 20, in 31 episodes); `content.detected_guests` entries removed 33; segment
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

## 5. Acceptance

Fresh agent, 2026-11-04: `minutes 4, tokens ~45k`. Rows added above: the reading-list items below,
the segment-row integer (674), commercial 95 stated, the #2200 / #2198 totals, the ids, the class
split, the KG population. Its reading of #2365 led to class C (m0016's recurrence hits). Added to
section 1's reading list: #2365 (closed; Kara Swisher's promo on The Gray Area, "4 served records
still carry the old name until a relabel": they are in class C), #2362 (closed; host seat never
typed commercial), #2331 (closed; sponsor reads take voices), #2200 and #2198 (open; insights hidden
by unnamed voices: not moved here, GI 0), #1227 (closed; pre-roll ad narrators as guests),
`docs/guides/NAMING_KNOWN_ISSUES.md` K14, `docs/guides/NAMING_LADDER.md`.

Spot checks for step 8:

- (a) the 3 class-B promo voices: the episode's real participants keep their names;
- (b) the 4 not-listed ids unchanged;
- (c) the class-B host keeps name and role;
- (d) one voice per reader (R1-R4, C1-C5): roster, segment labels, transcript prefix, bridge
  identity row unnamed and typed `commercial`;
- (e) the 31 KG Person nodes gone, no GI node touched;
- (f) one search-index transcript chunk per class: the prefix gone, rows outside the 52 episodes
  untouched (step 10, prod).

## 6. Build

_TODO_

## 7. Measure

_TODO_

## 8. Verdict

_TODO_

## 9. Operator review

- **Open call 1:** 0043 reverses 0034's third-ruling keeps for these 39 self-introduced promo and
  shout-out voices, aligning stored episodes with the shipped pipeline (#2123, #2364).
  Operator accepted: 2026-11-04 0043 reverses the 0034 third-ruling keeps for the 39 listed voices.
- **Open call 2:** class C (20 voices, 13 episodes, section 3) is the same ruling's keeps, found
  at step 5. Reverse those too (59 in all)? Until answered, the build lists class C separately so
  it can be dropped whole.

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
