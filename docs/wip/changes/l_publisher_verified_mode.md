---
id: l_publisher_verified_mode
class: full
status: open
issue: none (ingestion arc, tier L: L2, L3 ledger side, L4, L5, L6; #2056)
touches:
  - src/podcast_scraper/identity/ledger.py
  - src/podcast_scraper/identity/guards.py
  - src/podcast_scraper/identity/publisher_text.py
  - src/podcast_scraper/kg/entity_clusters.py
  - tests/unit/podcast_scraper/identity/
---

# Tier L — a ledger rule mode that reads the publisher's own text (shipped switched off)

Method: [CORPUS_CHANGE_METHOD.md](../../guides/CORPUS_CHANGE_METHOD.md) §4, full class. Arc list
`docs/wip/INGESTION-ARC-2026-11-03.md`, tier L. Operator, 2026-11-04: tier L is built in this arc
as a new named rule mode beside today's, measured on a snapshot clone and scored once on a fresh
unseen split; `PERSON_MODE` stays on today's rules, so the deploy changes no stored file; switching
on is its own one-line change and deploy. Ids only in this record.

## 1. Intake

- **What a ledger change does on prod:** since 0041 every `upgrade run` and `enrich` ends with
  `refresh_ledger`, which respells the episode files to the build's group spellings
  (`identity/refresh_guard.py`, `identity/ledger.py:437`). A rule that changes a join or a shown
  spelling rewrites stored files at the next refresh once it is the active mode.
- **The arc's tier L items, read again against the code (2026-11-04):**
  - **L1** (#2403, open): the roster's ~13 same-person rules, unified or justified (RFC-131
    Phase 5). These decide names inside one episode at naming time; the ledger reads their output,
    not the rules. Its prerequisites (the shadow namers' monitoring decision; a voice-level truth
    set) are not met. **Not in this mode;** a forward roster change of its own.
  - **L2** (handover §2.3; record d2 "accepted loss"; record s6h Not covered): a feed's stated
    host matched to a group of ASR spellings by first name and the host role.
  - **L3** (handover §2.4; record s6h 4b item 1): `known_hosts` is mixed evidence. The pipeline
    half (`workflow/stages/processing._recurrent_hosts_from_disk` writes ASR spellings into it) is
    a forward fix, **not in this mode**. The ledger half is: the ledger takes every `known_hosts`
    name as the publisher's.
  - **L4** (record s6h Not covered): strict-tier losses D1 -> D2, `sarah-gonzalez`-shaped 16 gold
    insights, `erika-barris` 8, `david-sanger` 8, one other 1.
  - **L5** (#2056, open): duplicate people from spelling variants; check against D2, close or
    name what is left.
  - **L6** (B5 / #2363): a voice's heard spelling joins the person its own episode's title names
    (eval-data branch `ingestion-arc-l6`, `data/eval/ingestion-arc/l6/`: 27 voices, 20 served
    episodes still carry the heard spelling).
- **Existing machinery read:** `identity/ledger.build_ledger` (mode `PERSON_MODE`, linkage
  `no_one_word_bridge`, guards ADR-167, reviewed pairs s5d), `_add_display_spellings` and
  `_publisher_spellings` (records s6e, s6g, s6h), `identity/guards.collect_episode_evidence` and
  `PUBLISHER_KEYS = ("known_hosts", "metadata_named", "detected_guests")` (`guards.py:50`).
- **Records:** s5c, s5d, s6a–s6h, d1, d2, 0041. ADR-166 (the ledger), ADR-167 (guards).
- **Scoring set:** eval-data `naming_speakers_v1/val_v7` (80 cases, 27 feeds, built 2026-11-04,
  labels_v1, 207 scored names), unread by anyone tuning a rule; design on dev_v1 and the snapshot's
  unlabelled data only.

## 2. Outcome and baseline

- **Outcome:** in the new mode, a group's shown spelling and its joins follow the publisher's own
  text where the publisher names the person, and an ASR spelling the pipeline put in `known_hosts`
  no longer counts as the publisher's.
- **Baseline, snapshot-prod-20261103, today's mode:** ledger 652 joins in 458 groups, 40 blocked
  (`identity_ledger.json`, rule `cluster-person-no-bridge-guarded/3`). Groups of this record's
  cases, shown spelling and its rule: `person:dara-obrey` (publisher, 15 episodes),
  `person:erica-barris` (frequency), `person:luisa-lemi` (publisher, 9), `person:sarah-gonzales`
  (frequency), `person:david-sanger` (publisher, 4), `person:john-glenn-hill` (tie).
- **Gold baseline:** S7 / S8 on dev_v1 and val_v7 with today's mode: to be taken with the scorer
  D1/D2 used (eval-data `d2out`), before any rule (section 4).

## 3. Diagnosis by loss

Scripts in the session scratchpad (`l2_diag.py`, `l3_pub.py`, `l4_diag.py`, `l_pubname.py`), over
snapshot-prod-20261103; counts and ids only.

- **The ledger's "publisher" names are the pipeline's extracted names, never checked against the
  publisher's text.** `guards._publisher_names_by_key` reads the diagnostics' `tried.known_hosts`,
  `metadata_named` and `detected_guests`. Of 4,307 `known_hosts` name-episodes (two-word names),
  **1,063 are not in the episode's own publisher text** (feed and episode title and description,
  folded). Examples by show: Curious Cases' stated host is in 33 of 33 episodes' text, its ASR
  spelling in 0 of 24 (the source of `person:dara-obrey`'s "publisher" spelling); Latin America in
  Focus's `known_hosts` spelling is in 0 of 17, while the publisher's own spelling of that host
  is in 20 of 44 episodes and in no `known_hosts`; Latent Space's two `known_hosts` spellings of
  one co-host are in 0 of 37 and 0 of 18.
- **The publisher's text names some groups the ledger never sees as stated.** Planet Money's text
  spells `person:sarah-gonzales`'s person in 10 of 89 episodes; that spelling is in no extracted
  list, so the frequency rule (6 episodes against 5) shows the other one.
- **First name plus host role is not enough to join (L2).** Pairs of (a show's stated host, a
  same-show person id in a host seat with the same first name, spelled otherwise, not joined):
  **29**. About 4 groups are ASR spellings of the stated host (`person:rj-hanaki` /
  `person:rj-honicky`, `person:dara-obrey`, `person:erica-barris`, `person:jeff-glow`'s spellings);
  the rest are other real people who share a first name in one show: several a16z hosts
  (`person:david-george`, `person:david-booth`, `person:david-clark`), `person:ben-horowitz` against
  another stated Ben, `person:patrick-mckenzie` against the stated Patrick. Two guards separate
  them on this data: the stated host is the **only** stated host of the show with that first name,
  and the surnames resemble (same Soundex, or a folded-surname ratio of at least 0.5). And one
  ASR spelling differs in the first name too (`person:john-glenn-hill` against the stated
  `Jonquilyn`-spelled host): a whole-name resemblance is needed there.
- **L4's three cases are three causes.** `erika-barris`: all ASR spellings of the stated host
  (L2's). `sarah-gonzalez`: the publisher's spelling is in the text but in no extracted list (the
  second bullet). `david-sanger`: shown with a middle initial by the publisher rule, the same
  person; the strict tier counts it wrong. That is M2's scoring split (same person misspelled vs
  another person), not a ledger error.
- **L6:** in the replay, 27 voices in 20 served episodes carry a heard spelling of the person their
  own episode title names, every one the same person; D2's s5d keeps the one-episode ids apart.
- **L5:** not measured yet (section 4 measures it on the new mode's ledger).

## 4. Prediction

Dated: 2026-11-04T14:00

**The mode** (prototyped in the scratchpad, `l_proto.py`, by patching the evidence and the join
test at runtime; no code in `identity/` yet). A new evidence mode beside today's, selected by a
constant like `PERSON_MODE`, default today's:

- **V1** an extracted name (`known_hosts`, `metadata_named`, `detected_guests`) counts as the
  publisher's only when the episode's own publisher text (feed and episode title and description,
  folded) contains it;
- **V2** the publisher's text supplies its own spelling of a person the episode names: a span
  "first name + one or two capitalised words" whose surname resembles (same Soundex, or folded
  ratio >= 0.5), the only such spelling in the text; a trailing possessive is dropped;
- **V3** joins inside one show: a host-seat person id joins a verified host by route A (same first
  name, the only verified host with it, surname Soundex equal or ratio >= 0.6), B (first names at
  ratio >= 0.75, the same surname test) or C (the same last word, the same first letter, the rest
  at ratio >= 0.5).

The prototype was iterated four times on the snapshot's unlabelled data, each run's joins read by
hand: run 1 joined two pairs of different people through a whole-name route (dropped), run 2 a
dozen through a loose first-name route (dropped), run 3 two through route C without the first
letter (fixed). Design data: the snapshot and dev_v1. val_v7 is not read.

**On snapshot-prod-20261103, the new mode against today's** (prototype run 4):

- **V1 drops** 1,063 `known_hosts`, 264 `metadata_named` and 1 `detected_guests` name-episodes
  as not the publisher's; **V2 finds** 187 publisher spellings (204 left: two fitting spellings in
  one text); **V3** 28 candidate join pairs.
- **Joins 652 -> 656** (+4, 0 removed, 0 moved): `person:erika-beras` -> `person:erica-barris`;
  `person:john-quillen-hill`, `person:jonklin-hill`, `person:jonquilyn-hill` ->
  `person:john-glenn-hill`.
- **Shown spellings: 18 groups move.** 12 take the publisher's spelling: `person:amanda-aronchik`,
  `person:chris-hustane`, `person:dara-obrey`, `person:erica-barris`, `person:greg-rosalski`,
  `person:jeff-glow`, `person:john-glenn-hill`, `person:luisa-lemi`, `person:luisa-lemme`,
  `person:sam-brazell`, `person:sarah-gonzales`, `person:yasmin-tayag` (7 of them had no spelling
  before, a tie or a plurality: `person:amanda-aronchik`, `person:chris-hustane`,
  `person:greg-rosalski`, `person:jeff-glow`, `person:john-glenn-hill`, `person:sam-brazell`,
  `person:yasmin-tayag`). 6 keep their spelling and change only the evidence label (publisher
  <-> frequency): `person:casey-newton`, `person:dan-saffer`, `person:heather-haddon`,
  `person:lily-ottinger`, `person:noah-kravitz`, `person:steve-hatherly`.
- **Respell when the mode is switched on:** 110 files written of 1,009 examined, 0 refused (on a
  copy, `l_proto.py --write-ledger`).
- **dev_v1 (design set), served names after the respell:** S7_correct 2,440 -> **2,452**,
  S7_wrong 392 -> **380** (the 12 are `person:dara-obrey`'s, D2's accepted loss), fold_exact
  2,251 -> **2,279**, fuzzy_or_title 25 -> 9, token_subset 164 -> 164, S7_unnamed 1,313 -> 1,313.
- **val_v7 (scored once, at step 7), integers:** S7_wrong 299 -> **290 to 299** (no increase),
  S7_correct 2,334 -> **2,334 to 2,345**, fold_exact 2,181 -> **2,181 or more**, S7_unnamed 799
  unchanged.
- **Must not move:** S7_unnamed (both sets); no join removed; no group loses a member; the
  shipped mode (today's) builds byte-identical ledgers on the snapshot before and after the build;
  `PERSON_MODE` and the active evidence mode unchanged, so the deploy writes no file.
- **L5 (#2056):** to be answered from the build's ledger: whether the two reported pairs are one
  group each in today's mode and in the new one.

### 4b. Rows added at step 5 (2026-11-04)

- **Baselines' source:** both today's-mode gold numbers above are one run of
  `scripts/measure/gold_insights.py --corpus <snap1103> --set dev_v1 --set val_v7 --labels
  <set>=labels_v1` (scratchpad `l_gold_base.txt`, 2026-11-04), which reads the cases and labels
  inside the scorer; no case or label file was opened by hand, and val_v7's output was read for
  its totals only. S8_undecidable baseline: dev_v1 77, val_v7 5.
- **val_v7 point predictions** (rule: dev_v1's net move per insight, +12 correct over 4,427
  insights, scaled to val_v7's 3,634 insights = +10; fold_exact +28 / 4,427 -> +23): S7_wrong
  299 -> **289**, S7_correct 2,334 -> **2,344**, fold_exact 2,181 -> **2,204**; bounds S7_wrong
  <= 299, S7_correct >= 2,334, fold_exact >= 2,181 (floors: the mode removes no join and moves a
  group only to a spelling the publisher's text gives). S8_undecidable 5 unchanged; S7_unnamed 799
  unchanged.
- **Must not move, added:** blocked joins 40 unchanged; groups 458 less the merges of the 4 added
  joins, no other group changes membership; the 6 label-only groups keep their spelling; the
  ADR-167 guards and the s5d reviewed pairs untouched (no code there); the one-episode ids s5d
  keeps apart (L6's 27 voices) stay apart, since this mode adds no title rule (L6 is not in it,
  below).
- **By identity, V1 and V2 samples:** V1 drops the `known_hosts` spellings `person:rj-hanaki` (0 of
  37 texts) and `person:rj-honicky` (0 of 18) on Latent Space, `person:luisa-lemi` (0 of 17) on
  Latin America in Focus, `person:dara-obrey` (0 of 24) on Curious Cases. V2 finds the
  publisher's spelling for `person:luisa-lemi`'s group (20 of 44 texts) and
  `person:sarah-gonzales`'s (10 of 89). The build lists every V1 drop and V2 spelling by episode
  and id in its measurement.
- **Denominators:** V2's 187 found and 204 ambiguous are of 391 roster or extracted names not in
  their episode's text whose first name occurs there; V3's 28 pairs are of 77 shows with a stated
  host; section 3's 29 pairs are of the same 77 shows.
- **L6 is not in this mode** (correction to section 1's scope): the title rule joins by an episode
  title naming the person, a fourth evidence source this record does not build; it stays a tier-L
  item for its own change.
- **Iteration stops at prototype run 4.** The build implements run 4; nothing is re-tuned after
  the val_v7 score.

## 5. Acceptance

Fresh agent, 2026-11-04: `minutes 4, tokens ~40k`. Rows added in 4b: point predictions with
their rule, the baselines' source, S8, blocked joins, group count, the label-only groups, the
guards and s5d untouched, samples by id, denominators, the discipline stated. Reading list
added to section 1: #2102 (tier sources by provenance: V1's subject), #2092 (show hosts and
episode hosts one field), #2219 (entity registry), #2097 (post-deploy epic), #2261 (feed-level
profiles), #2078 and #2105 (host-seat roles, which V3 reads); `docs/wip/HANDOVER-NAMING-2026-11-03.md`
(the "handover" of section 1), `docs/guides/NAMING_KNOWN_ISSUES.md`, `docs/guides/NAMING_LADDER.md`,
`docs/guides/SPEAKER_ATTRIBUTION_TRACE_GUIDE.md`, the arc list's "Tier L" section.

Spot checks for step 8: the 12 shown-spelling moves; the 4 added joins and 5 rejected V3 pairs;
10 V1-dropped `known_hosts` names (Latent Space among them); today's mode byte-identical on the
snapshot; none of the 110 respelled files gives a voice another person's name;
`person:sarah-gonzales` shows the publisher's spelling.

## 6. Build

- `identity/publisher_text.py` (new, pure): `fold`, `soundex`, the surname tests, `text_spans`,
  `verify_and_find` (V1, V2) and `host_route` (V3's routes A, B, C), each the prototype's run 4.
- `identity/guards.py`: `collect_episode_evidence(root, mode=...)`; in `publisher_verified` each
  episode's evidence carries V2's pairs, its host seats, its show and the show's verified names;
  `host_join_pair_rule` answers `"join"` for V3's pairs over the reviewed pair rule. The
  diagnostics path now resolves through `utils.run_relative` (it took the first occurrence of the
  run's name; every other reader takes the last).
- `kg/entity_clusters.py`: a pair rule's `"join"` matches a candidate to a cluster; the guards
  still decide.
- `identity/ledger.py`: `EVIDENCE_MODE = "extracted"` (shipped), `RULE_VERSION_BY_EVIDENCE`,
  `build_ledger(root, evidence_mode=None)`; `_publisher_spellings` takes V2's pairs as the same
  person. The default build passes `extracted` evidence and stamps today's `RULE_VERSION`.
- Tests: `tests/unit/podcast_scraper/identity/test_publisher_verified_mode.py`, 21: the shipped
  mode, unknown modes refused, spans without a possessive, V1 and V2 on one episode, V2's
  ambiguity, the routes including the six measured failures as negatives, the join answer with a
  guard still refusing, today's evidence on an ASR-host show, the verified evidence, the verified
  build's join and spelling, the default build equal to the extracted build, and the Dara shape
  only V2's pair reaches. Mutation (`scratchpad mut_l.py`): V1's filter, V2's spellings, V3's join
  and V2's pair each made a no-op fail a test, 4 of 4 (the pair survived the first round; its
  test was added). Identity and kg unit tests: 994 passed.
- `MISSED:` the tests were written after the code (the prototype ran first); the mutation run is
  the evidence they test it.

## 7. Measure

On snapshot-prod-20261103 with the committed build (`identity/` at the build commit; scratchpad
`l_build.py`, `l_compare.py`, `l_switch_on.py`, `l5.py`):

| Row (4 / 4b) | Predicted | Measured |
| --- | --- | --- |
| today's mode, old code vs new, ledger byte-identical | yes | **yes** (a worktree at the commit before the build; `code_version` equal) |
| joins in the new mode | 652 -> 656, the 4 named | **656**, the same 4 ids, equal to prototype run 4 |
| joins removed or moved | 0 | **0** |
| blocked joins | 40 unchanged | **40**, the same set |
| groups | 458 less the merges | **458**: the 4 joining ids were one-member ids, which are not groups |
| shown spellings moved | 18 (12 to the publisher's, 6 label-only) | **18**, the same set as run 4 |
| respell when switched on | 110 episodes of 1,009 files | **102 episodes, 980 files**, 0 refused (a copy) |
| dev_v1 S7_correct / S7_wrong / fold_exact | 2,452 / 380 / 2,279 | **2,452 / 380 / 2,279** |
| val_v7 S7_wrong | 289 (<= 299) | **299** |
| val_v7 S7_correct | 2,344 (>= 2,334) | **2,334** |
| val_v7 fold_exact | 2,204 (>= 2,181) | **2,181** |
| S7_unnamed, S8_undecidable (both sets) | unchanged | **unchanged** (1,313 / 77; 799 / 5) |

- **val_v7 is a miss on the point predictions and holds every bound: the mode moves nothing on
  the unseen split.** None of its 80 cases' served names changes. The point predictions scaled
  dev_v1's move by insight count; dev_v1's move is one group (`person:dara-obrey`, 12 insights),
  which says nothing about another set's feeds. Read as: on 27 unseen feeds the mode neither
  helps nor hurts; its gain is where the publisher's text and the extracted names disagree, which
  those feeds' cases do not touch. Scored once; nothing re-tuned.
- **Respell 102 against the prototype's 110:** the prototype patched the roster's `_same_person`
  for the whole process, so V2's pairs also reached the respell's own matching; the build keeps
  them inside the display rule. The 8 episodes are not traced by id.
- **L5 (#2056):** of its two pairs, "Theo Jaffee" / "Theo Jaffe" are one group in both modes, and
  only one spelling of the other pair exists on the snapshot (`person:lukasz-kaiser`, 1 episode;
  no `person:lucas-kaiser` node). Neither duplicate remains; closing #2056 is the operator's call.
- **Spot checks (5b):** the 12 shown-spelling moves and the 4 joins are listed in section 4 by id,
  each read: all are the publisher's or the stated host's own spelling of the same person. Of
  V3's 28 candidate pairs, 21 joined nothing, most because the pair's second id is the
  publisher's spelling, which no episode file carries as a node (for example
  `person:sarah-gonzales` / `person:sarah-gonzalez`; V2 fixes its spelling instead), the rest
  already one group: none joins two people. V1's drops include the Latent Space and Latin
  America in Focus spellings named in 4b. `person:sarah-gonzales` shows the publisher's spelling.
  Not read: each of the 102 respelled episodes' files.
- **The one change on the default path:** `guards._publisher_names_by_key` now resolves an
  absolute transcript path through `utils.run_relative` (the last occurrence of the run's name; it
  took the first, and a path ending at the run directory now gives no diagnostics instead of an
  odd path). It differs only for a path that repeats the run's name; on the snapshot the default
  ledger is byte-identical with it.

**Builder: GO for shipping switched off.** Deciding numbers: today's mode byte-identical, so the
deploy changes no stored file; switched on, the measured ledger moves (4 joins, 12 spellings, 0
removed, 40 blocked unchanged) are each the same person, dev_v1 recovers D2's accepted 12, and
val_v7 does not move. Switching on stays the operator's call, after reading this table.

## 8. Verdict

Verdict read 2026-11-04 by a fresh agent (reviewer, Sonnet; not the builder, not a step-5
reader), inputs as file paths, ran nothing over the corpus, read no val_v7 case or label.

Review budget: minutes 5, tokens 40000

```text
REVIEW l_publisher_verified_mode 2026-11-04 commit cf77b1ece reviewer Sonnet (fresh) minutes 4
PREDICTION vs GATE: 12 rows (section 7; no gate report), 9 equal, 3 val_v7 point misses stated as misses, bounds held; respell row in other units (110 files vs 102 episodes / 980 files), 8 episodes untraced
REQUIRED ITEMS: EVIDENCE_MODE "extracted"; default build passes extracted evidence and today's rule version; unknown modes refused; "join" only from host_join_pair_rule, only in publisher_verified; today's mode byte-identical; 21 tests, mutation 4 of 4; missing: ci-fast log, section 8 and Not covered, run_relative not named as a default-path change
AFTER-STATE READ: default path unchanged on the snapshot; switched-on ledger moves read by id (4 joins, 12 spellings); 102 respelled episodes not read; 3 of V3's 28 pairs not itemised
OPEN CALLS: L1 / L3 pipeline half / L6 out of this mode: agreed; switching on stays the operator's; closing #2056 the operator's
DOUBTED NUMBER: 102 vs 110 respelled episodes; trace by id before any switch-on (not blocking a switched-off ship)
VERDICT: GO WITH CONDITIONS  shipped switched off with the default path unchanged on the snapshot; val_v7 misses honestly stated
```

Conditions, each answered:

1. `make ci-fast` on the committed tree: **open**, run once with the arc's other changes before
   the PR (arc list, "Before the branch's PR").
2. `run_relative` named as the default-path change in section 7: **done**.
3. Section 8 and Not covered filled: **done**.
4. The 102 vs 110 respelled episodes traced by id: **open, before any switch-on change** (it is
   not written by this deploy).

Verdict: GO WITH CONDITIONS

- **Condition 1:** `make ci-fast` green on the committed tree before the branch's PR.
- **Condition 2:** before switching the mode on, the respell's episode set traced against the
  prototype's by id, and the switch-on measured again on a fresh snapshot.

### 8b. Conditions after the verdict (builder, 2026-11-04)

- **Condition 1, done:** `make ci-fast` ran on `8e8daac8` (the build plus 0043's rebuilt outcome)
  and ended `MAKE_EXIT=0`. The only commits after it are docs.
- **Condition 2, first half done:** the prototype was run again on the pre-build tree (`5f2a8baf`)
  and the build on `8e8daac8`, each on its own fresh copy of snapshot-prod-20261103. Each copy was
  then diffed against the snapshot (scratchpad `l_trace.sh`, `l_trace_diff.py`). The prototype
  wrote 110 episodes (1,009 files) and the build wrote 102 (980), as before. The build writes no
  episode the prototype does not. The 8 written only by the prototype:
  - **7 fold an unplaced listing into the placed voice of the same person.** `person:dara-obrey`
    in 5 BBC episodes (feed `rss_podcasts.files.bbci.co.uk_d95134c7`) is listed but unplaced
    beside the placed host whose spelling the publisher gives. The others are
    `person:jennifer-li` in 1 a16z episode (`rss_feeds.simplecast.com_e739a158`, run
    `3d25717e`), beside a placed guest with the heard spelling, and `person:nicolas-cerisier` in
    1 episode (`rss_feeds.megaphone.fm_370fb395`, run `7096bc13`). In that one the placed
    guest's heard spelling also takes the publisher's. The prototype's process-wide
    `_same_person` patch made the roster's own respell treat V2's pairs as one person. The build
    keeps V2's pairs inside the display rule, so these listed-but-unheard duplicates stay.
    That is arc item M4's shape, not a tier L loss.
  - **1 is a prototype defect the build avoids.** In `rss_rss.iono.fm_5cbbfd48` (run `c34e9535`)
    the prototype gave both placed voices `unplaced_*` ids and cut the first voice's surname. The
    build leaves the episode as stored.
  - The second half (the switch-on measured again on a fresh snapshot) is still open. It belongs
    to the switch-on change.

## 9. Operator review

- **Scope call:** L1 and L3's pipeline half move out of tier L into their own forward changes
  (section 1).

## 10. Before apply / deploy

_TODO_

## 11. After apply / deploy

_TODO_

## 12. Retrospective

_TODO_

## Not covered

- **The gain on unseen data is zero:** val_v7 does not move; the evidence for the mode is dev_v1
  (the design set, one group) and the ledger moves read by id on the snapshot.
- The 102 respelled episodes' files were not read one by one. The 8 episodes between the
  prototype's 110 and the build's 102 were traced by id (8b). `person:jennifer-li` beside the
  placed guest's heard spelling was not checked against the audio as one person.
- 3 of V3's 28 candidate pairs are not itemised; the 21 that joined nothing are listed by kind,
  not each read.
- V1 drops 1,063 `known_hosts` name-episodes; 4 are named by id, the rest counted. A V1 drop
  also loosens the publisher-list guard (b); blocked joins stayed 40 on the snapshot, but on a
  corpus with more ASR `known_hosts` a guard that held could stop holding.
- V2's span finder is English-capitalisation based; non-Latin scripts and lowercase-name
  languages give it nothing.
- L1, L3's pipeline half and L6 are not built (section 1, 4b).
- No gate report: a ledger mode has no migration; the measurement is section 7's table and the
  scratchpad scripts, which are not committed.
- `make ci-fast` ran before the branch's rebase onto PR B. It runs again after the rebase (arc
  list, "Before the branch's PR").
