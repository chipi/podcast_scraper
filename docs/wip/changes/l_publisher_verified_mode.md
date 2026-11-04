---
id: l_publisher_verified_mode
class: full
status: open
issue: none (ingestion arc, tier L: L2, L3 ledger side, L4, L5, L6; #2056)
touches:
  - src/podcast_scraper/identity/ledger.py
  - src/podcast_scraper/identity/guards.py
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
  `person:sam-brazell`, `person:sarah-gonzales`, `person:yasmin-tayag` (5 of them had no spelling
  before: tie or plurality). 6 keep their spelling and change only the evidence label (publisher
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

## 5. Acceptance

_TODO_

## 6. Build

_TODO_

## 7. Measure

_TODO_

## 8. Verdict

_TODO_

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

_TODO_
